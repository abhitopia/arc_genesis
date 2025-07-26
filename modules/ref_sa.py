from __future__ import annotations

import torch
from torch import einsum, nn
from torch.nn import init
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

class MultiHeadSlotAttentionRef(Module):
    def __init__(
        self,
        num_slots,
        dim,
        heads = 4,
        dim_head = 64,
        iters = 3,
        eps = 1e-8,
        hidden_dim = 128
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim_head ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)

        dim_inner = dim_head * heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_k = nn.Linear(dim, dim_inner, bias=False)
        self.to_v = nn.Linear(dim, dim_inner, bias=False)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = nn.Linear(dim_inner, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        # hidden_dim = max(dim, hidden_dim)

        self.norm_mlp = nn.LayerNorm(dim)   
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(
        self,
        inputs,
        num_slots: int | None = None
    ):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        # Reseed to ensure identical random number generation
        torch.manual_seed(0)
        
        mu = repeat(self.slots_mu, '1 1 d -> b s d', b = b, s = n_s)
        sigma = repeat(self.slots_logsigma.exp(), '1 1 d -> b s d', b = b, s = n_s)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        

        k, v = self.to_k(inputs), self.to_v(inputs)
        k, v = map(self.split_heads, (k, v))

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)

            q = self.to_q(slots)
            q = self.split_heads(q)

            dots = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

            attn = dots.softmax(dim = -2)
            attn = F.normalize(attn + self.eps, p = 1, dim = -1)

            updates = einsum('... j d, ... i j -> ... i d', v, attn)
            updates = self.merge_heads(updates)
            updates = self.combine_heads(updates)

            updates, packed_shape = pack([updates], '* d')
            slots_prev, _ = pack([slots_prev], '* d')

            slots = self.gru(updates, slots_prev)

            slots, = unpack(slots, packed_shape, '* d')
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots
    

if __name__ == '__main__':
    from src.modules.mh_slot_attention import MultiHeadSlotAttention, MultiHeadSlotAttentionImplicit
    torch.manual_seed(0)

    B, H, W = 2, 8, 8               # tiny test – 64 tokens
    C_enc   = 32                    # encoder feature dim
    K       = 5                     # slots
    D_slot  = 128                   # total slot dim
    heads   = 4
    d_head  = D_slot // heads
    hidden_dim = 256
    iters = 3

    print("🔬 Testing Multi-Head Slot Attention implementations...")
    print(f"Config: B={B}, K={K}, D={D_slot}, heads={heads}, iters={iters}")
    print("-" * 60)

    # ❶ Reference implementation
    ref_sa = MultiHeadSlotAttentionRef(num_slots=K, dim=D_slot, heads=heads, dim_head=d_head, hidden_dim=hidden_dim).eval()

    # ❷ Basic multi-head implementation
    basic_sa = MultiHeadSlotAttention(num_slots=K, dim=D_slot, heads=heads, iters=iters, hidden_dim=hidden_dim).eval()
    basic_sa.load_state_dict(ref_sa.state_dict(), strict=False)   # weight copy

    # ❸ Implicit gradient implementation (without implicit grads)
    impl_sa = MultiHeadSlotAttentionImplicit(num_slots=K, dim=D_slot, heads=heads, iters=iters, hidden_dim=hidden_dim, implicit_grads=False).eval()
    
    # Create parameter mapping for impl_sa
    ref_state = ref_sa.state_dict()
    impl_state = {}
    param_mapping = {
        'slots_mu': 'mu',
        'slots_logsigma': 'log_sigma', 
        'norm_input': 'norm_in',
        'norm_slots': 'norm_slot',
        'norm_mlp': 'norm_mlp'
    }
    
    for ref_key, ref_value in ref_state.items():
        # Map parameter names
        impl_key = ref_key
        for ref_name, impl_name in param_mapping.items():
            if ref_name in ref_key:
                impl_key = ref_key.replace(ref_name, impl_name)
                break
        impl_state[impl_key] = ref_value
    
    impl_sa.load_state_dict(impl_state, strict=False)

    # ❹ Implicit gradient implementation (with implicit grads)
    impl_grad_sa = MultiHeadSlotAttentionImplicit(num_slots=K, dim=D_slot, heads=heads, iters=iters, hidden_dim=hidden_dim, implicit_grads=True).eval()
    impl_grad_sa.load_state_dict(impl_state, strict=False)

    # input tensor
    x = torch.randn(B, H*W, D_slot)

    with torch.no_grad():
        ref_out = ref_sa(x)
        basic_out = basic_sa(x)
        impl_out = impl_sa(x)
        impl_grad_out = impl_grad_sa(x)

    # Compare all implementations
    tests = [
        ("Reference vs Basic", ref_out, basic_out),
        ("Reference vs Implicit (no grad)", ref_out, impl_out),
        ("Basic vs Implicit (no grad)", basic_out, impl_out),
        ("Implicit (no grad) vs Implicit (with grad)", impl_out, impl_grad_out),
    ]

    print("Results:")
    all_passed = True
    
    for name, out1, out2 in tests:
        diff = (out1 - out2).abs().max().item()
        status = "✅ PASS" if diff < 1e-5 else "❌ FAIL"
        print(f"{name:35} | max|Δ|: {diff:.2e} | {status}")
        
        if name != "Implicit (no grad) vs Implicit (with grad)" and diff >= 1e-5:
            all_passed = False

    print("-" * 60)
    
    # The implicit gradient version should be different from non-implicit versions
    impl_vs_no_impl_diff = (impl_out - impl_grad_out).abs().max().item()
    if impl_vs_no_impl_diff > 1e-5:
        print("✅ Implicit gradient step produces different results (as expected)")
    else:
        print("⚠️  Implicit gradient step produces identical results (unexpected)")

    if all_passed:
        print("🎉 All core implementations match within tolerance!")
    else:
        print("💥 Some implementations diverged beyond tolerance!")
        
    print(f"\nSample output shapes: {ref_out.shape}")
    print(f"Reference sample: {ref_out[0, 0, :5]}")
    print(f"Basic sample:     {basic_out[0, 0, :5]}")
    print(f"Implicit sample:  {impl_out[0, 0, :5]}")