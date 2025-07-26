import torch
from torch import nn
from einops import rearrange          # pip install einops

class MultiHeadSlotAttention(nn.Module):
    """
    • Multi-head flavour of Locatello et al. Slot Attention
    • Fixes:   scale by dim_head, avoid repeat(), fused normalisation
    • Minor speed/memory gains but identical maths
    """
    def __init__(self,
                 num_slots: int,
                 dim: int = 256,                 # total slot dim  (= heads * dim_head)
                 heads: int = 4,
                 iters: int = 3,
                 eps: float = 1e-8,
                 hidden_dim: int | None = None # if None, set to dim
                ):  # 
        super().__init__()

        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.dim = dim
        self.hidden_dim = hidden_dim or dim 

        assert self.dim % heads == 0, 'dim must be divisible by heads'
        self.dim_head = self.dim // heads

        self.scale = self.dim_head ** -0.5      # ★ correct per-head scaling

        # learnable Gaussian for slot initialisation
        self.slots_mu       = nn.Parameter(torch.randn(1, 1, self.dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        # layer norms
        self.norm_input = nn.LayerNorm(self.dim)
        self.norm_slots = nn.LayerNorm(self.dim)
        self.norm_mlp   = nn.LayerNorm(self.dim)

        # q-k-v projections
        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)

        # head helpers
        self.split  = lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads)
        self.merge  = lambda t: rearrange(t, 'b h n d -> b n (h d)')

        # combine-heads linear
        self.combine_heads = nn.Linear(self.dim, self.dim, bias=False)

        # GRU for slot refinement
        self.gru = nn.GRUCell(self.dim, self.dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.dim)
        )

    # -------------------------------------------------------------

    def forward(self, inputs: torch.Tensor, num_slots: int | None = None):
        """
        inputs: [B, N, D]  — N = HxW flattened pixels/features
        returns: slots [B, K, D]
        """
        b, n, d = inputs.shape
        k = num_slots or self.num_slots

        # Reseed to ensure identical random number generation
        torch.manual_seed(0)

        # 1 initial slots ϵ~N(0,1)
        mu     = self.slots_mu      .expand(b, k, -1)
        sigma  = self.slots_logsigma.expand(b, k, -1).exp()
        slots  = mu + sigma * torch.randn_like(mu)

        # 2 pre-normalise input and get K,V
        x = self.norm_input(inputs)
        k_feat = self.split(self.to_k(x))           # [B, H, N, d_h]
        v_feat = self.split(self.to_v(x))

        # 3 iterative refinement
        for _ in range(self.iters):
            slots_prev = slots                      # save for GRU

            q_feat = self.split(self.to_q(self.norm_slots(slots)))  # [B,H,K,d_h]

            dots  = (q_feat @ k_feat.transpose(-1, -2)) * self.scale   # [B,H,K,N]
            attn  = dots.softmax(dim=-2)                               # softmax over K
            attn  = attn / (attn.sum(dim=-1, keepdim=True) + self.eps) # normalise cols

            updates = attn @ v_feat                     # [B,H,K,d_h]
            updates = self.combine_heads(self.merge(updates))  # back to [B,K,D]

            # GRU expects 2-D (batch*K, dim)
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).view(b, k, d)

            slots = slots + self.mlp(self.norm_mlp(slots))    # residual FF

        return slots
    

class MultiHeadSlotAttentionImplicit(nn.Module):
    """Multi-head Slot Attention with optional implicit gradient step.

    Args:
        num_slots (int): number of slots K.
        dim (int): total slot dimensionality D (must be divisible by *heads*).
        heads (int): number of attention heads H.
        iters (int): number of iterative refinement steps T.
        eps (float): numerical stability term for normalisation.
        hidden_dim (int | None): hidden width of the slot MLP (defaults to *dim*).
        implicit_grads (bool): if *True*, performs **one extra, detached**
            refinement step at the end of the forward pass ("implicit
            differentiation" trick), which tends to stabilise training.

    Notes:
        • When *heads = 1* this reduces to the original Slot-Attention.
        • Biases on Q/K/V are omitted - LayerNorm zero-centres features anyway.
    """

    def __init__(self,
                 num_slots: int,
                 dim: int = 256,
                 heads: int = 4,
                 iters: int = 3,
                 eps: float = 1e-8,
                 hidden_dim: int | None = None,
                 implicit_grads: bool = False):
        super().__init__()
        assert dim % heads == 0, '`dim` must be divisible by `heads`'

        self.K = num_slots
        self.T = iters
        self.H = heads
        self.D = dim
        self.d_h = dim // heads
        self.eps = eps
        self.implicit_grads = implicit_grads
        self.hidden_dim = hidden_dim or dim

        self.scale = self.d_h ** -0.5  # correct per‑head √d scaling

        # learnable Gaussian for slot initialisation
        self.mu = nn.Parameter(torch.randn(1, 1, self.D))
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.xavier_uniform_(self.log_sigma)

        # projections
        self.norm_in = nn.LayerNorm(self.D)
        self.norm_slot = nn.LayerNorm(self.D)
        self.norm_mlp = nn.LayerNorm(self.D)

        self.to_q = nn.Linear(self.D, self.D, bias=False)
        self.to_k = nn.Linear(self.D, self.D, bias=False)
        self.to_v = nn.Linear(self.D, self.D, bias=False)
        self.combine_heads = nn.Linear(self.D, self.D, bias=False)

        # helpers for head reshape
        self.split = lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.H)
        self.merge = lambda t: rearrange(t, 'b h n d -> b n (h d)')

        # slot update modules
        self.gru = nn.GRUCell(self.D, self.D)
        self.mlp = nn.Sequential(
            nn.Linear(self.D, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.D)
        )

    # ---------------------------------------------------------------------
    def _attention_step(self, slots: torch.Tensor, k_feat: torch.Tensor,
                         v_feat: torch.Tensor) -> torch.Tensor:
        """Single refinement step (supports autodiff)."""
        b, k, d = slots.shape

        # 1) project queries from current slots (pre‑normed)
        q = self.split(self.to_q(self.norm_slot(slots)))          # [B,H,K,d_h]

        # 2) attention logits & weights
        dots = (q @ k_feat.transpose(-1, -2)) * self.scale        # [B,H,K,N]
        attn = dots.softmax(dim=2)                                # softmax over K
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps) # col‑norm

        # 3) updates: weighted sum of V over pixels
        updates = self.combine_heads(self.merge(attn @ v_feat))   # [B,K,D]

        # 4) GRU + residual MLP
        slots = self.gru(updates.reshape(-1, d),
                         slots.reshape(-1, d)).view(b, k, d)
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, num_slots: int | None = None):
        """Args:
             x: [B, N, D] flattened encoder features.
           Returns:
             slots: [B, K, D] final slot representations.
        """
        B, N, _ = x.shape
        K = num_slots or self.K

        torch.manual_seed(0)

        # initialise slots with learned Gaussian + noise (re‑parameterisation)
        mu = self.mu.expand(B, K, -1)
        sigma = self.log_sigma.exp().expand(B, K, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # pre‑compute key / value projections of inputs
        x_norm = self.norm_in(x)
        k_feat = self.split(self.to_k(x_norm))    # [B,H,N,d_h]
        v_feat = self.split(self.to_v(x_norm))

        # iterative refinement
        for _ in range(self.T):
            slots = self._attention_step(slots, k_feat, v_feat)

        # optional implicit gradient step (detached slots)
        if self.implicit_grads:
            slots = self._attention_step(slots.detach(), k_feat, v_feat)

        return slots
