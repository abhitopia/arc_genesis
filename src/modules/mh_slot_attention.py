import torch
from torch import nn
from einops import rearrange          # pip install einops
import torch.nn.functional as F


class MultiHeadSlotAttentionImplicit(nn.Module):
    """Multi-head Slot Attention with optional implicit gradient step.

    Args:
        num_slots (int): number of slots K.
        dim (int): total slot dimensionality D (must be divisible by *heads*).
        heads (int): number of attention heads H.
        iters (int): number of iterative refinement steps T.
        eps (float): numerical stability term for normalisation.
        slot_mlp_dim (int | None): hidden width of the slot MLP (defaults to *dim*).
        implicit_grads (bool): if *True*, performs **one extra, detached**
            refinement step at the end of the forward pass ("implicit
            differentiation" trick), which tends to stabilise training.
        ordered_slots (bool): if *True*, each slot gets unique mu/sigma parameters
            for better specialization, spaced from -1 to +1 in latent space.

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
                 slot_mlp_dim: int | None = None,
                 implicit_grads: bool = False,
                 ordered_slots: bool = True):
        super().__init__()
        assert dim % heads == 0, '`dim` must be divisible by `heads`'

        self.K = num_slots
        self.T = iters
        self.H = heads
        self.D = dim
        self.d_h = dim // heads
        self.eps = eps
        self.implicit_grads = implicit_grads
        self.slot_mlp_dim = slot_mlp_dim or dim
        self.ordered_slots = ordered_slots

        self.scale = self.d_h ** -0.5  # correct per‑head √d scaling

        # learnable Gaussian for slot initialisation
        if self.ordered_slots:
            # Each slot has unique mu and sigma for better specialization
            # Systematically space slots from -1 to +1 in latent space
            init_range = torch.linspace(-1, 1, self.K).unsqueeze(-1)  # K×1
            self.mu = nn.Parameter(init_range.repeat(1, self.D).unsqueeze(0))  # (1,K,D)
            self.log_sigma = nn.Parameter(torch.full((1, self.K, self.D), -1.0)) # (1,K,D)
            # softplus(-1) ≈ 0.3 stdev
        else:
            # Original shared initialization
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
            nn.Linear(self.D, self.slot_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.slot_mlp_dim, self.D)
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
        attn = dots.softmax(dim=-2)                               # softmax over K
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

        # initialise slots with learned Gaussian + noise (re‑parameterisation)
        if self.ordered_slots:
            # Each slot has unique parameters: (1,K,D) -> (B,K,D)
            mu = self.mu[:, :K, :].expand(B, -1, -1)
            sigma = self.log_sigma[:, :K, :].expand(B, -1, -1)
        else:
            # Original: repeat shared parameters along both batch and slot dimensions
            mu = self.mu.expand(B, K, -1)
            sigma = self.log_sigma.expand(B, K, -1)

        if self.training:
            sigma = F.softplus(sigma) + 1e-5  # Gradient-friendly softplus (as opposed to exp)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = mu

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
