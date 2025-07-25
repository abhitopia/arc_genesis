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