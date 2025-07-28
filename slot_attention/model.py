import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

# Import LatentDecoder from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.latent_decoder import LatentDecoder

from slot_attention.encoder import DownsampleEncoder


class SlotAttention(nn.Module):

    def __init__(self, in_dim, slot_size, num_slots, num_iters, mlp_hdim, 
                                                                epsilon =1e-8, implicit_grads=False, ordered_slots=True):

        super().__init__()

        self.num_slots = num_slots
        self.num_iters = num_iters
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.implicit_grads = implicit_grads
        self.ordered_slots = ordered_slots

        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(in_dim, slot_size, bias=False)
        self.project_v = nn.Linear(in_dim, slot_size, bias=False)

        self.norm_inputs = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)


        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hdim),
            nn.ReLU(),
            nn.Linear(mlp_hdim, slot_size)
            )

        if self.ordered_slots:
            # Each slot has unique mu and sigma for better specialization
            # Systematically space slots from -1 to +1 in latent space
            init_range = torch.linspace(-1, 1, self.num_slots).unsqueeze(-1)  # K×1
            self.slots_mu = nn.Parameter(init_range.repeat(1, self.slot_size).unsqueeze(0))  # (1,K,D)
            self.slots_logsigma = nn.Parameter(torch.full((1, num_slots, slot_size), -1.0)) # (1,K,D)
            # softplus(-1) ≈ 0.3 stdev

        else:
            # Original: all slots share the same mu and sigma
            self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.slot_size)))
            self.slots_logsigma = nn.Parameter(nn.init.xavier_uniform_(torch.ones( 1, self.slot_size)))


    def step(self, slots, k, v, batch_size):

        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.project_q(slots) # shape: [batch_size, num_slots, slot_size]
        scores = (self.slot_size ** -0.5) * torch.matmul(k, q.transpose(2, 1))
        attn = torch.softmax(scores, dim=-1) # shape: [batch_size, num_inputs, num_slots]

        #weighted mean 
        attn = attn + self.epsilon
        attn = attn/ torch.sum(attn, dim=1, keepdim=True) #shape: [batch_size, num_inputs, num_slots]

        updates = torch.matmul(attn.transpose(2, 1), v) #shape: [batch_size, num_slots, slot_size]

        slots = self.gru(updates.reshape(-1, self.slot_size), slots_prev.reshape(-1, self.slot_size))
        slots = slots.reshape(batch_size, self.num_slots, self.slot_size)
        slots = self.norm_mlp(slots)
        #slots = self.mlp(slots)
        slots = slots + self.mlp(slots)

        return slots

    def forward(self, x):

        batch_size, num_inputs, in_dim = x.shape
        x = self.norm_inputs(x)
        k = self.project_k(x) # shape:[batch_size, num_inputs, slot_size]
        v = self.project_v(x) # shape:[batch_size, num_inputs, slot_size]

        if self.ordered_slots:
            # Each slot has unique parameters: (1,K,D) -> (B,K,D)
            mu = self.slots_mu.repeat(batch_size, 1, 1)
            logsigma = self.slots_logsigma.repeat(batch_size, 1, 1)
        else:
            # Original: repeat shared parameters along both batch and slot dimensions
            mu = self.slots_mu.repeat(batch_size, self.num_slots, 1)
            logsigma = self.slots_logsigma.repeat(batch_size, self.num_slots, 1)
        logsigma = F.softplus(logsigma) + 1e-5
        slots_dist = dist.independent.Independent(dist.Normal(loc=mu, scale=logsigma), 1)
        slots = slots_dist.rsample()

        for _ in range(self.num_iters):
            slots = self.step(slots, k, v, batch_size)

        if self.implicit_grads:
            slots = self.step(slots.detach(), k, v, batch_size)

        return slots

class SlotAttentionModel(nn.Module):

    def __init__(self, 
           resolution, 
           num_slots,
           num_iters,
           in_channels =3, 
           base_ch = 32,
           bottleneck_hw = 8,
           slot_size = 64,
           slot_mlp_size = 128,
           decoder_num_layers = 6,
           use_encoder_pos_embed = True,
           ordered_slots = True,
           implicit_grads = False):

        super().__init__()

        self.resolution = resolution
        self.bottleneck_hw = bottleneck_hw
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.decoder_num_layers = decoder_num_layers
        self.base_ch = base_ch
        self.use_encoder_pos_embed = use_encoder_pos_embed
        self.ordered_slots = ordered_slots

        # 1) encoder
        self.encoder = DownsampleEncoder(in_ch=self.in_channels,
                                         base_ch=self.base_ch,
                                         image_hw=self.resolution,
                                         bottleneck_hw=self.bottleneck_hw,
                                         num_conv_per_res=2)
        
         # 2) position embedding that matches the encoder grid (optional)
        if self.use_encoder_pos_embed:
            self.enc_pos_emb = PositionEmbed(self.encoder.out_channels,
                                             (self.bottleneck_hw, self.bottleneck_hw),
                                             device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.enc_pos_emb = None
        
        # 3) projection → Slot‑Attention (identical to your old code)
        self.norm_pre_sa = nn.LayerNorm(self.encoder.out_channels)
        self.linear_pre_sa = nn.Linear(self.encoder.out_channels,
                                       self.encoder.out_channels)

        # self.norm_layer = nn.LayerNorm(hdim)
        # self.pre_slot_encode = nn.Sequential(
        #                             nn.Linear(hdim, hdim),
        #                             nn.ReLU(),
        #                             nn.Linear(hdim, hdim)
        #                         )

        self.slot_attention = SlotAttention(
                            in_dim      = self.encoder.out_channels,
                            slot_size   = self.slot_size,
                            num_slots   = self.num_slots,
                            num_iters   = num_iters,
                            mlp_hdim    = slot_mlp_size,
                            implicit_grads=implicit_grads,
                            ordered_slots=self.ordered_slots)

        # 4) latent decoder – broadcast starts at **bottleneck_hw**
        self.decoder = LatentDecoder(
                           input_channels  = self.slot_size,
                           output_channels = 4,                   # RGB+mask
                           output_size     = self.resolution,
                           broadcast_size  = 4,
                           num_layers      = self.decoder_num_layers,
                           add_coords_every_layer=False,
                           use_position_embed=True,
                           feat_dim        = self.base_ch)


    def forward(self, imgs):
        B = imgs.size(0)

        feats = self.encoder(imgs)                              # B,C,h,w
        # print("ENCODER OUT SHAPE:", feats.shape) # expect (B, C, 8, 8)
        feats = feats.permute(0, 2, 3, 1)                       # B,h,w,C
        
        # Apply position embedding if enabled
        if self.enc_pos_emb is not None:
            feats = self.enc_pos_emb(feats)
        
        feats = feats.view(B, -1, feats.size(-1))
        feats = self.linear_pre_sa(self.norm_pre_sa(feats))

        slots = self.slot_attention(feats)                      # B, K, D
        # ----- decode ----------------------------------------------------
        dec = self.decoder(slots.reshape(-1, slots.size(-1)))
        dec = dec.view(B, self.num_slots, 4, *dec.shape[-2:])

        recon, mask_logits = torch.split(dec, [3, 1], dim=2)
        
        # Choose normalization based on ordered_slots setting
        if self.ordered_slots:
            # Stable stick-breaking normalization using log-space computation
            log_sig  = F.logsigmoid(mask_logits)        # log σ(z_k)
            log_1sig = F.logsigmoid(-mask_logits)       # log (1‑σ(z_k))

            # log cumulative sum of the 'remaining stick'
            # cum_log = [0, log(1‑σ(z0)), log(1‑σ(z0))+log(1‑σ(z1)), …]
            cum_log  = torch.cumsum(log_1sig, dim=1)
            cum_log  = torch.cat([torch.zeros_like(cum_log[:, :1]),   # prepend 0 for k=0
                                  cum_log[:, :-1]], dim=1)

            log_masks = log_sig + cum_log          # log m_k = log σ + Σ_{<k} log(1‑σ)
            masks     = torch.exp(log_masks)       # back to probability space
        else:
            # Standard softmax normalization for unordered slots
            masks = torch.softmax(mask_logits, dim=1)
        
        recon_combined = (recon * masks).sum(dim=1)

        return recon_combined, recon, masks, slots

class PositionEmbed(nn.Module):

    def __init__(self, hdim, resolution, device):
        super().__init__()

        self.dense = nn.Linear(4, hdim)
        self.grid = build_grid(resolution).to(device)

    def forward(self, x):

        grid = self.dense(self.grid)
        return x + grid

def build_grid(resolution):

    grid = torch.meshgrid(*[torch.linspace(0.0, 1.0, r) for r in resolution])
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    grid = torch.cat([grid, 1.0-grid], dim=-1)
    return grid


if __name__ == "__main__":
    from pprint import pprint
    g = build_grid((8, 8))            # use the bottleneck resolution here
    print("grid[:, :, 0] (y‑coord) first row: ", g[0, 0, :, 0].tolist())
    print("grid[:, :, 0] (y‑coord) last  row: ", g[0, -1, :, 0].tolist())
    print("grid[:, :, 1] (x‑coord) first col: ", g[0, :, 0, 1].tolist())
    print("grid[:, :, 1] (x‑coord) last  col: ", g[0, :, -1, 1].tolist())