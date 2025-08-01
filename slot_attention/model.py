import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

# Import LatentDecoder from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.latent_decoder import LatentDecoder, PositionEmbed
from src.modules.mh_slot_attention import MultiHeadSlotAttentionImplicit

from slot_attention.encoder import DownsampleEncoder


def stick_breaking_normalization(mask_logits: torch.Tensor) -> torch.Tensor:
    """Stick-breaking over slot axis in log-space for stability.
        Input:  [B,S,1,H,W] logits
        Output: [B,S,1,H,W] masks that sum to 1
    """
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
    return masks

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
           sequential = False,
           ordered_slots = True,
           implicit_grads = False,
           use_vae = False,
           heads = 1):

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
        self.use_vae = use_vae
        self.heads = heads
        self.sequential = sequential

        if self.sequential:
            assert self.ordered_slots, "Sequential mode requires ordered slots"

        # 1) encoder
        self.encoder = DownsampleEncoder(in_ch=self.in_channels,
                                         base_ch=self.base_ch,
                                         image_hw=self.resolution,
                                         bottleneck_hw=self.bottleneck_hw,
                                         num_conv_per_res=2)
        
         # 2) position embedding that matches the encoder grid (optional)
        if self.use_encoder_pos_embed:
            self.enc_pos_emb = PositionEmbed(im_dim=self.bottleneck_hw,
                                             feat_dim=self.encoder.out_channels)
        else:
            self.enc_pos_emb = None
        
        # 3) projection → Slot‑Attention (map to slot dimensions)
        self.norm_pre_sa = nn.LayerNorm(self.encoder.out_channels)
        self.linear_pre_sa = nn.Linear(self.encoder.out_channels,
                                       self.slot_size)

        # self.norm_layer = nn.LayerNorm(hdim)
        # self.pre_slot_encode = nn.Sequential(
        #                             nn.Linear(hdim, hdim),
        #                             nn.ReLU(),
        #                             nn.Linear(hdim, hdim)
        #                         )

        self.slot_attention = MultiHeadSlotAttentionImplicit(
                            num_slots   = self.num_slots if not self.sequential else 1,
                            dim         = self.slot_size,
                            heads       = self.heads,
                            iters       = num_iters,
                            slot_mlp_dim = slot_mlp_size,
                            implicit_grads=implicit_grads,
                            ordered_slots=self.ordered_slots)

        # VAE components for slot regularization
        if self.use_vae:
            # Small MLP to jointly learn mu and logvar
            self.vae_mlp = nn.Sequential(
                nn.Linear(self.slot_size, 2*self.slot_size),
                nn.ReLU(),
                nn.Linear(2*self.slot_size, 2 * self.slot_size)
            )

        # 4) latent decoder – broadcast starts at **bottleneck_hw**
        self.decoder = LatentDecoder(
                           input_channels  = self.slot_size,             # Use slot_size directly
                           output_channels = 4,                   # RGB+mask
                           output_size     = self.resolution,
                           broadcast_size  = 4,
                           num_layers      = self.decoder_num_layers,
                           add_coords_every_layer=False,
                           use_position_embed=True,
                           feat_dim        = self.slot_size)            # Use slot_size for consistency


    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """B,3,H,W -> B,N,D_slot (flattened spatial features projected to slot dim)."""
        B = imgs.size(0)

        feats = self.encoder(imgs)                              # B,C,h,w
        # print("ENCODER OUT SHAPE:", feats.shape) # expect (B, C, 8, 8)
        
        # Apply position embedding if enabled (before permute, while in B,C,H,W format)
        if self.enc_pos_emb is not None:
            feats = self.enc_pos_emb(feats)
        
        feats = feats.permute(0, 2, 3, 1)                       # B,h,w,C
        
        feats = feats.view(B, -1, feats.size(-1))
        feats = self.linear_pre_sa(self.norm_pre_sa(feats))
        return feats
    
    def decode(self, slots: torch.Tensor) -> torch.Tensor:
        """Decode S slots to RGB and mask logits.
           Input:  slots  [B,S,D]
           Output: recon  [B,S,3,H,W], mask_logits [B,S,1,H,W]
        """        
        B, S, D = slots.shape
        slots = slots.reshape(-1, D)   # (B*S,D)
        dec = self.decoder(slots)
        dec = dec.view(B, S, 4, *dec.shape[-2:])
        recon, mask_logits = torch.split(dec, [3, 1], dim=2)
        return recon, mask_logits

    def forward_parallel(self, imgs):
        feats = self.encode(imgs)
        slots = self.slot_attention(feats)                      # B, K, D
        
        # ----- KL regularizer (VAE) --------------------------------------
        kl_loss = torch.tensor(0.0, device=slots.device)
        if self.use_vae:
            vae_output = self.vae_mlp(slots)                    # B, K, 2*D
            mu_post, logvar = torch.chunk(vae_output, 2, dim=-1)  # B, K, D each
            # KL divergence from N(0,I) prior: -0.5 * sum(1 + log(σ²) - μ² - σ²)

            # Sum over slots and latent dimensions, then mean over batch
            kl_loss = -0.5 * (1 + logvar - mu_post.pow(2) - logvar.exp()).sum(dim=(1, 2)).mean(dim=0)
        
        # ----- decode ----------------------------------------------------
        recon, mask_logits = self.decode(slots)
        
        # Choose normalization based on ordered_slots setting
        if self.ordered_slots:
            masks = stick_breaking_normalization(mask_logits)
        else:
            # Standard softmax normalization for unordered slots
            masks = torch.softmax(mask_logits, dim=1)
        
        recon_combined = (recon * masks).sum(dim=1)
        # No scopes in parallel mode
        scopes = None
        return recon_combined, recon, masks, mask_logits, scopes, slots, kl_loss


    def forward_sequential(self, imgs: torch.Tensor) -> torch.Tensor:
        B, _, H, W = imgs.shape

        # ================== sequential (MONet-style) path ==================
        K = self.num_slots

        # keep log_scope in float32 for stability (esp. under AMP)
        log_scope = torch.zeros(B, 1, H, W, device=imgs.device, dtype=torch.float32)

        recons = []
        masks  = []
        mask_logits_list = []
        scopes_list = []
        slots_list = []
        kl_terms = []

        for t in range(K):
            # 0) Scope the image for this step: scope = exp(log_scope) (clamped for AMP)
            scope = log_scope.clamp_min(-30.0).exp().to(imgs.dtype)   # -30 ~ 9e-14
            scoped = imgs * scope                                     # B,3,H,W
            
            # Store scope for visualization
            scopes_list.append(scope)

            # 1) Encode & one-slot attention
            feats  = self.encode(scoped)                             # B,N,D_slot
            slot_t = self.slot_attention(feats, num_slots=1)     # B,1,D
            slots_list.append(slot_t)

            # optional per-step VAE
            if self.use_vae:
                vae_out = self.vae_mlp(slot_t)                        # B,1,2D
                mu_post, logvar = torch.chunk(vae_out, 2, dim=-1)
                # Store raw KL terms without reduction for later consistency
                kl_terms.append(-0.5 * (1 + logvar - mu_post.pow(2) - logvar.exp()))

            # 2) Decode -> rgb and mask logits
            recon_t, mask_logits_t = self.decode(slot_t)       # recon_t: B,1,3,H,W
            recon_t = recon_t[:, 0]                                   # B,3,H,W
            logits_t = mask_logits_t[:, 0]                            # B,1,H,W

            # 3) Stick-breaking in log-space
            log_alpha    = F.logsigmoid(logits_t)                     # log σ(z)
            log_one_m_a  = F.logsigmoid(-logits_t)                    # log (1-σ(z))

            if t < K - 1:
                log_mask_t = log_scope + log_alpha                    # B,1,H,W
                mask_t = log_mask_t.clamp_min(-30.0).exp().to(imgs.dtype)
                log_scope = log_scope + log_one_m_a                   # update remaining scope
            else:
                # last takes the remainder exactly
                mask_t = log_scope.clamp_min(-30.0).exp().to(imgs.dtype)

            recons.append(recon_t)
            masks.append(mask_t)
            mask_logits_list.append(logits_t)

        recon = torch.stack(recons, dim=1)                            # B,K,3,H,W
        masks = torch.stack(masks,  dim=1)                            # B,K,1,H,W
        mask_logits = torch.stack(mask_logits_list, dim=1)            # B,K,1,H,W
        scopes = torch.stack(scopes_list, dim=1)                      # B,K,1,H,W
        slots = torch.cat(slots_list, dim=1)                          # B,K,D

        if self.use_vae and kl_terms:
            # Concatenate K tensors of shape [B, 1, D] along slot dimension  
            kl_concat = torch.cat(kl_terms, dim=1)  # [B, K, D]
            # Apply same reduction as parallel: sum over slots and latent dims, mean over batch
            kl_loss = kl_concat.sum(dim=(1, 2)).mean(dim=0)
        else:
            kl_loss = torch.tensor(0.0, device=imgs.device)

        recon_combined = (recon * masks).sum(dim=1)                   # B,3,H,W
        return recon_combined, recon, masks, mask_logits, scopes, slots, kl_loss
    

    def forward(self, imgs):
        if self.sequential:
            return self.forward_sequential(imgs)
        else:
            return self.forward_parallel(imgs)