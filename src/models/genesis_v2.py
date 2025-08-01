import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from dataclasses import dataclass
from ..modules.unet import UNet, ConvGNReLU
from ..modules.masks_stickbreaking import StickBreakingSegmentation
from ..modules.latent_decoder import LatentDecoder
from ..modules.losses import normal_mixture_loss, mse_reconstruction_loss, categorical_kl_loss
from ..modules.autoregressive_kl import AutoregressiveKLLoss

@dataclass
class GenesisV2Config:
    K: int = 5  # Number of segmentation steps (unified parameter)
    in_chnls: int = 3
    img_size: int = 64
    feat_dim: int = 64
    kernel: str = 'gaussian'
    broadcast_size: int = 4
    add_coords_every_layer: bool = False
    use_position_embed: bool = False  # Use learnable PositionEmbed instead of raw PixelCoords
    normal_std: float = 0.7     # std for normal distribution for the mixture model
    lstm_hidden_dim: int = 256    # hidden dimension for autoregressive KL loss LSTM
    detach_recon_masks: bool = True  # whether to detach reconstructed masks in KL loss
    use_vae: bool = True  # whether to use VAE (stochastic) latents or deterministic latents
    num_layers: int | None = 4  # Number of layers in LatentDecoder, None = auto (num upsampling stages)

class GenesisV2(nn.Module):
    def __init__(self, config: GenesisV2Config):
        super(GenesisV2, self).__init__()
        self.config = config
        self.in_channels = config.in_chnls
        self.img_size = config.img_size
        self.encoder = nn.Sequential(UNet(in_chnls=self.in_channels, 
                        out_chnls=config.feat_dim, 
                        img_size=config.img_size),
                        nn.ReLU())
        
        # Segment into K masks
        self.seg_head = ConvGNReLU(config.feat_dim, config.feat_dim, 3, 1, 1)
        self.segmenter = StickBreakingSegmentation(
                            inp_channels=config.feat_dim, 
                            img_size=config.img_size, 
                            K_steps=config.K,
                            out_channels=8,
                            kernel=config.kernel)
        
        # Object feature extraction heads
        self.feat_head = nn.Sequential(
            ConvGNReLU(config.feat_dim, config.feat_dim, 3, 1, 1),
            nn.Conv2d(config.feat_dim, 2*config.feat_dim, 1))
        
        # Latent heads - output size depends on whether we use VAE
        z_output_dim = 2 * config.feat_dim if config.use_vae else config.feat_dim
        self.z_head = nn.Sequential(
            nn.LayerNorm(2*config.feat_dim),
            nn.Linear(2*config.feat_dim, 2*config.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*config.feat_dim, z_output_dim))
        
        # Decoder
        self.decoder = LatentDecoder(
            input_channels=config.feat_dim,
            output_channels=self.in_channels + 1, # RGB + alpha
            output_size=config.img_size,
            num_layers=config.num_layers,
            broadcast_size=config.broadcast_size,
            feat_dim=config.feat_dim,
            add_coords_every_layer=config.add_coords_every_layer,
            use_position_embed=config.use_position_embed)
        
        # Autoregressive KL loss module - only when using VAE
        if config.use_vae:
            self.latent_kl_loss = AutoregressiveKLLoss(
                latent_dim=config.feat_dim,
                hidden_dim=config.lstm_hidden_dim)
        else:
            self.latent_kl_loss = None
     

    def encode(self, x, max_steps=None, dynamic=False):
        x_enc = self.encoder(x) # [B, feat_dim, H, W]

        x_seg = self.seg_head(x_enc) # [B, feat_dim, H, W]
        masks_k, scopes_k = self.segmenter(x_seg, max_steps=max_steps, dynamic=dynamic) # 2x [B, K, H, W]
        
        # Vectorized object feature extraction and latent computation
        z_k, q_z_k = self.compute_latents(x_enc, masks_k)

        return z_k, q_z_k, masks_k, scopes_k
    

    def decode(self, z_k):
        # decode latents to RGBA images
        B, K, D = z_k.shape
        z_k_flat = z_k.view(-1, D)  # [B*K, D]
        dec = self.decoder(z_k_flat)  # [B*K, in_channels + 1, H, W]
        dec = dec.view(B, K, -1, self.img_size, self.img_size)  # [B, K, C, H, W]

        # Split into RGB and alpha
        recon_k_logits, alpha_k_logits = dec.split([self.in_channels, 1], dim=2) # [B, K, C, H, W]

        # Ensure that the range of color is [0, 1] for each of K objects
        recon_k = torch.sigmoid(recon_k_logits) # [B, K, C, H, W]

        # Normalize Alpha across K objects
        log_alpha_k = F.log_softmax(alpha_k_logits, dim=1) # [B, K, 1, H, W]

        return recon_k, log_alpha_k


    def forward(self, x, max_steps=None, dynamic=False):
        """
        x: [B, C, H, W]  # C = 3 for RGB input channels
        """
        z_k, q_z_k, masks_k, scopes_k = self.encode(x, max_steps=max_steps, dynamic=dynamic) # [B, K, F], [B, K, H, W], Normal([B, K, F])
        recon_k, log_alpha_k = self.decode(z_k) # [B, K, C/1, H, W]

        # Reconstruct image by marginalizing over K objects
        recon = (recon_k * log_alpha_k.exp()).sum(dim=1) # [B, C, H, W]

        # Compute dual reconstruction losses
        mixture_loss = normal_mixture_loss(x, recon_k, log_alpha_k, std=self.config.normal_std) # [B]
        mse_loss = mse_reconstruction_loss(x, recon_k, log_alpha_k) # [B]
        latent_kl_loss_per_slot = None
        latent_kl_loss = None

        # Compute latent KL loss only if using VAE and have the necessary components
        if q_z_k is not None:
            latent_kl_loss_per_slot, _ = self.latent_kl_loss(q_z_k, z_k, sum_k=False) # [B, K]
            latent_kl_loss = latent_kl_loss_per_slot.sum(dim=1) # [B] - sum over slots

        # Convert decoder's log-alpha to probability space for consistency
        alpha_k = log_alpha_k.squeeze(2).exp() # [B, K, H, W]

        # Compute mask KL loss (always computed as it's cheap)
        mask_kl_loss_val = categorical_kl_loss(
                q_probs=masks_k, 
                p_probs=alpha_k,    # [B, K, H, W]
                detach_p=self.config.detach_recon_masks
            ) # [B]

        return {
            # Losses
            'mixture_loss': mixture_loss,
            'mse_loss': mse_loss,
            'latent_kl_loss': latent_kl_loss,
            'latent_kl_loss_per_slot': latent_kl_loss_per_slot,
            'mask_kl_loss': mask_kl_loss_val,

            # Main Outputs for downstream use
            'recon': recon,                         # [B, C, H, W]
            'latents_k': z_k,                       # [B, K, F]
            'recon_k': recon_k,                     # [B, K, C, H, W]
            'masks_k': masks_k,                     # [B, K, H, W] (Attention Masks, PROBABILITY space)
            'scopes_k': scopes_k,                   # [B, K, H, W] (Attention Scopes, PROBABILITY space)
            'alpha_k': alpha_k,                     # [B, K, H, W] (Decoder Masks, PROBABILITY space)
        }
    

    
    def compute_latents(self, enc_feat, masks):
        """
        Vectorized computation of object features and latents
        
        Args:
            enc_feat: [B, F, H, W] - encoded features
            masks: [B, K, H, W] - probability masks
            
        Returns:
            z_k: [B, K, F] - latents for each object (sampled if use_vae=True, deterministic if False)
            q_z_k: Vectorized Normal distribution [B, K, F] or None - posterior distribution (None if deterministic)
        """
        # masks are already in regular probability space
        
        # Get object features from encoder
        obj_features = self.feat_head(enc_feat)  # [B, 2*F, H, W]
        
        # Expand dimensions for broadcasting: masks [B, K, H, W] -> [B, K, 1, H, W]
        # obj_features [B, 2*F, H, W] -> [B, 1, 2*F, H, W]
        masks_expanded = masks.unsqueeze(2)  # [B, K, 1, H, W]
        obj_features_expanded = obj_features.unsqueeze(1)  # [B, 1, 2*F, H, W]
        
        # Masked feature extraction (vectorized across all K objects)
        masked_features = masks_expanded * obj_features_expanded  # [B, K, 2*F, H, W]
        
        # Sum over spatial dimensions
        obj_feat_sum = masked_features.sum(dim=(3, 4))  # [B, K, 2*F]
        
        # Normalize by mask sum (add small epsilon for numerical stability)
        mask_sum = masks.sum(dim=(2, 3))  # [B, K]
        obj_feat_normalized = obj_feat_sum / (mask_sum.unsqueeze(-1) + 1e-5)  # [B, K, 2*F]
        
        # Apply z_head to get parameters
        z_out = self.z_head(obj_feat_normalized)  # [B, K, output_dim]
        
        if self.config.use_vae:
            # VAE mode: z_head outputs [mu, sigma_logits]
            mu, sigma_logits = z_out.chunk(2, dim=2)  # Each: [B, K, F]
            sigma = torch.nn.functional.softplus(sigma_logits + 0.5) + 1e-8 # [B, K, F]
            q_z_k = Normal(mu, sigma)  # Vectorized Normal distribution [B, K, F]
            z_k = q_z_k.rsample()  # [B, K, F] - stochastic sampling
        else:
            # Deterministic mode: z_head outputs just mu
            z_k = z_out  # [B, K, F] - deterministic latents
            q_z_k = None  # No distribution needed
        
        return z_k, q_z_k


if __name__ == "__main__":
    config = GenesisV2Config()
    model = GenesisV2(config)
    print(model)

    # Test the model
    x = torch.randn(1, 3, 64, 64)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    results = model(x)
    
    print(f"Total loss: {results['loss'].shape}")
    print(f"Mixture loss: {results['mixture_loss'].shape}")
    print(f"Reconstruction: {results['reconstruction'].shape}")
    print(f"Masks: {results['masks'].shape}")
    print(f"Object reconstructions: {results['object_reconstructions'].shape}")
    print(f"Log alpha: {results['log_alpha'].shape}")
    print(f"Latents: {results['latents'].shape}")
    
    if results['kl_losses'] is not None:
        print(f"Number of KL losses: {len(results['kl_losses'])}")
        print(f"KL loss shapes: {[kl.shape for kl in results['kl_losses']]}")
        
    print(f"Total loss value: {results['loss'].item()}")
    print(f"Mixture loss value: {results['mixture_loss'].item()}")