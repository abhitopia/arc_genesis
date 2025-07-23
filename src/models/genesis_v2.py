import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from ..modules.unet import UNet, ConvGNReLU
from ..modules.masks_stickbreaking import StickBreakingSegmentation
from ..modules.latent_decoder import LatentDecoder
from ..modules.losses import normal_mixture_loss, categorical_kl_loss
from ..modules.autoregressive_kl import AutoregressiveKLLoss

class GenesisV2Config:
    K_steps: int = 5
    in_chnls: int = 3
    img_size: int = 64
    feat_dim: int = 64
    kernel: str = 'gaussian'
    broadcast_size: int = 4
    add_coords_every_layer: bool = False
    normal_std: float = 0.7     # std for normal distribution for the mixture model
    lstm_hidden_dim: int = 256    # hidden dimension for autoregressive KL loss LSTM
    detach_recon_masks: bool = True  # whether to detach reconstructed masks in KL loss

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
                            K_steps=config.K_steps,
                            out_channels=8,
                            kernel=config.kernel)
        
        # Object feature extraction heads
        self.feat_head = nn.Sequential(
            ConvGNReLU(config.feat_dim, config.feat_dim, 3, 1, 1),
            nn.Conv2d(config.feat_dim, 2*config.feat_dim, 1))
        
        # Latent heads
        self.z_head = nn.Sequential(
            nn.LayerNorm(2*config.feat_dim),
            nn.Linear(2*config.feat_dim, 2*config.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*config.feat_dim, 2*config.feat_dim))
        
        # Decoder
        self.decoder = LatentDecoder(
            input_channels=config.feat_dim,
            output_channels=self.in_channels + 1, # RGB + alpha
            output_size=config.img_size,
            broadcast_size=config.broadcast_size,
            feat_dim=config.feat_dim,
            add_coords_every_layer=config.add_coords_every_layer)
        
        # Autoregressive KL loss module
        self.latent_kl_loss = AutoregressiveKLLoss(
            latent_dim=config.feat_dim,
            hidden_dim=config.lstm_hidden_dim)
     

    def encode(self, x, max_steps=None, dynamic=False):
        x_enc = self.encoder(x) # [B, feat_dim, H, W]

        x_seg = self.seg_head(x_enc) # [B, feat_dim, H, W]
        masks, scopes = self.segmenter(x_seg, max_steps=max_steps, dynamic=dynamic) # 2x [B, K_steps, H, W]
        
        # Vectorized object feature extraction and latent computation
        z_k, q_z_k = self.compute_latents(x_enc, masks)

        return z_k, q_z_k, masks
    

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
        z_k, q_z_k, masks = self.encode(x, max_steps=max_steps, dynamic=dynamic) # [B, K, F], [B, K, H, W], Normal([B, K, F])
        recon_k, log_alpha_k = self.decode(z_k) # [B, K, C/1, H, W]

        # Reconstruct image by marginalizing over K objects
        recon = (recon_k * log_alpha_k.exp()).sum(dim=1) # [B, C, H, W]

        # Compute mixture loss
        mixture_loss = normal_mixture_loss(x, recon_k, log_alpha_k, std=self.config.normal_std) # [B]

        # Compute KL loss if autoregressive prior
        latent_kl_loss, _ = self.latent_kl_loss(q_z_k, z_k, sum_k=True) # [B]

        # Compute mask KL loss
        mask_kl_loss_val = categorical_kl_loss(
                q_probs=masks, 
                p_probs=log_alpha_k.squeeze(2).exp(),  # [B, K, H, W]
                detach_p=self.config.detach_recon_masks
            ) # [B]

        return {
            'mixture_loss': mixture_loss,
            'latent_kl_loss': latent_kl_loss,
            'mask_kl_loss': mask_kl_loss_val,
            'reconstruction': recon,
            'masks': masks,
            'object_reconstructions': recon_k,
            'log_alpha': log_alpha_k,
            'latents': z_k
        }
    

    
    def compute_latents(self, enc_feat, masks):
        """
        Vectorized computation of object features and latents
        
        Args:
            enc_feat: [B, F, H, W] - encoded features
            masks: [B, K, H, W] - probability masks
            
        Returns:
            z_k: [B, K, F] - sampled latents for each object
            q_z_k: Vectorized Normal distribution [B, K, F] - posterior distribution
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
        
        # Apply z_head to get posterior parameters
        # z_head can handle [B, K, 2*F] directly - applies to last dimension
        z_out = self.z_head(obj_feat_normalized)  # [B, K, 2*F]
        
        # Split into mean and sigma parameters
        mu, sigma_logits = z_out.chunk(2, dim=2)  # Each: [B, K, F]
 
        # Convert logits to positive sigma values using softplus + small epsilon
        sigma = torch.nn.functional.softplus(sigma_logits + 0.5) + 1e-8 # [B, K, F]
        
        # Sample latents using reparameterization trick (vectorized)
        q_z_k = Normal(mu, sigma)  # Vectorized Normal distribution [B, K, F]
        z_k = q_z_k.rsample()  # [B, K, F]
        
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