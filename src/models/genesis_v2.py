import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from ..modules.unet import UNet, ConvGNReLU
from ..modules.masks_stickbreaking import StickBreakingSegmentation
from ..modules.latent_decoder import LatentDecoder
from ..modules.mixture_model import normal_mixture_loss

class GenesisV2Config:
    K_steps: int = 5
    in_chnls: int = 3
    img_size: int = 64
    feat_dim: int = 64
    kernel: str = 'gaussian'
    broadcast_size: int = 4
    add_coords_every_layer: bool = False
    normal_std: float = 0.7     # std for normal distribution for the mixture model

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
        

    def encode(self, x, max_steps=None, dynamic=False):
        x_enc = self.encoder(x) # [B, feat_dim, H, W]

        x_seg = self.seg_head(x_enc) # [B, feat_dim, H, W]
        masks, scopes = self.segmenter(x_seg, max_steps=max_steps, dynamic=dynamic) # 2x [B, K_steps, H, W]
        
        # Vectorized object feature extraction and latent computation
        z_k, mu_k, sigma_k = self.compute_latents(x_enc, masks)

        return z_k, masks
    

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
        z_k, masks = self.encode(x, max_steps=max_steps, dynamic=dynamic) # [B, K, F], [B, K, H, W]
        recon_k, log_alpha_k = self.decode(z_k) # [B, K, C/1, H, W]

        # Reconstruct image by marginalizing over K objects
        recon = (recon_k * log_alpha_k.exp()).sum(dim=1) # [B, C, H, W]

        # Compute mixture loss
        loss = normal_mixture_loss(x, recon_k, log_alpha_k, std=self.config.normal_std) # [B]

        return loss
    

    
    def compute_latents(self, enc_feat, masks):
        """
        Vectorized computation of object features and latents
        
        Args:
            enc_feat: [B, F, H, W] - encoded features
            masks: [B, K, H, W] - probability masks
            
        Returns:
            mu_k: [B, K, F] - means for each object 
            sigma_k: [B, K, F] - sigmas for each object
            z_k: [B, K, F] - sampled latents for each object
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
        
        # Sample latents using reparameterization trick
        q_z = Normal(mu, sigma)
        z = q_z.rsample()  # [B, K, F]
        
        return z, mu, sigma


if __name__ == "__main__":
    config = GenesisV2Config()
    model = GenesisV2(config)
    print(model)

    # Test the model
    x = torch.randn(1, 3, 64, 64)
    print(x.shape)
    recon, masks, scopes, mu_k, sigma_k, z_k, recon_k, log_alpha_k, loss = model(x)
    print(f"Reconstruction: {recon.shape}")
    print(f"Masks: {masks.shape}")
    print(f"Scopes: {scopes.shape}")
    print(f"Mu: {mu_k.shape}")
    print(f"Sigma: {sigma_k.shape}")
    print(f"Z: {z_k.shape}")
    print(f"Object reconstructions: {recon_k.shape}")
    print(f"Log alpha: {log_alpha_k.shape}")
    print(f"Mixture loss: {loss.shape}")
    print(f"Mixture loss value: {loss.item()}")