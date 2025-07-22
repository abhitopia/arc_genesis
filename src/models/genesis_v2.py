import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ..modules.unet import UNet, ConvGNReLU
from ..modules.masks_stickbreaking import StickBreakingSegmentation

class GenesisV2Config:
    K_steps: int = 5
    in_chnls: int = 3
    img_size: int = 64
    feat_dim: int = 64
    kernel: str = 'gaussian'

class GenesisV2(nn.Module):
    def __init__(self, config: GenesisV2Config):
        super(GenesisV2, self).__init__()
        self.config = config
        self.encoder = nn.Sequential(UNet(in_chnls=config.in_chnls, 
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

    def forward(self, x, max_steps=None, dynamic=False):
        """
        x: [B, C, H, W]  # C = 3 for RGB input channels
        """
        x_enc = self.encoder(x) # [B, feat_dim, H, W]

        x_seg = self.seg_head(x_enc) # [B, feat_dim, H, W]
        log_masks, log_scopes = self.segmenter(x_seg, max_steps=max_steps, dynamic=dynamic) # 2x [B, K_steps, H, W]
        
        # Vectorized object feature extraction and latent computation
        mu_k, sigma_k, z_k = self.extract_object_features(x_enc, log_masks)
        
        return log_masks, log_scopes, mu_k, sigma_k, z_k
    
    def extract_object_features(self, enc_feat, log_masks):
        """
        Vectorized computation of object features and latents
        
        Args:
            enc_feat: [B, F, H, W] - encoded features
            log_masks: [B, K, H, W] - log probability masks
            
        Returns:
            mu_k: [B, K, F] - means for each object 
            sigma_k: [B, K, F] - sigmas for each object
            z_k: [B, K, F] - sampled latents for each object
        """
        B, F, H, W = enc_feat.shape
        B, K, H, W = log_masks.shape
        
        # Convert log masks to masks
        masks = log_masks.exp()  # [B, K, H, W]
        
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
        
        return mu, sigma, z


if __name__ == "__main__":
    config = GenesisV2Config()
    model = GenesisV2(config)
    print(model)

    # Test the model
    x = torch.randn(1, 3, 64, 64)
    print(x.shape)
    log_masks, log_scopes, mu_k, sigma_k, z_k = model(x)
    print(log_masks.shape)
    print(log_scopes.shape)
    print(mu_k.shape)
    print(sigma_k.shape)
    print(z_k.shape)
    import ipdb; ipdb.set_trace()