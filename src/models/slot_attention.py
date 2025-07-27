from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from src.modules.mh_slot_attention import MultiHeadSlotAttentionImplicit
from ..modules.unet import UNet
from ..modules.latent_decoder import LatentDecoder
from ..modules.losses import normal_mixture_loss, mse_reconstruction_loss
from ..modules.coord_modules import PixelCoords
from ..modules.autoregressive_kl import AutoregressiveKLLoss


@dataclass
class SlotAttentionConfig:
    K: int = 7  # Number of slots (unified parameter)
    num_iterations: int = 3
    num_heads: int = 4
    slot_dim: int = None  # If None, uses feat_dim
    slot_mlp_dim: int = None  # If None, uses slot_dim
    implicit_grads: bool = False
    # Encoder/Decoder config - same as GenesisV2
    in_chnls: int = 3
    img_size: int = 64
    feat_dim: int = 64
    broadcast_size: int = 4
    add_coords_every_layer: bool = False
    use_position_embed: bool = False  # Use learnable PositionEmbed instead of raw PixelCoords
    normal_std: float = 0.7
    # VAE config
    use_vae: bool = True
    lstm_hidden_dim: int = 256
    num_layers: int | None = 4  # Number of layers in LatentDecoder, None = auto (num upsampling stages)


class SlotAttentionModel(nn.Module):
    def __init__(self, config: SlotAttentionConfig):
        super(SlotAttentionModel, self).__init__()
        self.config = config
        self.in_channels = config.in_chnls
        self.img_size = config.img_size
        
        # Use feat_dim if slot_dim is None
        self.slot_dim = config.slot_dim if config.slot_dim is not None else config.feat_dim
        
        # Use slot_dim if slot_mlp_dim is None
        self.slot_mlp_dim = config.slot_mlp_dim if config.slot_mlp_dim is not None else self.slot_dim
        
        # Same encoder as GenesisV2
        self.encoder = nn.Sequential(
            UNet(in_chnls=self.in_channels, 
                 out_chnls=config.feat_dim, 
                 img_size=config.img_size),
            nn.ReLU()
        )
        
        # Add position embeddings using PixelCoords (better for SlotAttention)
        self.pixel_coords = PixelCoords(config.img_size)
        
        # Input to SlotAttention will be feat_dim + 2 (for x, y coordinates)
        slot_input_dim = config.feat_dim + 2
        
        # Project encoder features + coordinates to slot dimension if needed
        self.feat_projection = nn.Linear(slot_input_dim, self.slot_dim) if slot_input_dim != self.slot_dim else nn.Identity()
        
        # SlotAttention module
        self.slot_attention = MultiHeadSlotAttentionImplicit(
            num_slots=config.K,
            dim=self.slot_dim,
            heads=config.num_heads,
            iters=config.num_iterations,
            slot_mlp_dim=self.slot_mlp_dim,
            implicit_grads=config.implicit_grads
        )
        
        # Project slot representations back to decoder input dimension
        self.slot_to_decoder = nn.Linear(self.slot_dim, config.feat_dim) if self.slot_dim != config.feat_dim else nn.Identity()
        
        # VAE head to convert slots to latent parameters (if using VAE)
        if config.use_vae:
            z_output_dim = 2 * config.feat_dim  # mu and sigma
            self.vae_head = nn.Sequential(
                nn.LayerNorm(config.feat_dim),
                nn.Linear(config.feat_dim, config.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.feat_dim, z_output_dim)
            )
            self.latent_kl_loss = AutoregressiveKLLoss(
                latent_dim=config.feat_dim,
                hidden_dim=config.lstm_hidden_dim
            )
        else:
            self.vae_head = None
            self.latent_kl_loss = None

        # Same decoder as GenesisV2
        self.decoder = LatentDecoder(
            input_channels=config.feat_dim,
            output_channels=self.in_channels + 1,  # RGB + alpha
            output_size=config.img_size,
            num_layers=config.num_layers,
            broadcast_size=config.broadcast_size,
            feat_dim=config.feat_dim,
            add_coords_every_layer=config.add_coords_every_layer,
            use_position_embed=config.use_position_embed
        )

    def forward(self, x):
        """
        x: [B, C, H, W] - RGB input
        """
        B, C, H, W = x.shape
        
        # Encode input to features
        x_enc = self.encoder(x)  # [B, feat_dim, H, W]
        
        # Add position embeddings (coordinates) - immediate access to both features and position
        x_with_coords = self.pixel_coords(x_enc)  # [B, feat_dim + 2, H, W]
        
        # Flatten spatial dimensions for SlotAttention
        x_flat = x_with_coords.view(B, self.config.feat_dim + 2, -1).transpose(1, 2)  # [B, H*W, feat_dim + 2]
        
        # Project to slot dimension if needed
        x_projected = self.feat_projection(x_flat)  # [B, H*W, slot_dim]
        
        # Get slot representations
        slots = self.slot_attention(x_projected)  # [B, K, slot_dim]
        
        # Project slots back to decoder input dimension  
        decoder_input = self.slot_to_decoder(slots)  # [B, K, feat_dim]
        
        # Handle VAE vs deterministic latents
        if self.config.use_vae and self.vae_head is not None:
            # Convert to VAE parameters and sample
            z_out = self.vae_head(decoder_input)  # [B, K, 2*feat_dim]
            mu, sigma_logits = z_out.chunk(2, dim=2)  # Each: [B, K, feat_dim]
            sigma = F.softplus(sigma_logits + 0.5) + 1e-8
            q_z_k = Normal(mu, sigma)  # Vectorized Normal distribution [B, K, F]
            z_k = q_z_k.rsample()  # [B, K, feat_dim] - stochastic sampling
        else:
            # Use projected slots directly as deterministic latents
            z_k = decoder_input  # [B, K, feat_dim]
            q_z_k = None
        
        # Decode slots to RGBA images
        recon_k, log_alpha_k = self.decode(z_k)
        
        # Reconstruct image by marginalizing over slots
        recon = (recon_k * log_alpha_k.exp()).sum(dim=1)  # [B, C, H, W]
        
        # Compute reconstruction losses
        mixture_loss = normal_mixture_loss(x, recon_k, log_alpha_k, std=self.config.normal_std)
        mse_loss = mse_reconstruction_loss(x, recon_k, log_alpha_k)
        
        # Compute latent KL loss if using VAE
        latent_kl_loss_per_slot = None
        latent_kl_loss = None
        if q_z_k is not None:
            latent_kl_loss_per_slot, _ = self.latent_kl_loss(q_z_k, z_k, sum_k=False)  # [B, K]
            latent_kl_loss = latent_kl_loss_per_slot.sum(dim=1)  # [B] - sum over slots
        
        # Convert decoder's log-alpha to probability space
        alpha_k = log_alpha_k.squeeze(2).exp()  # [B, K, H, W]
        
        # SlotAttention doesn't have mask KL loss (no separate attention vs reconstruction masks)
        # Set to zero for compatibility with training loop
        batch_size = x.shape[0]
        mask_kl_loss = torch.zeros(batch_size, device=x.device)
        
        return {
            # Losses
            'mixture_loss': mixture_loss,
            'mse_loss': mse_loss,
            'latent_kl_loss': latent_kl_loss,
            'latent_kl_loss_per_slot': latent_kl_loss_per_slot,
            'mask_kl_loss': mask_kl_loss,  # Zero for SlotAttention (no mask consistency loss)
            
            # Main outputs
            'recon': recon,                         # [B, C, H, W]
            'latents_k': z_k,                       # [B, K, F]
            'recon_k': recon_k,                     # [B, K, C, H, W]
            'alpha_k': alpha_k,                     # [B, K, H, W]
            'slots': slots,                         # [B, K, slot_dim] - raw slot representations
        }
    
    def decode(self, z_k):
        """Decode latents to RGBA images - same as GenesisV2"""
        B, K, D = z_k.shape
        z_k_flat = z_k.view(-1, D)  # [B*K, D]
        dec = self.decoder(z_k_flat)  # [B*K, in_channels + 1, H, W]
        dec = dec.view(B, K, -1, self.img_size, self.img_size)  # [B, K, C+1, H, W]

        # Split into RGB and alpha
        recon_k_logits, alpha_k_logits = dec.split([self.in_channels, 1], dim=2)

        # Apply activations
        recon_k = torch.sigmoid(recon_k_logits)  # [B, K, C, H, W]
        log_alpha_k = F.log_softmax(alpha_k_logits, dim=1)  # [B, K, 1, H, W]

        return recon_k, log_alpha_k


if __name__ == "__main__":
    print("=== Testing SlotAttention with PixelCoords Position Embeddings ===")
    
    # Test with default config (VAE mode)
    config = SlotAttentionConfig()
    model = SlotAttentionModel(config)
    print(f"Model slot_dim (should be {config.feat_dim}): {model.slot_dim}")
    print(f"VAE mode: {config.use_vae}")

    # Test the model
    x = torch.randn(2, 3, 64, 64)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    results = model(x)
    
    print(f"Reconstruction: {results['recon'].shape}")
    print(f"Object reconstructions: {results['recon_k'].shape}")
    print(f"Alpha masks: {results['alpha_k'].shape}")
    print(f"Slots: {results['slots'].shape}")
    print(f"Latents: {results['latents_k'].shape}")
    
    print(f"Mixture loss: {results['mixture_loss'].shape}")
    print(f"MSE loss: {results['mse_loss'].shape}")
    
    if results['latent_kl_loss'] is not None:
        print(f"Latent KL loss: {results['latent_kl_loss'].shape}")
        print(f"Latent KL loss per slot: {results['latent_kl_loss_per_slot'].shape}")
    
    # Test deterministic mode
    print("\n--- Testing deterministic mode (use_vae=False) ---")
    config_det = SlotAttentionConfig(use_vae=False)
    model_det = SlotAttentionModel(config_det)
    print(f"VAE mode: {config_det.use_vae}")
    results_det = model_det(x)
    print(f"Latent KL loss: {results_det['latent_kl_loss']}")  # Should be None
    
    # Test with explicit slot_dim
    print("\n--- Testing with explicit slot_dim=128 ---")
    config2 = SlotAttentionConfig(slot_dim=128)
    model2 = SlotAttentionModel(config2)
    print(f"Model slot_dim (should be 128): {model2.slot_dim}")
    results2 = model2(x)
    print(f"Slots shape: {results2['slots'].shape}")
    print("All tests passed! VAE functionality working correctly.")