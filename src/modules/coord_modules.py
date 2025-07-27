import torch
import torch.nn as nn

class PixelCoords(nn.Module):
    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim),
                                  torch.linspace(-1, 1, im_dim),
                                  indexing='ij')
        g_1 = g_1.view((1, 1) + g_1.shape)
        g_2 = g_2.view((1, 1) + g_2.shape)
        # Concatenate once and register as buffer so it automatically moves to correct device
        coords = torch.cat((g_1, g_2), dim=1)  # [1, 2, H, W]
        self.register_buffer('coords', coords, persistent=False)
        
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        # Just expand coordinates to batch size and concatenate with input
        coords = self.coords.expand(batch_size, -1, -1, -1)
        return torch.cat((x, coords), dim=1)


class PositionEmbed(nn.Module):
    """
    Learnable positional embedding that adds coordinate-based features to input.
    Same interface as PixelCoords but adds instead of concatenating.
    """
    def __init__(self, im_dim, feat_dim):
        super().__init__()
        self.dense = nn.Linear(4, feat_dim)
        self.feat_dim = feat_dim
        
        grid = self._build_grid(im_dim)
        self.register_buffer('grid', grid, persistent=False)

    def _build_grid(self, im_dim):
        """Build coordinate grid exactly like reference implementation"""
        # Use [0, 1] range like the reference  
        grid = torch.meshgrid(
            torch.linspace(0.0, 1.0, im_dim),
            torch.linspace(0.0, 1.0, im_dim),
            indexing='ij'
        )
        grid = torch.stack(grid, dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0)  # [1, H, W, 2]
        
        # Add complement coordinates: [x, y, 1-x, 1-y]
        grid = torch.cat([grid, 1.0 - grid], dim=-1)  # [1, H, W, 4]
        return grid

    def forward(self, x):
        """
        Args:
            x: [B, feat_dim, H, W] input tensor (assumes channels == feat_dim)
        Returns:
            x + positional_embedding: [B, feat_dim, H, W]
        """
        batch_size = x.shape[0]
        
        # Get coordinate embedding: [1, H, W, 4] -> [1, H, W, feat_dim]
        pos_embed = self.dense(self.grid)  # [1, H, W, feat_dim]
        
        # Reshape to match input: [1, feat_dim, H, W]
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        
        # Expand to batch size and add to input
        pos_embed = pos_embed.expand(batch_size, -1, -1, -1)
        
        return x + pos_embed


class SemiConv(nn.Module):
    """
    SemiConv is gated convolution layer along with normalized pixel coordinates. 
    Coordinates break the translation invariance of the convolution and embeds positional information.
    """
    def __init__(self, inp_channels, out_channels, img_size, gate_init=0.0):
        super(SemiConv, self).__init__()
        self.conv = nn.Conv2d(inp_channels, out_channels, 1)
        assert out_channels > 2, "out_channels must be greater than 2"

        # Biased to use pixel coordinates initially by setting gate to 0.0
        self.gate = nn.Parameter(torch.tensor(gate_init)) 
        
        # Use PixelCoords module for coordinate generation
        self.pixel_coords = PixelCoords(img_size)
        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, shape [B, inp_channels, H, W]
        Returns:
            out: torch.Tensor, shape [B, out_channels, H, W]
            delta: torch.Tensor, shape [B, 2, H, W]
        """
        batch_size, _, H, W = x.size()
        
        # gate is initialized to 0.0 which means convolution is initiatially disabled
        # and the output relies solely on the pixel coordinates but that should change during training
        out = self.gate * self.conv(x)

        # Create zero tensor with out_channels - 2 so pixel_coords gives us out_channels total
        zero_tensor = torch.zeros(batch_size, self.out_channels - 2, H, W, device=x.device, dtype=x.dtype)
        static_coords_full = self.pixel_coords(zero_tensor)  # [B, out_channels, H, W]

        # delta is the last 2 channels of the output which additively modify the static pixel coordinates
        delta = out[:, -2:, :, :] # [B, 2, H, W]

        # Add static coords directly to out (non-inplace)
        out = out + static_coords_full
        return out, delta