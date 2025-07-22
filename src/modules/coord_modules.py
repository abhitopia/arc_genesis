import torch
import torch.nn as nn

class PixelCoords(nn.Module):
    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim),
                                  torch.linspace(-1, 1, im_dim))
        g_1 = g_1.view((1, 1) + g_1.shape)
        g_2 = g_2.view((1, 1) + g_2.shape)
        # Concatenate once and register as buffer so it automatically moves to correct device
        coords = torch.cat((g_1, g_2), dim=1)  # [1, 2, H, W]
        self.register_buffer('coords', coords, persistent=False)
        
    def forward(self, batch_size):
        # Just expand coordinates to batch size and concatenate with input
        coords = self.coords.expand(batch_size, -1, -1, -1)
        return coords


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
        batch_size = x.size(0)
        
        # gate is initialized to 0.0 which means convolution is initiatially disabled
        # and the output relies solely on the pixel coordinates but that should change during training
        out = self.gate * self.conv(x)

        # Get coordinate tensor and add to the last 2 channels of out
        static_coords = self.pixel_coords(batch_size)  # [B, 2, H, W]

        # delta is the last 2 channels of the output which additively modify the static pixel coordinates
        delta = out[:, -2:, :, :] # [B, 2, H, W]

        # Set the last 2 channels of out to the static coordinates plus the delta
        out[:, -2:, :, :] = delta + static_coords
        return out, delta
    

class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        self.pixel_coords = PixelCoords(dim)
    def forward(self, x):
        assert x.dim() == 2, "Input must be a 2D tensor"
        b_sz = x.size(0)

        # Broadcast
        x = x.view(b_sz, -1, 1, 1)
        x = x.expand(-1, -1, self.dim, self.dim)

        # Concatenate with pixel coordinates
        coords = self.pixel_coords(b_sz)
        return torch.cat((x, coords), dim=1)