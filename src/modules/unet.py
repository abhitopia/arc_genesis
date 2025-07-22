import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConvReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding),
            nn.ReLU(inplace=True)
        ) 

class ConvINReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvINReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(nout, affine=True),
            nn.ReLU(inplace=True)
        )

class ConvGNReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.ReLU(inplace=True)
        )

class UNet(nn.Module):
    """
    Optimized U-Net architecture with improved memory efficiency, better interpolation,
    proper output channel mapping, and MLP bottleneck.
    """

    def __init__(self, in_chnls, out_chnls, img_size, num_blocks=None, norm='gn'):
        """
        Initialises an optimized U-Net model with MLP bottleneck.

        Args:
            num_blocks (int): Number of blocks in the encoder/decoder.
                Supported values: 4, 5, 6. ( defaults to int(np.log2(cfg.img_size)-1))
            img_size (int, optional): The size of the input image.
                Defaults to 64.
            in_chnls (int, optional): The number of channels in the input tensor.
                Defaults to 4.
            out_chnls (int, optional): The number of channels in the output
                tensor.
            norm (str, optional): The type of normalization to use.
                Supported values: 'in' (InstanceNorm), 'gn' (GroupNorm),
                anything else for no normalization. Defaults to 'in'.
        """
        super(UNet, self).__init__()

        if num_blocks is None:
            num_blocks = int(np.log2(img_size)-1) # Min img_size is 32, max is 64

        divisor = 2**(num_blocks - 1)
        assert img_size % divisor == 0, (
            f"Image size ({img_size}) must be divisible by 2**(num_blocks - 1), "
            f"which is {divisor} for {num_blocks} blocks."
        )

        c = out_chnls
        # Select the convolutional block type based on normalization
        if norm == 'in':
            conv_block = ConvINReLU
        elif norm == 'gn':
            conv_block = ConvGNReLU
        else:
            conv_block = ConvReLU
            
        # Define channel dimensions for encoder and decoder based on num_blocks
        if num_blocks == 4:
            enc_in = [in_chnls, c, 2*c, 2*c]
            enc_out = [c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c]
            dec_out = [2*c, 2*c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2*c, 2*c]
            enc_out = [c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c, c]
        else:
            raise ValueError(f"Unsupported number of blocks: {num_blocks}")
            
        # Create encoder and decoder layers
        self.down = nn.ModuleList([conv_block(i, o, 3, 1, 1) 
                                  for i, o in zip(enc_in, enc_out)])
        self.up = nn.ModuleList([conv_block(i, o, 3, 1, 1) 
                                for i, o in zip(dec_in, dec_out)])
        
        # Calculate the feature map size at the bottleneck
        self.featuremap_size = img_size // 2**(num_blocks-1)
        
        # MLP bottleneck
        self.bottleneck = nn.Sequential(
            Flatten(),
            nn.Linear(2*c*self.featuremap_size**2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*c*self.featuremap_size**2), nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Store only skip connections (memory optimization)
        skip_connections = []
        
        # Down-sampling path (encoder)
        current = x
        for i, block in enumerate(self.down):
            # Apply convolutional block
            current = block(current)
            # Store activation for skip connection
            skip_connections.append(current)
            # Down-sample for all but the last block
            if i < len(self.down) - 1:
                current = F.interpolate(current, scale_factor=0.5, mode='bilinear', 
                                      align_corners=False)
        
        # MLP bottleneck
        bottleneck_out = self.bottleneck(current)
        current = bottleneck_out.view(batch_size, -1,
                                    self.featuremap_size, self.featuremap_size)
        
        # Up-sampling path (decoder)
        for i, block in enumerate(self.up):
            # Concatenate with skip connection (reverse order)
            skip_idx = len(skip_connections) - 1 - i
            features = torch.cat([current, skip_connections[skip_idx]], dim=1)
            current = block(features)
            # Up-sample for all but the last block
            if i < len(self.up) - 1:
                current = F.interpolate(current, scale_factor=2.0, mode='bilinear', 
                                      align_corners=False)
        
        output = current
        return output

if __name__ == "__main__":
    # Test the optimized U-Net
    print("=== Optimized U-Net with MLP Bottleneck ===")
    model = UNet(in_chnls=4, out_chnls=64, img_size=64)
    
    # Create a random input tensor
    x = torch.randn(1, 4, 64, 64)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    out = model(x)
    print(f"Output shape: {out.shape}")