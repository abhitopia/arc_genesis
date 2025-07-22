import torch
import torch.nn as nn
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

    def forward(self, x):
        """
        x: [B, C, H, W]  # C = 3 for RGB input channels
        """
        x_enc = self.encoder(x) # [B, feat_dim, H, W]

        x_seg = self.seg_head(x_enc) # [B, feat_dim, H, W]
        log_masks, log_scopes = self.segmenter(x_seg) # [B, K_steps, H, W], [B, K_steps, H, W]
        return log_masks, log_scopes
    


if __name__ == "__main__":
    config = GenesisV2Config()
    model = GenesisV2(config)
    print(model)

    # Test the model
    x = torch.randn(1, 3, 64, 64)
    print(x.shape)
    log_masks, log_scopes = model(x)
    print(log_masks.shape)
    print(log_scopes.shape)
    import ipdb; ipdb.set_trace()