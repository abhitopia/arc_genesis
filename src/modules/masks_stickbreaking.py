import numpy as np
import torch
import torch.nn as nn
from .coord_modules import SemiConv


class StickBreakingSegmentation(nn.Module):
    """
    Ref: https://arxiv.org/pdf/2104.09958v2 (Genesis-V2 Section 3.1)
    """
    def __init__(self, inp_channels, img_size, K_steps, out_channels=8, kernel='gaussian'):
        super(StickBreakingSegmentation, self).__init__()
        self.K_steps = K_steps
        self.img_size = img_size
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.kernel = kernel

        # Initialise kernel sigma
        if self.kernel == 'laplacian':
            sigma_init = 1.0 / (np.sqrt(K_steps)*np.log(2))
        elif self.kernel == 'gaussian':
            sigma_init = 1.0 / (K_steps*np.log(2))
        elif self.kernel == 'epanechnikov':
            sigma_init = 2.0 / K_steps
        else:
            return ValueError("No valid kernel.")
        
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())
        self.feature_head = SemiConv(inp_channels, out_channels, img_size)


    def forward(self, x, max_steps=None, dynamic=False):
        """
        Args:
            x: torch.Tensor, shape [B, inp_channels, H, W]
        Returns:
            log_masks: torch.Tensor, shape [B, K_steps, H, W]
            log_scopes: torch.Tensor, shape [B, K_steps, H, W]
        """

        if max_steps is None:
            max_steps = self.K_steps

        batch_size, C, H, W = x.shape
        device = x.device
        if dynamic:
            assert batch_size == 1, "Dynamic mode only supports batch size 1"

        feat_out, delta = self.feature_head(x)
        feat_flat = feat_out.view(batch_size, self.out_channels, H*W)  # [B, O, H*W]

        # Sample from uniform to select random pixels as seeds
        rand_pixel = torch.rand(batch_size, 1, H, W, device=device, dtype=feat_out.dtype) # rand_pixel: [B, 1, H, W]
        log_scope = torch.zeros(batch_size, 1, H, W, device=device, dtype=feat_out.dtype)

        log_scopes = [log_scope]
        log_masks = []

        for step in range(max_steps -1):  # -1 because we add the final residual mask at the end
            scope = log_scope.exp()  # [B, 1, H, W]
            pixel_probs = rand_pixel * scope # [B, 1, H, W]
            flat_idx = pixel_probs.view(batch_size, -1).argmax(dim=1) # [B]

            # Find coordinates of max probability pixel
            seed = torch.gather(feat_flat,                   # (B,O,H*W)
                    2,
                    flat_idx.unsqueeze(1)                    # (B, 1)
                    .expand(-1, self.out_channels)           # (B, O)
                    .unsqueeze(-1)                           # (B, O, 1)
                    ).unsqueeze(-1)                          # (B, O, 1, 1)
 
             #  Distance and alpha
            sq_dist = (feat_out - seed).square().sum(dim=1) # [B, H, W]
            if self.kernel == 'laplacian':
                dist = sq_dist.sqrt()
                alpha = torch.exp(- dist / self.log_sigma.exp())
            elif self.kernel == 'gaussian':
                alpha = torch.exp(- sq_dist / self.log_sigma.exp()) # alpha: [B, H, W]
            elif self.kernel == 'epanechnikov':
                alpha = (1 - sq_dist / self.log_sigma.exp()).relu()
            else:
                raise ValueError("No valid kernel.")
            alpha = alpha.unsqueeze(1) # alpha: [B, 1, H, W]
        
            # Clamp mask values to [0.01, 0.99] for numerical stability 
            alpha = alpha + (alpha.clamp(min=0.01, max=0.99) - alpha).detach()

            # Stick breaking update in log space
            log_a = torch.log(alpha)        # log_a: [B, 1, H, W]
            log_neg_a = torch.log1p(-alpha) # ln(1‑α): [B, 1, H, W]
            log_m = log_scope + log_a # log_m: [B, 1, H, W]
            log_scope = log_scope + log_neg_a # log_scope: [B, 1, H, W]
            if dynamic and log_m.exp().sum() < 20: # If the mask is too small, stop the loop
                break # Consequtive masks are by design smaller than previous, 20 ~ 0.5% of the image size

            log_masks.append(log_m)
            log_scopes.append(log_scope)

        # final residual mask
        log_masks.append(log_scopes[-1])

        log_masks = torch.stack(log_masks, dim=1).squeeze(2) # [B, K_steps, H, W]
        log_scopes = torch.stack(log_scopes, dim=1).squeeze(2) # [B, K_steps, H, W]
        
        # Convert back to regular probability space
        masks = log_masks.exp() # [B, K_steps, H, W]
        scopes = log_scopes.exp() # [B, K_steps, H, W]
        
        return masks, scopes


if __name__ == "__main__":
    model = StickBreakingSegmentation(inp_channels=64, img_size=64, K_steps=5)
    x = torch.randn(1, 64, 64, 64)
    out, delta = model(x)
    print(out.shape)
    print(delta.shape)