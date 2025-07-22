import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .coord_modules import PixelCoords


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def _groupnorm(num_ch: int) -> nn.GroupNorm:
    """8 groups if possible, else fall back to 1 (InstanceNorm)."""
    groups = 8 if num_ch >= 8 else 1
    return nn.GroupNorm(groups, num_ch)


class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        assert x.dim() == 2, "Input must be a 2D tensor"
        b_sz = x.size(0)

        # Broadcast
        x = x.view(b_sz, -1, 1, 1)
        x = x.expand(-1, -1, self.dim, self.dim)
        return x

# ---------------------------------------------------------------------------
# LatentDecoder
# ---------------------------------------------------------------------------
class LatentDecoder(nn.Module):
    """
    Parameters
    ----------
    input_channels          : int  - channels of latent input (default = 256)
    output_channels         : int  - channels of final image / logits (default = 4)
    output_size             : int  (power of 2) - desired image size (default = 16)
    broadcast_size          : int  (power of 2) - side length after the BroadcastLayer (default = 4)
    feat_dim                : int | None - hidden width; default = `input_channels`
    add_coords_every_layer  : bool - True → inject coords at every scale (default = False)
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        output_size: int,
        broadcast_size: int = 4,
        feat_dim: int | None = None,
        add_coords_every_layer: bool = False,
    ):
        super().__init__()

        # ---- sanity checks -------------------------------------------------
        assert _is_pow2(broadcast_size), "`broadcast_size` must be a power of 2"
        assert _is_pow2(output_size), "`output_size` must be a power of 2"
        assert broadcast_size <= output_size, "`broadcast_size` > `output_size`"

        feat_dim = feat_dim or input_channels               # hidden width
        num_ups = int(math.log2(output_size // broadcast_size))
 

        # import ipdb; ipdb.set_trace()
        # ---- construct network --------------------------------------------
        layers: list[nn.Module] = [BroadcastLayer(broadcast_size)]
        in_c = input_channels
        cur_size = broadcast_size

        for i in range(num_ups):
            # inject coords: always on first iteration, or every iteration if requested
            if i == 0 or add_coords_every_layer:
                layers.append(PixelCoords(cur_size))
                in_c += 2

            # ConvTranspose2d‑block that doubles H & W
            out_c = feat_dim

            # print(f"layer {i}: in_c: {in_c}, out_c: {out_c}")
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_c,
                        out_c,
                        kernel_size=5,     # receptive field 5 × 5
                        stride=2,          # ←   doubles spatial size
                        padding=2,
                        output_padding=1,  # ensures exact 2× up‑sample
                    ),
                    _groupnorm(out_c),
                    nn.ReLU(inplace=True),
                ]
            )
            in_c = out_c
            cur_size *= 2  # track spatial size

        # 1×1 projection to the requested number of channels
        layers.append(nn.Conv2d(in_c, output_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # # Example 1: old behaviour, coords only once
    # dec16 = LatentDecoder(
    #     broadcast_size=1,
    #     output_size=16,
    #     input_channels=256,
    #     output_channels=4,           # e.g. RGBA
    #     add_coords_every_layer=False
    # )
    # print(dec16(torch.randn(8, 256)).shape)      # ⇒ torch.Size([8, 4, 16, 16])

    # Example 2: coords every scale
    dec64 = LatentDecoder(
        broadcast_size=4,
        output_size=64,
        input_channels=128,
        output_channels=3,           # RGB
        feat_dim=192,
        add_coords_every_layer=True
    )
    x = torch.randn(8, 128)
    print(dec64(x).shape)                           # ⇒ torch.Size([8, 3, 64, 64])
