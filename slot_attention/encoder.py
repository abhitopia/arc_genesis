# encoder.py
import math
import torch
import torch.nn as nn


def _groupnorm(num_ch: int) -> nn.GroupNorm:
    groups = 8 if num_ch >= 8 else 1          # InstanceNorm fallback
    return nn.GroupNorm(groups, num_ch)


class DownsampleEncoder(nn.Module):
    """
    Generic encoder that downsamples an H×W image to `bottleneck_hw`×`bottleneck_hw`
    feature maps using a stack of stride‑2 convolutions.

    Example
    -------
    >>> enc = DownsampleEncoder(in_ch=3, base_ch=32,
    ...                         image_hw=32, bottleneck_hw=8)
    >>> x = torch.randn(4, 3, 32, 32)
    >>> feats = enc(x)                         # (B, C_out, 8, 8)
    """

    def __init__(self,
                 in_ch: int,
                 base_ch: int,
                 image_hw: int,
                 bottleneck_hw: int,
                 num_conv_per_res: int = 1):
        """
        Parameters
        ----------
        in_ch            : channels of the input image (3 for RGB)
        base_ch          : #channels after the first conv; doubled every stage
        image_hw         : input resolution (must be power of 2, e.g. 32, 64)
        bottleneck_hw    : target bottleneck resolution (power of 2, ≤ image_hw)
        num_conv_per_res : extra 3×3 convs **at each resolution** (default 1)
        """
        super().__init__()

        assert (image_hw & (image_hw - 1)) == 0,        "`image_hw` must be power of 2"
        assert (bottleneck_hw & (bottleneck_hw - 1)) == 0, "`bottleneck_hw` must be power of 2"
        assert bottleneck_hw <= image_hw,               "bottleneck must not exceed input size"

        # How many times do we need to halve H and W?
        num_down = int(math.log2(image_hw // bottleneck_hw))

        layers = []
        ch_in = in_ch
        ch = base_ch

        for stage in range(num_down + 1):               # +1 so last stage keeps spatial size
            # ❶ stride‑1 conv(s) at current resolution
            for _ in range(num_conv_per_res):
                layers += [
                    nn.Conv2d(ch_in, ch, 3, 1, 1, bias=False),
                    _groupnorm(ch),
                    nn.ReLU(inplace=True),
                ]
                ch_in = ch

            # ❷ stride‑2 down‑sample, except after final stage
            if stage < num_down:
                layers += [
                    nn.Conv2d(ch, ch, 4, 2, 1, bias=False),  # exact 2× down‑sample
                    _groupnorm(ch),
                    nn.ReLU(inplace=True),
                ]
                # optionally widen channels each down‑step
                ch = min(ch * 2, base_ch * 4)

        self.encoder = nn.Sequential(*layers)
        self.out_channels = ch_in
        self.bottleneck_hw = bottleneck_hw

    # ---------------------------------------------------------------------
    def forward(self, x):
        """
        Returns
        -------
        torch.Tensor with shape (B, C_out, bottleneck_hw, bottleneck_hw)
        """
        return self.encoder(x)
