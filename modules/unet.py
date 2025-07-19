# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.blocks as B


class UNet(nn.Module):
    """
    A U-Net architecture with a configurable number of blocks, normalization
    layers, and a fully-connected bottleneck.
    """

    def __init__(self, num_blocks, img_size=64,
                 filter_start=32, in_chnls=4, out_chnls=1,
                 norm='in'):
        """
        Initialises a U-Net model.

        Args:
            num_blocks (int): Number of blocks in the encoder/decoder.
                Supported values: 4, 5, 6.
            img_size (int, optional): The size of the input image.
                Defaults to 64.
            filter_start (int, optional): The number of filters in the first
                convolutional layer. Defaults to 32.
            in_chnls (int, optional): The number of channels in the input tensor.
                Defaults to 4.
            out_chnls (int, optional): The number of channels in the output
                tensor. Defaults to 1.
            norm (str, optional): The type of normalization to use.
                Supported values: 'in' (InstanceNorm), 'gn' (GroupNorm),
                anything else for no normalization. Defaults to 'in'.
        """
        super(UNet, self).__init__()
        # TODO(martin): make more general

        divisor = 2**(num_blocks - 1)
        assert img_size % divisor == 0, (
            f"Image size ({img_size}) must be divisible by 2**(num_blocks - 1), "
            f"which is {divisor} for {num_blocks} blocks."
        )

        c = filter_start
        # Select the convolutional block type based on normalization
        if norm == 'in':
            conv_block = B.ConvINReLU
        elif norm == 'gn':
            conv_block = B.ConvGNReLU
        else:
            conv_block = B.ConvReLU
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
        self.down = []
        self.up = []
        # Create encoder layers (down-sampling path)
        # 3x3 kernels, stride 1, padding 1
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        # Create decoder layers (up-sampling path)
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        # Calculate the feature map size at the bottleneck
        self.featuremap_size = img_size // 2**(num_blocks-1)
        # Bottleneck MLP
        self.mlp = nn.Sequential(
            B.Flatten(),
            nn.Linear(2*c*self.featuremap_size**2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*c*self.featuremap_size**2), nn.ReLU()
        )
        # Final 1x1 convolution to map to output channels
        self.final_conv = nn.Conv2d(c, out_chnls, 1)
        self.out_chnls = out_chnls

    def forward(self, x):
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        # Down-sampling path (encoder)
        for i, block in enumerate(self.down):
            # Apply convolutional block
            act = block(x_down[-1])
            # Store activation for skip connection
            skip.append(act)
            # Down-sample for all but the last block
            if i < len(self.down)-1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest')
            x_down.append(act)
        # Bottleneck
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size, self.featuremap_size)
        # Up-sampling path (decoder)
        for i, block in enumerate(self.up):
            # Concatenate feature map from up-sampling path with skip connection
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            # Up-sample for all but the last block
            if i < len(self.up)-1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest')
        # Final 1x1 convolution
        return self.final_conv(x_up), None
