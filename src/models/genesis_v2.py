import torch
import torch.nn as nn

class GenesisV2Config:
    K_steps: int = 10
    img_size: int = 64

class GenesisV2(nn.Module):
    def __init__(self, config: GenesisV2Config):
        super(GenesisV2, self).__init__()
        self.config = config

    def forward(self, x):
        return x
    


if __name__ == "__main__":
    config = GenesisV2Config()
    model = GenesisV2(config)
    print(model)