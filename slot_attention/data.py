import numpy as np
import torch
import h5py
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Import the VariableDSpritesDataset from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.d_sprites import VariableDSpritesDataset, VariableDSpritesConfig

class DSpritesDataset(Dataset):
    """
    Wrapper around VariableDSpritesDataset for slot attention training.
    """
    
    def __init__(self, config, masks=False, factors=False, d_set='train'):
        super(DSpritesDataset, self).__init__()
        
        self.masks = masks
        self.factors = factors
        self.d_set = d_set.lower()
        self.config = config
        
        # Get the appropriate dataset split
        self.dataset = config.get_dataset(self.d_set)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        # Get sample from VariableDSpritesDataset
        sample = self.dataset[i]
        
        # Convert the interface to match original TetrominoesDataset
        outs = {}
        
        # Convert image from [0, 1] to [-1, 1] range as expected by slot attention
        image = sample['image']  # Shape: (C, H, W) in [0, 1] range
        image_normalized = (image - 0.5) * 2  # Convert to [-1, 1] range
        outs['imgs'] = image_normalized
        
        # Add masks if requested (convert from target)
        if self.masks:
            # sample['target'] has shape (num_colors, H, W)
            # Convert to expected format if needed
            outs['masks'] = sample['target']
            
        # Add factors if requested (placeholder - not available in d_sprites)
        if self.factors:
            # Create dummy factors since d_sprites doesn't have explicit factors
            outs['factors'] = torch.zeros(10)  # Placeholder
            
        return outs
