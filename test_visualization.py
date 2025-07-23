#!/usr/bin/env python3
"""
Test script for the visualization function.
Creates a simple synthetic sample and tests the slot visualization.
"""

import torch
import matplotlib.pyplot as plt
from src.models.genesis_v2 import GenesisV2Config, GenesisV2
from utils.visualisation import make_slot_figure

def create_synthetic_sample():
    """Create a simple synthetic RGB image for testing."""
    # Create a 64x64 RGB image with some simple patterns
    img = torch.zeros(3, 64, 64)
    
    # Red square in top-left
    img[0, 10:30, 10:30] = 1.0
    
    # Green circle in center (approximate)
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
    center_mask = ((x - 32)**2 + (y - 32)**2) < 15**2
    img[1] = center_mask.float() * 0.8
    
    # Blue rectangle in bottom-right
    img[2, 40:55, 40:60] = 0.9
    
    return img

def extract_slot_stats_for_sample(output, sample_idx=0):
    """Extract per-slot statistics for a single sample from model output."""
    B, K = output['masks_k'].shape[:2]
    
    slot_stats = {
        'scopes_k': [],
        'masks_k': [],
        'alpha_k': [],
        'recon_k': []
    }
    
    for k in range(K):
        # Scopes: [B, K, H, W] -> (1, H, W)
        slot_stats['scopes_k'].append(output['scopes_k'][sample_idx, k:k+1])
        
        # Attention masks: [B, K, H, W] -> (1, H, W)
        slot_stats['masks_k'].append(output['masks_k'][sample_idx, k:k+1])
        
        # Decoder masks: [B, K, H, W] -> (1, H, W)
        slot_stats['alpha_k'].append(output['alpha_k'][sample_idx, k:k+1])
        
        # Appearance reconstructions: [B, K, C, H, W] -> (C, H, W)
        slot_stats['recon_k'].append(output['recon_k'][sample_idx, k])
    
    return slot_stats

def test_visualization():
    """Test the visualization function with GenesisV2."""
    print("Testing slot visualization...")
    
    # Create model
    config = GenesisV2Config(
        K_steps=5,
        in_chnls=3,
        img_size=64,
        feat_dim=64,
    )
    model = GenesisV2(config)
    model.eval()
    
    # Create test input
    image = create_synthetic_sample()
    batch = image.unsqueeze(0)  # Add batch dimension: (1, 3, 64, 64)
    
    print(f"Input shape: {batch.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    
    print("Model output keys:", list(output.keys()))
    print(f"Reconstruction shape: {output['recon'].shape}")
    print(f"Attention masks shape: {output['masks_k'].shape}")
    print(f"Decoder masks shape: {output['alpha_k'].shape}")
    print(f"Object reconstructions shape: {output['recon_k'].shape}")
    
    # Extract data for first sample
    slot_stats = extract_slot_stats_for_sample(output, sample_idx=0)
    
    # Create visualization
    fig = make_slot_figure(
        image=image,  # (3, 64, 64)
        recon=output['recon'][0],  # (3, 64, 64) 
        slot_stats=slot_stats,
        max_slots=None,  # use all slots
        figsize_per_slot=(2, 2)
    )
    
    # Save and show
    fig.savefig('test_slot_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'test_slot_visualization.png'")
    
    # Display if possible
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display figure: {e}")
        print("Check the saved 'test_slot_visualization.png' file")
    
    plt.close(fig)
    print("Test completed!")

if __name__ == "__main__":
    test_visualization() 