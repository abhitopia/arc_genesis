"""Visualisation helpers for Genesis-V2.

Currently only a single public helper is provided:
    • make_slot_figure(...) –
      Returns a matplotlib Figure that visualises one sample (input image,
      full reconstruction, per-slot masks / reconstructions).

The helper is self-contained so it can be called from notebooks or integrated
later into the Lightning training loop.
"""
from typing import List, Dict, Tuple
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def _to_numpy(img: torch.Tensor) -> "np.ndarray":  # type: ignore
    """Convert (C,H,W) tensor in [0,1] or [0,255] to (H,W,C) float32 [0,1]."""
    if img.max() > 1.01:
        img = img / 255.0
    return img.detach().cpu().float().permute(1, 2, 0).clamp(0, 1).numpy()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def extract_slot_stats_for_sample(
    output: Dict[str, torch.Tensor],
    sample_idx: int = 0
) -> Dict[str, List[torch.Tensor]]:
    """Extract per-slot statistics for a single sample from a batched model output."""
    # alpha_k is always present in both GenesisV2 and SlotAttention
    K = output['alpha_k'].shape[1]

    slot_stats = {
        'scopes_k': [],
        'masks_k': [],
        'alpha_k': [],
        'recon_k': []
    }

    # Check what's available in the output
    has_masks = 'masks_k' in output  # Only GenesisV2 has separate attention masks
    has_scopes = 'scopes_k' in output  # Only GenesisV2 has attention scopes
    has_alpha = 'alpha_k' in output
    has_recon = 'recon_k' in output
    
    if not has_alpha or not has_recon:
        raise KeyError(f"Required keys 'alpha_k' and 'recon_k' not found in model output.")

    for k in range(K):
        # Scopes: [B, K, H, W] -> (1, H, W) [GenesisV2 only]
        if has_scopes:
            slot_stats['scopes_k'].append(output['scopes_k'][sample_idx, k:k+1])
        
        # Attention masks: [B, K, H, W] -> (1, H, W) [GenesisV2 only]
        if has_masks:
            slot_stats['masks_k'].append(output['masks_k'][sample_idx, k:k+1])
        
        # Decoder masks: [B, K, H, W] -> (1, H, W) [Both models]
        slot_stats['alpha_k'].append(output['alpha_k'][sample_idx, k:k+1])
        
        # Appearance reconstructions: [B, K, C, H, W] -> (C, H, W) [Both models]
        slot_stats['recon_k'].append(output['recon_k'][sample_idx, k])
    
    return slot_stats

def make_slot_figure(
    image: torch.Tensor,
    recon: torch.Tensor,
    slot_stats: Dict[str, List[torch.Tensor]],
    max_slots: int | None = None,
    figsize_per_slot: Tuple[int, int] = (2, 2),
) -> Figure:
    """Create a per-slot visualisation figure.

    Parameters
    ----------
    image
        `(C,H,W)` input tensor.
    recon
        `(C,H,W)` full reconstruction.
    slot_stats
        Dictionary containing per-slot tensors. Expected keys:
            - "masks_k"    : list[K] of (1,H,W)   – attention masks (PROBABILITY space) [GenesisV2 only]
            - "alpha_k"    : list[K] of (1,H,W)   – decoder masks (PROBABILITY space) [Both models]
            - "recon_k"    : list[K] of (C,H,W)   – appearance reconstructions [Both models]
            - "scopes_k"   : (optional) list[K] of (1,H,W) – attention scopes (PROBABILITY space) [GenesisV2 only]
    max_slots
        Visualise at most this many slots.
    figsize_per_slot
        Width and height in inches per slot column.

    Returns
    -------
    matplotlib.figure.Figure
        Ready-to-log figure.
    """
    # ------------------------------------------------------------------
    # Determine number of slots - use masks_k if available (GenesisV2), else alpha_k (SlotAttention)
    if slot_stats["masks_k"]:
        total_slots = len(slot_stats["masks_k"])
        has_attention_masks = True
    else:
        total_slots = len(slot_stats["alpha_k"])
        has_attention_masks = False
        
    K = total_slots if max_slots is None else min(max_slots, total_slots)
    
    # Determine number of rows based on available data
    has_scopes = "scopes_k" in slot_stats and slot_stats["scopes_k"]
    
    # Row calculation:
    # - Scopes (if available): +1
    # - Attention masks (if available): +1  
    # - Decoder masks: +1
    # - Reconstructions: +1
    # - Masked reconstructions: +1
    rows = 3  # decoder mask, recon, masked (always present)
    if has_scopes:
        rows += 1
    if has_attention_masks:
        rows += 1
    
    # +1 extra column for global images (input/recon)
    cols = K + 1

    fig_w = figsize_per_slot[0] * cols
    fig_h = figsize_per_slot[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.reshape(rows, cols)

    # --- Column 0: Input & Full Reconstruction ---
    axes[0, 0].imshow(_to_numpy(image)); axes[0, 0].set_title("Input")
    axes[1, 0].imshow(_to_numpy(recon)); axes[1, 0].set_title("Recon")

    # Hide unused axes in the first column for a cleaner look
    for i in range(2, rows):
        axes[i, 0].axis("off")

    # --- Loop over slots for remaining columns ---
    for k in range(K):
        col = k + 1  # Offset because col-0 is global
        
        # --- Prepare data for current slot ---
        dec_mask = slot_stats["alpha_k"][k].squeeze(0) # (H,W), already in prob space
        x_r      = slot_stats["recon_k"][k]                      # (C,H,W)
        masked_recon = x_r * dec_mask.unsqueeze(0)

        # --- Plot per-slot data ---
        row_idx = 0
        
        # Row 0: Attention Scope (if available - GenesisV2 only)
        if has_scopes:
            scope = slot_stats["scopes_k"][k].squeeze(0) # (H,W), already in prob space
            axes[row_idx, col].imshow(scope.cpu(), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col].set_title(f"Scope {k}")
            row_idx += 1
        
        # Row: Attention Mask (if available - GenesisV2 only)
        if has_attention_masks:
            att_mask = slot_stats["masks_k"][k].squeeze(0)          # (H,W), already in prob space
            axes[row_idx, col].imshow(att_mask.cpu(), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col].set_title(f"Att-mask {k}")
            row_idx += 1
        
        # Row: Decoder Mask (always present)
        axes[row_idx, col].imshow(dec_mask.cpu(), cmap="gray", vmin=0, vmax=1)
        axes[row_idx, col].set_title(f"Dec-mask {k}")
        row_idx += 1

        # Row: Appearance Reconstruction
        axes[row_idx, col].imshow(_to_numpy(x_r))
        axes[row_idx, col].set_title(f"Recon {k}")
        row_idx += 1

        # Row: Masked Reconstruction
        axes[row_idx, col].imshow(_to_numpy(masked_recon))
        axes[row_idx, col].set_title(f"Masked {k}")

    # --- Final Touches ---
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout(pad=0.1)
    return fig 