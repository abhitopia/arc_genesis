"""
GECO: Generalized ELBO Constrained Optimization

This module implements the GECO algorithm for automatically balancing reconstruction
and KL terms in VAE training by dynamically adjusting the KL weight (β) to maintain
a target reconstruction error.

References:
    - "Taming VAEs" (Rezende et al., 2018)
    - https://arxiv.org/abs/1810.00597

Author: Adapted for GenesisV2 training
"""

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn


class GECO(nn.Module):
    """
    GECO (Generalized ELBO Constrained Optimization) implementation.
    
    GECO automatically adjusts the KL weight (β) in VAE training to maintain a target
    reconstruction error. Instead of using a fixed β, GECO dynamically updates β
    to keep the reconstruction error close to a desired goal.
    
    The algorithm works by:
    1. Maintaining an exponential moving average (EMA) of the reconstruction error
    2. Computing the constraint violation: (goal - error_ema)
    3. Updating β using: β(t+1) = β(t) * exp(λ * constraint)
    
    This ensures that:
    - If reconstruction error > goal: β decreases (focus more on reconstruction)
    - If reconstruction error < goal: β increases (focus more on regularization)
    
    Args:
        goal (float): Target reconstruction error (usually per pixel/channel)
        step_size (float): GECO learning rate (λ) controlling β update speed
        alpha (float, optional): EMA momentum for reconstruction error. Higher values
            provide more smoothing. Defaults to 0.99.
        beta_init (float, optional): Initial β value. Defaults to 1.0.
        beta_min (float, optional): Minimum β value to prevent β → 0. Defaults to 1e-10.
        beta_max (float, optional): Maximum β value to prevent β → ∞. Defaults to 1e10.
        speedup (Optional[float], optional): Speedup factor when constraint is satisfied
            (error < goal). When set, multiplies step_size by this factor for faster
            β reduction. Defaults to None.
        
    Example:
        >>> # For 64x64 RGB images with goal of 0.5655 per pixel/channel
        >>> num_elements = 3 * 64 * 64  # pixels * channels
        >>> geco_goal = 0.5655 * num_elements
        >>> geco = GECO(goal=geco_goal, step_size=1e-5)
        >>> 
        >>> # In training loop
        >>> loss = geco.loss(reconstruction_error, kl_divergence)
        >>> loss.backward()
    """
    
    def __init__(
        self,
        goal: float,
        step_size: float,
        alpha: float = 0.99,
        beta_init: float = 1.0,
        beta_min: float = 1e-10,
        beta_max: float = 1e10,
        speedup: Optional[float] = None
    ):
        super().__init__()
        
        # Validate inputs
        self._validate_inputs(goal, step_size, alpha, beta_init, beta_min, beta_max)
        
        # Target reconstruction error
        self.goal = goal
        
        # GECO hyperparameters
        self.step_size = step_size
        self.alpha = alpha
        self.speedup = speedup
        
        # β bounds
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Initialize β as a parameter (will be moved to correct device automatically)
        self.register_buffer('beta', torch.tensor(beta_init, dtype=torch.float32))
        
        # Reconstruction error EMA (initialized to None, set on first call)
        self.register_buffer('err_ema', torch.tensor(0.0, dtype=torch.float32))
        self._err_ema_initialized = False
        
        # Track statistics for monitoring
        self.register_buffer('_num_updates', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_constraint_history', torch.zeros(100, dtype=torch.float32))
        self._history_idx = 0
    
    def _validate_inputs(
        self,
        goal: float,
        step_size: float,
        alpha: float,
        beta_init: float,
        beta_min: float,
        beta_max: float
    ) -> None:
        """Validate GECO hyperparameters."""
        if goal <= 0:
            raise ValueError(f"goal must be positive, got {goal}")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if beta_init <= 0:
            raise ValueError(f"beta_init must be positive, got {beta_init}")
        if beta_min <= 0:
            raise ValueError(f"beta_min must be positive, got {beta_min}")
        if beta_max <= beta_min:
            raise ValueError(f"beta_max ({beta_max}) must be > beta_min ({beta_min})")
    
    def loss(
        self,
        reconstruction_error: torch.Tensor,
        kl_divergence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GECO loss and update β.
        
        This method:
        1. Computes the current loss using the current β
        2. Updates the reconstruction error EMA
        3. Updates β based on the constraint violation
        
        Args:
            reconstruction_error (torch.Tensor): Current reconstruction error (scalar)
            kl_divergence (torch.Tensor): Current KL divergence (scalar)
            
        Returns:
            torch.Tensor: Total loss (reconstruction_error + β * kl_divergence)
        """
        # Ensure inputs are scalars
        if reconstruction_error.numel() != 1:
            raise ValueError(f"reconstruction_error must be scalar, got shape {reconstruction_error.shape}")
        if kl_divergence.numel() != 1:
            raise ValueError(f"kl_divergence must be scalar, got shape {kl_divergence.shape}")
        
        # Compute current loss with current β
        current_loss = reconstruction_error + self.beta * kl_divergence
        
        # Update β (no gradients needed for this update)
        with torch.no_grad():
            self._update_beta(reconstruction_error)
        
        return current_loss
    
    def _update_beta(self, reconstruction_error: torch.Tensor) -> None:
        """
        Update β based on reconstruction error constraint.
        
        Args:
            reconstruction_error (torch.Tensor): Current reconstruction error
        """
        # Update reconstruction error EMA
        if not self._err_ema_initialized:
            self.err_ema.copy_(reconstruction_error)
            self._err_ema_initialized = True
        else:
            self.err_ema.mul_(self.alpha).add_(reconstruction_error, alpha=(1.0 - self.alpha))
        
        # Compute constraint violation: positive means error < goal (good)
        constraint = self.goal - self.err_ema
        
        # Determine step size (with optional speedup)
        current_step_size = self.step_size
        if self.speedup is not None and constraint > 0:
            # Error is below goal, speed up β reduction
            current_step_size *= self.speedup
        
        # Update β: β(t+1) = β(t) * exp(λ * constraint)
        update_factor = torch.exp(current_step_size * constraint)
        self.beta.mul_(update_factor).clamp_(self.beta_min, self.beta_max)
        
        # Update statistics
        self._num_updates += 1
        self._constraint_history[self._history_idx] = constraint.item()
        self._history_idx = (self._history_idx + 1) % 100
    
    def get_stats(self) -> dict:
        """
        Get current GECO statistics for monitoring.
        
        Returns:
            dict: Dictionary containing current GECO state and statistics
        """
        # Compute constraint statistics from history
        valid_history = self._constraint_history[:min(self._num_updates.item(), 100)]
        
        stats = {
            'beta': self.beta.item(),
            'err_ema': self.err_ema.item() if self._err_ema_initialized else 0.0,
            'goal': self.goal,
            'constraint': (self.goal - self.err_ema).item() if self._err_ema_initialized else 0.0,
            'num_updates': self._num_updates.item(),
            'constraint_mean': valid_history.mean().item() if len(valid_history) > 0 else 0.0,
            'constraint_std': valid_history.std().item() if len(valid_history) > 1 else 0.0,
            'beta_at_min': self.beta <= self.beta_min * 1.01,  # Small tolerance
            'beta_at_max': self.beta >= self.beta_max * 0.99,  # Small tolerance
        }
        
        return stats
    
    def reset(self) -> None:
        """Reset GECO state (useful for new training runs)."""
        self.err_ema.zero_()
        self._err_ema_initialized = False
        self._num_updates.zero_()
        self._constraint_history.zero_()
        self._history_idx = 0
        # Keep current β value (don't reset to init)
    
    def set_beta(self, beta: float) -> None:
        """
        Manually set β value (useful for debugging or initialization).
        
        Args:
            beta (float): New β value
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta.fill_(beta)


def create_geco_for_image_size(
    img_size: int,
    channels: int = 3,
    goal_per_pixel: float = 0.5655,
    base_step_size: float = 1e-5,
    reference_size: int = 64,
    **kwargs
) -> GECO:
    """
    Create a GECO instance configured for specific image dimensions.
    
    This helper function automatically scales the goal and step size based on image
    resolution, making it easier to transfer hyperparameters across different
    image sizes.
    
    Args:
        img_size (int): Image size (assuming square images)
        channels (int, optional): Number of channels. Defaults to 3.
        goal_per_pixel (float, optional): Target reconstruction error per pixel/channel.
            Defaults to 0.5655.
        base_step_size (float, optional): Base GECO learning rate. Defaults to 1e-5.
        reference_size (int, optional): Reference image size for scaling. Defaults to 64.
        **kwargs: Additional arguments passed to GECO constructor
        
    Returns:
        GECO: Configured GECO instance
        
    Example:
        >>> # For 128x128 RGB images
        >>> geco = create_geco_for_image_size(img_size=128, channels=3)
        >>> 
        >>> # For 64x64 grayscale images
        >>> geco = create_geco_for_image_size(img_size=64, channels=1)
    """
    # Calculate total goal scaled by image dimensions
    num_elements = channels * img_size * img_size
    goal = goal_per_pixel * num_elements
    
    # Scale step size to maintain similar update dynamics across resolutions
    step_size = base_step_size * (reference_size ** 2) / (img_size ** 2)
    
    return GECO(goal=goal, step_size=step_size, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    geco = GECO(goal=100.0, step_size=1e-5)
    
    # Simulate training steps
    for step in range(10):
        recon_err = torch.tensor(95.0 + torch.randn(1).item() * 5)  # ~95 ± 5
        kl_div = torch.tensor(10.0)
        
        loss = geco.loss(recon_err, kl_div)
        print(f"Step {step}: loss={loss.item():.2f}, β={geco.beta.item():.6f}, "
              f"err_ema={geco.err_ema.item():.2f}")
    
    # Print final statistics
    print("\nFinal GECO stats:")
    for key, value in geco.get_stats().items():
        print(f"  {key}: {value}") 