import torch
from torch.distributions.normal import Normal

def normal_mixture_loss(x, recon_k, log_alpha_k, std=0.7):
    """
    Compute mixture model loss for object-centric learning.
    Args:
        x: [B, C, H, W] - input image
        recon_k: [B, K, C, H, W] - object reconstructions  
        log_alpha_k: [B, K, 1, H, W] - log mixing coefficients
        std: float - standard deviation for reconstruction likelihood
        
    Returns:
        loss: reconstruction loss (negative log-likelihood) per batch item
    """
    B, K, C, H, W = recon_k.shape
    
    # Expand input to match object dimension: [B, C, H, W] -> [B, 1, C, H, W]
    x_expanded = x.unsqueeze(1).expand(-1, K, -1, -1, -1)  # [B, K, C, H, W]
    
    # Compute reconstruction likelihood for each object
    p_x_given_k = Normal(recon_k, std)  # [B, K, C, H, W]
    log_p_x_given_k = p_x_given_k.log_prob(x_expanded)  # [B, K, C, H, W]
    
    # Expand alpha to match channels
    # log_alpha_k: [B, K, 1, H, W] -> [B, K, C, H, W]
    log_alpha_expanded = log_alpha_k.expand(-1, -1, C, -1, -1)
    
    # Combine reconstruction likelihood with mixing coefficients
    log_weighted = log_alpha_expanded + log_p_x_given_k  # [B, K, C, H, W]
    
    # Compute mixture likelihood using log-sum-exp for numerical stability
    log_likelihood = torch.logsumexp(log_weighted, dim=1)  # [B, C, H, W]
    
    # Convert to loss (negative log-likelihood)
    loss_per_pixel = -log_likelihood  # [B, C, H, W]
    
    # Sum over spatial and channel dimensions
    return loss_per_pixel.sum(dim=(1, 2, 3))  # [B]