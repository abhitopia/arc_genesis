import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

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

def categorical_kl_loss(q_probs, p_probs, detach_p=True):
    """
    Compute KL divergence between two categorical probability distributions.
    Computes KL(q || p) summed over spatial dimensions.
    
    Args:
        q_probs: [B, K, H, W] - categorical probabilities (distribution q)
        p_probs: [B, K, H, W] - categorical probabilities (distribution p)
        detach_p: bool - whether to detach p_probs from gradients
        
    Returns:
        kl_loss: [B] - KL divergence loss per batch item
    """
    B, K, H, W = q_probs.shape
    
    # Optionally detach p_probs to prevent gradient flow
    if detach_p:
        p_probs = p_probs.detach()
    
    # Ensure probabilities are bounded away from 0 to avoid infinities
    q_probs = torch.clamp(q_probs, min=1e-5)
    p_probs = torch.clamp(p_probs, min=1e-5)
    
    # Reshape for categorical distribution: [B*H*W, K]
    q_categorical = q_probs.permute(0, 2, 3, 1).reshape(-1, K)  # [B*H*W, K]
    p_categorical = p_probs.permute(0, 2, 3, 1).reshape(-1, K)  # [B*H*W, K]
    
    # Create categorical distributions
    q_dist = Categorical(q_categorical)
    p_dist = Categorical(p_categorical)
    
    # Compute KL divergence KL(q || p) per location
    kl_per_location = kl_divergence(q_dist, p_dist)  # [B*H*W]
    
    # Reshape back and sum over spatial dimensions
    kl_per_location = kl_per_location.view(B, H, W)  # [B, H, W]
    kl_loss = kl_per_location.sum(dim=(1, 2))  # [B]
    
    return kl_loss