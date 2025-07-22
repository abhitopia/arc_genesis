import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class AutoregressiveKLLoss(nn.Module):
    """
    Autoregressive KL loss module for computing KL divergence between 
    posterior and autoregressive prior distributions.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        """
        Args:
            latent_dim: Dimension of latent variables
            hidden_dim: Hidden dimension for LSTM
        """
        super(AutoregressiveKLLoss, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Prior LSTM that predicts parameters for autoregressive prior
        self.prior_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            batch_first=True  # Input shape: (batch, seq_len, input_size)
        )
        
        # Linear layer to predict prior parameters (mean and std)
        self.prior_linear = nn.Linear(hidden_dim, 2 * latent_dim)
        
    def to_prior_sigma(self, sigma_raw, sigmoid_bias=4.0, eps=1e-4):
        """
        This parameterisation bounds sigma of a learned prior to [eps, 1+eps].
        The default sigmoid_bias of 4.0 initialises sigma to be close to 1.0.
        The default eps prevents instability as sigma -> 0.
        """
        return torch.sigmoid(sigma_raw + sigmoid_bias) + eps
        
    def forward(self, q_z_k, z_k, sum_k=False):
        """
        Compute autoregressive KL loss between posterior and prior distributions.
        
        Args:
            q_z_k: Vectorized posterior Normal distribution with shape [B, K, F]
            z_k: Tensor of shape [B, K, F] - sampled latents for each component
            sum_k: Whether to sum over the KL divergence for each component
            
        Returns:
            kl_losses: List of K KL divergence tensors, each of shape [B]
            p_z_k: Vectorized prior Normal distribution with shape [B, K, F]
        """
        B, K, F = z_k.shape
        num_steps = K
        
        # -- Build vectorized autoregressive prior --
        if num_steps > 1:
            # Prepare sequence for LSTM: exclude last component
            # Shape: [B, K-1, F]
            z_seq = z_k[:, :-1, :]
            
            # LSTM forward pass
            # lstm_out shape: [B, K-1, hidden_dim]
            lstm_out, _ = self.prior_lstm(z_seq)
            
            # Predict prior parameters for components 2 to K
            # linear_out shape: [B, K-1, 2*F]
            linear_out = self.prior_linear(lstm_out)
            mu_raw, sigma_raw = torch.chunk(linear_out, 2, dim=2)
            
            # Process parameters
            mu_autoregressive = torch.tanh(mu_raw)  # [B, K-1, F] - Keep latent in [-1, 1]
            sigma_autoregressive = self.to_prior_sigma(sigma_raw)  # [B, K-1, F] - Valid sigma range
            
            # Create prior parameters for first component (standard normal)
            mu_first = torch.zeros_like(z_k[:, :1, :])  # [B, 1, F]
            sigma_first = torch.ones_like(z_k[:, :1, :])  # [B, 1, F]
            
            # Concatenate to create vectorized prior parameters
            mu_prior = torch.cat([mu_first, mu_autoregressive], dim=1)  # [B, K, F]
            sigma_prior = torch.cat([sigma_first, sigma_autoregressive], dim=1)  # [B, K, F]
        else:
            # Only one step, use standard normal
            mu_prior = torch.zeros_like(z_k)  # [B, K, F]
            sigma_prior = torch.ones_like(z_k)  # [B, K, F]
        
        # Create vectorized prior distribution
        p_z_k = Normal(mu_prior, sigma_prior)  # [B, K, F]
        
        # -- Compute vectorized KL divergence --
        # Posterior log probabilities: [B, K, F] -> [B, K]
        log_q_all = q_z_k.log_prob(z_k).sum(dim=2)  # [B, K]
        
        # Prior log probabilities: [B, K, F] -> [B, K]  
        log_p_all = p_z_k.log_prob(z_k).sum(dim=2)  # [B, K]
        
        # KL divergence: KL(q||p) = log q - log p
        kl_all = log_q_all - log_p_all  # [B, K]
        
        if sum_k:
            kl_losses = kl_all.sum(dim=1)  # [B]
        else:
            kl_losses = kl_all  # [B, K]
                    
        return kl_losses, p_z_k 