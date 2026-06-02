import torch
import torch.nn.functional as F

def nll_loss(y_mu: torch.Tensor, y_logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative Log-Likelihood for heteroscedastic regression.
    y_mu: (N, 1) predictive mean
    y_logvar: (N, 1) predictive log-variance
    target: (N, 1) true values
    Returns scalar loss.
    """
    # NLL = 0.5 * e^(-logvar) * (target - mu)^2 + 0.5 * logvar
    # To prevent division by zero / exploding gradients, e^(-logvar) is robust.
    precision = torch.exp(-y_logvar)
    squared_error = (target - y_mu) ** 2
    
    loss = 0.5 * precision * squared_error + 0.5 * y_logvar
    return loss.mean()

def kl_divergence_loss(mu_z: torch.Tensor, logvar_z: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence between approximate posterior q(z|S) and prior p(z) = N(0, I).
    mu_z: (B, latent_dim)
    logvar_z: (B, latent_dim)
    Returns scalar loss averaged over batch size.
    """
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_element = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    # Sum over latent dim, mean over batch
    kl_batch = kl_per_element.sum(dim=1).mean()
    return kl_batch

def compute_vae_loss(y_mu, y_logvar, target, mu_z, logvar_z, beta=1e-3):
    """
    Combined Information Bottleneck Loss.
    beta: Weight for the KL divergence (Information Bottleneck constraint).
    """
    nll = nll_loss(y_mu, y_logvar, target)
    kld = kl_divergence_loss(mu_z, logvar_z)
    total_loss = nll + beta * kld
    return total_loss, nll, kld
