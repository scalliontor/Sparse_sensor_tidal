import torch
import torch.nn.functional as F

def nll_loss(y_mu: torch.Tensor, y_logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    precision = torch.exp(-y_logvar)
    squared_error = (target - y_mu) ** 2
    loss = 0.5 * precision * squared_error + 0.5 * y_logvar
    return loss.sum() # Sum instead of mean so we can mean over P properly later

def kl_divergence_loss(mu_z: torch.Tensor, logvar_z: torch.Tensor) -> torch.Tensor:
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_element = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    # Sum over latent dim
    kl_batch = kl_per_element.sum(dim=1) # (B,)
    return kl_batch

def compute_ovcno_loss(y_mu, y_logvar, target, mu_z, logvar_z, o_i, d_s, 
                       beta_0=1e-4, beta_1=1e-3, lambda_obs=1.0, margin=0.1):
    """
    Combined Observability-Aware Loss.
    o_i: (B*P, 1) learned observability score [0,1]
    d_s: (B*P, 1) distance from point to nearest sensor
    """
    B_times_P = y_mu.shape[0]
    B = mu_z.shape[0]
    P = B_times_P // B
    
    # 1. NLL Loss
    # We mean over all points
    precision = torch.exp(-y_logvar)
    sq_err = (target - y_mu) ** 2
    nll_pt = 0.5 * precision * sq_err + 0.5 * y_logvar  # (B*P, 1)
    L_nll = nll_pt.mean()

    # 2. Adaptive KL Loss
    # beta_i = beta_0 + beta_1 * (1 - o_i)  => (B*P, 1)
    beta_i = beta_0 + beta_1 * (1.0 - o_i)
    
    kl_b = kl_divergence_loss(mu_z, logvar_z) # (B,)
    kl_bp = kl_b.repeat_interleave(P).view(-1, 1) # (B*P, 1)
    
    L_kl_adaptive = (beta_i * kl_bp).mean()
    
    # 3. Observability Consistency Loss (Ranking)
    # Target: d_s gives the geometrical distance.
    # o_i gives observability proxy.
    # We want predicted std (or variance) to be ranked properly. Or directly o_i ranking. The user paper states:
    # max(0, margin - (sigma_j - sigma_i)) where j is harder than i.
    # Harder usually implies larger distance d_s.
    
    # Subsample pairs to keep memory and computation low.
    num_pairs = min(1000, B_times_P // 2)
    idx1 = torch.randperm(B_times_P)[:num_pairs]
    idx2 = torch.randperm(B_times_P)[:num_pairs]
    
    d1, d2 = d_s[idx1], d_s[idx2]
    sigma = torch.exp(0.5 * y_logvar)
    sig1, sig2 = sigma[idx1], sigma[idx2]
    
    # We only care when point 1 is strictly further than point 2
    mask = (d1 > d2 + 0.1).float() # margin in distance to define "harder"
    
    # If d1 > d2, sig1 should be >= sig2. Thus penalty is max(0, margin - (sig1 - sig2))
    ranking_penalty = F.relu(margin - (sig1 - sig2))
    
    L_obs = (ranking_penalty * mask).sum() / (mask.sum() + 1e-6)
    
    # Optional constraint: tie o_i directly explicitly? The adaptive beta already ties o_i to KL, 
    # and Decoder ties o_i to predicting y_mu and y_logvar. This is inherently sufficient.
    
    total_loss = L_nll + L_kl_adaptive + lambda_obs * L_obs
    
    return total_loss, L_nll, L_kl_adaptive, L_obs
