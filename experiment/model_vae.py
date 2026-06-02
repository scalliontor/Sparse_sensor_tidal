"""
ForecastDeepONet-VAE — Variational Information-Bottleneck Causal Operator.
Branch: Causal LSTM encodes history into a probabilistic latent distribution: z ~ N(mu_z, var_z).
Trunk: Fourier PE outputting dual functions for mean and log-variance.
Result: Models both information bottleneck (KL loss) and heteroscedastic uncertainty (NLL loss).
"""
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 3, n_freqs: int = 8):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs)
        self.in_dim = in_dim
        self.n_freqs = n_freqs

    @property
    def out_dim(self) -> int:
        return self.in_dim + self.in_dim * self.n_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = x.unsqueeze(-1) * math.pi * self.freqs.view(1, 1, -1)
        fourier = torch.stack([torch.sin(args), torch.cos(args)], dim=-1)
        fourier = fourier.reshape(x.shape[0], -1)
        return torch.cat([x, fourier], dim=-1)


def build_mlp(in_dim, hidden, depth, out_dim, act="gelu"):
    Act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[act]
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), Act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ForecastDeepONetVAE(nn.Module):
    """
    Variational Causal operator: 
    Encodes sensor_history[0..T_obs] -> z ~ N(mu_z, var_z)
    Predicts eta_mu(x,y,t) and eta_logvar(x,y,t) -> for point-wise uncertainty.
    """
    def __init__(self, n_sensors: int = 16, lstm_hidden: int = 256,
                 lstm_layers: int = 2, latent_dim: int = 256,
                 width: int = 256, depth: int = 4, n_fourier_freqs: int = 8):
        super().__init__()

        # Variational Branch: causal LSTM encoder
        self.lstm = nn.LSTM(n_sensors, lstm_hidden, lstm_layers, batch_first=True, dropout=0.0)
        self.branch_mu = nn.Linear(lstm_hidden, latent_dim)
        self.branch_logvar = nn.Linear(lstm_hidden, latent_dim)

        # Dual Trunk: Fourier PE + MLP (one for Mean, one for LogVar)
        self.fourier = FourierFeatures(in_dim=3, n_freqs=n_fourier_freqs)
        self.trunk_mu = build_mlp(self.fourier.out_dim, width, depth, latent_dim)
        self.trunk_logvar = build_mlp(self.fourier.out_dim, width, depth, latent_dim)

        self.bias_mu = nn.Parameter(torch.zeros(1))
        self.bias_logvar = nn.Parameter(torch.zeros(1))
        
        # We start logvar slightly negative to encourage low variance initialization
        nn.init.constant_(self.bias_logvar, -2.0)

    def encode(self, sensor_hist: torch.Tensor):
        """
        sensor_hist: (B, T_obs, n_sensors)
        returns mu_z: (B, latent_dim), logvar_z: (B, latent_dim)
        """
        _, (h_n, _) = self.lstm(sensor_hist)   # h_n: (layers, B, hidden)
        last_h = h_n[-1]                        # (B, hidden)
        mu_z = self.branch_mu(last_h)
        logvar_z = self.branch_logvar(last_h)
        return mu_z, logvar_z

    def reparameterize(self, mu_z, logvar_z):
        if self.training:
            std_z = torch.exp(0.5 * logvar_z)
            eps = torch.randn_like(std_z)
            return mu_z + eps * std_z
        else:
            return mu_z

    def forward(self, sensor_hist: torch.Tensor, trunk_x: torch.Tensor):
        """
        sensor_hist: (B, T_obs, n_sensors)   — past observations
        trunk_x:     (B*P, 3)               — future query (x_norm, y_norm, t_norm)
        returns:     
            y_mu:      (B*P, 1) predictive mean
            y_logvar:  (B*P, 1) predictive variance
            mu_z:      (B, latent_dim) information bottleneck mean
            logvar_z:  (B, latent_dim) information bottleneck variance
        """
        B = sensor_hist.shape[0]
        P = trunk_x.shape[0] // B

        # Encode & Sample
        mu_z, logvar_z = self.encode(sensor_hist)
        z = self.reparameterize(mu_z, logvar_z)         # (B, latent)
        z_exp = z.unsqueeze(1).expand(-1, P, -1).reshape(B * P, -1)

        # Evaluate Trunks
        t_enc = self.fourier(trunk_x)                   # (B*P, fourier_out)
        
        t_mu = self.trunk_mu(t_enc)                     # (B*P, latent)
        y_mu = torch.sum(z_exp * t_mu, dim=-1, keepdim=True) + self.bias_mu
        
        t_logvar = self.trunk_logvar(t_enc)             # (B*P, latent)
        y_logvar = torch.sum(z_exp * t_logvar, dim=-1, keepdim=True) + self.bias_logvar

        return y_mu, y_logvar, mu_z, logvar_z
