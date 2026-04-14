"""
ForecastDeepONet — Causal LSTM branch + Fourier trunk.
Branch: LSTM processes sensor history t=0..T_obs → latent vector
Trunk:  Fourier PE on (x, y, t_future) → spatial basis
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


class ForecastDeepONet(nn.Module):
    """
    Causal operator: sensor_history[0..T_obs] -> eta(x, y, t>T_obs)

    Args:
        n_sensors:      number of sensors (input_size of LSTM)
        lstm_hidden:    LSTM hidden size
        lstm_layers:    LSTM depth
        latent_dim:     dot-product dimension (branch out = trunk out)
        width / depth:  trunk MLP width and depth
        n_fourier_freqs: Fourier PE frequencies for trunk
    """
    def __init__(self, n_sensors: int = 16, lstm_hidden: int = 256,
                 lstm_layers: int = 2, latent_dim: int = 256,
                 width: int = 256, depth: int = 4, n_fourier_freqs: int = 8):
        super().__init__()

        # Branch: causal LSTM encoder
        self.lstm = nn.LSTM(n_sensors, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=0.0)
        self.branch_proj = nn.Linear(lstm_hidden, latent_dim)

        # Trunk: Fourier PE + MLP
        self.fourier = FourierFeatures(in_dim=3, n_freqs=n_fourier_freqs)
        self.trunk = build_mlp(self.fourier.out_dim, width, depth, latent_dim)

        self.bias = nn.Parameter(torch.zeros(1))

    def encode(self, sensor_hist: torch.Tensor) -> torch.Tensor:
        """
        sensor_hist: (B, T_obs, n_sensors)
        returns b:   (B, latent_dim)
        """
        _, (h_n, _) = self.lstm(sensor_hist)   # h_n: (layers, B, hidden)
        return self.branch_proj(h_n[-1])        # (B, latent_dim)

    def forward(self, sensor_hist: torch.Tensor, trunk_x: torch.Tensor) -> torch.Tensor:
        """
        sensor_hist: (B, T_obs, n_sensors)   — past observations
        trunk_x:     (B*P, 3)               — future query (x_norm, y_norm, t_norm)
        returns:     (B*P, 1)
        """
        B = sensor_hist.shape[0]
        P = trunk_x.shape[0] // B

        b = self.encode(sensor_hist)                    # (B, latent)
        b_exp = b.unsqueeze(1).expand(-1, P, -1).reshape(B * P, -1)

        t_enc = self.fourier(trunk_x)                   # (B*P, fourier_out)
        t = self.trunk(t_enc)                           # (B*P, latent)

        y = torch.sum(b_exp * t, dim=-1, keepdim=True) + self.bias  # (B*P, 1)
        return y
