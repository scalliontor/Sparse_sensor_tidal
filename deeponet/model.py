# model.py
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random/fixed Fourier positional encoding for trunk (x, y, t).
    Output dim: in_dim + in_dim * n_freqs * 2
    """
    def __init__(self, in_dim: int = 3, n_freqs: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        # Fixed frequencies: 1, 2, 4, 8, ..., 2^(n_freqs-1)
        freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)  # (n_freqs,)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return self.in_dim + self.in_dim * self.n_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        # args: (B, in_dim, n_freqs)
        args = x.unsqueeze(-1) * math.pi * self.freqs.view(1, 1, -1)
        sin_f = torch.sin(args)  # (B, in_dim, n_freqs)
        cos_f = torch.cos(args)
        fourier = torch.stack([sin_f, cos_f], dim=-1).reshape(x.shape[0], -1)  # (B, in_dim*n_freqs*2)
        return torch.cat([x, fourier], dim=-1)  # (B, out_dim)


def build_mlp(in_dim: int, hidden_dim: int, depth: int, out_dim: int,
              activation: str = "gelu", dropout: float = 0.0) -> nn.Sequential:
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    if activation.lower() not in acts:
        raise ValueError(f"Unsupported activation: {activation}")
    Act = acts[activation.lower()]

    layers = []
    d = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(Act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """
    DeepONet with Fourier positional encoding on trunk.
      branch(ssh_flat) -> b in R^p
      trunk(FF(x,y,t)) -> t in R^p
      output = sum_k b_k * t_k + bias
    """
    def __init__(
        self,
        branch_in: int,
        trunk_in: int = 3,
        width: int = 256,
        depth: int = 4,
        latent_dim: int = 128,
        activation: str = "gelu",
        dropout: float = 0.0,
        n_fourier_freqs: int = 0,   # 0 = no Fourier features
    ):
        super().__init__()
        self.fourier = FourierFeatures(trunk_in, n_fourier_freqs) if n_fourier_freqs > 0 else None
        trunk_enc_in = self.fourier.out_dim if self.fourier is not None else trunk_in

        self.branch = build_mlp(branch_in, width, depth, latent_dim, activation, dropout)
        self.trunk  = build_mlp(trunk_enc_in, width, depth, latent_dim, activation, dropout)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_x: torch.Tensor, trunk_x: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch_x)
        t_enc = self.fourier(trunk_x) if self.fourier is not None else trunk_x
        t = self.trunk(t_enc)
        y = torch.sum(b * t, dim=-1, keepdim=True) + self.bias
        return y
