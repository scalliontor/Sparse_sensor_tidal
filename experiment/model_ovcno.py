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

class ObservabilityAwareVCNO(nn.Module):
    def __init__(self, lstm_hidden: int = 256, lstm_layers: int = 2, latent_dim: int = 256,
                 width: int = 256, depth: int = 4, n_fourier_freqs: int = 8):
        super().__init__()
        
        # 1. Sensor Geometry Encoder: inputs (x, y, s)
        self.mlp_s = build_mlp(3, 64, 2, 64, act="gelu")
        # 2. Set Encoder Network (after mean-pool)
        self.set_enc = build_mlp(64, 128, 1, 128, act="gelu")
        
        # 3. Causal LSTM Encoder over summarized states
        self.lstm = nn.LSTM(128, lstm_hidden, lstm_layers, batch_first=True, dropout=0.0)
        
        # 4. Latent Parameterization
        self.branch_mu = nn.Linear(lstm_hidden, latent_dim)
        self.branch_logvar = nn.Linear(lstm_hidden, latent_dim)
        
        # 5. Learned Observability Field (takes h_T, x, y, t, d_s)
        # We process point components. h_T is (B, lstm_hidden), we will concatenate them per point.
        self.obs_net = build_mlp(lstm_hidden + 4, 128, 3, 1, act="relu")
        
        # 6. Final Decoder
        self.fourier = FourierFeatures(in_dim=3, n_freqs=n_fourier_freqs)
        # Decoder takes (z, fourier_enc, o_i)
        decoder_in_dim = latent_dim + self.fourier.out_dim + 1
        
        self.trunk_mu = build_mlp(decoder_in_dim, width, depth, 1)
        self.trunk_logvar = build_mlp(decoder_in_dim, width, depth, 1)
        
        self.bias_logvar = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.bias_logvar, -2.0)

    def reparameterize(self, mu_z, logvar_z, force_sample: bool = False):
        if self.training or force_sample:
            std_z = torch.exp(0.5 * logvar_z)
            eps = torch.randn_like(std_z)
            return mu_z + eps * std_z
        else:
            return mu_z

    def forward(self, sensor_hist: torch.Tensor, sensor_pts: torch.Tensor, trunk_x: torch.Tensor,
                sample_z: bool = False):
        """
        sensor_hist: (B, T_obs, K)
        sensor_pts: (B, K, 2)
        trunk_x: (B*P, 4) -> (x, y, t, d_s)
        sample_z: if True, force latent sampling even in eval mode (for MC inference)
        """
        B, T, K = sensor_hist.shape
        P = trunk_x.shape[0] // B
        
        # Step 1 & 2: Process Set of Sensors per timestep
        # We need to form (B, T, K, 3) where the 3 is [x, y, s]
        pts_expanded = sensor_pts.unsqueeze(1).expand(B, T, K, 2) # (B, T, K, 2)
        s_expanded = sensor_hist.unsqueeze(-1) # (B, T, K, 1)
        sensor_tokens = torch.cat([pts_expanded, s_expanded], dim=-1) # (B, T, K, 3)
        
        # Pass through MLP_s
        tokens_flat = sensor_tokens.view(B*T*K, 3)
        e_k = self.mlp_s(tokens_flat).view(B, T, K, 64)
        
        # DeepSets Mean Pooling
        e_pool = e_k.mean(dim=2) # (B, T, 64)
        r_t = self.set_enc(e_pool.view(B*T, 64)).view(B, T, 128)
        
        # Step 3: Temporal Causal Encoding
        _, (h_n, _) = self.lstm(r_t)
        h_T = h_n[-1] # (B, lstm_hidden)
        
        # Step 4: Latent code
        mu_z = self.branch_mu(h_T)
        logvar_z = self.branch_logvar(h_T)
        z = self.reparameterize(mu_z, logvar_z, force_sample=sample_z) # (B, latent)
        
        # Expand z and h_T to match query points
        z_exp = z.unsqueeze(1).expand(-1, P, -1).reshape(B*P, -1)
        h_T_exp = h_T.unsqueeze(1).expand(-1, P, -1).reshape(B*P, -1)
        
        # Step 5: Learned Observability Field
        # trunk_x contains [x, y, t, d_s], which is shape (B*P, 4)
        obs_input = torch.cat([h_T_exp, trunk_x], dim=-1)
        o_i = torch.sigmoid(self.obs_net(obs_input)) # (B*P, 1) in [0,1]
        
        # Step 6: Decoder
        # Positional encoding on (x,y,t) only
        coords = trunk_x[:, :3]
        f_enc = self.fourier(coords)
        
        dec_input = torch.cat([z_exp, f_enc, o_i], dim=-1)
        
        y_mu = self.trunk_mu(dec_input)
        y_logvar = self.trunk_logvar(dec_input) + self.bias_logvar
        
        return y_mu, y_logvar, mu_z, logvar_z, o_i
