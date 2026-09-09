"""
The two encoders of Eq. 6,  q_phi(z_b, z_a | x, c, a) = q_{phi_b}(z_b|x,c) q_{phi_a}(z_a|z_b,x,a).

BasalEncoder   q_{phi_b}(z_b | x, c)
    x (2,000 HVG) --MLP--> 2000 -> 1024 -> 512, then ADD the learned context embedding
    (nn.Embedding[n_context, 512]), then a 2-layer MLP at constant width into the
    Gaussian head. z_b is the basal cellular state: everything about the cell that is
    not the perturbation.

PertEncoder    q_{phi_a}(z_a | z_b, a)
    z_b + the learned perturbation embedding (nn.Embedding[n_pert, 512]), a 2-layer MLP
    at constant width, Gaussian head. Conditioning on z_b (rather than on x) is what
    makes the paper's conditional prior p_alpha(z_a | z_b, a) the matching prior.

Both embedding tables are owned by the top-level model and SHARED: the perturbation
table is also the energy function's view of `a`, so `a` enters the latent prior and the
posterior through the same parameters.
"""
import torch
import torch.nn as nn

from .layers import GaussianHead, mlp, reparameterize


class BasalEncoder(nn.Module):
    def __init__(self, n_genes, d, z_dim, x_hidden=(1024,), trunk_hidden=(512, 512),
                 dropout=0.1, norm="layer", act="silu", logvar_min=-6.0, logvar_max=4.0,
                 use_context=True):
        super().__init__()
        self.use_context = use_context
        # 2000 -> 1024 -> d : the paper's "two layer MLP" into the unified space
        self.x_proj = mlp(n_genes, list(x_hidden), out_dim=d, dropout=dropout,
                          norm=norm, act=act)
        self.head = GaussianHead(d, list(trunk_hidden), z_dim, dropout, norm, act,
                                 logvar_min, logvar_max)

    def forward(self, x, u_c=None, sample=True):
        h = self.x_proj(x)
        if self.use_context and u_c is not None:
            h = h + u_c
        mu, lv = self.head(h)
        z = reparameterize(mu, lv) if sample else mu
        return z, mu, lv, h


class PertEncoder(nn.Module):
    def __init__(self, d, z_b_dim, z_dim, trunk_hidden=(512, 512), dropout=0.1,
                 norm="layer", act="silu", logvar_min=-6.0, logvar_max=4.0):
        super().__init__()
        # z_b and the perturbation embedding are added, so they must share a width; a
        # projection is inserted only when the config forces them apart.
        self.z_b_proj = nn.Linear(z_b_dim, d, bias=False) if z_b_dim != d else nn.Identity()
        self.head = GaussianHead(d, list(trunk_hidden), z_dim, dropout, norm, act,
                                 logvar_min, logvar_max)

    def forward(self, z_b, u_a, sample=True, std_scale=1.0):
        mu, lv = self.head(self.z_b_proj(z_b) + u_a)
        z = reparameterize(mu, lv, std_scale) if sample else mu
        return z, mu, lv
