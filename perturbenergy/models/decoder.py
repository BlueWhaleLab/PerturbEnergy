"""
p_theta(x | z_b, z_a): the generator of Eq. 1.

Two-layer MLP mirroring the basal encoder's x-projection (512 -> 1024 -> 2000), with a
linear output because the data is log-normalised expression, not counts. `combine`
decides how the two latents enter: `add` (default, keeps the input width at z_dim, and
matches the additive way perturbations are composed in scGen/CPA) or `concat`.
"""
import torch
import torch.nn as nn

from .layers import mlp


class Decoder(nn.Module):
    def __init__(self, z_b_dim, z_a_dim, n_genes, hidden=(1024,), combine="add",
                 dropout=0.1, norm="layer", act="silu"):
        super().__init__()
        if combine not in ("add", "concat"):
            raise ValueError(f"combine must be 'add' or 'concat', got {combine!r}")
        if combine == "add" and z_b_dim != z_a_dim:
            raise ValueError(f"combine='add' needs z_b_dim == z_a_dim "
                             f"({z_b_dim} != {z_a_dim}); use combine='concat'")
        self.combine = combine
        in_dim = z_b_dim if combine == "add" else z_b_dim + z_a_dim
        self.net = mlp(in_dim, list(hidden), out_dim=n_genes, dropout=dropout,
                       norm=norm, act=act)

    def forward(self, z_b, z_a):
        h = z_b + z_a if self.combine == "add" else torch.cat([z_b, z_a], dim=-1)
        return self.net(h)
