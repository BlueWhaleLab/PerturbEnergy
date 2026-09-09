"""Shared building blocks: MLP stacks, the Gaussian posterior head, reparameterisation."""
import torch
import torch.nn as nn

ACTS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU, "leakyrelu": nn.LeakyReLU}


def norm_layer(kind, width):
    if kind == "batch":
        return nn.BatchNorm1d(width)
    if kind == "layer":
        return nn.LayerNorm(width)
    return nn.Identity()


def mlp(in_dim, hidden, out_dim=None, dropout=0.0, norm="layer", act="silu",
        out_act=False):
    """`hidden` is the list of hidden widths; `out_dim` adds a final bare Linear.

    mlp(2000, [1024, 512])              -> 2000->1024->512, both blocks normed/activated
    mlp(512, [1024], out_dim=2000)      -> 512->1024 (normed/activated) ->2000 (linear)
    """
    A = ACTS[act]
    layers, d = [], in_dim
    for w in hidden:
        layers += [nn.Linear(d, w), norm_layer(norm, w), A(), nn.Dropout(dropout)]
        d = w
    if out_dim is not None:
        layers += [nn.Linear(d, out_dim)]
        if out_act:
            layers += [A()]
    return nn.Sequential(*layers)


def reparameterize(mu, logvar, std_scale=1.0):
    return mu + std_scale * torch.randn_like(mu) * torch.exp(0.5 * logvar)


class GaussianHead(nn.Module):
    """Two linear heads on a shared trunk -> (mu, logvar).

    The logvar clamp guards exp(). Its LOWER bound sets the minimum posterior std
    (exp(logvar_min/2)); push it too low and the "posterior" is a deterministic encoder
    with a KL term that no longer means anything.
    """

    def __init__(self, in_dim, hidden, z_dim, dropout=0.0, norm="layer", act="silu",
                 logvar_min=-6.0, logvar_max=4.0):
        super().__init__()
        self.trunk = mlp(in_dim, hidden, dropout=dropout, norm=norm, act=act)
        d = hidden[-1] if hidden else in_dim
        self.fc_mu = nn.Linear(d, z_dim)
        self.fc_lv = nn.Linear(d, z_dim)
        self.logvar_min, self.logvar_max = logvar_min, logvar_max

    def forward(self, h):
        h = self.trunk(h)
        return self.fc_mu(h), self.fc_lv(h).clamp(self.logvar_min, self.logvar_max)
