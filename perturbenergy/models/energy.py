"""
E_alpha(z_a, z_b, a): the scalar energy of the conditional prior in Eq. 4,

    p_alpha(z_a | z_b, a) = p_0(z_a) exp(-E_alpha(z_a, z_b, a)) / F_alpha(z_b, a).

LOW energy = this perturbation representation is compatible with the condition, so it
gets more prior mass; high energy = incompatible, and the prior suppresses it.

The energy is a similarity between the perturbation representation and the perturbation
embedding -- the same table the perturbation encoder uses, so lowering E(z_a, a) is
literally "make z_a say which perturbation this is":

    kind="mlp_cos"  (default)  E = -cos( g(z_a), u_a )   g = 2-layer MLP, width preserved
    kind="cos"                 E = -cos( z_a,   u_a )    no parameters of its own
    kind="mlp"                 E = MLP([z_a, u_a]) -> scalar   (unbounded, needs the L2 reg)

`use_z_b=True` restores the paper's full conditioning E(z_a, z_b, a): z_b is projected
and added into the perturbation embedding before the similarity (or concatenated, for
kind="mlp"). It is off by default -- a cosine energy conditioned on z_b makes the target
of the Langevin chain move with every cell, which is much harder to fit -- but the option
is here because Eq. 4 is written that way.

Scale: cosine energies live in [-1, 1], so a hinge margin of ~0.2-0.5 is meaningful and
1.0 is already extreme. `scale` multiplies the similarity if a sharper energy is wanted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import mlp


class EnergyNet(nn.Module):
    def __init__(self, z_a_dim, d, kind="mlp_cos", hidden=(512,), dropout=0.0,
                 norm="layer", act="silu", use_z_b=False, z_b_dim=None, scale=1.0):
        super().__init__()
        self.kind, self.use_z_b, self.scale = kind, use_z_b, float(scale)
        z_b_dim = z_b_dim or d
        self.z_b_proj = (nn.Linear(z_b_dim, d, bias=False) if use_z_b and kind != "mlp"
                         else None)
        if kind == "mlp_cos":
            # 2-layer MLP on z_a that preserves the width, then the cosine
            self.g = mlp(z_a_dim, list(hidden), out_dim=d, dropout=dropout, norm=norm,
                         act=act)
        elif kind == "cos":
            self.g = None
            if z_a_dim != d:
                raise ValueError(f"kind='cos' compares z_a with the perturbation "
                                 f"embedding directly, so z_a_dim ({z_a_dim}) must equal "
                                 f"the embedding width ({d}); use kind='mlp_cos'")
        elif kind == "mlp":
            in_dim = z_a_dim + d + (z_b_dim if use_z_b else 0)
            self.g = mlp(in_dim, list(hidden), out_dim=1, dropout=dropout, norm=norm,
                         act=act)
        else:
            raise ValueError(f"unknown energy kind {kind!r}")

    def forward(self, z_a, u_a, z_b=None):
        """-> [B] energies. `z_b` is ignored unless use_z_b."""
        if self.kind == "mlp":
            parts = [z_a, u_a] + ([z_b] if self.use_z_b else [])
            return self.g(torch.cat(parts, dim=-1)).squeeze(-1) * self.scale
        h = self.g(z_a) if self.g is not None else z_a
        cond = u_a
        if self.use_z_b and z_b is not None:
            cond = cond + self.z_b_proj(z_b)
        return -self.scale * F.cosine_similarity(h, cond, dim=-1)

    @property
    def is_parametric(self):
        return any(p.requires_grad for p in self.parameters())
