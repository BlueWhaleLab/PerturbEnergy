"""
PerturbEnergy: a conditional-energy prior over the perturbation latent space.

    x  in R^2000   log-normalised expression of a cell
    c  in C        cellular context (cell type / cell line)
    a  in A        perturbation label
    z_b            basal state, q_{phi_b}(z_b | x, c)          (encoders.BasalEncoder)
    z_a            perturbation representation, q_{phi_a}(z_a | z_b, a)  (PertEncoder)
    x_hat          p_theta(x | z_b, z_a)                        (decoder.Decoder)
    E_alpha        the energy of Eq. 4                          (energy.EnergyNet)

Two embedding tables -- context and perturbation, both `d` wide -- are owned here and
shared by the encoders and the energy, so `a` reaches the posterior and the prior through
the same parameters.

Forward (training) reconstructs the OBSERVED cell: that is what the ELBO of Eq. 7 is a
bound on, and the observed cell already carries its own perturbation label.

Inference (`predict`, the paper's Inference paragraph) is the counterfactual direction:
an UNPERTURBED cell (x0, c0) and a target perturbation a ->
    z_b0 ~ q(z_b|x0,c0);  M candidates z_a ~ q(z_a|z_b0, a);
    keep the candidate with the LOWEST energy;  x = p_theta(x|z_b0, z_a*).
"""
import torch
import torch.nn as nn

from .decoder import Decoder
from .encoders import BasalEncoder, PertEncoder
from .energy import EnergyNet
from .layers import reparameterize
from .pert_embedding import PertEmbedding


class PerturbEnergy(nn.Module):
    def __init__(self, n_genes, n_context, pert_tokens, pert_mask, cfg):
        """`pert_tokens`/`pert_mask` come from the data loader: [n_labels, max_targets].
        For a single-target dataset they are one column and the embedding is the plain
        lookup table the design specifies."""
        super().__init__()
        d = int(cfg["d_model"])
        z_b = int(cfg.get("z_b_dim") or d)
        z_a = int(cfg.get("z_a_dim") or d)
        dp, nm, act = cfg.get("dropout", 0.1), cfg.get("norm", "layer"), cfg.get("act", "silu")
        lo, hi = cfg.get("logvar_min", -6.0), cfg.get("logvar_max", 4.0)
        self.d, self.z_b_dim, self.z_a_dim = d, z_b, z_a
        self.n_genes, self.n_context = n_genes, n_context
        self.use_energy = bool(cfg.get("use_energy", True))
        self.use_context = bool(cfg.get("use_context", True))
        self.n_candidates = int(cfg.get("n_candidates", 1))
        self.select_mode = cfg.get("select_mode", "min")
        self.select_tau = float(cfg.get("select_tau", 1.0))
        self.candidate_std = float(cfg.get("candidate_std_scale", 1.0))

        self.context_emb = nn.Embedding(n_context, d) if self.use_context else None
        self.pert_emb = PertEmbedding(pert_tokens, pert_mask, d,
                                      cfg.get("pert_reduce", "sum"))
        self.n_pert = self.pert_emb.n_labels

        self.enc_b = BasalEncoder(n_genes, d, z_b, cfg.get("x_hidden", [1024]),
                                  cfg.get("enc_hidden", [512, 512]), dp, nm, act, lo, hi,
                                  use_context=self.use_context)
        self.enc_a = PertEncoder(d, z_b, z_a, cfg.get("enc_hidden", [512, 512]),
                                 dp, nm, act, lo, hi)
        self.dec = Decoder(z_b, z_a, n_genes, cfg.get("dec_hidden", [1024]),
                           cfg.get("combine", "add"), dp, nm, act)
        e = dict(cfg.get("energy", {}))
        self.energy_net = EnergyNet(z_a, d, e.get("kind", "mlp_cos"),
                                    e.get("hidden", [512]), e.get("dropout", 0.0), nm, act,
                                    e.get("use_z_b", False), z_b,
                                    e.get("scale", 1.0)) if self.use_energy else None

    # ---- parameter groups --------------------------------------------------- #
    def energy_parameters(self):
        """alpha: only the energy network. The shared embedding tables belong to the VAE
        group -- phase A sees them detached, so the two phases never fight over them."""
        return list(self.energy_net.parameters()) if self.use_energy else []

    def vae_parameters(self):
        ids = {id(p) for p in self.energy_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    # ---- pieces ------------------------------------------------------------- #
    def embed(self, c, a):
        """Context ids of UNSEEN (-1) -- a context that never occurred in training -- fall
        back to the MEAN of the learned context embeddings rather than to row 0, which
        would silently be some other cell line."""
        u_c = None
        if self.use_context and c is not None:
            miss = c < 0
            u_c = self.context_emb(c.clamp(min=0))
            if bool(miss.any()):
                u_c = torch.where(miss.unsqueeze(-1),
                                  self.context_emb.weight.mean(0, keepdim=True), u_c)
        return u_c, self.pert_emb(a)

    def energy(self, z_a, u_a, z_b=None):
        if not self.use_energy:
            return torch.zeros(z_a.shape[0], device=z_a.device, dtype=z_a.dtype)
        return self.energy_net(z_a, u_a, z_b)

    def decode(self, z_b, z_a):
        return self.dec(z_b, z_a)

    def select_z_a(self, mu_a, lv_a, u_a, z_b=None, n=None, mode=None, tau=None):
        """Draw `n` candidates from q(z_a|z_b,a) and let the energy choose one (Inference
        step iii). n=1 is plain reparameterised sampling. Returns (z_a, E(z_a), E_all)."""
        n = int(n or self.n_candidates)
        if not self.use_energy or n <= 1:
            z = reparameterize(mu_a, lv_a, self.candidate_std)
            return z, self.energy(z, u_a, z_b), None
        mode = mode or self.select_mode
        tau = float(tau or self.select_tau)
        B = mu_a.shape[0]
        Z = torch.stack([reparameterize(mu_a, lv_a, self.candidate_std) for _ in range(n)], 1)
        flat = Z.reshape(B * n, -1)
        # `reshape` is row-major over (B, n); repeat_interleave matches it row for row
        E = self.energy(flat, u_a.repeat_interleave(n, 0),
                        None if z_b is None else z_b.repeat_interleave(n, 0)).view(B, n)
        idx = (torch.multinomial(torch.softmax(-E / tau, 1), 1).squeeze(1)
               if mode == "softmax" else E.argmin(1))
        rows = torch.arange(B, device=Z.device)
        return Z[rows, idx], E[rows, idx], E

    # ---- forward ------------------------------------------------------------ #
    def forward(self, x, c, a, sample=True, n_candidates=None):
        u_c, u_a = self.embed(c, a)
        z_b, mu_b, lv_b, h_x = self.enc_b(x, u_c, sample=sample)
        _, mu_a, lv_a = self.enc_a(z_b, u_a, sample=False)
        if sample:
            z_a, e_sel, e_all = self.select_z_a(mu_a, lv_a, u_a, z_b, n_candidates)
        else:
            z_a, e_all = mu_a, None
            e_sel = self.energy(z_a, u_a, z_b)
        return dict(x_hat=self.decode(z_b, z_a), z_b=z_b, z_a=z_a, mu_b=mu_b, lv_b=lv_b,
                    mu_a=mu_a, lv_a=lv_a, u_a=u_a, u_c=u_c, h_x=h_x,
                    e_sel=e_sel, e_all=e_all)

    # ---- counterfactual prediction ------------------------------------------ #
    @torch.no_grad()
    def predict(self, x0, c0, a, m_samples=16):
        """Unperturbed cells (x0, c0) + a target perturbation -> predicted perturbed
        expression. Paper's Inference: sample M candidates, keep the lowest-energy one."""
        u_c, u_a = self.embed(c0, a)
        z_b, mu_b, _, _ = self.enc_b(x0, u_c, sample=False)
        _, mu_a, lv_a = self.enc_a(z_b, u_a, sample=False)
        z_a, _, _ = self.select_z_a(mu_a, lv_a, u_a, z_b, n=m_samples, mode="min")
        return self.decode(z_b, z_a)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def describe(self):
        e = (f"{self.energy_net.kind}"
             f"{'+z_b' if self.energy_net.use_z_b else ''}" if self.use_energy else "off")
        return (f"PerturbEnergy(genes={self.n_genes} d={self.d} z_b={self.z_b_dim} "
                f"z_a={self.z_a_dim} contexts={self.n_context} "
                f"perts={self.n_pert}/{self.pert_emb.n_targets}targets "
                f"energy={e} combine={self.dec.combine} "
                f"params={self.n_params()/1e6:.2f}M)")
