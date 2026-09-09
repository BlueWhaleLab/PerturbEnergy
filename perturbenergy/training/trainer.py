"""
Training loop for PerturbEnergy -- Algorithm 1 of the paper, plus the two schedules the
project needs.

What a training step is (`train.pairing`)
----------------------------------------
control    (default) The basal encoder is fed a CONTROL cell of the group's context and
           the generator must produce the PERTURBED population -- the task itself, and
           the setting the paper's Inference paragraph describes. Cells are unpaired (the
           assay is destructive), so the reconstruction term is a distance between
           populations: the squared error of the condition means plus the MMD the paper
           uses. Batches come from data.loader.ConditionSampler, shaped as
           `n_conditions` blocks of `set_size` cells.
observed   The literal reading of the Eq. 7 ELBO: reconstruct the OBSERVED cell from its
           own (x, c, a). It trains a better autoencoder and a much worse predictor --
           z_b already contains the whole cell, so the generator can ignore z_a, and at
           inference (where z_b comes from a control cell) the output is close to the
           control. Kept for the ablation; not the default.

Schedules (`train.mode`)
------------------------
joint      encoders, generator and energy are trained together from step 0.
finetune   stage 1 trains the plain VAE (reconstruction + the two KL terms, energy off)
           for `train.stage1_steps`; stage 2 switches the energy on and continues for
           `train.stage2_steps`. The encoders start from a representation that already
           reconstructs, so the energy only has to reshape it.

Energy update (`loss.energy_update`)
------------------------------------
alternating  Algorithm 1 verbatim -- two phases per step:
             A. freeze {theta, phi_b, phi_a}; update alpha with Eq. 12 on detached
                positives/negatives.
             B. freeze alpha; update {theta, phi_b, phi_a} with L_ELBO.
joint        one optimiser, the energy term added to L_ELBO with `loss.energy_weight`.
             Cheaper and often just as good, but it is not the paper's algorithm.

Negatives (`loss.negatives`)
----------------------------
langevin   Eq. 15: K steps of annealed Langevin from `langevin.init` (N(0,I) is
           Algorithm 1 line 2; `pert_emb` starts from ANOTHER perturbation's embedding,
           a harder negative, and `batch_z_a` from another cell's z_a).
mismatch   pair each z_a with ANOTHER perturbation's embedding: E(z_a, u_a'). Free, and
           for a similarity energy it is exactly the contrast that matters -- z_a must
           sit closer to its own perturbation than to any other. The paper's chain is
           the principled sampler; this is the cheap, stable alternative.
both       hinge against whichever negative currently has the LOWER energy (the harder
           of the two).
"""
import csv
import json
import os
import time

import numpy as np
import torch

from ..data.loader import ConditionSampler
from ..models.perturbenergy import PerturbEnergy
from .langevin import sample_negative
from .losses import (energy_elbo_term, energy_objective, kl_standard, recon_loss,
                     set_recon_loss)


def _set_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def _mismatched(a, u_a, pert_emb):
    """Another perturbation's embedding for every row, never the row's own."""
    n = len(a)
    perm = torch.randperm(n, device=a.device)
    same = a[perm] == a
    if same.any():                                   # roll the clashes by one position
        roll = (torch.arange(n, device=a.device) + 1) % n
        perm = torch.where(same, roll, perm)
    return u_a[perm], perm


class Trainer:
    def __init__(self, model, data, cfg, out_dir, device="cuda"):
        self.model, self.data, self.cfg = model, data, cfg
        self.out_dir = out_dir
        self.device = device
        self.L, self.O, self.LG, self.T = cfg.loss, cfg.optim, cfg.langevin, cfg.train
        os.makedirs(out_dir, exist_ok=True)

        self.pairing = cfg.train.get("pairing", "control")
        self.set_size = int(cfg.data.get("set_size", 32))
        if self.pairing == "control":
            n_cond = max(1, cfg.data.batch_size // self.set_size)
            self.train_sampler = ConditionSampler(data, n_cond, self.set_size, "train",
                                                  cfg.train.seed)
            try:
                self.val_sampler = ConditionSampler(data, n_cond, self.set_size, "val",
                                                    cfg.train.seed + 1)
            except ValueError:
                self.val_sampler = None
                print("[warn] no usable val conditions -- validating on train conditions")

        Opt = (torch.optim.AdamW if str(self.O.get("name", "adamw")).lower() == "adamw"
               else torch.optim.Adam)
        self.alternating = (self.L.energy_update == "alternating"
                            and model.use_energy and model.energy_parameters())
        if self.L.energy_update == "alternating" and model.use_energy \
                and not model.energy_parameters():
            print("[warn] energy_update='alternating' but the energy has no parameters "
                  "of its own (kind='cos') -> using the joint update instead")
        vae_p = model.vae_parameters()
        if self.alternating:
            self.vae_opt = Opt(vae_p, lr=self.O.vae_lr, weight_decay=self.O.weight_decay)
            self.e_opt = Opt(model.energy_parameters(), lr=self.O.energy_lr,
                             weight_decay=self.O.weight_decay)
        else:
            self.vae_opt = Opt(vae_p + model.energy_parameters(), lr=self.O.vae_lr,
                               weight_decay=self.O.weight_decay)
            self.e_opt = None
        self.vae_params = vae_p
        self.hist_path = os.path.join(out_dir, "metrics.csv")
        self.f = open(self.hist_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["step", "stage", "loss", "recon_mse", "recon_mmd", "kl_b",
                         "kl_a", "e_pos", "e_neg", "e_term", "val_mse", "sec"])
        self.best = float("inf")

    # ------------------------------------------------------------------ #
    def _negatives(self, o, a, energy_on):
        """(e_pos, e_neg) for the current batch, plus the raw negative sample if any."""
        m = self.model
        if not energy_on:
            z = torch.zeros((), device=o["z_a"].device)
            return z, z, None
        z_pos, u_a = o["z_a"], o["u_a"]
        z_b = o["z_b"] if m.energy_net.use_z_b else None
        kind = self.L.negatives
        e_pos = m.energy(z_pos, u_a, z_b)

        e_mis = z_neg = None
        if kind in ("mismatch", "both"):
            u_neg, _ = _mismatched(a, u_a, m.pert_emb)
            e_mis = m.energy(z_pos, u_neg.detach(), z_b)
        if kind in ("langevin", "both"):
            init = self._langevin_init(o, a)
            z_neg = sample_negative(m, u_a, init, z_b, self.LG.steps, self.LG.step_size,
                                    self.LG.noise_scale, self.LG.clamp, self.LG.noise_decay)
            e_lan = m.energy(z_neg, u_a, z_b)
        if kind == "mismatch":
            return e_pos, e_mis, None
        if kind == "langevin":
            return e_pos, e_lan, z_neg
        return e_pos, torch.minimum(e_mis, e_lan), z_neg    # 'both': the harder negative

    def _langevin_init(self, o, a):
        """Algorithm 1 line 2 initialises the chain from p_0 = N(0,I); the other two
        starts are informative negatives that a short chain can actually refine."""
        init = self.LG.init
        if init == "pert_emb":
            u_neg, _ = _mismatched(a, o["u_a"], self.model.pert_emb)
            return u_neg.detach()
        if init == "batch_z_a":
            return o["z_a"].detach()[torch.randperm(len(a), device=a.device)]
        return torch.randn_like(o["z_a"])

    # ------------------------------------------------------------------ #
    def step(self, batch, step, energy_on):
        """`batch` is (x_in, x_target, c, a); for pairing='observed' the two x are the
        same tensor."""
        x_in, x_tgt, c, a = batch
        m, L, O = self.model, self.L, self.O
        x = x_in
        m.train()
        beta = min(1.0, step / max(1, L.kl_warmup))
        stats = {}

        # ---- phase A: alpha <- Eq. 12 (Algorithm 1 line 4) ------------------ #
        if energy_on and self.alternating:
            with torch.no_grad():
                o0 = m(x, c, a, sample=True)
            o0 = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in o0.items()}
            _set_grad(m.energy_net, True)
            e_pos, e_neg, _ = self._negatives(o0, a, True)
            e_loss = energy_objective(e_pos, e_neg, O.margin, O.use_margin, O.energy_l2_reg)
            self.e_opt.zero_grad(set_to_none=True)
            e_loss.backward()
            torch.nn.utils.clip_grad_norm_(m.energy_parameters(), O.grad_clip)
            self.e_opt.step()
            stats["e_pos"] = float(e_pos.mean().detach())
            stats["e_neg"] = float(e_neg.mean().detach())
            _set_grad(m.energy_net, False)          # frozen for phase B

        # ---- phase B: {theta, phi} <- L_ELBO (Algorithm 1 line 5) ----------- #
        o = m(x, c, a, sample=True)
        if self.pairing == "control":
            rec, mse_v, mmd_v = set_recon_loss(x_tgt, o["x_hat"], self.set_size,
                                               L.set_recon, L.mmd_weight, L.var_weight,
                                               tuple(L.mmd_scales))
        else:
            rec, mse_v, mmd_v = recon_loss(x_tgt, o["x_hat"], L.recon, L.mmd_weight,
                                           L.recon_group_size, tuple(L.mmd_scales),
                                           L.recon_reduce)
        kl_b = kl_standard(o["mu_b"], o["lv_b"])
        kl_a = kl_standard(o["mu_a"], o["lv_a"])
        loss = rec + beta * (L.beta_kl_b * kl_b + L.beta_kl_a * kl_a)

        e_term = torch.zeros((), device=x.device)
        if energy_on:
            e_pos, e_neg, _ = self._negatives(o, a, True)
            if self.alternating:
                # alpha is frozen: this term only pulls the POSTERIOR towards low energy
                e_term = energy_elbo_term(e_pos, e_neg.detach(), L.energy_term, O.margin)
            else:
                # one optimiser: the same hinge trains alpha and the posterior together
                e_term = energy_objective(e_pos, e_neg, O.margin, O.use_margin,
                                          O.energy_l2_reg)
                stats["e_pos"] = float(e_pos.mean().detach())
                stats["e_neg"] = float(e_neg.mean().detach())
            loss = loss + L.energy_weight * e_term

        self.vae_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.vae_params + ([] if self.alternating else m.energy_parameters()),
            O.grad_clip)
        self.vae_opt.step()
        if energy_on and self.alternating:
            _set_grad(m.energy_net, True)

        stats.update(loss=float(loss.detach()), recon_mse=float(mse_v),
                     recon_mmd=float(mmd_v), kl_b=float(kl_b.detach()),
                     kl_a=float(kl_a.detach()), e_term=float(e_term.detach()))
        return stats

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self, max_batches=40, batch_size=1024):
        """Checkpoint-selection score, on the same footing as training.

        pairing='control': the mean squared error of the predicted CONDITION MEAN against
        the real one, per gene -- the quantity every delta metric is built on, and not
        something a model can win by echoing its input.
        pairing='observed': per-cell reconstruction MSE.
        """
        self.model.eval()
        if self.pairing == "control":
            smp = self.val_sampler or self.train_sampler
            tot = 0.0
            for _ in range(max_batches):
                x_in, x_tgt, c, a = smp.batch()
                x_hat = self.model.predict(x_in, c, a,
                                           m_samples=self.cfg.eval.m_samples)
                n = x_tgt.shape[0] // self.set_size
                T = x_tgt[:n * self.set_size].view(n, self.set_size, -1).mean(1)
                P = x_hat[:n * self.set_size].view(n, self.set_size, -1).mean(1)
                tot += float(((P - T) ** 2).mean())
            return tot / max_batches
        tot = n = 0.0
        for i, (x, c, a) in enumerate(self.data.batches("val", batch_size, shuffle=False)):
            o = self.model(x, c, a, sample=False)
            tot += float(((o["x_hat"] - x) ** 2).mean()) * len(x)
            n += len(x)
            if i + 1 >= max_batches:
                break
        return tot / max(n, 1)

    def save(self, step, tag="last", extra=None):
        p = os.path.join(self.out_dir, f"model_{tag}.pt")
        torch.save({"model": self.model.state_dict(), "step": step,
                    "cfg": self.cfg.to_dict(), "n_genes": self.model.n_genes,
                    "n_context": self.model.n_context,
                    "pert_tokens": self.model.pert_emb.tokens.cpu(),
                    "pert_mask": self.model.pert_emb.mask.cpu(),
                    "dataset": self.data.dataset, **(extra or {})}, p)
        return p

    # ------------------------------------------------------------------ #
    def run(self):
        T = self.T
        mode = T.mode
        if mode == "finetune":
            plan = [("vae", T.stage1_steps, False), ("energy", T.stage2_steps, True)]
        else:
            plan = [("joint", T.max_steps, self.model.use_energy)]
        stream = (self.train_sampler.stream() if self.pairing == "control"
                  else ((x, x, c, a) for x, c, a
                        in self.data.sampler(self.cfg.data.batch_size, T.seed)))
        step = 0
        t0 = time.time()
        acc, nacc = {}, 0
        for stage, n_steps, energy_on in plan:
            if n_steps <= 0:
                continue
            print(f"\n=== stage '{stage}': {n_steps} steps, energy "
                  f"{'ON' if energy_on else 'OFF'} ===", flush=True)
            for _ in range(n_steps):
                step += 1
                s = self.step(next(stream), step, energy_on)
                for k, v in s.items():
                    acc[k] = acc.get(k, 0.0) + v
                nacc += 1
                if step % T.log_every == 0 or step == 1:
                    avg = {k: v / max(nacc, 1) for k, v in acc.items()}
                    print(f"step {step:7d} [{stage:6s}] loss {avg.get('loss', 0):9.2f} "
                          f"mse {avg.get('recon_mse', 0):9.2f} "
                          f"mmd {avg.get('recon_mmd', 0):7.4f} "
                          f"kl_b {avg.get('kl_b', 0):8.2f} kl_a {avg.get('kl_a', 0):8.2f} "
                          f"| E+ {avg.get('e_pos', float('nan')):7.4f} "
                          f"E- {avg.get('e_neg', float('nan')):7.4f} "
                          f"Eterm {avg.get('e_term', 0):.4f} "
                          f"| {time.time() - t0:6.0f}s", flush=True)
                    self.w.writerow([step, stage, avg.get("loss", ""),
                                     avg.get("recon_mse", ""), avg.get("recon_mmd", ""),
                                     avg.get("kl_b", ""), avg.get("kl_a", ""),
                                     avg.get("e_pos", ""), avg.get("e_neg", ""),
                                     avg.get("e_term", ""), "", round(time.time() - t0)])
                    self.f.flush()
                    acc, nacc = {}, 0
                if step % T.eval_every == 0 or step == (sum(p[1] for p in plan)):
                    v = self.validate()
                    lbl = ("val condition-mean MSE/gene" if self.pairing == "control"
                           else "val recon MSE/gene")
                    print(f"          {lbl} {v:.6f}"
                          f"{'   <- best' if v < self.best else ''}", flush=True)
                    self.w.writerow([step, stage, "", "", "", "", "", "", "", "", v,
                                     round(time.time() - t0)])
                    self.f.flush()
                    self.save(step, "last")
                    if v < self.best:
                        self.best = v
                        self.save(step, "best", extra={"val_mse": v})
        self.save(step, "last")
        self.f.close()
        print(f"\nTRAIN_DONE {self.out_dir}  steps={step}  best_val_mse={self.best:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return self.out_dir


def build_model(data, cfg, device="cuda"):
    m = PerturbEnergy(data.n_genes, data.n_context, data.pert_tokens, data.pert_mask,
                      cfg.model).to(device)
    print("[model]", m.describe(), flush=True)
    return m


def load_checkpoint(path, device="cuda"):
    """Rebuild a model from a checkpoint. Returns (model, cfg_dict)."""
    from ..config import Cfg
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Cfg(ck["cfg"])
    m = PerturbEnergy(ck["n_genes"], ck["n_context"], ck["pert_tokens"],
                      ck["pert_mask"], cfg.model).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg, ck
