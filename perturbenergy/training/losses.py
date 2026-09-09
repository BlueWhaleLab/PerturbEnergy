"""
The pieces of Eq. 16,

    L_ELBO = E_q[log p_theta(x|z_b,z_a)]                       <- reconstruction
             - KL(q_{phi_b} || p_beta)                          <- basal KL
             - E[ KL(q_{phi_a} || p_0) ]                        <- perturbation KL
             - E[ E_q[E_alpha] + log F_alpha ]                  <- the energy terms

and the energy objective of Eq. 12,

    grad_alpha J_E = E_{q_{phi_a}}[grad E_alpha(z_a)]  -  E_{p_alpha}[grad E_alpha(z_a)]
                     \___ positive phase ___/             \___ negative phase ___/

which is what makes `log F_alpha` tractable: its gradient is exactly the negative phase
(Lemma 1), so the intractable partition function never has to be evaluated.

Both priors p_beta(z_b|c) and p_0(z_a) are the fixed N(0, I) the paper assumes when it
discretises the Langevin update (Eq. 15).
"""
import torch
import torch.nn.functional as F


def kl_standard(mu, logvar):
    """KL( N(mu, exp(logvar)) || N(0, I) ): summed over latent dims, averaged over cells."""
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()


def mse_recon(x, x_hat, reduce="sum"):
    """`sum`: per-cell squared error summed over genes, averaged over the batch -- the
    Gaussian log-likelihood up to a constant, and the scale the KL weights assume.
    `mean` divides by the number of genes as well."""
    e = (x_hat - x) ** 2
    return e.sum(1).mean() if reduce == "sum" else e.mean()


def mmd(x, y, scales=(1.0, 2.0, 5.0, 10.0, 20.0)):
    """Multi-scale RBF MMD^2 with a median-heuristic bandwidth."""
    xy = torch.cat([x, y], 0)
    d2 = torch.cdist(xy, xy).pow(2)
    med = d2.detach().median().clamp(min=1e-6)
    k = sum(torch.exp(-d2 / (s * med)) for s in scales)
    n = x.shape[0]
    return k[:n, :n].mean() + k[n:, n:].mean() - 2.0 * k[:n, n:].mean()


def grouped_mmd(x, x_hat, group_size=64, scales=(1.0, 2.0, 5.0, 10.0, 20.0)):
    """MMD between the real and reconstructed POPULATIONS, in chunks of `group_size`
    cells. This is the paper's population-level stand-in for E_q[log p_theta]; a full
    batch-sized kernel matrix would be 512x512 per step and dominated by one bandwidth."""
    n = x.shape[0]
    vals = [mmd(x_hat[s:s + group_size], x[s:s + group_size], scales)
            for s in range(0, n, group_size) if n - s >= 2]
    return torch.stack(vals).mean() if vals else x.sum() * 0.0


def recon_loss(x, x_hat, kind="mse+mmd", mmd_weight=1.0, group_size=64,
               scales=(1.0, 2.0, 5.0, 10.0, 20.0), reduce="sum"):
    """kind: 'mse' | 'mmd' | 'mse+mmd'. Returns (total, mse_detached, mmd_detached).

    The paper approximates E_q[log p_theta] with an MMD between the real and generated
    populations. Pure MMD has no per-cell gradient, so it reconstructs a population with
    the right shape but individually poor cells; keeping the MSE alongside it (the
    default) preserves the per-cell signal. `kind='mmd'` reproduces the paper literally.
    """
    zero = torch.zeros((), device=x.device)
    m = mse_recon(x, x_hat, reduce) if "mse" in kind else zero
    d = grouped_mmd(x, x_hat, group_size, tuple(scales)) if "mmd" in kind else zero
    return m + mmd_weight * d, m.detach(), d.detach()


def energy_objective(e_pos, e_neg, margin=0.2, use_margin=True, l2_reg=1e-3):
    """Eq. 12 as a loss to MINIMISE: positives (z_a inferred from real cells) get LOW
    energy, negatives (samples from the current p_alpha, or mismatched perturbations)
    get HIGH energy.

    `use_margin` swaps the raw difference for the hinge max(0, margin + E+ - E-), which
    stops pushing once the two are separated and so removes the runaway energy scale a
    plain difference produces. `l2_reg` keeps the absolute energies near zero, which
    matters for the unbounded `kind='mlp'` energy and is harmless for the cosine one.
    """
    loss = (F.relu(margin + e_pos - e_neg).mean() if use_margin
            else e_pos.mean() - e_neg.mean())
    if l2_reg:
        loss = loss + l2_reg * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
    return loss


def energy_elbo_term(e_pos, e_neg=None, kind="hinge", margin=0.2):
    """The energy contribution to the VAE-side update (the `E_q[E_alpha]` of Eq. 16).

    kind='raw'    the literal Eq. 16 term, E_q[E_alpha]: pushes the posterior straight at
                  the energy minimum. Unbounded below for a cosine energy, so with a
                  large weight it collapses every z_a of a perturbation onto one point.
    kind='hinge'  the same pressure, but bounded: it stops once the posterior's energy is
                  a margin below the negatives'. Default.
    kind='none'   drop the term; the energy then only shapes z_a through candidate
                  selection in the forward pass.
    """
    if kind == "none" or e_pos is None:
        return torch.zeros((), device=e_pos.device if e_pos is not None else None)
    if kind == "raw" or e_neg is None:
        return e_pos.mean()
    return F.relu(margin + e_pos - e_neg).mean()


def set_recon_loss(x_tgt, x_pred, set_size, kind="mean+mmd", mmd_weight=1.0,
                   var_weight=0.0, scales=(1.0, 2.0, 5.0, 10.0, 20.0)):
    """Unpaired, population-level reconstruction -- the term the paper approximates with
    an MMD.

    The batch is `n_conditions` blocks of `set_size` cells (data.loader.ConditionSampler),
    so predicted and real populations line up block by block and the loss can be a
    distance between DISTRIBUTIONS instead of between arbitrarily paired cells:

        mean   ||mean(pred) - mean(real)||^2 summed over genes, averaged over conditions.
               A strong, low-variance signal for the condition's average response -- and
               exactly what the delta metrics measure.
        mmd    multi-scale RBF MMD^2 between the two populations. Keeps the prediction a
               population rather than `set_size` copies of one profile.
        var    ||std(pred) - std(real)||^2: matches the spread gene by gene. Off by
               default (the MMD already covers it) but cheap and sometimes steadier.

    Returns (total, mean_mse, mmd) with the last two detached for logging.
    """
    n = x_tgt.shape[0] // set_size
    T = x_tgt[:n * set_size].view(n, set_size, -1)
    P = x_pred[:n * set_size].view(n, set_size, -1)
    zero = torch.zeros((), device=x_tgt.device)

    m = ((P.mean(1) - T.mean(1)) ** 2).sum(1).mean() if "mean" in kind else zero
    if var_weight:
        m = m + var_weight * ((P.std(1) - T.std(1)) ** 2).sum(1).mean()
    d = (torch.stack([mmd(P[i], T[i], scales) for i in range(n)]).mean()
         if "mmd" in kind else zero)
    return m + mmd_weight * d, m.detach(), d.detach()
