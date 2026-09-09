"""
Langevin negative sampling from p_alpha(z_a | z_b, a)  --  paper Eq. 13-15.

The negative phase of Eq. 12 needs samples from the model's own prior, which is where the
partition function would normally be required. It is not: F_alpha(z_b, a) does not depend
on z_a, so

    grad_{z_a} log p_alpha(z_a|z_b,a) = grad log p_0(z_a) - grad E_alpha(z_a, z_b, a),

and with p_0 = N(0, I) the Euler-Maruyama step of Eq. 15 is

    z^{k+1} = z^k - (eta/2) [ grad_{z_a} E_alpha(z^k, z_b, a) + z^k ] + sqrt(eta) * eps.

Only z_a carries gradient here: the energy parameters are frozen for the chain, and the
returned sample is detached, so the negative phase contributes exactly the term Lemma 1
prescribes and nothing else.

`noise_decay` anneals the Brownian term to zero along the chain. Eq. 15 is written with a
constant noise scale; annealing makes a short chain (K=20, not K=1000) actually settle
into the energy's basin instead of jittering around it, which is the standard short-run
MCMC practice for EBMs.
"""
import contextlib
import math

import torch


@contextlib.contextmanager
def _frozen(net):
    """No parameter gradients and no dropout/BatchNorm noise inside the chain."""
    was_training = net.training
    flags = [p.requires_grad for p in net.parameters()]
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(net.parameters(), flags):
            p.requires_grad_(f)
        if was_training:
            net.train()


def _noise_scale(k, steps, base, decay):
    if decay == "linear":
        return base * (1.0 - k / max(1, steps - 1))
    if decay == "cosine":
        return base * 0.5 * (1.0 + math.cos(math.pi * k / max(1, steps - 1)))
    return base


@torch.enable_grad()
def sample_negative(model, u_a, init, z_b=None, steps=20, step_size=0.1,
                    noise_scale=1.0, clamp=10.0, noise_decay="linear"):
    """Run K Langevin steps from `init` and return z_a^(K), detached.

    `init` (Algorithm 1 line 2) is drawn from p_0 = N(0, I) by the caller; other choices
    (another perturbation's embedding, another cell's z_a) are harder negatives and are
    selected in the trainer.
    """
    u_a = u_a.detach()
    z_b = None if z_b is None else z_b.detach()
    with _frozen(model.energy_net):
        z = init.detach().clone()
        for k in range(steps):
            z = z.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(model.energy(z, u_a, z_b).sum(), z)
            z = z - 0.5 * step_size * (grad + z)
            s = _noise_scale(k, steps, noise_scale, noise_decay)
            if s > 0:
                z = z + s * (step_size ** 0.5) * torch.randn_like(z)
            if clamp:
                z = z.clamp(-clamp, clamp)
    return z.detach()
