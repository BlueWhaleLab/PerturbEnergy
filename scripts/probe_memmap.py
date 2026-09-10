"""
Sanity-check a memmap-backed dataset and measure what a training step will cost.

    python scripts/probe_memmap.py --path $PE_PREPPED/tahoe_full

Prints the dataset's shape and split, then times ConditionSampler batches both cold
(rows drawn across the whole file) and warm (the page cache has them), because the two
differ by a lot on a 765 GB array and only the pair of them says what a run will cost.
Ends with the step budget needed for a given number of passes over the conditions.
"""
import argparse
import os
import time

import _bootstrap  # noqa: F401
import numpy as np

from perturbenergy.data.loader import ConditionSampler, PerturbData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--set_size", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--passes", type=int, nargs="*", default=[15, 25, 40, 55],
                    help="times each condition should be sampled")
    a = ap.parse_args()

    t0 = time.time()
    d = PerturbData(a.path, device="cuda")
    print(f"[probe] loader ready in {time.time() - t0:.0f}s")
    gb = d.X_np.nbytes / 1e9
    print(f"[probe] X {d.X_np.shape} = {gb:.0f} GB on {d.store}")
    print(f"[probe] contexts {len(d.c2i)} | perturbations {len(d.a2i)} "
          f"| control {d.control!r}")
    for s in ("train", "val", "test"):
        print(f"[probe]   {s:5s} {len(d.idx[s]):>12,}")
    ho = d.uns.get("split_info", {}).get("held_out")
    print(f"[probe] held out: {list(ho) if ho is not None else '?'}")

    n_cond = max(1, a.batch_size // a.set_size)
    t0 = time.time()
    smp = ConditionSampler(d, n_cond, a.set_size, "train", 0)
    print(f"[probe] sampler built in {time.time() - t0:.0f}s over "
          f"{len(smp):,} trainable conditions")

    rows_per_step = 2 * n_cond * a.set_size          # target set + control set
    mb = rows_per_step * d.n_genes * 4 / 1e6
    for label in ("cold", "warm"):
        t0 = time.time()
        for _ in range(a.iters):
            smp.batch()
        dt = (time.time() - t0) / a.iters
        print(f"[probe] {label:4s}: {dt * 1000:7.1f} ms/step   "
              f"({rows_per_step} rows, {mb:.0f} MB, {mb / dt:7.0f} MB/s)")

    print(f"\n[probe] {len(smp):,} conditions, {n_cond} per step")
    print(f"  {'passes':>7s} {'steps':>10s} {'per variant':>13s} {'6 variants':>12s}")
    for p in a.passes:
        steps = int(round(len(smp) * p / n_cond))
        h = steps * dt / 3600
        print(f"  {p:>7d} {steps:>10,} {h:>11.1f} h {h * 6:>10.1f} h")
    print("  (energy variants run 2-4x slower than this floor: Langevin adds K inner steps)")


if __name__ == "__main__":
    main()
