"""
Comparison tables over every eval_*.json / eval.json under the run root.

    python scripts/report.py                     # every dataset
    python scripts/report.py --dataset replogle
    python scripts/report.py --csv out.csv

Two groups, and the HVG one follows the metric taxonomy of "Benchmarking virtual cell
models for in-the-wild perturbation response" (the suite PerturbDiff reports):

    (a) correlation-based measures   PCC R2 PDCorr R2_delta LFCSpear PCA_LFCSpear
                                     LogFC_cos DirAgr RLogFC PDS_cos PDS_l1 PDS_l2
    (b) DEG-related scores           DEPrec DERec DEOverlap ES AUROC AUPRC
    (c) distribution-level distances RMSE MSE MAE RMSE_cell R2_cell EDist MMD
                                     Sinkhorn PCA_Sinkhorn

Metrics we compute beyond that paper are slotted into the family they belong to
(R2/R2_delta -> correlation, ES/AUROC/AUPRC -> DEG, MAE/MSE/cell-level -> distribution,
which is where the paper itself puts RMSE).

The DEG group is printed one metric family at a time so each row reads as a clean sweep
over K in {25, 50, 100, 200}. The (a)/(b)/(c) split does NOT apply there -- every DEG
number is already restricted to the true top-K genes.

`*` marks the best value in a column; the arrow row says which direction is better.
"""
import argparse
import csv
import glob as globmod
import json
import os

import _bootstrap  # noqa: F401
import numpy as np

from perturbenergy import paths as P
from perturbenergy.evaluation.metrics import DEG_METRICS, HVG_METRICS

HVG_CORR = ["PCC", "R2", "PDCorr", "R2_delta", "LFCSpear", "PCA_LFCSpear",
            "LogFC_cos", "DirAgr", "RLogFC", "PDS_cos", "PDS_l1", "PDS_l2"]
HVG_DEG = ["DEPrec", "DERec", "DEOverlap", "ES", "AUROC", "AUPRC"]
HVG_DIST = ["RMSE", "MSE", "MAE", "RMSE_cell", "R2_cell",
            "EDist", "MMD", "Sinkhorn", "PCA_Sinkhorn"]
HVG_BLOCKS = [("(a) correlation-based measures", HVG_CORR),
              ("(b) DEG-related scores", HVG_DEG),
              ("(c) distribution-level distances", HVG_DIST)]
DEG_FAMILIES = ["PCC", "R2", "R2_delta", "MAE", "MSE", "RMSE", "RMSE_cell", "R2_cell"]

# run-name suffix -> the label and the order rows appear in
VARIANTS = [("ref_control", "control (no change)"),
            ("ref_mean_shift", "mean_shift (linear)"),
            ("ref_additive", "additive (linear)"),
            ("noenergy", "VAE, energy OFF"),
            ("energy_lang", "energy: Langevin"),
            ("energy_mis", "energy: mismatch"),
            ("energy_sel", "energy: Langevin+select"),
            ("ft_lang", "energy: 2-stage finetune"),
            ("obs_pairing", "[abl] observed pairing")]


def collect(patterns, dataset=None):
    rows = {}
    for pat in patterns:
        for f in sorted(globmod.glob(pat, recursive=True)):
            try:
                s = json.load(open(f))["summary"]
            except Exception as e:                                     # noqa: BLE001
                print(f"[skip] {f}: {e}")
                continue
            if dataset and s.get("dataset") != dataset:
                continue
            s["_run"] = os.path.basename(os.path.dirname(f))
            rows.setdefault(s.get("dataset", "?"), []).append(s)
    return rows


def _order(rows, ds):
    """Rows in VARIANTS order, labelled; anything unrecognised goes last."""
    out, seen = [], set()
    for suf, label in VARIANTS:
        for r in rows:
            if r["_run"] == f"{ds}_{suf}" and r["_run"] not in seen:
                seen.add(r["_run"])
                out.append((label, r))
    for r in rows:
        if r["_run"] not in seen:
            out.append((r["_run"], r))
    return out


def _table(labelled, cols, direction, title, w=26):
    cols = [c for c in cols if any(c in r for _, r in labelled)]
    if not cols:
        return
    best = {}
    for c in cols:
        v = [r[c] for _, r in labelled
             if isinstance(r.get(c), (int, float)) and np.isfinite(r[c])]
        if v:
            best[c] = max(v) if direction.get(c, True) else min(v)
    head = f"{'':<{w}}" + "".join(f"{c:>13s}" for c in cols)
    print(f"\n  {title}")
    print("  " + "-" * len(head))
    print("  " + head)
    print("  " + "-" * len(head))
    for label, r in labelled:
        line = f"{label:<{w}}"
        for c in cols:
            v = r.get(c)
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                line += f"{'-':>13s}"
            else:
                star = "*" if c in best and abs(v - best[c]) < 1e-12 else " "
                line += f"{v:>12.4f}{star}"
        print("  " + line)
    print("  " + "-" * len(head))
    print("  " + f"{'direction':<{w}}"
          + "".join(f"{('^ hi' if direction.get(c, True) else 'v lo'):>13s}" for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="*", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--de_ks", type=int, nargs="*", default=[25, 50, 100, 200])
    ap.add_argument("--only", default="both", choices=["hvg", "deg", "both"],
                    help="print just one of the two groups")
    ap.add_argument("--csv", default=None, help="also write a wide CSV here")
    args = ap.parse_args()

    pats = args.glob or [os.path.join(P.MODELS_ROOT, "**", "eval*.json"),
                         os.path.join(P.PREPPED, "eval_*.json")]
    by_ds = collect(pats, args.dataset)
    if not by_ds:
        raise SystemExit(f"no eval json under {pats}")

    for ds in sorted(by_ds):
        rows = by_ds[ds]
        labelled = _order(rows, ds)
        r0 = rows[0]
        print(f"\n{'=' * 118}")
        print(f"{ds.upper()}   split={r0.get('split_kind')} | "
              f"{r0.get('n_groups')} test groups | DE ranked on real data "
              f"({r0.get('de_method')})")
        print("=" * 118)
        if args.only in ("hvg", "both"):
            print("\nHVG group -- all 2,000 genes")
            for title, cols in HVG_BLOCKS:
                _table(labelled, cols, {c: HVG_METRICS[c][1] for c in cols}, title)
        if args.only == "hvg":
            continue

        print(f"\n\nDEG group -- true top-K DE genes, K in {args.de_ks}")
        for fam in DEG_FAMILIES:
            cols = [f"de{k}_{fam}" for k in args.de_ks]
            hib = DEG_METRICS[fam][0]
            _table(labelled, cols, {c: hib for c in cols},
                   f"{fam}  ({'higher better' if hib else 'lower better'})")

    if args.csv:
        allr = [r for v in by_ds.values() for r in v]
        keys = sorted({k for r in allr for k in r if not isinstance(r[k], (list, dict))})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(allr)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
