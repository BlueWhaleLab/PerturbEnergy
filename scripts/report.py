"""
Collect every eval_*.json under the run root into two comparison tables -- the HVG group
and the DEG group -- one row per (dataset, predictor/run).

    python scripts/report.py                          # everything under $PE_MODELS_ROOT
    python scripts/report.py --dataset replogle
    python scripts/report.py --glob '/path/**/eval_*.json' --out report.csv

Metrics are printed with an arrow for their direction; the best value in each column is
starred. Nothing is ranked automatically -- HVG-group and DEG-group numbers can disagree,
and which one matters is a judgement about the question being asked.
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

HVG_SHOW = ["MSE", "RMSE", "MAE", "R2", "R2_delta", "PCC", "PDCorr", "LFCSpear",
            "LogFC_cos", "DirAgr", "ES", "DEPrec", "DEOverlap", "AUROC", "AUPRC",
            "EDist", "MMD", "Sinkhorn", "PDS_cos"]


def collect(patterns):
    rows = []
    for pat in patterns:
        for f in sorted(globmod.glob(pat, recursive=True)):
            try:
                s = json.load(open(f))["summary"]
            except Exception as e:                                     # noqa: BLE001
                print(f"[skip] {f}: {e}")
                continue
            s["_file"] = f
            s["_run"] = os.path.basename(os.path.dirname(f))
            rows.append(s)
    return rows


def _table(rows, cols, direction, label, name_w=34):
    if not cols:
        return ""
    head = f"{'run':<{name_w}}" + "".join(f"{c:>12s}" for c in cols)
    best = {}
    for c in cols:
        v = [r[c] for r in rows if isinstance(r.get(c), (int, float)) and np.isfinite(r[c])]
        if v:
            best[c] = max(v) if direction.get(c, True) else min(v)
    out = [f"\n{label}", "-" * len(head), head, "-" * len(head)]
    for r in rows:
        line = f"{(r['_run'] + '/' + str(r.get('predictor', '')))[:name_w]:<{name_w}}"
        for c in cols:
            v = r.get(c)
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                line += f"{'-':>12s}"
            else:
                line += f"{v:>11.4f}{'*' if c in best and abs(v - best[c]) < 1e-12 else ' '}"
        out.append(line)
    arrows = f"{'(higher better)':<{name_w}}" + "".join(
        f"{('^' if direction.get(c, True) else 'v'):>12s}" for c in cols)
    out.append("-" * len(head))
    out.append(arrows)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="*", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--de_ks", type=int, nargs="*", default=[50, 100, 200])
    ap.add_argument("--out", default=None, help="also write a wide CSV here")
    args = ap.parse_args()

    pats = args.glob or [os.path.join(P.MODELS_ROOT, "**", "eval*.json"),
                         os.path.join(P.PREPPED, "eval_*.json")]
    rows = collect(pats)
    if args.dataset:
        rows = [r for r in rows if r.get("dataset") == args.dataset]
    if not rows:
        raise SystemExit(f"no eval json found under {pats}")
    rows.sort(key=lambda r: (str(r.get("dataset")), str(r.get("_run"))))

    for ds in sorted({str(r.get("dataset")) for r in rows}):
        sub = [r for r in rows if str(r.get("dataset")) == ds]
        print(f"\n{'=' * 100}\n{ds}   ({sub[0].get('split_kind')} split, "
              f"{sub[0].get('n_groups')} groups, DE by {sub[0].get('de_method')})\n{'=' * 100}")
        cols = [c for c in HVG_SHOW if any(c in r for r in sub)]
        print(_table(sub, cols, {c: HVG_METRICS[c][1] for c in cols}, "HVG group (all 2,000 genes)"))
        for base, (hib, _d) in DEG_METRICS.items():
            cs = [f"de{k}_{base}" for k in args.de_ks if any(f"de{k}_{base}" in r for r in sub)]
            if cs:
                print(_table(sub, cs, {c: hib for c in cs}, f"DEG group -- {base}"))

    if args.out:
        keys = sorted({k for r in rows for k in r if not isinstance(r[k], (list, dict))})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
