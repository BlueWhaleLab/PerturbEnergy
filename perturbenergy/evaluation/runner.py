"""
The evaluation protocol -- identical for every dataset and every predictor, which is the
only way the numbers compare.

    groups        (context, perturbation) pairs in the TEST split with >= min_cells cells,
                  the control perturbation excluded; the largest `max_groups` are scored.
    input         the control population of the SAME context -- and, where the split is
                  taken on another column (PBMC holds out DONORS while the model conditions
                  on CELL TYPE), of the same held-out donors. That is what a virtual cell
                  is given at inference: unperturbed cells of the sample in front of you,
                  plus a target perturbation.
    prediction    predictor(control cells, context, perturbation) -> predicted cells.
    DE genes      ranked on the REAL data only (perturbed vs control, Wilcoxon), so every
                  predictor is scored on the same genes.
    metrics       metrics.group_metrics per group; averaged over groups, plus the
                  cross-group discrimination scores (PDS / RLogFC).

Writes `eval_<dataset>_<predictor>.json` (summary + per-group rows) and a one-line
`.csv` next to it.
"""
import json
import os
import time

import numpy as np

from ..data.loader import PerturbData
from .deg import de_sets
from .metrics import (fit_pca, format_tables, group_metrics, perturbation_discrimination,
                      reference_scale, summarise)
from .predictors import build_predictor


def evaluate(data, predictor, *, split="test", min_cells=30, max_groups=200, n_ctrl=300,
             de_ks=(50, 100, 200), de_method="wilcoxon", top_k=50, distribution=True,
             seed=0, verbose=True):
    """-> (summary dict, per-group rows)."""
    X, obs = data.X_np, data.obs
    is_tr = data.split == "train"
    predictor.fit(X[is_tr], obs[is_tr], data.control, data)

    ctl_all = X[data.pert == data.control]
    pca = fit_pca(ctl_all[:4000], 50, seed)
    dist_scale = reference_scale(ctl_all[:512], seed=seed)
    pca_scale = reference_scale(pca.transform(ctl_all[:512]), seed=seed)

    groups = data.groups(split, min_cells, max_groups)
    if verbose:
        print(f"[eval] {data.dataset} / {predictor.name}: {len(groups)} {split} groups "
              f"(>= {min_cells} cells), {n_ctrl} control cells each", flush=True)

    rows, dp, dr, skipped = [], [], [], []
    t0 = time.time()
    for gi, (ctx, prt, n_real) in enumerate(groups):
        if not predictor.can_predict(ctx, prt):
            skipped.append(f"{ctx}|{prt}")
            continue
        hold = (None if data.split_on_is_context
                else data.hold_units_of(ctx, prt, split))
        ctl = data.control_cells(ctx, max_cells=n_ctrl, seed=seed, hold=hold)
        if len(ctl) < 5:
            skipped.append(f"{ctx}|{prt} (no control cells)")
            continue
        X_real = data.cells_of(ctx, prt, split)
        X_pred = predictor.predict(ctl, ctx, prt)
        de_idx, _, _ = de_sets(X_real, ctl, de_ks, de_method)
        r = group_metrics(X_pred, X_real, ctl, de_idx, top_k, pca, distribution,
                          seed=seed, dist_scale=dist_scale, pca_scale=pca_scale)
        r.update(group=f"{ctx}|{prt}", context=ctx, pert=prt, n_real=int(n_real),
                 n_ctrl=int(len(ctl)))
        rows.append(r)
        dp.append(X_pred.mean(0) - ctl.mean(0))
        dr.append(X_real.mean(0) - ctl.mean(0))
        if verbose and (gi < 3 or gi % 50 == 0):
            print(f"  [{gi:4d}/{len(groups)}] {r['group'][:40]:40s} "
                  f"PDCorr {r['PDCorr']:6.3f}  de50_MSE {r.get('de50_MSE', float('nan')):7.4f}  "
                  f"DEOverlap {r['DEOverlap']:5.3f}   {time.time()-t0:5.0f}s", flush=True)

    disc = perturbation_discrimination(dp, dr) if len(rows) > 1 else {}
    drop = ("group", "context", "pert", "n_real", "n_ctrl")
    summary = summarise([{k: v for k, v in r.items() if k not in drop} for r in rows], disc)
    summary.update(dataset=data.dataset, predictor=predictor.name, split=split,
                   split_kind=data.split_kind, de_method=de_method, top_k=top_k,
                   n_skipped=len(skipped), seconds=round(time.time() - t0))
    if skipped and verbose:
        print(f"[eval] skipped {len(skipped)} groups the predictor cannot handle: "
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    return summary, rows


def write_results(summary, rows, out_json, de_ks=(50, 100, 200)):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump({"summary": summary, "per_group": rows}, open(out_json, "w"),
              indent=2, default=float)
    import csv
    with open(out_json.replace(".json", ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = [k for k in summary if not isinstance(summary[k], (list, dict))]
        w.writerow(keys)
        w.writerow([summary[k] for k in keys])
    print(f"\n=== {summary['dataset']} / {summary['predictor']} "
          f"({summary['n_groups']} groups, split={summary['split']}) ===")
    print(format_tables(summary, de_ks))
    print(f"\nwrote {out_json}")
    return out_json


def run(path, predictor_name, *, ckpt=None, out=None, **kw):
    """Convenience entry point used by scripts/evaluate.py."""
    device = kw.pop("device", "cuda")
    data = PerturbData(path, device="cpu")            # metrics are numpy-side
    pred = build_predictor(predictor_name, ckpt, kw.pop("m_samples", 16),
                           data.combo_sep, device)
    de_ks = tuple(kw.get("de_ks", (50, 100, 200)))
    summary, rows = evaluate(data, pred, **kw)
    out = out or os.path.join(os.path.dirname(path),
                              f"eval_{data.dataset}_{pred.name}.json")
    write_results(summary, rows, out, de_ks)
    return summary
