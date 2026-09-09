"""
Score a predictor on a prepped dataset with the full metric suite.

    python scripts/evaluate.py --dataset norman --predictor control
    python scripts/evaluate.py --dataset norman --predictor mean_shift
    python scripts/evaluate.py --dataset norman --predictor additive
    python scripts/evaluate.py --dataset replogle --predictor perturbenergy \
        --ckpt $PE_MODELS_ROOT/rep_joint/model_best.pt

Results land next to the checkpoint (or beside the prepped file for the reference
predictors) as eval_<dataset>_<predictor>.json/.csv.
"""
import argparse
import os

import _bootstrap  # noqa: F401

from perturbenergy import config as C
from perturbenergy import paths as P
from perturbenergy.data.loader import PerturbData
from perturbenergy.evaluation.predictors import PREDICTORS, build_predictor
from perturbenergy.evaluation.runner import evaluate, write_results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--path", default=None)
    ap.add_argument("--predictor", default="control", choices=list(PREDICTORS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default=None)
    ap.add_argument("--min_cells", type=int, default=None)
    ap.add_argument("--max_groups", type=int, default=None)
    ap.add_argument("--n_ctrl", type=int, default=None)
    ap.add_argument("--m_samples", type=int, default=None)
    ap.add_argument("--de_ks", type=int, nargs="*", default=None)
    ap.add_argument("--de_method", default=None, choices=["wilcoxon", "delta"])
    ap.add_argument("--no_distribution", action="store_true",
                    help="skip the O(n^2) population distances (energy/MMD/Sinkhorn)")
    C.add_args(ap)
    args = ap.parse_args()

    cfg = C.load(args.config or os.path.join(P.PROJECT_ROOT, "configs",
                                             f"{args.dataset}.yaml"), args.set)
    e = cfg.eval
    path = args.path or P.prepped_path(args.dataset)
    data = PerturbData(path, device="cpu")
    pred = build_predictor(args.predictor, args.ckpt,
                           args.m_samples or e.m_samples, data.combo_sep, args.device)
    de_ks = tuple(args.de_ks if args.de_ks is not None else e.de_ks)
    summary, rows = evaluate(
        data, pred, split=args.split or e.split,
        min_cells=args.min_cells if args.min_cells is not None else e.min_cells,
        max_groups=args.max_groups if args.max_groups is not None else e.max_groups,
        n_ctrl=args.n_ctrl or e.n_ctrl, de_ks=de_ks,
        de_method=args.de_method or e.de_method, top_k=e.top_k,
        distribution=(not args.no_distribution) and e.distribution, seed=e.seed)
    out = args.out or os.path.join(
        os.path.dirname(args.ckpt) if args.ckpt else os.path.dirname(path),
        f"eval_{data.dataset}_{pred.name}.json")
    write_results(summary, rows, out, de_ks)


if __name__ == "__main__":
    main()
