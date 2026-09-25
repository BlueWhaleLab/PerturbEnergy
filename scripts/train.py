"""
Train PerturbEnergy on one prepped dataset.

    python scripts/train.py --dataset replogle --run rep_joint
    python scripts/train.py --dataset replogle --run rep_noenergy --set model.use_energy=false
    python scripts/train.py --dataset norman   --run nm_ft --set train.mode=finetune
    python scripts/train.py --dataset tahoe    --run th --set loss.negatives=mismatch \
                                                          langevin.steps=40

Config: configs/default.yaml <- configs/<dataset>.yaml <- --set overrides.
Everything a run produces (checkpoints, metrics.csv, vocab.json, config_used.json) goes
to $PE_MODELS_ROOT/<run>, never into the project directory.
"""
import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np
import torch

from perturbenergy import config as C
from perturbenergy import paths as P
from perturbenergy.data.loader import PerturbData
# Imported up front, not inside `if args.evaluate`. A deferred import binds whatever is on
# disk HOURS later, when training ends -- if the code changed meanwhile, the run dies at
# the finish line against a half-updated module set. Importing here pins one version for
# the life of the process.
from perturbenergy.evaluation.predictors import build_predictor
from perturbenergy.evaluation.runner import evaluate, write_results
from perturbenergy.training.trainer import Trainer, build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--path", default=None, help="override the prepped .h5ad")
    ap.add_argument("--run", default=None, help="run name under $PE_MODELS_ROOT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--evaluate", action="store_true",
                    help="run the full evaluation on the best checkpoint when training ends")
    C.add_args(ap)
    args = ap.parse_args()

    cfg = C.load(args.config or os.path.join(P.PROJECT_ROOT, "configs",
                                             f"{args.dataset}.yaml"), args.set)
    run = args.run or f"{args.dataset}_{'energy' if cfg.model.use_energy else 'noenergy'}"
    out_dir = cfg.train.out_dir or P.run_dir(run)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    path = args.path or P.prepped_path(args.dataset)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- run:\n  python -m perturbenergy.data.prepare "
                         f"--dataset {args.dataset}")
    # `device` is where BATCHES land -- always the compute device. Whether the matrix
    # itself goes resident on the GPU is a separate question, answered by the budget:
    # gpu_resident=false just means "keep it off the card", not "hand the trainer CPU
    # tensors while the model sits on cuda".
    data = PerturbData(path, device=dev,
                       gpu_budget_gb=(cfg.data.gpu_budget_gb
                                      if cfg.data.gpu_resident else 0.0))
    data.save_vocab(os.path.join(out_dir, "vocab.json"))
    C.save(cfg, os.path.join(out_dir, "config_used.json"))

    model = build_model(data, cfg, dev)
    print(f"[run] {run} -> {out_dir}  (device {dev}, mode {cfg.train.mode}, "
          f"energy {'on' if cfg.model.use_energy else 'off'})", flush=True)
    Trainer(model, data, cfg, out_dir, dev).run()

    if args.evaluate:
        ck = os.path.join(out_dir, "model_best.pt")
        pred = build_predictor("perturbenergy", ck, cfg.eval.m_samples, data.combo_sep, dev)
        e = cfg.eval
        summary, rows = evaluate(PerturbData(path, device="cpu", verbose=False), pred,
                                 split=e.split, min_cells=e.min_cells,
                                 max_groups=e.max_groups, n_ctrl=e.n_ctrl,
                                 de_ks=tuple(e.de_ks), de_method=e.de_method,
                                 top_k=e.top_k, distribution=e.distribution, seed=e.seed,
                                 n_real=e.get("n_real", 0))
        write_results(summary, rows, os.path.join(out_dir, "eval.json"), tuple(e.de_ks))


if __name__ == "__main__":
    main()
