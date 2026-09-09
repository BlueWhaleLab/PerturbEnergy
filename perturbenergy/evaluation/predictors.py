"""
Everything that turns (control cells, context, perturbation) into predicted perturbed
cells, so the model and its reference points go through exactly the same protocol.

    control      predict no change. The floor: on the delta metrics this is the honest
                 zero point, and a model that cannot beat it has learned nothing.
    mean_shift   control cells + the mean delta of that perturbation measured on TRAIN.
                 The linear baseline recent benchmarks find hard to beat. For a
                 perturbation absent from train it falls back to the mean delta over all
                 training perturbations.
    additive     control cells + the SUM of the training deltas of the perturbation's
                 individual targets. Only meaningful where perturbations compose
                 (Norman's pairs); it is the "no interaction" hypothesis, and beating it
                 is what predicting a combination actually means.
    perturbenergy   a trained checkpoint, run through the paper's Inference procedure:
                 z_b from the control cell, M candidate z_a, keep the lowest-energy one,
                 decode.
"""
import os

import numpy as np


class Predictor:
    name = "base"

    def fit(self, X_train, obs_train, control, data=None):
        return self

    def predict(self, X_ctrl, context, pert):
        raise NotImplementedError

    def can_predict(self, context, pert):
        return True


class ControlPredictor(Predictor):
    name = "control"

    def predict(self, X_ctrl, context, pert):
        return X_ctrl.copy()


class MeanShiftPredictor(Predictor):
    name = "mean_shift"

    def fit(self, X_train, obs_train, control, data=None):
        prt = obs_train["pert"].astype(str).values
        ctx = obs_train["context"].astype(str).values
        ctl_mu = {c: X_train[(ctx == c) & (prt == control)].mean(0)
                  for c in np.unique(ctx) if ((ctx == c) & (prt == control)).any()}
        glob = (X_train[prt == control].mean(0) if (prt == control).any()
                else X_train.mean(0))
        self.delta = {}
        for p in np.unique(prt):
            if p == control:
                continue
            m = prt == p
            # compare against the control of the SAME contexts this perturbation was
            # measured in, otherwise the delta absorbs the context difference
            base = np.mean([ctl_mu.get(c, glob) for c in np.unique(ctx[m])], axis=0)
            self.delta[p] = X_train[m].mean(0) - base
        self.fallback = (np.mean(list(self.delta.values()), axis=0) if self.delta
                         else np.zeros(X_train.shape[1], np.float32))
        return self

    def predict(self, X_ctrl, context, pert):
        return X_ctrl + self.delta.get(pert, self.fallback)[None]


class AdditivePredictor(MeanShiftPredictor):
    """delta(A_B) = delta(A) + delta(B): the no-interaction hypothesis."""
    name = "additive"

    def __init__(self, sep="_"):
        self.sep = sep

    def predict(self, X_ctrl, context, pert):
        if pert in self.delta:
            return X_ctrl + self.delta[pert][None]
        parts = pert.split(self.sep) if self.sep else [pert]
        known = [self.delta[p] for p in parts if p in self.delta]
        d = np.sum(known, axis=0) if known else self.fallback
        return X_ctrl + d[None]


class PerturbEnergyPredictor(Predictor):
    name = "perturbenergy"

    def __init__(self, ckpt, m_samples=16, device=None, batch_size=4096):
        import torch

        from ..training.trainer import load_checkpoint
        self.torch = torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg, self.ck = load_checkpoint(ckpt, dev)
        self.device, self.m, self.batch_size = dev, m_samples, batch_size
        self.ckpt = ckpt
        self.c2i = self.a2i = None
        self.predictable = None
        v = os.path.join(os.path.dirname(ckpt), "vocab.json")
        if os.path.exists(v):
            import json
            d = json.load(open(v))
            self.c2i, self.a2i = d["context"], d["pert"]
            self.predictable = dict(zip(d["pert"], d.get("predictable", [])))

    def fit(self, X_train, obs_train, control, data=None):
        if self.c2i is None and data is not None:
            self.c2i, self.a2i = data.c2i, data.a2i
            self.predictable = dict(zip(data.labels, data.pert_predictable.tolist()))
        return self

    def can_predict(self, context, pert):
        return bool(self.a2i is not None and str(pert) in self.a2i
                    and (self.predictable or {}).get(str(pert), True))

    def predict(self, X_ctrl, context, pert):
        torch = self.torch
        ci = self.c2i.get(str(context), -1)          # -1 -> the mean context embedding
        ai = self.a2i[str(pert)]
        out = []
        with torch.no_grad():
            for s in range(0, len(X_ctrl), self.batch_size):
                x = torch.from_numpy(X_ctrl[s:s + self.batch_size]).float().to(self.device)
                n = len(x)
                c = torch.full((n,), ci, dtype=torch.long, device=self.device)
                a = torch.full((n,), ai, dtype=torch.long, device=self.device)
                out.append(self.model.predict(x, c, a, self.m).cpu().numpy())
        return np.concatenate(out)


PREDICTORS = {"control": ControlPredictor, "mean_shift": MeanShiftPredictor,
              "additive": AdditivePredictor, "perturbenergy": PerturbEnergyPredictor}


def build_predictor(name, ckpt=None, m_samples=16, combo_sep="_", device=None):
    if name == "perturbenergy":
        if not ckpt:
            raise ValueError("predictor 'perturbenergy' needs --ckpt")
        return PerturbEnergyPredictor(ckpt, m_samples, device)
    if name == "additive":
        return AdditivePredictor(combo_sep or "_")
    return PREDICTORS[name]()
