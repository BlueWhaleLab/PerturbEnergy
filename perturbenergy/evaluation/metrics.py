"""
Two metric groups, reported separately.

HVG group -- all 2,000 genes
    The suite recommended by "Benchmarking virtual cell models for in-the-wild
    perturbation response" [B] and the cell-eval suite PerturbDiff reports [P]:

      PCC, R2, R2_delta, MAE, MSE, RMSE            mean-profile accuracy
      PDCorr, LFCSpear, PCA_LFCSpear, LogFC_cos    is the SHIFT right, not just the level
      DirAgr, ES                                   sign / magnitude agreement
      DEPrec, DERec, DEOverlap, AUROC, AUPRC       does it find the right DE genes
      EDist, MMD, Sinkhorn, PCA_Sinkhorn           distribution-level distance
      PDS_l1/l2/cos (= RLogFC)                     is a prediction closer to ITS OWN
                                                   ground truth than to other conditions'
      RMSE_cell, R2_cell                           the predicted mean against every real
                                                   cell (reported alongside, never instead
                                                   of, the condition-level numbers)

DEG group -- the true top-50 / 100 / 200 DE genes
      de{k}_MSE, de{k}_RMSE, de{k}_MAE, de{k}_R2   plus de{k}_R2_delta and de{k}_PCC.
    Errors over all 2,000 HVGs are dominated by the ~95% of genes a perturbation barely
    moves, so the two groups can rank models differently; the DEG group is the one that
    says whether the actual response was predicted.

Everything runs on LOG-NORMALISED expression, so a log fold change is just a difference
of means: lfc = mean(log-perturbed) - mean(log-control), which equals log2FC up to 1/ln2
(and Spearman/cosine are scale-invariant anyway).
"""
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

HVG_METRICS = {
    "PCC":          ("B",   True,  "Pearson of mean expression profiles"),
    "R2":           ("P",   True,  "R^2 of mean expression"),
    "R2_delta":     ("P",   True,  "R^2 of the delta vs control"),
    "MAE":          ("P",   False, "mean |error| on the mean profile"),
    "MSE":          ("P",   False, "mean squared error on the mean profile"),
    "RMSE":         ("B",   False, "root MSE on the mean profile"),
    "PDCorr":       ("B,P", True,  "Pearson of (pred-ctrl) vs (real-ctrl)"),
    "LFCSpear":     ("B,P", True,  "Spearman of log fold changes"),
    "PCA_LFCSpear": ("B",   True,  "LFCSpear in 50-dim PCA space"),
    "LogFC_cos":    ("B",   True,  "cosine similarity of log fold changes"),
    "DirAgr":       ("P",   True,  "sign agreement of the delta (genes with a nonzero true delta)"),
    "ES":           ("P",   True,  "Pearson of effect sizes |delta|"),
    "DEPrec":       ("B,P", True,  "precision of the predicted top-k DE genes"),
    "DERec":        ("B",   True,  "recall of the true top-k DE genes"),
    "DEOverlap":    ("B,P", True,  "Jaccard of predicted/true top-k DE sets"),
    "AUROC":        ("P",   True,  "ranking true DE genes by |predicted delta|"),
    "AUPRC":        ("P",   True,  "ditto, average precision"),
    "EDist":        ("B",   False, "energy distance between cell populations"),
    "MMD":          ("B",   False, "multi-scale RBF MMD between populations"),
    "Sinkhorn":     ("B",   False, "entropic OT divergence (eps=0.05)"),
    "PCA_Sinkhorn": ("B",   False, "Sinkhorn in 50-dim PCA space"),
    "PDS_l1":       ("P",   False, "perturbation discrimination, L1 (0=perfect, .5=random)"),
    "PDS_l2":       ("P",   False, "perturbation discrimination, L2"),
    "PDS_cos":      ("P",   False, "perturbation discrimination, cosine"),
    "RLogFC":       ("B",   False, "rank of the true perturbation by cosine LFC"),
    "RMSE_cell":    ("+",   False, "predicted mean vs EVERY real cell"),
    "R2_cell":      ("+",   True,  "R^2 of the predicted mean against real cells"),
}
DEG_METRICS = {
    "MSE":      (False, "mean squared error over the true top-k DE genes"),
    "RMSE":     (False, "root MSE over the true top-k DE genes"),
    "MAE":      (False, "mean |error| over the true top-k DE genes"),
    "R2":       (True,  "R^2 of mean expression over the true top-k DE genes"),
    "R2_delta": (True,  "R^2 of the delta over the true top-k DE genes"),
    "PCC":      (True,  "Pearson of mean expression over the true top-k DE genes"),
}


def metric_info(name):
    if name.startswith("de") and "_" in name:
        k, base = name.split("_", 1)
        if base in DEG_METRICS:
            hib, desc = DEG_METRICS[base]
            return ("DEG", hib, f"{desc} (k={k[2:]})")
    if name in HVG_METRICS:
        return HVG_METRICS[name]
    return ("?", True, "")


# --------------------------------------------------------------------------- #
def _pearson(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _pairwise_sq(A, B):
    return np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0)


def _subsample(X, n, rng):
    return X[rng.choice(len(X), n, replace=False)] if len(X) > n else X


def energy_distance(X, Y, max_cells=512, seed=0):
    """E(X,Y) = 2 E||x-y|| - E||x-x'|| - E||y-y'||; zero iff the distributions match."""
    rng = np.random.default_rng(seed)
    X, Y = _subsample(X, max_cells, rng), _subsample(Y, max_cells, rng)
    return float(2 * np.sqrt(_pairwise_sq(X, Y)).mean()
                 - np.sqrt(_pairwise_sq(X, X)).mean()
                 - np.sqrt(_pairwise_sq(Y, Y)).mean())


def mmd_rbf(X, Y, scales=(0.5, 1.0, 2.0, 5.0), max_cells=512, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = _subsample(X, max_cells, rng), _subsample(Y, max_cells, rng)
    Z = np.concatenate([X, Y])
    d2 = _pairwise_sq(Z, Z)
    med = max(np.median(d2), 1e-8)
    k = sum(np.exp(-d2 / (s * med)) for s in scales)
    n = len(X)
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean())


def reference_scale(X, max_cells=512, seed=0):
    """One cost scale for the whole dataset (e.g. from the control pool). Sinkhorn must
    use a SHARED scale: normalising each pair by its own spread would make the divergence
    scale-invariant and unable to tell a shifted population from a resampled one."""
    rng = np.random.default_rng(seed)
    return float(max(_pairwise_sq(_subsample(X, max_cells, rng),
                                  _subsample(X, max_cells, rng)).mean(), 1e-12))


def sinkhorn_divergence(X, Y, eps=0.05, iters=100, max_cells=256, seed=0, scale=None):
    """S(X,Y) = OT_eps(X,Y) - .5 OT_eps(X,X) - .5 OT_eps(Y,Y), self-implemented."""
    rng = np.random.default_rng(seed)
    X, Y = _subsample(X, max_cells, rng), _subsample(Y, max_cells, rng)
    scale = scale or max(_pairwise_sq(X, X).mean(), 1e-12)

    def ot(A, B):
        C = _pairwise_sq(A, B) / scale
        n, m = len(A), len(B)
        a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
        K = np.exp(-C / eps)
        u, v = np.ones(n), np.ones(m)
        for _ in range(iters):
            u = a / np.maximum(K @ v, 1e-300)
            v = b / np.maximum(K.T @ u, 1e-300)
        return float(((u[:, None] * K * v[None, :]) * C).sum())

    return float(ot(X, Y) - 0.5 * ot(X, X) - 0.5 * ot(Y, Y))


def _pca_delta(pca, d):
    """Project a DELTA into PCA space with the components only.

    `PCA.transform` subtracts the basis mean, which is right for cells and wrong for a
    difference: it injects a constant that is identical for the prediction and the truth
    and far larger than either, so the rank correlation between them comes out ~0.98
    whatever was predicted. A delta is mean-free by construction.
    """
    return np.asarray(d) @ pca.components_.T


def _cell_level(X_real, mu_pred, idx=None):
    """The predicted MEAN profile scored against every individual real cell.

    cell_MSE = within-group variance + condition_MSE, and on real data the first term is
    ~95% of it -- so this mostly measures whether a model produces a plausible population
    at all. R^2 follows the sklearn convention on the flattened (cell, gene) entries.
    """
    Xr = X_real if idx is None else X_real[:, idx]
    mp = mu_pred if idx is None else mu_pred[idx]
    res = (Xr - mp[None]) ** 2
    ss = float(((Xr - Xr.mean()) ** 2).sum())
    return float(np.sqrt(res.mean())), (float(1.0 - res.sum() / ss) if ss > 0 else float("nan"))


# --------------------------------------------------------------------------- #
def group_metrics(X_pred, X_real, X_ctrl, de_idx=None, top_k=50, pca=None,
                  distribution=True, eps=0.05, seed=0, dist_scale=None, pca_scale=None):
    """Every metric for ONE (context, perturbation) group. Returns a flat dict.

    `de_idx` is {k: gene indices} from deg.de_sets, computed on the REAL data.
    """
    mu_p, mu_r, mu_c = X_pred.mean(0), X_real.mean(0), X_ctrl.mean(0)
    d_p, d_r = mu_p - mu_c, mu_r - mu_c              # log fold changes
    out = {}

    err = mu_p - mu_r
    out["PCC"] = _pearson(mu_p, mu_r)
    out["R2"] = float(r2_score(mu_r, mu_p))
    out["R2_delta"] = float(r2_score(d_r, d_p))
    out["MAE"] = float(np.abs(err).mean())
    out["MSE"] = float((err ** 2).mean())
    out["RMSE"] = float(np.sqrt(out["MSE"]))
    out["PDCorr"] = _pearson(d_p, d_r)
    out["LFCSpear"] = float(spearmanr(d_p, d_r).statistic)
    out["LogFC_cos"] = _cosine(d_p, d_r)
    # restricted to genes the perturbation actually moves: with log-normalised counts a
    # good fraction of HVGs have a TRUE delta of exactly 0 in a small group, and those can
    # only "agree" with a prediction of exactly 0, which deflates every real model's score
    # towards the do-nothing baseline's.
    moved = d_r != 0
    out["DirAgr"] = (float((np.sign(d_p[moved]) == np.sign(d_r[moved])).mean())
                     if moved.any() else float("nan"))
    out["ES"] = _pearson(np.abs(d_p), np.abs(d_r))
    if pca is not None:
        out["PCA_LFCSpear"] = float(spearmanr(_pca_delta(pca, d_p),
                                              _pca_delta(pca, d_r)).statistic)
    out["RMSE_cell"], out["R2_cell"] = _cell_level(X_real, mu_p)

    # --- DEG group: the true top-k DE genes ------------------------------- #
    for k, idx in (de_idx or {}).items():
        e = mu_p[idx] - mu_r[idx]
        out[f"de{k}_MSE"] = float((e ** 2).mean())
        out[f"de{k}_RMSE"] = float(np.sqrt(out[f"de{k}_MSE"]))
        out[f"de{k}_MAE"] = float(np.abs(e).mean())
        out[f"de{k}_R2"] = float(r2_score(mu_r[idx], mu_p[idx]))
        out[f"de{k}_R2_delta"] = float(r2_score(d_r[idx], d_p[idx]))
        out[f"de{k}_PCC"] = _pearson(mu_p[idx], mu_r[idx])

    # --- DE-set retrieval at top_k ---------------------------------------- #
    true_de = set((de_idx or {}).get(top_k, np.argsort(-np.abs(d_r))[:top_k]).tolist())
    pred_de = set(np.argsort(-np.abs(d_p))[:len(true_de)].tolist())
    inter = len(true_de & pred_de)
    out["DEPrec"] = inter / max(len(pred_de), 1)
    out["DERec"] = inter / max(len(true_de), 1)
    out["DEOverlap"] = inter / max(len(true_de | pred_de), 1)
    y = np.zeros(len(d_r)); y[list(true_de)] = 1.0
    if 0 < y.sum() < len(y):
        out["AUROC"] = float(roc_auc_score(y, np.abs(d_p)))
        out["AUPRC"] = float(average_precision_score(y, np.abs(d_p)))

    if distribution:
        out["EDist"] = energy_distance(X_pred, X_real, seed=seed)
        out["MMD"] = mmd_rbf(X_pred, X_real, seed=seed)
        out["Sinkhorn"] = sinkhorn_divergence(X_pred, X_real, eps=eps, seed=seed,
                                              scale=dist_scale)
        if pca is not None:
            out["PCA_Sinkhorn"] = sinkhorn_divergence(pca.transform(X_pred),
                                                      pca.transform(X_real),
                                                      eps=eps, seed=seed, scale=pca_scale)
    return out


def perturbation_discrimination(pred_deltas, real_deltas):
    """PDS / RLogFC: is a prediction closer to ITS OWN ground truth than to the other
    perturbations'? For each group, rank all real deltas by distance to the predicted
    delta and take the normalised rank of the true one -- 0 perfect, ~0.5 random."""
    P, R = np.asarray(pred_deltas), np.asarray(real_deltas)
    n = len(P)
    if n < 2:
        return {}
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    Rn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    dists = {"PDS_l1": np.abs(P[:, None, :] - R[None, :, :]).sum(-1),
             "PDS_l2": np.sqrt(_pairwise_sq(P, R)),
             "PDS_cos": 1.0 - Pn @ Rn.T}
    out = {}
    for name, D in dists.items():
        out[name] = float(np.mean([(rankdata(D[i], method="average")[i] - 1) / (n - 1)
                                   for i in range(n)]))
    out["RLogFC"] = out["PDS_cos"]
    return out


def summarise(rows, disc=None):
    """nan-safe average of per-group dicts, merged with the discrimination scores."""
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        out[k] = float(np.mean(v)) if v else float("nan")
    out["n_groups"] = len(rows)
    if disc:
        out.update(disc)
    return out


def fit_pca(X, n_components=50, seed=0):
    n = max(2, min(n_components, min(X.shape) - 1))
    return PCA(n_components=n, random_state=seed).fit(X)


def format_tables(summary, de_ks=(50, 100, 200)):
    """The two tables the report asks for: HVG group, then DEG group."""
    lines = ["  -- HVG group (all genes) " + "-" * 44]
    for k in HVG_METRICS:
        if k not in summary or not np.isfinite(summary[k]):
            continue
        src, hib, desc = HVG_METRICS[k]
        lines.append(f"  {k:14s} [{src:3s}] {'^' if hib else 'v'} {summary[k]:10.4f}  {desc}")
    lines.append("  -- DEG group (true top-k DE genes) " + "-" * 34)
    head = "  " + " " * 12 + "".join(f"{'top' + str(k):>12s}" for k in de_ks)
    lines.append(head)
    for base, (hib, _desc) in DEG_METRICS.items():
        vals = [summary.get(f"de{k}_{base}", float("nan")) for k in de_ks]
        if not any(np.isfinite(v) for v in vals):
            continue
        lines.append(f"  {base:10s} {'^' if hib else 'v'} "
                     + "".join(f"{v:12.4f}" for v in vals))
    return "\n".join(lines)
