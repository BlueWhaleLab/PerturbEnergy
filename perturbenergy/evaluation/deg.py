"""
Which genes count as differentially expressed for a (context, perturbation) group.

The DE set is always derived from the REAL data -- the perturbed population against its
matched control population -- never from a prediction. Every model being compared is
therefore scored on exactly the same genes, which is the only way the DEG table means
anything across methods.

    wilcoxon  (default)  rank-sum test per gene, real vs control, ranked by p-value.
                         This is what `scanpy.tl.rank_genes_groups(method='wilcoxon')`
                         and the cell-eval suite used by STATE / PerturbDiff do.
    delta                rank by |mean(real) - mean(control)|, i.e. by |log fold change|
                         since the data is log-normalised. No test, no sample-size
                         dependence -- useful when groups are small.
"""
import numpy as np


def rank_de_genes(X_real, X_ctrl, method="wilcoxon", seed=0):
    """Return gene indices ordered MOST differentially expressed first."""
    delta = X_real.mean(0) - X_ctrl.mean(0)
    if method == "delta" or len(X_real) < 3 or len(X_ctrl) < 3:
        return np.argsort(-np.abs(delta)), np.abs(delta)
    from scipy.stats import mannwhitneyu
    with np.errstate(invalid="ignore"):
        res = mannwhitneyu(X_real, X_ctrl, axis=0, alternative="two-sided",
                           method="asymptotic")
    p = np.asarray(res.pvalue, dtype=np.float64)
    p[~np.isfinite(p)] = 1.0
    # p-value ascending; |delta| breaks the many exact ties at p == 0 or p == 1
    order = np.lexsort((-np.abs(delta), p))
    score = -np.log10(np.clip(p, 1e-300, None))
    return order, score


def de_sets(X_real, X_ctrl, ks=(50, 100, 200), method="wilcoxon"):
    """{k: indices of the top-k DE genes} plus the full ranking."""
    order, score = rank_de_genes(X_real, X_ctrl, method)
    g = X_real.shape[1]
    return {int(k): np.sort(order[:min(int(k), g)]) for k in ks}, order, score
