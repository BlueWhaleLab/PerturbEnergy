"""
Train / val / test splits, following PerturbDiff App. A.2.3 (which follows STATE).

    unseen_context   whole contexts are held out for TEST, and `leak_frac` of the
                     held-out contexts' perturbations are moved back into TRAIN, so the
                     model sees the context under partial perturbation coverage.
                       replogle : hold out the cell line `hepg2`, leak 30%
                       pbmc     : hold out 4 of the 12 donors, leak 30%
                       tahoe    : hold out 5 of the 50 cell lines, chosen by PCA on the
                                  pseudobulk profiles, leak 30%
    unseen_pert      whole perturbations are held out for TEST (`holdout_frac` of them).
                     A model whose perturbation embedding is a plain lookup table cannot
                     predict these at all -- there is no row for a label it never saw --
                     so this is the honest setting only for models that embed a
                     perturbation from something other than its identity.
    unseen_combo     Norman2019's version of the same question, and the one this project
                     uses for it. Norman has a single context (K562), which makes the
                     held-out-context protocol degenerate, but its perturbations are 106
                     single CRISPRa targets and 131 PAIRS of them. Holding out
                     `holdout_frac` of the PAIRS gives perturbations that never occur in
                     training while both of their targets do -- so a compositional
                     embedding (models.pert_embedding) can still represent them, and the
                     test measures exactly what it should: generalisation to an unseen
                     perturbation, not a missing lookup row.
    random           cells split at random; a sanity baseline, not a paper setting.

Control cells
-------------
Cells carrying the control label are kept in TRAIN whatever the protocol
(`control_split="train"`, the default). They are the unperturbed reference population
that inference is fed at test time, and PerturbDiff reports them as a pool separate from
its train/valid/test counts. `control_split="context"` is the stricter variant: controls
follow their context, so a held-out context's controls land in test as well.

Validation
----------
`val_frac` of the TRAIN cells, drawn at random, are relabelled `val`. It exists for
checkpoint selection on reconstruction, not for measuring generalisation -- the held-out
contexts/perturbations in `test` are what the paper's numbers come from. PerturbDiff's
validation sets are similarly tiny (4,825 of 604,248 perturbed Replogle cells).
"""
import numpy as np


def select_holdout_pca(X, ctx, n_hold, seed=0):
    """Pick `n_hold` contexts spanning the transcriptional space, the way PerturbDiff
    selects Tahoe's five held-out cell lines ("PCA was first performed on pseudobulked
    expression profiles of the 50 cell lines to identify distinct phenotypic clusters;
    from these clusters, five cell lines were chosen"): pseudobulk per context, PCA,
    k-means into `n_hold` clusters, take the context closest to each centre."""
    names = np.array(sorted(set(ctx)))
    pb = np.stack([X[ctx == c].mean(0) for c in names])
    return select_holdout_pca_pb(pb, names, n_hold, seed)


def select_holdout_pca_pb(pb, names, n_hold, seed=0):
    """The same selection from an ALREADY pseudobulked matrix [n_contexts, n_genes].

    Full Tahoe never fits in memory, so its prep accumulates the per-cell-line sums as it
    streams the shards and hands them here -- identical arithmetic, no second pass.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    names = np.asarray(names)
    n_pc = max(2, min(10, len(names) - 1, pb.shape[1]))
    Z = PCA(n_components=n_pc, random_state=seed).fit_transform(pb)
    km = KMeans(n_clusters=n_hold, n_init=10, random_state=seed).fit(Z)
    held = []
    for k in range(n_hold):
        idx = np.where(km.labels_ == k)[0]
        if len(idx):
            d = np.linalg.norm(Z[idx] - km.cluster_centers_[k], axis=1)
            held.append(names[idx[int(np.argmin(d))]])
    return sorted(held)


def make_split(context, pert, control, kind="unseen_context", *, held_out=None,
               n_holdout=0, holdout_frac=0.30, leak_frac=0.30, val_frac=0.02,
               split_values=None, control_split="train", seed=0, X=None,
               holdout_by="random", combo_sep=None):
    """Return (split array of 'train'/'val'/'test', info dict).

    `split_values` is the column the hold-out is taken on when it differs from the
    model's context column -- PBMC holds out DONORS while conditioning on cell type.
    """
    rng = np.random.default_rng(seed)
    context = np.asarray(context).astype(str)
    pert = np.asarray(pert).astype(str)
    hold_on = np.asarray(split_values).astype(str) if split_values is not None else context
    n = len(pert)
    split = np.full(n, "train", dtype=object)
    is_ctrl = pert == control
    info = {"kind": kind, "control_split": control_split}

    if kind == "random":
        split[rng.random(n) < holdout_frac] = "test"

    elif kind == "unseen_pert":
        perts = np.setdiff1d(np.unique(pert), [control])
        k = max(1, int(round(len(perts) * holdout_frac)))
        held_p = set(rng.choice(perts, k, replace=False).tolist())
        split[np.isin(pert, list(held_p))] = "test"
        info["held_out_perts"] = sorted(held_p)

    elif kind == "unseen_combo":
        sep = str(combo_sep or "_")
        targets = {p: p.split(sep) for p in np.unique(pert) if p != control}
        singles = {p for p, t in targets.items() if len(t) == 1}
        pairs = [p for p, t in targets.items() if len(t) > 1 and all(x in singles for x in t)]
        orphans = [p for p, t in targets.items() if len(t) > 1 and not all(x in singles for x in t)]
        if not pairs:
            raise ValueError("unseen_combo: no multi-target perturbation whose targets "
                             "all appear on their own -- use kind='unseen_pert'")
        k = max(1, int(round(len(pairs) * holdout_frac)))
        held_p = set(rng.choice(np.array(pairs), k, replace=False).tolist())
        split[np.isin(pert, list(held_p))] = "test"
        info.update(n_singles=len(singles), n_combos=len(pairs),
                    n_uncomposable_combos=len(orphans),
                    held_out_perts=sorted(held_p))

    elif kind == "unseen_context":
        names = np.unique(hold_on)
        if held_out:
            held = [h for h in held_out if h in set(names)]
            missing = sorted(set(held_out) - set(names))
            if missing:
                raise ValueError(f"held-out contexts not in the data: {missing} "
                                 f"(available: {sorted(names)})")
        elif holdout_by == "pca" and X is not None:
            held = select_holdout_pca(X, hold_on, n_holdout or 1, seed)
        else:
            k = n_holdout or max(1, int(round(len(names) * holdout_frac)))
            held = sorted(rng.choice(names, k, replace=False).tolist())
        in_held = np.isin(hold_on, held)
        # leak_frac of the held-out contexts' perturbations go back to train
        hp = np.setdiff1d(np.unique(pert[in_held]), [control])
        leaked = set(rng.choice(hp, int(round(len(hp) * leak_frac)), replace=False).tolist()) \
            if len(hp) else set()
        split[in_held] = "test"
        split[in_held & np.isin(pert, list(leaked))] = "train"
        info["held_out"] = list(held)
        info["n_leaked_perts"] = len(leaked)
        info["n_test_perts"] = int(len(hp) - len(leaked))
    else:
        raise ValueError(f"unknown split kind {kind!r}")

    if control_split == "train":
        split[is_ctrl] = "train"

    # carve a small random validation set out of whatever ended up in train
    tr = np.where(split == "train")[0]
    if val_frac > 0 and len(tr) > 10:
        v = rng.choice(tr, max(1, int(round(len(tr) * val_frac))), replace=False)
        split[v] = "val"

    info.update({s: int((split == s).sum()) for s in ("train", "val", "test")})
    info["n_control"] = int(is_ctrl.sum())
    return split.astype(str), info
