"""
Low-level readers for the raw .h5ad sources.

None of these files fit comfortably in memory as written (Replogle is 22 GB dense, the
Parse PBMC atlas ~800 GB sparse), so every reader takes a row selection and a gene
selection and materialises only that. AnnData stores X three different ways and they are
NOT interchangeable -- in CSR `indptr` indexes rows, in CSC it indexes columns -- so the
readers dispatch on the `encoding-type` attribute rather than guessing.
"""
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_cat(obs_grp, col):
    """An obs column (categorical or plain) as a string array."""
    g = obs_grp[col]
    if isinstance(g, h5py.Group) and "categories" in g:
        cats = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in g["categories"][:]])
        return cats[g["codes"][:]]
    v = g[:]
    if v.dtype.kind in "SO":
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])
    return v.astype(str)


def var_names(f, prefer=("gene_name_index", "gene_symbol", "_index")):
    """Gene symbols, whichever column this file happens to use."""
    v = f["var"]
    key = next((k for k in prefer if k in v), list(v.keys())[0])
    g = v[key]
    if isinstance(g, h5py.Group) and "categories" in g:
        cats = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in g["categories"][:]])
        return cats[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def x_encoding(f):
    X = f["X"]
    if isinstance(X, h5py.Dataset):
        return "dense"
    k = dict(X.attrs).get("encoding-type", "csr_matrix")
    return k.decode() if isinstance(k, bytes) else k


def x_shape(f):
    X = f["X"]
    if isinstance(X, h5py.Dataset):
        return tuple(X.shape)
    return tuple(int(v) for v in np.asarray(dict(X.attrs)["shape"]).ravel())


def read_rows(f, rows, gene_idx, chunk=20000):
    """Materialise `rows` x `gene_idx` from f['X'] as a dense float32 array.

    Rows are read in sorted order (HDF5 fancy indexing requires it and random access over
    a 22 GB dataset is otherwise pathological) and restored to the caller's order at the
    end.
    """
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    srt = rows[order]
    gene_idx = np.asarray(gene_idx, dtype=np.int64)
    kind = x_encoding(f)
    n_rows_total, n_genes_total = x_shape(f)
    out = np.zeros((len(rows), len(gene_idx)), np.float32)

    if kind == "dense":
        X = f["X"]
        for s in range(0, len(srt), chunk):
            blk = srt[s:s + chunk]
            # a contiguous run reads ~10x faster as a slice than as a fancy index
            raw = (X[blk[0]:blk[-1] + 1] if blk[-1] - blk[0] + 1 == len(blk)
                   else X[blk])
            out[s:s + len(blk)] = np.asarray(raw[:, gene_idx], dtype=np.float32)
    elif kind == "csc_matrix":
        # indptr runs over GENES; `indices` are cell ids. Reading gene by gene means one
        # seek per selected gene instead of one per selected cell.
        X = f["X"]
        indptr = X["indptr"][:]
        if int(indptr[-1]) <= 200_000_000:              # fits: let scipy do the indexing
            M = sp.csc_matrix((X["data"][:], X["indices"][:], indptr),
                              shape=(n_rows_total, n_genes_total)).tocsr()
            out = np.asarray(M[srt][:, gene_idx].todense(), dtype=np.float32)
            inv = np.empty(len(order), np.int64)
            inv[order] = np.arange(len(order))
            return out[inv]
        pos = np.full(n_rows_total, -1, np.int64)
        pos[srt] = np.arange(len(srt))
        for j, g in enumerate(gene_idx):
            s, e = int(indptr[g]), int(indptr[g + 1])
            if e <= s:
                continue
            ridx = X["indices"][s:e]
            m = pos[ridx] >= 0
            if m.any():
                out[pos[ridx[m]], j] = X["data"][s:e][m]
    else:                                                    # csr_matrix
        X = f["X"]
        indptr = X["indptr"][:]
        keep = np.full(n_genes_total, -1, np.int64)
        keep[gene_idx] = np.arange(len(gene_idx))
        nnz = int(indptr[-1])
        if nnz <= 60_000_000:                                # small enough: let scipy index
            M = sp.csr_matrix((X["data"][:], X["indices"][:], indptr),
                              shape=(n_rows_total, n_genes_total))
            out = np.asarray(M[srt][:, gene_idx].todense(), dtype=np.float32)
        else:
            for i, r in enumerate(srt):
                s, e = int(indptr[r]), int(indptr[r + 1])
                if e <= s:
                    continue
                cols = X["indices"][s:e]
                m = keep[cols] >= 0
                if m.any():
                    out[i, keep[cols[m]]] = X["data"][s:e][m]

    inv = np.empty(len(order), np.int64)
    inv[order] = np.arange(len(order))
    return out[inv]


def sample_rows(keys, cap, max_cells=0, seed=0, control_mask=None, cap_control=None):
    """Cap the number of cells per group, then cap the total.

    `keys` is one label per cell (the group it belongs to, e.g. context|donor|pert).
    The CONTROL population gets its own, larger cap: it is the reference every group is
    compared against, so starving it would bias every delta. cap<=0 means "no cap".
    """
    rng = np.random.default_rng(seed)
    n = len(keys)
    if cap <= 0 and not max_cells:
        return np.arange(n)
    df = pd.DataFrame({"i": np.arange(n), "k": keys,
                       "ctl": (control_mask if control_mask is not None
                               else np.zeros(n, bool))})
    picked = []
    for (k, ctl), sub in df.groupby(["k", "ctl"], sort=False):
        idx = sub["i"].values
        c = (cap_control if ctl and cap_control else cap)
        picked.append(idx if c <= 0 or len(idx) <= c else rng.choice(idx, c, replace=False))
    rows = np.concatenate(picked) if picked else np.array([], np.int64)
    if max_cells and len(rows) > max_cells:
        rows = rng.choice(rows, max_cells, replace=False)
    return np.sort(rows)


def lognorm(counts, target=1e4):
    """scanpy's normalize_total(1e4) + log1p, on a dense array."""
    lib = np.asarray(counts.sum(-1)).reshape(-1, 1).clip(min=1.0)
    return np.log1p(counts / lib * target).astype(np.float32)


def looks_like_counts(X, n=200000):
    """Raw counts (large, integral) vs already log-normalised values."""
    s = X[:n] if len(X) > n else X
    return bool(s.max() > 50 and np.abs(s - np.round(s)).mean() < 1e-3)


def top_hvg(X, n_top=2000, counts=True):
    """Top-N highly variable genes, as PerturbDiff does ('sc.pp.highly_variable_genes').

    `seurat_v3` is the flavour that runs on raw counts; on already-normalised data we use
    the default (mean/dispersion) flavour, which is what `sc.pp.highly_variable_genes`
    expects post-log1p. Falls back to plain variance ranking if scanpy is unavailable.
    """
    try:
        import anndata as ad
        import scanpy as sc
        A = ad.AnnData(X=np.asarray(X, np.float32))
        if counts:
            sc.pp.highly_variable_genes(A, n_top_genes=n_top, flavor="seurat_v3")
        else:
            sc.pp.highly_variable_genes(A, n_top_genes=n_top)
        idx = np.where(A.var["highly_variable"].values)[0]
        if len(idx) >= n_top:
            return np.sort(idx[:n_top])
        print(f"[hvg] scanpy returned {len(idx)} genes; topping up by variance")
    except Exception as e:                                            # noqa: BLE001
        print(f"[hvg] scanpy unavailable/failed ({e}); ranking by variance")
        idx = np.array([], np.int64)
    v = X.var(0)
    v[idx] = np.inf
    return np.sort(np.argsort(v)[-n_top:])
