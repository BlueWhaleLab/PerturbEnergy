"""
Turn each raw dataset into the ONE uniform prepped .h5ad everything downstream reads
(schema in registry.py, split protocol in splits.py).

    python -m perturbenergy.data.prepare --dataset replogle
    python -m perturbenergy.data.prepare --dataset norman
    python -m perturbenergy.data.prepare --dataset tahoe --max_cells 1000000
    python -m perturbenergy.data.prepare --dataset pbmc  --max_cells 1000000

Every dataset ends up as log1p(CP10K) expression on 2,000 genes -- PerturbDiff's shipped
panel where one exists, otherwise the top-2,000 HVGs -- with a train/val/test column.

Per-dataset defaults (cell caps, held-out contexts) come from registry.py and PREP_DEFAULTS
below; the command line overrides them.
"""
import argparse
import glob
import json
import os
import sys
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from .. import paths as P
from . import h5utils as H
from .registry import DATASETS, load_panel, spec
from .splits import make_split

# cells kept per (context[, split_on], perturbation) group, and overall.
# 0 = no cap: Replogle and Norman are small enough to prep whole.
PREP_DEFAULTS = {
    "replogle": dict(cap_per_group=0, cap_control=0, max_cells=0),
    "norman":   dict(cap_per_group=0, cap_control=0, max_cells=0),
    # Tahoe and PBMC are far too big to prep whole. The caps buy CONDITION COVERAGE over
    # depth: a benchmark wants many (context, perturbation) groups with enough cells to
    # form a population, not a few groups with thousands.
    "tahoe":    dict(cap_per_group=100, cap_control=2000, max_cells=0),
    "pbmc":     dict(cap_per_group=100, cap_control=2000, max_cells=0),
}


# --------------------------------------------------------------------------- #
# per-dataset readers -> (X, obs, gene_names, control_label)
# --------------------------------------------------------------------------- #
def read_h5ad_generic(s, args):
    """Replogle and anything else that is one .h5ad with the columns named in the spec."""
    with h5py.File(s["path"], "r") as f:
        obs = f["obs"]
        ctx = H.read_cat(obs, s["context"])
        prt = H.read_cat(obs, s["pert"])
        if s.get("control_col"):                       # control flagged in another column
            flag = H.read_cat(obs, s["control_col"])
            prt = np.where(flag == s["control"], s["control"], prt)
        hold = H.read_cat(obs, s["split_col"]) if s.get("split_col") else None
        genes = H.var_names(f)
        gene_idx = _panel_index(s, genes)
        keys = _group_keys(ctx, prt, hold)
        rows = H.sample_rows(keys, args.cap_per_group, args.max_cells, args.seed,
                             control_mask=(prt == s["control"]),
                             cap_control=args.cap_control)
        print(f"[{args.dataset}] {len(ctx)} cells, {len(genes)} genes -> "
              f"panel {len(gene_idx)} | contexts {len(set(ctx))} "
              f"| perturbations {len(set(prt))} | keeping {len(rows)} cells", flush=True)
        X = H.read_rows(f, rows, gene_idx)
    O = pd.DataFrame({"context": ctx[rows], "pert": prt[rows]})
    if hold is not None:
        O["split_on"] = hold[rows]
    return X, O, genes[gene_idx], s["control"]


def read_norman(s, args):
    """Norman2019 CRISPRa. Two shapes exist: scPerturb's single filtered atlas (controls
    inside, `perturbation == 'control'`) and the HuggingFace pair (perturbed + controls in
    separate files, capped at 120 cells per perturbation). Both are raw counts."""
    files = [s["path"]] + ([s["control_path"]] if s.get("control_path") else [])
    Xs, C, Pp, genes = [], [], [], None
    for p in files:
        with h5py.File(p, "r") as f:
            prt = H.read_cat(f["obs"], s["pert"])
            ctx = (H.read_cat(f["obs"], s["context"]) if s["context"] in f["obs"]
                   else np.full(len(prt), "K562"))
            g = H.var_names(f)
            genes = g if genes is None else genes
            if not np.array_equal(g, genes):
                raise ValueError(f"{p}: gene order differs from the first Norman file")
            keys = _group_keys(ctx, prt, None)
            rows = H.sample_rows(keys, args.cap_per_group, 0, args.seed,
                                 control_mask=(prt == s["control"]),
                                 cap_control=args.cap_control)
            Xs.append(H.read_rows(f, rows, np.arange(len(genes))))
            C.append(ctx[rows]); Pp.append(prt[rows])
    X = np.concatenate(Xs)
    O = pd.DataFrame({"context": np.concatenate(C), "pert": np.concatenate(Pp)})
    # scPerturb spells the control 'control'; normalise a few other spellings too
    O["pert"] = O["pert"].replace({"ctrl": "control", "non-targeting": "control"})
    n_ctl = int((O["pert"] == s["control"]).sum())
    print(f"[norman] {len(X)} cells ({n_ctl} control) | {O['pert'].nunique()} "
          f"perturbations | {len(genes)} genes"
          f"{'  [HuggingFace subsample]' if s.get('is_fallback') else ''}", flush=True)
    return X, O, genes, s["control"]


def read_tahoe(s, args):
    """Tahoe-100M parquet shards: each cell is a list of (gene token, count). We keep the
    2,000 PerturbDiff panel genes, mapped from symbols to Tahoe token ids."""
    import pyarrow.parquet as pq
    import scipy.sparse as sp

    meta = os.path.join(P.TAHOE_DIR, "metadata", "gene_metadata.parquet")
    if not os.path.exists(meta):
        raise FileNotFoundError(f"{meta} -- run `python -m perturbenergy.data.download "
                                f"--what tahoe` first")
    gm = pq.read_table(meta).to_pandas()
    n_vocab = int(gm.token_id.max()) + 1
    panel = set(load_panel(s) or [])
    sym2tok = {str(a): int(b) for a, b in zip(gm.gene_symbol, gm.token_id)}
    toks = np.array(sorted({sym2tok[g] for g in panel if g in sym2tok}))
    syms = gm.set_index("token_id").loc[toks, "gene_symbol"].astype(str).values
    print(f"[tahoe] panel: {len(toks)}/{len(panel)} PerturbDiff genes mapped to token ids")

    shards = sorted(glob.glob(os.path.join(s["path"], "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {s['path']}")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(shards)
    cols = ["genes", "expressions", s["context"], s["pert"]]
    Xs, C, Pp, seen, total = [], [], [], {}, 0
    for si, path in enumerate(shards):
        tbl = pq.read_table(path, columns=cols)
        ga, ea = tbl["genes"].combine_chunks(), tbl["expressions"].combine_chunks()
        offs = np.asarray(ga.offsets)
        Xc = sp.csr_matrix((np.clip(np.asarray(ea.values).astype(np.float32), 0, None),
                            np.asarray(ga.values), offs),
                           shape=(len(offs) - 1, n_vocab))
        cl = tbl[s["context"]].to_pandas().astype(str).values
        dr = tbl[s["pert"]].to_pandas().astype(str).values
        take = []
        for j in range(len(cl)):
            k = (cl[j], dr[j])
            cap = (args.cap_control if dr[j] == s["control"] else args.cap_per_group)
            if cap <= 0 or seen.get(k, 0) < cap:
                seen[k] = seen.get(k, 0) + 1
                take.append(j)
        if take:
            take = np.asarray(take)
            Xs.append(Xc[take][:, toks].toarray().astype(np.float32))
            C.append(cl[take]); Pp.append(dr[take])
            total += len(take)
        if si % 25 == 0:
            print(f"  [tahoe] shard {si}/{len(shards)}: {total} cells kept", flush=True)
        if args.max_cells and total >= args.max_cells:
            break
    X = np.concatenate(Xs)[: args.max_cells or None]
    O = pd.DataFrame({"context": np.concatenate(C)[: args.max_cells or None],
                      "pert": np.concatenate(Pp)[: args.max_cells or None]})
    print(f"[tahoe] {len(X)} cells | contexts {O.context.nunique()} "
          f"| perts {O.pert.nunique()} | {len(shards)} shards on disk", flush=True)
    return X, O, syms, s["control"]


def read_pbmc(s, args):
    """Stream the ~800 GB Parse PBMC CSR file in CONTIGUOUS row blocks.

    Rows are interleaved across samples, so picking scattered rows would mean hundreds of
    thousands of random reads over a file this size. Instead the file is walked in
    contiguous blocks (in a shuffled order, so the cell budget is spread over the whole
    atlas rather than concentrated at the front), and each block is filtered against
    per-(cell type, donor, cytokine) quotas.

    Both inner loops are vectorised: the quota check is a within-block rank against a
    running counter, and the expression rows are pulled out by building a scipy CSR over
    the block and indexing it, instead of one Python iteration per cell.
    """
    import scipy.sparse as sp
    with h5py.File(s["path"], "r") as f:
        obs = f["obs"]
        ctx = H.read_cat(obs, s["context"])
        hold = H.read_cat(obs, s["split_col"])
        prt = H.read_cat(obs, s["pert"])
        flag = H.read_cat(obs, s["control_col"])
        prt = np.where(flag == s["control"], s["control"], prt)
        genes = H.var_names(f)
        gene_idx = _panel_index(s, genes)
        print(f"[pbmc] {len(ctx)} cells, {len(genes)} genes -> panel {len(gene_idx)} | "
              f"cell types {len(set(ctx))} donors {len(set(hold))} "
              f"perts {len(set(prt))}", flush=True)

        # one integer code per (cell type, donor, perturbation) group, and its quota
        codes, uniq = pd.factorize(pd.Series(ctx) + "|" + pd.Series(hold) + "|"
                                   + pd.Series(prt), sort=False)
        is_ctl_group = np.array([u.rsplit("|", 1)[1] == s["control"] for u in uniq])
        cap = np.where(is_ctl_group, args.cap_control or (1 << 30),
                       args.cap_per_group or (1 << 30)).astype(np.int64)
        seen = np.zeros(len(uniq), np.int64)
        print(f"[pbmc] {len(uniq)} (cell type, donor, cytokine) groups; quota "
              f"{args.cap_per_group or 'all'} / {args.cap_control or 'all'} for controls",
              flush=True)

        X = f["X"]
        indptr = X["indptr"][:]
        n_rows = len(indptr) - 1
        rng = np.random.default_rng(args.seed)
        starts = np.arange(0, n_rows, args.block_rows)
        rng.shuffle(starts)
        rows_out, Xs, total = [], [], 0
        t0 = time.time()
        for bi, s0 in enumerate(starts):
            e0 = min(s0 + args.block_rows, n_rows)
            cb = codes[s0:e0]
            # rank of each row within its group inside this block (stable, so the order
            # along the file is preserved), then keep the ones still under quota
            order = np.argsort(cb, kind="stable")
            sc = cb[order]
            bounds = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
            sizes = np.diff(np.r_[bounds, len(sc)])
            rank = np.arange(len(sc)) - np.repeat(bounds, sizes)
            within = np.empty_like(rank)
            within[order] = rank
            keep_m = (seen[cb] + within) < cap[cb]
            if keep_m.any():
                take = np.flatnonzero(keep_m)
                np.add.at(seen, cb[take], 1)
                lo, hi = int(indptr[s0]), int(indptr[e0])
                blk = sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi],
                                     indptr[s0:e0 + 1] - lo),
                                    shape=(e0 - s0, len(genes)))
                Xs.append(np.asarray(blk[take][:, gene_idx].todense(), np.float32))
                rows_out.append(take + s0)
                total += len(take)
            if bi % 50 == 0:
                print(f"  [pbmc] block {bi}/{len(starts)} (row {s0}): kept {total} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if args.max_cells and total >= args.max_cells:
                break
    rows = np.concatenate(rows_out)[: args.max_cells or None]
    Xd = np.concatenate(Xs)[: args.max_cells or None]
    O = pd.DataFrame({"context": ctx[rows], "pert": prt[rows], "split_on": hold[rows]})
    print(f"[pbmc] {len(Xd)} cells | contexts {O.context.nunique()} "
          f"| donors {O.split_on.nunique()} | perts {O.pert.nunique()}", flush=True)
    return Xd, O, genes[gene_idx], s["control"]


def stream_tahoe_memmap(s, args, out_dir):
    """Prepare FULL Tahoe-100M without ever holding it in memory.

    101M cells x 2,000 genes of float32 is 808 GB, so the matrix is written straight to a
    raw `X.f32` on disk and later opened with np.memmap. Two passes over the shards:

      pass 1  read only the (cell_line, drug) columns -- cheap, a few seconds per hundred
              shards -- to size every group after the caps and so the final row count.
      pass 2  read the expression lists, keep the 2,000 panel genes, log-normalise, and
              append the rows in shard order. Sequential writes; the per-cell-line sums
              are accumulated on the way so the held-out cell lines can be chosen by the
              paper's PCA procedure without a third pass.

    The split is then computed on the obs columns alone. Output is a directory
    (X.f32 + obs.parquet + meta.json) that data.loader.PerturbData opens memory-mapped.
    """
    import pyarrow.parquet as pq
    import scipy.sparse as sp

    from .splits import select_holdout_pca_pb

    meta_p = os.path.join(P.TAHOE_DIR, "metadata", "gene_metadata.parquet")
    gm = pq.read_table(meta_p).to_pandas()
    n_vocab = int(gm.token_id.max()) + 1
    panel = set(load_panel(s) or [])
    sym2tok = {str(a): int(b) for a, b in zip(gm.gene_symbol, gm.token_id)}
    toks = np.array(sorted({sym2tok[g] for g in panel if g in sym2tok}))
    syms = gm.set_index("token_id").loc[toks, "gene_symbol"].astype(str).values
    G = len(toks)

    shards = sorted(glob.glob(os.path.join(s["path"], "*.parquet")))
    np.random.default_rng(args.seed).shuffle(shards)
    if getattr(args, "max_shards", 0):
        shards = shards[: args.max_shards]
    ctx_col, prt_col, ctl = s["context"], s["pert"], s["control"]
    dose_col = s.get("dose_col")
    read_cols = [ctx_col, prt_col] + ([dose_col] if dose_col else [])
    cap_g = args.cap_per_group or (1 << 62)
    cap_c = args.cap_control or (1 << 62)
    print(f"[tahoe] {len(shards)} shards | panel {G} genes | caps {cap_g}/{cap_c}",
          flush=True)

    # ---- pass 1: how many rows survive the caps -------------------------------- #
    # ---- pass 0: the (drug, dose) vocabulary ----------------------------------- #
    # A perturbation is a drug AT A DOSE. Reading two small columns over every shard
    # costs under a minute and is what lets pass 1 count the real groups.
    lab_of = None
    if dose_col:
        t0, pairs = time.time(), set()
        for path in shards:
            t = pq.read_table(path, columns=[prt_col, dose_col])
            pairs.update(zip(t[prt_col].to_pandas().astype(str).values,
                             t[dose_col].to_pandas().astype(str).values))
        lab_of = {}
        for d in sorted({d for d, _ in pairs}):
            ss = sorted(x for dd, x in pairs if dd == d)
            for j, x in enumerate(ss, 1):
                lab_of[x] = ctl if d == ctl else f"{d}@{j}"
        print(f"[tahoe] pass0: {len({d for d, _ in pairs})} drugs x doses -> "
              f"{len(set(lab_of.values()))} perturbations ({time.time()-t0:.0f}s)",
              flush=True)

    def labels_of(tbl):
        dr = tbl[prt_col].to_pandas().astype(str).values
        if lab_of is None:
            return dr, dr
        sm = tbl[dose_col].to_pandas().astype(str).values
        return np.array([lab_of[x] for x in sm]), dr

    t0, seen, N = time.time(), {}, 0
    lines, drugs = set(), set()
    for i, path in enumerate(shards):
        t = pq.read_table(path, columns=read_cols)
        cl = t[ctx_col].to_pandas().astype(str).values
        dr, _ = labels_of(t)
        lines.update(np.unique(cl).tolist()); drugs.update(np.unique(dr).tolist())
        for k, c in pd.Series(map("|".join, zip(cl, dr))).value_counts().items():
            cap = cap_c if k.rsplit("|", 1)[1] == ctl else cap_g
            room = max(0, cap - seen.get(k, 0))
            add = min(int(c), room)
            seen[k] = seen.get(k, 0) + add
            N += add
        if i % 250 == 0:
            print(f"  [pass1] {i}/{len(shards)} rows so far {N:,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # obs is carried as int32 codes, not strings: 101M Python str objects per column is
    # ~10 GB of pointers and objects each, and writing that to parquet is worse.
    LN = np.array(sorted(lines)); DR = np.array(sorted(drugs))
    print(f"[tahoe] pass1: {N:,} rows in {len(seen):,} groups | "
          f"{len(LN)} cell lines, {len(DR)} perturbations -> "
          f"{N * G * 4 / 1e9:.0f} GB on disk ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass 2: write the matrix ---------------------------------------------- #
    os.makedirs(out_dir, exist_ok=True)
    xpath = os.path.join(out_dir, "X.f32")
    X = np.memmap(xpath, np.float32, "w+", shape=(N, G))
    seen, pos, C, Pp, DG = {}, 0, [], [], []
    pb_sum, pb_n = {}, {}
    t0 = time.time()
    for i, path in enumerate(shards):
        tbl = pq.read_table(path, columns=["genes", "expressions"] + read_cols)
        cl = tbl[ctx_col].to_pandas().astype(str).values
        dr, raw_drug = labels_of(tbl)
        take = []
        for j in range(len(cl)):
            k = cl[j] + "|" + dr[j]
            cap = cap_c if dr[j] == ctl else cap_g
            if seen.get(k, 0) < cap:
                seen[k] = seen.get(k, 0) + 1
                take.append(j)
        if take:
            take = np.asarray(take)
            ga = tbl["genes"].combine_chunks()
            ea = tbl["expressions"].combine_chunks()
            offs = np.asarray(ga.offsets)
            M = sp.csr_matrix((np.clip(np.asarray(ea.values).astype(np.float32), 0, None),
                               np.asarray(ga.values), offs),
                              shape=(len(offs) - 1, n_vocab))
            blk = M[take][:, toks].toarray().astype(np.float32)
            blk = H.lognorm(blk)                      # per-cell: safe to do shard by shard
            X[pos:pos + len(blk)] = blk
            ct = cl[take]
            for c in np.unique(ct):
                m = ct == c
                pb_sum[c] = pb_sum.get(c, 0.0) + blk[m].sum(0)
                pb_n[c] = pb_n.get(c, 0) + int(m.sum())
            C.append(np.searchsorted(LN, cl[take]).astype(np.int32))
            Pp.append(np.searchsorted(DR, dr[take]).astype(np.int32))
            if lab_of is not None:
                DG.append(raw_drug[take])
            pos += len(blk)
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  [pass2] {i}/{len(shards)} rows {pos:,}/{N:,} "
                  f"({pos*G*4/1e9:.0f} GB, {el:.0f}s, eta {el/max(i,1)*(len(shards)-i)/60:.0f}m)",
                  flush=True)
    X.flush(); del X
    print(f"[tahoe] pass2 done: {pos:,} rows written ({time.time()-t0:.0f}s)", flush=True)

    cc, pc = np.concatenate(C), np.concatenate(Pp)
    del C, Pp
    assert len(cc) == pos

    # ---- held-out cell lines: the paper's PCA on the streamed pseudobulk -------- #
    names = np.array(sorted(pb_sum))
    pb = np.stack([pb_sum[c] / max(pb_n[c], 1) for c in names])
    held = select_holdout_pca_pb(pb, names, s.get("n_holdout", 5), args.seed)
    print(f"[tahoe] PCA over {len(names)} cell lines -> held out {list(held)}", flush=True)

    # make_split wants labels; LN/DR are small so expanding through the codes gives a
    # compact fixed-width array rather than an object array of 101M Python strings
    split, info = make_split(
        LN[cc], DR[pc], ctl, "unseen_context", held_out=list(held),
        leak_frac=(args.leak_frac if args.leak_frac is not None else s.get("leak_frac", .30)),
        val_frac=args.val_frac, control_split=args.control_split, seed=args.seed)
    O = pd.DataFrame({
        "context": pd.Categorical.from_codes(cc, LN),
        "pert": pd.Categorical.from_codes(pc, DR),
        "split": pd.Categorical(split)})
    if DG:                       # the drug behind each 'Drug@k', for aggregate analysis
        O["drug"] = pd.Categorical(np.concatenate(DG))
    print(f"[tahoe] split=unseen_context: " +
          " ".join(f"{k}={v}" for k, v in info.items() if k in ("train", "val", "test")) +
          f" | leaked perts {info.get('n_leaked_perts', 0)}", flush=True)

    O.to_parquet(os.path.join(out_dir, "obs.parquet"), index=False)
    json.dump({"n_cells": int(pos), "n_genes": int(G), "genes": [str(x) for x in syms],
               "uns": {"dataset": "tahoe", "control": ctl, "n_genes": int(G),
                       "split_kind": "unseen_context", "rescale": 1.0, "combo_sep": "",
                       "source": str(s["path"]), "seed": int(args.seed),
                       "split_info": {k: (list(v) if isinstance(v, (list, np.ndarray)) else v)
                                      for k, v in info.items()}}},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2, default=str)
    print(f"[tahoe] wrote {out_dir}  X=({pos}, {G})  control={ctl!r}", flush=True)
    return out_dir


READERS = {"replogle": read_h5ad_generic, "norman": read_norman,
           "tahoe": read_tahoe, "pbmc": read_pbmc}


# --------------------------------------------------------------------------- #
def _panel_index(s, genes):
    panel = load_panel(s)
    if not panel:
        return np.arange(len(genes))
    want = set(panel)
    idx = np.array([i for i, g in enumerate(genes) if g in want], np.int64)
    if len(idx) < len(panel):
        print(f"[panel] {len(idx)}/{len(panel)} panel genes found in this file")
    return idx


def _group_keys(ctx, prt, hold):
    """One label per cell naming the group the per-group cap applies to."""
    if hold is None:
        return np.char.add(np.char.add(ctx.astype(str), "|"), prt.astype(str))
    return np.char.add(np.char.add(np.char.add(np.char.add(
        ctx.astype(str), "|"), hold.astype(str)), "|"), prt.astype(str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--path", default=None, help="override the raw source path")
    ap.add_argument("--out", default=None, help="override the output .h5ad path")
    ap.add_argument("--tag", default=None, help="suffix for the output file stem")
    ap.add_argument("--cap_per_group", type=int, default=None,
                    help="cells kept per (context[,split_on],pert) group; 0 = all")
    ap.add_argument("--cap_control", type=int, default=None,
                    help="separate, larger cap for the control population; 0 = all")
    ap.add_argument("--max_cells", type=int, default=None, help="overall cap; 0 = all")
    ap.add_argument("--block_rows", type=int, default=20000,
                    help="pbmc: contiguous CSR rows read in one go")
    ap.add_argument("--n_hvg", type=int, default=2000,
                    help="genes kept when the dataset has no shipped panel")
    ap.add_argument("--rescale", type=float, default=1.0,
                    help="divide the log-normalised values by this. PerturbDiff uses 10 to "
                         "push log1p values into ~[0,1) for its diffusion model; it changes "
                         "the recon/KL/energy balance, so it is a real training difference")
    # split
    ap.add_argument("--split_kind", default=None,
                    choices=["unseen_context", "unseen_pert", "unseen_combo", "random"])
    ap.add_argument("--held_out", nargs="*", default=None,
                    help="explicit held-out context names (default: the paper's)")
    ap.add_argument("--leak_frac", type=float, default=None)
    ap.add_argument("--holdout_frac", type=float, default=None)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--control_split", default="train", choices=["train", "context"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_shards", type=int, default=0,
                    help="tahoe: only use the first N shards (smoke-testing the stream)")
    ap.add_argument("--backend", default="h5ad", choices=["h5ad", "memmap"],
                    help="memmap: stream the matrix to a directory (X.f32 + obs.parquet + "
                         "meta.json) instead of building it in RAM. Required for full "
                         "Tahoe-100M, whose 101M x 2,000 float32 matrix is 808 GB.")
    args = ap.parse_args(argv)

    s = spec(args.dataset)
    if args.path:
        s["path"] = args.path
    if not os.path.exists(s["path"]):
        raise SystemExit(f"raw source missing: {s['path']}\n"
                         f"run: python -m perturbenergy.data.download "
                         f"--what {args.dataset}")
    d = PREP_DEFAULTS.get(args.dataset, {})
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    for k in ("cap_per_group", "cap_control", "max_cells"):
        if getattr(args, k) is None:
            setattr(args, k, 0)

    if args.backend == "memmap":
        if args.dataset != "tahoe":
            raise SystemExit("--backend memmap is only implemented for tahoe")
        out = args.out or os.path.join(P.PREPPED, f"tahoe{'_' + args.tag if args.tag else ''}")
        return stream_tahoe_memmap(s, args, out)

    X, O, genes, control = READERS[args.dataset](s, args)
    genes = np.asarray([str(g) for g in genes])

    # --- normalise ---------------------------------------------------------- #
    is_counts = H.looks_like_counts(X) and not s.get("already_lognorm")
    if is_counts:
        print(f"[{args.dataset}] raw counts (max {X.max():.0f}) -> normalize_total(1e4)+log1p")
        X = H.lognorm(X)
    else:
        print(f"[{args.dataset}] already normalised (max {X.max():.2f})")
    if args.rescale and args.rescale != 1.0:
        X = (X / args.rescale).astype(np.float32)
        print(f"[{args.dataset}] rescaled by 1/{args.rescale:g} -> max {X.max():.3f}")

    # --- 2,000 genes -------------------------------------------------------- #
    if X.shape[1] > args.n_hvg:
        keep = H.top_hvg(X, args.n_hvg, counts=False)
        X, genes = X[:, keep], genes[keep]
        print(f"[{args.dataset}] selected top {args.n_hvg} HVGs from "
              f"{len(keep)} candidates")

    # --- split -------------------------------------------------------------- #
    kind = args.split_kind or s["split_kind"]
    if kind == "unseen_context" and O["context"].nunique() < 2 and "split_on" not in O:
        kind = "unseen_pert"
        print(f"[{args.dataset}] a single context -> 'unseen_context' is degenerate; "
              f"using 'unseen_pert'")
    split, info = make_split(
        O["context"].values, O["pert"].values, control, kind,
        held_out=(args.held_out if args.held_out is not None else s.get("held_out")),
        n_holdout=s.get("n_holdout", 0),
        holdout_frac=(args.holdout_frac if args.holdout_frac is not None
                      else s.get("holdout_frac", 0.30)),
        leak_frac=(args.leak_frac if args.leak_frac is not None else s.get("leak_frac", 0.30)),
        val_frac=args.val_frac,
        split_values=(O["split_on"].values if "split_on" in O else None),
        control_split=args.control_split, seed=args.seed, X=X,
        holdout_by=s.get("holdout_by", "random"), combo_sep=s.get("combo_sep"))
    O["split"] = split
    print(f"[{args.dataset}] split={kind}: " +
          " ".join(f"{k}={v}" for k, v in info.items() if k in ("train", "val", "test")) +
          f" | held out {info.get('held_out', info.get('held_out_perts', '-'))[:8] if isinstance(info.get('held_out', info.get('held_out_perts')), list) else '-'}"
          f" | leaked perts {info.get('n_leaked_perts', 0)}")

    # --- write -------------------------------------------------------------- #
    A = ad.AnnData(X=np.ascontiguousarray(X), obs=O.reset_index(drop=True))
    A.var_names = list(genes)
    A.uns.update(dataset=args.dataset, control=str(control), n_genes=int(X.shape[1]),
                 split_kind=kind, rescale=float(args.rescale),
                 split_info={k: v for k, v in info.items()},
                 combo_sep=(s.get("combo_sep") or ""),
                 source=str(s["path"]), seed=int(args.seed))
    os.makedirs(P.PREPPED, exist_ok=True)
    out = args.out or P.prepped_path(args.dataset, args.tag)
    A.write_h5ad(out, compression="gzip")
    print(f"[{args.dataset}] wrote {out}  X={X.shape}  control={control!r}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
