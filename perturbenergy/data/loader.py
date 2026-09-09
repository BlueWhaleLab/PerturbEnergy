"""
Loader over a prepped dataset, resident or memory-mapped.

Two backends, chosen by what `path` points at:

    <name>.h5ad   the whole expression matrix is read into RAM (and onto the GPU when it
                  fits `gpu_budget_gb`). Everything up to a few tens of GB.
    <name>/       a DIRECTORY written by prepare.py --backend memmap: `X.f32` is a raw
                  float32 array left ON DISK and opened with np.memmap, plus obs.parquet
                  and meta.json. This is the only way to hold full Tahoe-100M -- 101M
                  cells x 2,000 genes is 808 GB, which no machine here will keep resident.
                  A training step touches 1,024 rows (8 MB), so the page cache carries the
                  working set and the read cost stays off the critical path.

`take(rows)` is the one place rows become a tensor, so both backends look identical to
the sampler, the trainer and the evaluator.

Everything the model needs is three aligned arrays -- X (cells x 2,000), context ids and
perturbation ids -- so there is no need for a DataLoader with workers: random index
batches are far faster than collating.

`device="cuda"` puts the whole expression matrix on the GPU when it fits inside
`gpu_budget_gb` (these cards have 97 GB, and Replogle is 5 GB), which removes the host
copy from every step. A memmap-backed dataset never goes resident.

Vocabularies
------------
Contexts are indexed from the TRAIN split only; a context that appears solely in test has
no id and gets `UNSEEN` (-1). Under the default protocol this never happens -- the
held-out cell line's control cells stay in train, so its embedding row exists -- but
mapping an unknown context silently to id 0 would quietly evaluate the wrong one, so it
is made explicit instead.

Perturbations are indexed over EVERY label in the file, because the held-out ones still
have to be embeddable. What is learned from train is the TARGET vocabulary: a label is
split on `uns["combo_sep"]` (only Norman sets one) and embedded as the sum of its
targets. `pert_predictable` marks the labels whose targets are all in that vocabulary --
for Norman's held-out pairs that is true, which is the point of the split.
"""
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import torch

from ..models.pert_embedding import build_tables

UNSEEN = -1


class PerturbData:
    def __init__(self, path, device="cpu", gpu_budget_gb=40.0, verbose=True):
        self.memmap = os.path.isdir(path)
        if self.memmap:
            meta = json.load(open(os.path.join(path, "meta.json")))
            n, g = int(meta["n_cells"]), int(meta["n_genes"])
            self.X_np = np.memmap(os.path.join(path, "X.f32"), np.float32, "r", shape=(n, g))
            obs = pd.read_parquet(os.path.join(path, "obs.parquet")).reset_index(drop=True)
            self.genes = np.array([str(x) for x in meta["genes"]])
            self.uns = meta.get("uns", {})
        else:
            A = ad.read_h5ad(path)
            X = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
            self.X_np = np.ascontiguousarray(X, dtype=np.float32)
            obs = A.obs.reset_index(drop=True)
            self.genes = np.array([str(x) for x in A.var_names])
            self.uns = dict(A.uns)
        self.obs = obs
        self.control = str(self.uns["control"])
        self.dataset = str(self.uns.get("dataset", os.path.basename(path.rstrip("/"))))
        self.split_kind = str(self.uns.get("split_kind", "?"))
        self.context = obs["context"].astype(str).values
        self.pert = obs["pert"].astype(str).values
        self.split = obs["split"].astype(str).values
        self.split_on = (obs["split_on"].astype(str).values if "split_on" in obs
                         else self.context)

        tr = self.split == "train"
        self.combo_sep = (str(A.uns.get("combo_sep", "")) or None)
        self.c2i = {c: i for i, c in enumerate(sorted(set(self.context[tr])))}
        self.cid = np.array([self.c2i.get(c, UNSEEN) for c in self.context], np.int64)

        # perturbations: labels over the whole file, targets learned from train only
        self.labels = sorted(set(self.pert))
        self.a2i = {p: i for i, p in enumerate(self.labels)}
        self.i2a = {i: p for p, i in self.a2i.items()}
        self.aid = np.array([self.a2i[p] for p in self.pert], np.int64)
        train_targets = set()
        for p in sorted(set(self.pert[tr])):
            train_targets.update(p.split(self.combo_sep) if self.combo_sep else [p])
        self.pert_tokens, self.pert_mask, self.pert_predictable, self.t2i = build_tables(
            self.labels, train_targets, self.combo_sep)
        self.trainable_pert = np.array([p in set(self.pert[tr]) for p in self.labels])

        gb = self.X_np.nbytes / 1e9
        self.device = ("cuda" if (device == "cuda" and torch.cuda.is_available()
                                  and not self.memmap and gb <= gpu_budget_gb) else "cpu")
        # a memmap stays on disk: materialising it would defeat the point
        self.X = None if self.memmap else torch.from_numpy(self.X_np).to(self.device)
        self.C = torch.from_numpy(self.cid).to(self.device)
        self.A = torch.from_numpy(self.aid).to(self.device)
        self.idx = {s: np.where(self.split == s)[0] for s in ("train", "val", "test")}
        if verbose:
            print(f"[data] {self.dataset}: {len(self.X_np)} cells x {self.n_genes} genes "
                  f"({gb:.1f} GB {'memmapped' if self.memmap else 'on ' + self.device}) "
                  f"| contexts {len(self.c2i)} "
                  f"| perts {len(self.a2i)} ({int(self.pert_predictable.sum())} "
                  f"embeddable from {len(self.t2i)} targets) | control={self.control!r} "
                  f"| split={self.split_kind} "
                  + " ".join(f"{s}={len(v)}" for s, v in self.idx.items()), flush=True)
            n_un_c = int((self.cid[self.split == "test"] == UNSEEN).sum())
            if n_un_c:
                print(f"[data] {n_un_c} test cells are in a context never seen in train "
                      f"(the held-out contexts) -- expected for this protocol")
            n_new = int((~self.trainable_pert & self.pert_predictable).sum())
            if n_new:
                print(f"[data] {n_new} perturbations never occur in train yet are "
                      f"embeddable from targets that do -- the unseen-perturbation test")
            n_bad = int((~self.pert_predictable).sum())
            if n_bad:
                print(f"[data] {n_bad} perturbations have no known target and cannot be "
                      f"embedded; the evaluation skips them")

    # ---- shapes ------------------------------------------------------------- #
    @property
    def n_genes(self):
        return self.X_np.shape[1]

    @property
    def n_context(self):
        return len(self.c2i)

    @property
    def n_pert(self):
        return len(self.a2i)

    @property
    def n_targets(self):
        return len(self.t2i)

    # ---- batching ----------------------------------------------------------- #
    def take(self, rows):
        """Rows (numpy int array) -> a [len(rows), n_genes] tensor on self.device.

        The only place the two backends differ. For the memmap the rows are fetched in
        SORTED order and put back afterwards: the pages a batch needs then come off the
        array front-to-back instead of seeking around it, which is most of the difference
        on a file this size.
        """
        rows = np.asarray(rows)
        if not self.memmap:
            return self.X[torch.from_numpy(rows).to(self.device)]
        order = np.argsort(rows, kind="stable")
        block = np.asarray(self.X_np[rows[order]], dtype=np.float32)
        back = np.empty_like(block)
        back[order] = block
        return torch.from_numpy(back).to(self.device)

    def batches(self, split="train", batch_size=512, shuffle=True, seed=0, drop_last=True):
        """Yield (x, c, a) tensors. Cells whose context/perturbation is unseen in train
        are skipped for training splits -- they have no embedding row to look up."""
        idx = self.idx[split]
        if split != "test":
            idx = idx[(self.cid[idx] != UNSEEN) & self.trainable_pert[self.aid[idx]]]
        rng = np.random.default_rng(seed)
        order = rng.permutation(idx) if shuffle else idx
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            if drop_last and len(b) < 2:
                continue
            t = torch.from_numpy(np.asarray(b)).to(self.device)
            yield self.take(b), self.C[t], self.A[t]

    def sampler(self, batch_size=512, seed=0, split="train"):
        """Infinite stream of random training batches (steps, not epochs)."""
        idx = self.idx[split]
        idx = idx[(self.cid[idx] != UNSEEN) & self.trainable_pert[self.aid[idx]]]
        if len(idx) == 0:
            raise ValueError(f"split {split!r} has no usable cells")
        rng = np.random.default_rng(seed)
        while True:
            sel = idx[rng.integers(0, len(idx), min(batch_size, len(idx)))]
            t = torch.from_numpy(sel).to(self.device)
            yield self.take(sel), self.C[t], self.A[t]

    # ---- evaluation helpers -------------------------------------------------- #
    def control_cells(self, context, split=None, max_cells=None, seed=0, hold=None):
        """The control population of one context (numpy) -- what inference is fed.

        `hold` restricts it to particular hold-out units. It matters whenever the split is
        taken on a column other than the context: PBMC holds out DONORS while the model
        conditions on CELL TYPE, so the unperturbed cells a prediction starts from have to
        be the held-out donors' own, not some training donor's. Where the two coincide
        (Replogle's and Tahoe's cell lines) this is a no-op.
        """
        m = (self.context == context) & (self.pert == self.control)
        if hold is not None:
            m &= np.isin(self.split_on, np.asarray(list(hold), dtype=object).astype(str))
        if split:
            m &= self.split == split
        rows = np.where(m)[0]
        if max_cells and len(rows) > max_cells:
            rows = np.random.default_rng(seed).choice(rows, max_cells, replace=False)
        return self.X_np[rows]

    @property
    def split_on_is_context(self):
        return bool(np.array_equal(self.split_on, self.context))

    def hold_units_of(self, context, pert, split="test"):
        """Which hold-out units (donors / cell lines) the cells of this group come from."""
        m = ((self.context == context) & (self.pert == pert) & (self.split == split))
        return sorted(set(self.split_on[m].tolist()))

    def groups(self, split="test", min_cells=30, max_groups=None, exclude_control=True):
        """(context, pert) groups to score, largest first."""
        m = self.split == split
        if exclude_control:
            m &= self.pert != self.control
        keys = np.array([f"{c}|{p}" for c, p in zip(self.context, self.pert)])
        u, n = np.unique(keys[m], return_counts=True)
        order = np.argsort(-n)
        out = [(u[i].split("|", 1)[0], u[i].split("|", 1)[1], int(n[i]))
               for i in order if n[i] >= min_cells]
        return out[:max_groups] if max_groups else out

    def cells_of(self, context, pert, split=None):
        m = (self.context == context) & (self.pert == pert)
        if split:
            m &= self.split == split
        return self.X_np[m]

    def save_vocab(self, path):
        json.dump({"context": self.c2i, "pert": self.a2i, "target": self.t2i,
                   "combo_sep": self.combo_sep, "control": self.control,
                   "genes": self.genes.tolist(),
                   "predictable": self.pert_predictable.tolist(),
                   "in_train": self.trainable_pert.tolist()}, open(path, "w"))
        return path


def load_vocab(path):
    v = json.load(open(path))
    return v["context"], v["pert"], v.get("control"), v.get("genes")


# --------------------------------------------------------------------------- #
# condition-structured batches: the control -> perturbed translation setting
# --------------------------------------------------------------------------- #
class ConditionSampler:
    """Batches shaped like the task, not like an autoencoder.

    Perturbation response prediction is an UNPAIRED population problem: the assay is
    destructive, so the same cell is never seen before and after. The model is therefore
    fed a control cell of the right context and asked for a cell from the perturbed
    population -- which is what the paper's Inference paragraph does, and the only setting
    in which its MMD reconstruction term makes sense (a per-cell MSE against an arbitrary
    partner would be noise).

    A batch is `n_conditions` groups of `set_size` cells:

        x_ctrl [B, G]   control cells of the group's context      -> the encoder's input
        x_tgt  [B, G]   real cells of (context, perturbation)     -> the target population
        c, a   [B]      the group's context / perturbation ids
        gsize           set_size, so the loss can split the batch back into populations

    Training on whole populations (rather than shuffled cells) is what lets the
    reconstruction term be a distance between distributions.
    """

    def __init__(self, data, n_conditions=16, set_size=32, split="train", seed=0,
                 min_cells=2, include_control=True):
        self.d, self.n_cond, self.set_size = data, int(n_conditions), int(set_size)
        rows = np.where(data.split == split)[0]
        rows = rows[(data.cid[rows] != UNSEEN) & data.trainable_pert[data.aid[rows]]]
        # A condition is (context, perturbation) -- and, when the hold-out is taken on a
        # different column, (context, hold-out unit, perturbation). PBMC conditions on
        # cell type but was collected from 12 donors, and donor batch effects are large:
        # pairing a donor's perturbed cells with ANOTHER donor's controls would put that
        # batch effect straight into the target delta. The model still only sees the
        # context id, so this changes what is PAIRED, not what is conditioned on.
        #
        # Grouped with pandas rather than a mask per group: Tahoe has 19,000 conditions
        # over 1.9M cells, and one boolean scan each would be an hour of set-up.
        by_unit = not data.split_on_is_context
        unit = data.split_on if by_unit else data.context
        df = pd.DataFrame({"ctx": data.context[rows], "u": unit[rows],
                           "p": data.pert[rows]}, index=rows)
        target = df.groupby(["ctx", "u", "p"], sort=False).indices     # -> positions in df

        cm = (data.pert == data.control) & (data.split != "test")
        crows = np.where(cm)[0]
        cdf = pd.DataFrame({"ctx": data.context[crows], "u": unit[crows]}, index=crows)
        ctl_of = {k: crows[v] for k, v in
                  cdf.groupby(["ctx", "u"] if by_unit else ["ctx"], sort=False).indices.items()}

        self.groups = []
        for (ctx, u, prt), pos in target.items():
            if len(pos) < min_cells or (not include_control and prt == data.control):
                continue
            ctl = ctl_of.get((ctx, u) if by_unit else ctx)
            if ctl is None or len(ctl) < min_cells:
                continue                      # no reference population -> not trainable
            self.groups.append((rows[pos], ctl, data.c2i[ctx], data.a2i[prt]))
        self.rng = np.random.default_rng(seed)
        print(f"[data] {len(self.groups)} trainable conditions in '{split}' "
              f"({self.n_cond} x {self.set_size} cells per batch)", flush=True)

    def __len__(self):
        return len(self.groups)

    def _draw(self, idx, n):
        return self.rng.choice(idx, n, replace=len(idx) < n)

    def batch(self):
        gi = self.rng.choice(len(self.groups), self.n_cond,
                             replace=len(self.groups) < self.n_cond)
        tgt, ctl, cs, as_ = [], [], [], []
        for g in gi:
            idx, cidx, ci, ai = self.groups[g]
            tgt.append(self._draw(idx, self.set_size))
            ctl.append(self._draw(cidx, self.set_size))
            cs += [ci] * self.set_size
            as_ += [ai] * self.set_size
        d = self.d
        c = torch.as_tensor(cs, dtype=torch.long, device=d.device)
        a = torch.as_tensor(as_, dtype=torch.long, device=d.device)
        return (d.take(np.concatenate(ctl)), d.take(np.concatenate(tgt)), c, a)

    def stream(self):
        while True:
            yield self.batch()

    def epoch(self, n_batches=None):
        for _ in range(n_batches or max(1, len(self.groups) // self.n_cond)):
            yield self.batch()
