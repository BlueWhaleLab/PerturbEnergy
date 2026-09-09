"""
The perturbation embedding table.

For Replogle, Tahoe and PBMC a perturbation is one label, and this is exactly the
`nn.Embedding[n_perturbations, d]` the design calls for.

Norman2019 is different: its 237 conditions are 106 single CRISPRa targets and 131 PAIRS
of them, and the split holds out pairs so that the test perturbations are ones the model
never saw. A plain per-label table cannot represent those at all -- there is no row. So
the table is over TARGETS, and a label's embedding is the sum of its targets':

    u_a = sum_{t in targets(a)} W[t]

With single-target datasets every label has exactly one target and the two formulations
are identical, so nothing changes for them. With Norman, a held-out pair 'AHR_FEV' is
embedded from the 'AHR' and 'FEV' rows that training did fit -- which is what makes the
unseen-perturbation evaluation a test of generalisation rather than of a missing row.

The label -> targets map is a buffer, so a checkpoint round-trips it and inference never
has to rebuild the vocabulary.
"""
import torch
import torch.nn as nn


class PertEmbedding(nn.Module):
    def __init__(self, tokens, mask, d, reduce="sum"):
        """`tokens` [n_labels, max_targets] target ids (0 where masked out),
        `mask` [n_labels, max_targets] 1.0 for a real target."""
        super().__init__()
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        self.register_buffer("tokens", tokens)
        self.register_buffer("mask", mask)
        self.n_labels, self.max_targets = tokens.shape
        self.n_targets = int(tokens.max().item()) + 1 if tokens.numel() else 1
        self.reduce = reduce
        self.table = nn.Embedding(self.n_targets, d)

    @property
    def weight(self):
        """The underlying target table (what an L2 penalty or an inspection wants)."""
        return self.table.weight

    def forward(self, a):
        """a: [B] label ids -> [B, d]."""
        tok, m = self.tokens[a], self.mask[a]                 # [B, T], [B, T]
        e = self.table(tok) * m.unsqueeze(-1)                 # [B, T, d]
        s = e.sum(1)
        if self.reduce == "mean":
            return s / m.sum(1, keepdim=True).clamp(min=1.0)
        return s

    def extra_repr(self):
        return (f"labels={self.n_labels}, targets={self.n_targets}, "
                f"max_targets={self.max_targets}, reduce={self.reduce}")


def build_tables(labels, known_targets, sep=None, max_targets=None):
    """(tokens, mask, predictable) for a list of labels.

    `known_targets` is the target vocabulary built from the TRAINING labels; a label with
    no known target is not predictable and the evaluation reports it as skipped rather
    than scoring a random embedding.
    """
    import numpy as np
    t2i = {t: i for i, t in enumerate(sorted(known_targets))}
    parts = [(l.split(sep) if sep else [l]) for l in labels]
    width = max_targets or max((len(p) for p in parts), default=1)
    tokens = np.zeros((len(labels), width), np.int64)
    mask = np.zeros((len(labels), width), np.float32)
    predictable = np.zeros(len(labels), bool)
    for i, ps in enumerate(parts):
        known = [t2i[p] for p in ps[:width] if p in t2i]
        if known:
            tokens[i, :len(known)] = known
            mask[i, :len(known)] = 1.0
            predictable[i] = len(known) == len(ps)
    return tokens, mask, predictable, t2i
