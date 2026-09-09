"""
Where each raw dataset lives and how its columns map onto the project's uniform schema.

Uniform prepped format (what everything downstream reads):

    X    float32 [n_cells, 2000]   log1p(CP10K) expression on the 2,000-HVG panel
    obs  context   cell type / cell line / donor-independent covariate the model conditions on
         pert      perturbation label ("non-targeting" / "control" / "DMSO_TF" = control)
         split     train | val | test
         split_on  (optional) the column the hold-out was taken on, when it is not `context`
    var_names       gene symbols
    uns  dataset, control, split_kind, n_genes, held_out, rescale

Gene panel: the 2,000 genes are PerturbDiff's shipped panel where one exists
(`selected_genes/*.pkl`), otherwise the top-2,000 HVGs computed on the prepared subset
with `scanpy.pp.highly_variable_genes`. Both give 2,000 genes, which is what every
baseline in the paper is fed.

Splits follow PerturbDiff App. A.2.3 (which follows STATE) -- see splits.py.
"""
import os
import pickle

from .. import paths as P

REGISTRY = {
    # ---- genetic CRISPRi, 4 cell lines. PerturbDiff's own processed file. -------- #
    "replogle": dict(
        path=os.path.join(P.PD_DATA, "finetune_data", "nadig_processed_data",
                          "replogle.h5ad"),
        context="cell_line",           # k562 / jurkat / rpe1 / hepg2
        pert="gene",                   # 2,023 targets + "non-targeting"
        control="non-targeting",
        panel=os.path.join(P.GENE_PANELS, "replogle_real_selected_genes.pkl"),
        split_kind="unseen_context",
        held_out=["hepg2"],            # PerturbDiff: "one cell line (hepg2) was held out"
        leak_frac=0.30,                # "30% of perturbations randomly sampled from the
                                       #  held-out cell line" go back into train
        already_lognorm=True,          # STATE's processed file is normalised + log1p
    ),
    # ---- cytokine signalling, 18 cell types x 12 donors ------------------------- #
    "pbmc": dict(
        path=os.path.join(P.PD_DATA, "finetune_data", "pbmc_new",
                          "Parse_10M_PBMC_cytokines_processed_Xselected.h5ad"),
        context="cell_type",           # the model conditions on cell type (18)
        pert="cytokine",               # 90 cytokines
        control="PBS",
        control_col="treatment",       # control is flagged in a separate column
        split_col="donor",             # ... but the hold-out is taken on DONORS
        panel=os.path.join(P.GENE_PANELS, "pbmc_real_selected_genes.pkl"),
        split_kind="unseen_context",
        n_holdout=4,                   # PerturbDiff: "4 of the 12 donor cell lines"
        leak_frac=0.30,
        already_lognorm=False,
        streaming="pbmc",              # 800 GB: prep reads it in contiguous CSR blocks
    ),
    # ---- drug screen, 50 cancer cell lines -------------------------------------- #
    "tahoe": dict(
        path=os.path.join(P.TAHOE_DIR, "data"),
        context="cell_line_id",
        pert="drug",
        control="DMSO_TF",
        panel=os.path.join(P.GENE_PANELS, "tahoe100m_real_selected_genes.pkl"),
        split_kind="unseen_context",
        n_holdout=5,                   # PerturbDiff: five cell lines chosen by PCA on
        holdout_by="pca",              #  pseudobulk profiles of the 50 lines
        leak_frac=0.30,
        already_lognorm=False,
        streaming="tahoe",             # parquet shards of sparse token/count lists
    ),
    # ---- Norman2019 CRISPRa, single cell line ----------------------------------- #
    # One context (K562), so PerturbDiff's held-out-context protocol is degenerate here.
    # We keep the SPIRIT of it -- generalisation to conditions never seen in training --
    # by holding out whole PERTURBATIONS instead, at the same 30% rate PerturbDiff leaves
    # unseen in its held-out contexts.
    "norman": dict(
        path=os.path.join(P.NORMAN_DIR, "NormanWeissman2019_filtered.h5ad"),
        fallback=dict(                 # HuggingFace copy: 120 cells per perturbation
            path=os.path.join(P.NORMAN_DIR, "perturbations_singlecell.h5ad"),
            control_path=os.path.join(P.NORMAN_DIR, "controls_singlecell.h5ad"),
        ),
        context="cell_line",
        pert="perturbation",
        control="control",
        panel=None,                    # no shipped panel -> top-2,000 HVGs
        combo_sep="_",                 # 'FOXA3_FOXL2' is a DOUBLE CRISPRa perturbation:
                                       #  106 singles and 131 pairs, and every pair's two
                                       #  targets also appear on their own. Only Norman
                                       #  has this; a Tahoe drug called 'DMSO_TF' must NOT
                                       #  be split, so the separator is per dataset.
        split_kind="unseen_combo",
        holdout_frac=0.30,             # 30% of the pairs never seen in training
        already_lognorm=False,
    ),
}

DATASETS = tuple(REGISTRY)


def spec(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {DATASETS}")
    s = dict(REGISTRY[name])
    # fall back to the alternative source when the preferred file is not on disk
    if not os.path.exists(s["path"]) and s.get("fallback"):
        fb = s["fallback"]
        if all(os.path.exists(v) for v in fb.values()):
            s.update(fb)
            s["is_fallback"] = True
    return s


def load_panel(s):
    """The 2,000-gene panel for this dataset, or None if it has to be computed."""
    p = s.get("panel")
    if not p or not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return [str(g) for g in pickle.load(f)]


def available():
    """Which datasets have their raw source on disk right now."""
    out = {}
    for n in DATASETS:
        s = spec(n)
        out[n] = os.path.exists(s["path"])
    return out


if __name__ == "__main__":
    for n, ok in available().items():
        s = spec(n)
        print(f"{n:10s} {'OK     ' if ok else 'MISSING'} {s['path']}"
              f"{'   [fallback]' if s.get('is_fallback') else ''}")
