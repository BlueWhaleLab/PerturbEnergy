"""
Single source of truth for every on-disk location this project uses.

Nothing outside this module may hard-code a path: add the location here and import it.
Every value is overridable by an environment variable, so moving the project to another
machine needs no code edit -- just export the roots.

    raw data   ->  $PE_DATA_ROOT     (default /data/projects/yyc/datasets)
    prepped    ->  $PE_PREPPED       (default $PE_DATA_ROOT/perturb_bench)
    runs       ->  $PE_MODELS_ROOT   (default /data/projects/yyc/models/perturbenergy)

Layout under DATA_ROOT
----------------------
    perturbdiff/perturb_data/...   the PerturbDiff release (replogle / pbmc / gene panels)
    norman/                        Norman2019 CRISPRa single cells (HuggingFace)
    tahoe/data/*.parquet           Tahoe-100M shards (already on disk, 419 of 3388)
    perturb_bench/*.h5ad           the uniform prepped files this project trains on

Checkpoints, metrics and eval artefacts go under MODELS_ROOT, one directory per run --
never into the project directory.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
    """Env override, with ~ expanded and trailing slashes stripped."""
    return os.path.expanduser(os.environ.get(name) or default).rstrip("/")


# --------------------------------------------------------------------------- #
# roots
# --------------------------------------------------------------------------- #
DATA_ROOT   = _env("PE_DATA_ROOT",   "/data/projects/yyc/datasets")
MODELS_ROOT = _env("PE_MODELS_ROOT", "/data/projects/yyc/models/perturbenergy")
WORK_ROOT   = _env("PE_WORK_ROOT",   os.path.dirname(DATA_ROOT))
TMP_DIR     = _env("PE_TMP",         os.path.join(WORK_ROOT, "tmp"))
HF_CACHE    = _env("PE_HF_CACHE",    os.path.join(WORK_ROOT, "hf_cache"))

# --------------------------------------------------------------------------- #
# raw sources
# --------------------------------------------------------------------------- #
PERTURBDIFF_DIR = _env("PE_PERTURBDIFF", os.path.join(DATA_ROOT, "perturbdiff"))
NORMAN_DIR      = _env("PE_NORMAN",      os.path.join(DATA_ROOT, "norman"))
TAHOE_DIR       = _env("PE_TAHOE",       os.path.join(DATA_ROOT, "tahoe"))

PD_DATA      = os.path.join(PERTURBDIFF_DIR, "perturb_data")
GENE_PANELS  = os.path.join(PD_DATA, "selected_genes")

# --------------------------------------------------------------------------- #
# derived / written
# --------------------------------------------------------------------------- #
PREPPED   = _env("PE_PREPPED", os.path.join(DATA_ROOT, "perturb_bench"))
REPORTS   = _env("PE_REPORTS", os.path.join(MODELS_ROOT, "reports"))


def prepped_path(dataset, tag=None):
    """The uniform .h5ad this project trains on. `tag` distinguishes split variants."""
    stem = f"{dataset}_{tag}" if tag else dataset
    return os.path.join(PREPPED, f"{stem}.h5ad")


def run_dir(name):
    """Everything a single training run writes."""
    return os.path.join(MODELS_ROOT, name)


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0] if len(dirs) == 1 else dirs


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.isupper() and isinstance(v, str):
            print(f"{k:16s} {v}{'' if os.path.exists(v) else '   [missing]'}")
