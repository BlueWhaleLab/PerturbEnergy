"""
Fetch every raw source this project needs.

    python -m perturbenergy.data.download --what panels norman replogle
    python -m perturbenergy.data.download --what pbmc          # 52 GB, hours

Sources
-------
panels    PerturbDiff's 2,000-gene panels (`selected_genes/*.pkl`). Tiny, and needed by
          replogle / pbmc / tahoe prep, so always fetch these first.
replogle  PerturbDiff's `nadig_processed_data/replogle.h5ad` (2.5 GB zstd -> ~22 GB).
pbmc      PerturbDiff's Parse 10M PBMC cytokine atlas (53 GB zstd -> ~800 GB). The prep
          step streams it, but the download itself is the long pole.
norman    Norman2019 CRISPRa K562 from HuggingFace `nicolas-lynn/norman-perturb`
          (perturbed + control single cells, already log-normalised).
tahoe     Tahoe-100M parquet shards. Already on disk under $PE_TAHOE/data -- this only
          reports what is there; pass --tahoe_shards N to pull more from `tahoebio/Tahoe-100M`.

`.zst` files are decompressed in place with the system `zstd` (streaming, so the archive
never has to be held in memory) and the archive is removed afterwards unless --keep_zst.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

from .. import paths as P

PD_REPO = "katarinayuan/PerturbDiff_data"
NORMAN_REPO = "nicolas-lynn/norman-perturb"
TAHOE_REPO = "tahoebio/Tahoe-100M"

PANEL_FILES = [
    "perturb_data/selected_genes/replogle_real_selected_genes.pkl",
    "perturb_data/selected_genes/pbmc_real_selected_genes.pkl",
    "perturb_data/selected_genes/tahoe100m_real_selected_genes.pkl",
    "perturb_data/selected_genes/vocab_id2name.csv",
]
REPLOGLE_FILE = "perturb_data/finetune_data/nadig_processed_data/replogle.h5ad.zst"
PBMC_FILE = ("perturb_data/finetune_data/pbmc_new/"
             "Parse_10M_PBMC_cytokines_processed_Xselected.h5ad.zst")
NORMAN_FILES = ["perturbations_singlecell.h5ad", "controls_singlecell.h5ad"]
# scPerturb's curated Norman2019 -- the FULL atlas (111k cells, raw counts, controls
# included in the same file), unlike the HuggingFace copy which caps every perturbation at
# 120 cells. Preferred; the HuggingFace pair is the fallback.
NORMAN_ZENODO = ("https://zenodo.org/records/13350497/files/"
                 "NormanWeissman2019_filtered.h5ad?download=1")
NORMAN_ZENODO_NAME = "NormanWeissman2019_filtered.h5ad"
TAHOE_META = ["metadata/gene_metadata.parquet", "metadata/cell_line_metadata.parquet",
              "metadata/drug_metadata.parquet"]


def _hf_get(repo, filename, local_dir):
    """Download one file from a HuggingFace dataset repo into `local_dir`, resumable."""
    from huggingface_hub import hf_hub_download
    os.makedirs(local_dir, exist_ok=True)
    out = os.path.join(local_dir, filename)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"  [skip] {filename} already present ({os.path.getsize(out)/1e9:.2f} GB)")
        return out
    print(f"  [get ] {repo}:{filename}", flush=True)
    got = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                          local_dir=local_dir, cache_dir=P.HF_CACHE)
    return got


def _unzst(path, keep=False):
    """Stream-decompress `foo.h5ad.zst` -> `foo.h5ad`. Returns the decompressed path."""
    assert path.endswith(".zst"), path
    out = path[:-4]
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"  [skip] {os.path.basename(out)} already decompressed")
        return out
    tool = shutil.which("pzstd") or shutil.which("zstd")
    if tool is None:
        raise RuntimeError("neither pzstd nor zstd is on PATH -- cannot decompress "
                           f"{path}. Install zstd or decompress it by hand.")
    print(f"  [unzst] {os.path.basename(path)} -> {os.path.basename(out)} "
          f"(this writes ~10x the archive size)", flush=True)
    # -o instead of a shell redirect so a failed run leaves no truncated output behind
    subprocess.run([tool, "-d", "-f", "-o", out, path], check=True)
    if not keep:
        os.remove(path)
    return out


# --------------------------------------------------------------------------- #
def get_panels(_):
    for f in PANEL_FILES:
        _hf_get(PD_REPO, f, P.PERTURBDIFF_DIR)
    print(f"[panels] -> {P.GENE_PANELS}")


def get_replogle(args):
    z = _hf_get(PD_REPO, REPLOGLE_FILE, P.PERTURBDIFF_DIR)
    print(f"[replogle] -> {_unzst(z, args.keep_zst)}")


def get_pbmc(args):
    z = _hf_get(PD_REPO, PBMC_FILE, P.PERTURBDIFF_DIR)
    print(f"[pbmc] -> {_unzst(z, args.keep_zst)}")


def _http_get(url, out):
    """Resumable plain-HTTP download (Zenodo has no HF-style client)."""
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"  [skip] {os.path.basename(out)} already present "
              f"({os.path.getsize(out)/1e9:.2f} GB)")
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"  [get ] {url}", flush=True)
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10",
                    "-C", "-", "-o", out + ".part", url], check=True)
    os.replace(out + ".part", out)
    return out


def get_norman(_):
    """scPerturb's full atlas first; the 120-cells-per-perturbation HuggingFace copy is
    kept as a fallback so prep still works if Zenodo is unreachable."""
    try:
        _http_get(NORMAN_ZENODO, os.path.join(P.NORMAN_DIR, NORMAN_ZENODO_NAME))
    except Exception as e:                                  # noqa: BLE001
        print(f"  [warn] Zenodo copy unavailable ({e}); falling back to HuggingFace")
    for f in NORMAN_FILES:
        _hf_get(NORMAN_REPO, f, P.NORMAN_DIR)
    print(f"[norman] -> {P.NORMAN_DIR}")


def get_tahoe(args):
    for f in TAHOE_META:                              # gene vocabulary: needed by prep
        _hf_get(TAHOE_REPO, f, P.TAHOE_DIR)
    have = sorted(glob.glob(os.path.join(P.TAHOE_DIR, "data", "*.parquet")))
    print(f"[tahoe] {len(have)} parquet shards already under {P.TAHOE_DIR}/data")
    if not args.tahoe_stride and args.tahoe_shards <= len(have):
        return
    from huggingface_hub import list_repo_files
    allf = sorted(f for f in list_repo_files(TAHOE_REPO, repo_type="dataset")
                  if f.startswith("data/") and f.endswith(".parquet"))
    hv = {os.path.basename(p) for p in have}
    if args.tahoe_stride:
        want = allf[::args.tahoe_stride]
        print(f"[tahoe] stride {args.tahoe_stride}: {len(want)} of {len(allf)} shards "
              f"spread over the whole atlas")
    else:
        want = allf[: args.tahoe_shards]
    todo = [f for f in want if os.path.basename(f) not in hv]
    print(f"[tahoe] {len(todo)} to fetch", flush=True)
    for i, f in enumerate(todo):
        _hf_get(TAHOE_REPO, f, P.TAHOE_DIR)
        if i % 50 == 0:
            print(f"[tahoe] {i}/{len(todo)}", flush=True)


GETTERS = {"panels": get_panels, "replogle": get_replogle, "pbmc": get_pbmc,
           "norman": get_norman, "tahoe": get_tahoe}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", default=["panels", "norman", "replogle"],
                    choices=list(GETTERS) + ["all"])
    ap.add_argument("--keep_zst", action="store_true",
                    help="keep the .zst archive after decompressing (needs 2x the space)")
    ap.add_argument("--tahoe_shards", type=int, default=0,
                    help="total number of Tahoe parquet shards to have locally")
    ap.add_argument("--tahoe_stride", type=int, default=0,
                    help="take every N-th shard of the 3,388 instead of the first ones. "
                         "Tahoe is ordered BY DRUG, so a contiguous prefix covers very few "
                         "drugs: the 419 shards already on disk hold 11.8M cells but only "
                         "95 of the 1,137 drugs. A stride spreads the download over the "
                         "whole atlas and buys drug coverage instead of depth.")
    args = ap.parse_args()
    what = list(GETTERS) if "all" in args.what else args.what
    for w in what:
        print(f"=== {w} ===", flush=True)
        GETTERS[w](args)
    print("DOWNLOAD_DONE", " ".join(what))


if __name__ == "__main__":
    sys.exit(main())
