#!/bin/bash
# Shared environment for every job in this project. There is no scheduler on this box, so
# the run scripts are plain bash: they source this file and are launched by hand.
#
#     CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_all.sh replogle > logs/rep.out 2>&1 &
#
# Every path default lives in perturbenergy/paths.py; the exports below only mirror the
# ones the shell itself needs. Override any of them before sourcing.

PE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PE_ROOT PYTHONPATH="$PE_ROOT:${PYTHONPATH:-}"

export PE_DATA_ROOT="${PE_DATA_ROOT:-/data/projects/yyc/datasets}"
export PE_MODELS_ROOT="${PE_MODELS_ROOT:-/data/projects/yyc/models/perturbenergy}"
export PE_WORK_ROOT="${PE_WORK_ROOT:-/data/projects/yyc}"
mkdir -p "$PE_MODELS_ROOT"

# conda: the `squidiff` env is python 3.11 + torch 2.11.0+cu128. These GPUs are Blackwell
# (sm_120), so anything built against cu121 or older will NOT run on them.
export CONDA_ROOT="${CONDA_ROOT:-/data/projects/yyc/miniconda3}"
export PE_ENV="${PE_ENV:-/data/projects/yyc/conda_envs/squidiff}"
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$PE_ENV"

export TMPDIR="${TMPDIR:-$PE_WORK_ROOT/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export HF_HOME="${HF_HOME:-$PE_WORK_ROOT/hf_cache}"
mkdir -p "$TMPDIR" "$HF_HOME"

export HDF5_USE_FILE_LOCKING=FALSE          # h5py/AnnData on this filesystem
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"   # several jobs share the box

echo "[env] $(python -V 2>&1) @ $PE_ENV"
echo "[env] data=$PE_DATA_ROOT models=$PE_MODELS_ROOT"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
