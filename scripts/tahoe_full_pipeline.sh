#!/bin/bash
# 全量 Tahoe-100M 流式 prep（95.6M cells x 2000 genes -> 765 GB memmap）。
# 扰动单位是「药物@剂量」而非药物：Tahoe 有 380 种药，但 1,344 个 (drug, sample) 组合，
# PerturbDiff 的 #Perturbations=1,137 数的是后者。
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
echo "[pipe] $(date '+%F %T') 开始全量 prep"
python -m perturbenergy.data.prepare --dataset tahoe --backend memmap --tag full \
       --cap_per_group 0 --cap_control 0
echo "PREP_COMPLETE $(date '+%F %T')"
