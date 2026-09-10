#!/bin/bash
# 用已有 checkpoint 重跑评估（不重训），为的是把 DEG 组补到 K=25/50/100/200
# 并加上 DEG 组的 cell-level 指标。
#     CUDA_VISIBLE_DEVICES=0 bash scripts/reeval.sh replogle
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
DS="${1:?usage: reeval.sh <dataset>}"
M="$PE_MODELS_ROOT"

for p in control mean_shift; do
  echo "########## $DS / $p ##########"
  python scripts/evaluate.py --dataset "$DS" --predictor "$p" \
      --out "$M/${DS}_ref_${p}/eval.json" || echo "FAILED $DS/$p"
done
if [ "$DS" = "norman" ]; then
  echo "########## norman / additive ##########"
  python scripts/evaluate.py --dataset norman --predictor additive \
      --out "$M/norman_ref_additive/eval.json" || echo "FAILED norman/additive"
fi

for v in noenergy energy_lang energy_mis energy_sel ft_lang obs_pairing; do
  ck="$M/${DS}_${v}/model_best.pt"
  [ -f "$ck" ] || { echo "skip $v (no ckpt)"; continue; }
  echo "########## $DS / $v ##########"
  python scripts/evaluate.py --dataset "$DS" --predictor perturbenergy --ckpt "$ck" \
      --out "$M/${DS}_${v}/eval.json" || echo "FAILED $DS/$v"
done
echo "REEVAL_DONE $DS"
