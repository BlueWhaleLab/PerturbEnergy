#!/bin/bash
# 全量 Tahoe-100M（memmap，765 GB）上的六个变体 + 参考基线。
#
#     STEPS=120000 bash scripts/run_tahoe_full.sh            # 六个变体并行，每个一张卡
#     STEPS=120000 SERIAL=1 bash scripts/run_tahoe_full.sh   # 串行（GPU 紧张时）
#     ONLY="noenergy energy_mis" STEPS=120000 bash scripts/run_tahoe_full.sh
#
# 并行是默认：六个进程读同一个 memmap，page cache 是共享的，所以并行不会让 IO 变成
# 六倍，反而摊薄了把热数据读进缓存的一次性成本。
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh

PATH_FULL="$PE_DATA_ROOT/perturb_bench/tahoe_full"
CFG=configs/tahoe_full.yaml
STEPS="${STEPS:?必须给 STEPS，先跑 scripts/probe_memmap.py 定}"
HALF=$((STEPS/2))
VARIANTS=${ONLY:-"noenergy energy_lang energy_mis energy_sel ft_lang obs_pairing"}

[ -d "$PATH_FULL" ] || { echo "缺少 $PATH_FULL"; exit 1; }

opts_for () {                       # 变体名 -> --set 覆盖项
  case "$1" in
    noenergy)    echo "model.use_energy=false" ;;
    energy_lang) echo "model.use_energy=true loss.negatives=langevin" ;;
    energy_mis)  echo "model.use_energy=true loss.negatives=mismatch" ;;
    energy_sel)  echo "model.use_energy=true loss.negatives=langevin model.n_candidates=8" ;;
    ft_lang)     echo "model.use_energy=true loss.negatives=langevin train.mode=finetune" ;;
    obs_pairing) echo "model.use_energy=true loss.negatives=langevin train.pairing=observed" ;;
    *) echo ""; return 1 ;;
  esac
}

i=0
for v in $VARIANTS; do
  o=$(opts_for "$v") || { echo "未知变体 $v"; exit 1; }
  echo "########## tahoe_full / $v  (steps=$STEPS, GPU $i) ##########"
  cmd=(python scripts/train.py --dataset tahoe --config "$CFG" --path "$PATH_FULL"
       --run "tahoe_full_${v}" --evaluate --set $o
       train.max_steps=$STEPS train.stage1_steps=$HALF train.stage2_steps=$HALF)
  if [ -n "${SERIAL:-}" ]; then
    CUDA_VISIBLE_DEVICES=$i "${cmd[@]}"
  else
    CUDA_VISIBLE_DEVICES=$i nohup "${cmd[@]}" > "logs/tahoe_full_${v}.log" 2>&1 &
    echo "  -> logs/tahoe_full_${v}.log  PID $!"
  fi
  i=$(((i+1) % 8))
done

if [ -z "${SERIAL:-}" ]; then
  echo "六个变体已并行启动；等待全部结束 ..."
  wait
fi

echo "########## 参考基线 ##########"
for p in control mean_shift; do
  python scripts/evaluate.py --dataset tahoe --config "$CFG" --path "$PATH_FULL" \
      --predictor "$p" --out "$PE_MODELS_ROOT/tahoe_full_ref_${p}/eval.json" \
      || echo "FAILED $p"
done
echo "TAHOE_FULL_DONE"
