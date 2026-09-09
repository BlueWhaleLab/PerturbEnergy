#!/bin/bash
# End-to-end for one dataset: prepare (if needed) -> reference predictors -> the four
# PerturbEnergy variants -> comparison report.
#
#     CUDA_VISIBLE_DEVICES=0 bash scripts/run_all.sh replogle
#     CUDA_VISIBLE_DEVICES=1 STEPS=40000 bash scripts/run_all.sh norman
#     ONLY=energy_lang bash scripts/run_all.sh tahoe        # just one variant
#
# Variants
#   noenergy     the same VAE with the energy switched off -- the ablation the paper's
#                claim rests on. Everything else is identical.
#   energy_lang  joint training, negatives from the Langevin chain (paper Eq. 15).
#   energy_mis   joint training, negatives = the same z_a paired with ANOTHER
#                perturbation's embedding. Free, and the contrast a similarity energy
#                actually needs.
#   ft_lang      two-stage: plain VAE first, then the energy is switched on.
#   energy_sel   like energy_lang, but the energy also SELECTS z_a during training
#                (n_candidates=8), so training matches the paper's Inference procedure
#                instead of only inference doing the selection.
#   obs_pairing  train.pairing=observed: reconstruct the observed cell instead of mapping
#                a control population to a perturbed one. Included because it is the
#                literal reading of the Eq. 7 ELBO and because the gap it leaves shows
#                why the control pairing is the right training objective.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source env.sh

DS="${1:?usage: run_all.sh <replogle|norman|tahoe|pbmc> }"
ONLY="${ONLY:-}"
EXTRA="${EXTRA:-}"
STEPS_ARG=""
[ -n "${STEPS:-}" ] && STEPS_ARG="train.max_steps=$STEPS train.stage1_steps=$((STEPS/2)) train.stage2_steps=$((STEPS/2))"

PREP="$PE_DATA_ROOT/perturb_bench/$DS.h5ad"
[ -f "$PREP" ] || python -m perturbenergy.data.prepare --dataset "$DS"

run () {                       # run <name> <config overrides...>
  local name="$1"; shift
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  echo "########## $DS / $name ##########"
  python scripts/train.py --dataset "$DS" --run "${DS}_${name}" --evaluate \
      --set "$@" $STEPS_ARG $EXTRA
}

echo "########## $DS / reference predictors ##########"
for p in control mean_shift; do
  if [ -z "$ONLY" ] || [ "$ONLY" = "$p" ]; then
    python scripts/evaluate.py --dataset "$DS" --predictor "$p" \
        --out "$PE_MODELS_ROOT/${DS}_ref_${p}/eval.json"
  fi
done
if [ "$DS" = "norman" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "additive" ]; }; then
  python scripts/evaluate.py --dataset norman --predictor additive \
      --out "$PE_MODELS_ROOT/norman_ref_additive/eval.json"
fi

run noenergy    model.use_energy=false
run energy_lang model.use_energy=true  loss.negatives=langevin
run energy_mis  model.use_energy=true  loss.negatives=mismatch
run ft_lang     model.use_energy=true  loss.negatives=langevin train.mode=finetune
run energy_sel  model.use_energy=true  loss.negatives=langevin model.n_candidates=8
run obs_pairing model.use_energy=true  loss.negatives=langevin train.pairing=observed

python scripts/report.py --dataset "$DS" --out "$PE_MODELS_ROOT/report_${DS}.csv"
echo "RUN_ALL_DONE $DS"
