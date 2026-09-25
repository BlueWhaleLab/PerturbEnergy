# PerturbEnergy

Implementation of *Energy-based Model for Enhancing Perturbation Signal in Cell Response
Prediction*: a VAE for single-cell perturbation response prediction whose **perturbation
latent space carries a conditional energy-based prior**, so that `a` (the perturbation)
and `z_a` (its representation) are directly coupled instead of being coupled only
indirectly through a reconstruction loss.

```
p_alpha(z_a | z_b, a) = p_0(z_a) exp(-E_alpha(z_a, z_b, a)) / F_alpha(z_b, a)      (Eq. 4)
```

The intractable partition function never has to be evaluated: its gradient is exactly the
negative phase of the contrastive update (Lemma 1), and the negatives come from a short
Langevin chain (Eq. 15).

---

## 1. What the model is

Input: an **unperturbed** cell's expression, its context, and a target perturbation.
Output: the predicted **post-perturbation** expression.

| module | mapping | file |
|---|---|---|
| `BasalEncoder` `q_{phi_b}(z_b\|x,c)` | x (2000 HVG) → 1024 → 512, **+ context embedding**, → 2-layer MLP → (mu, logvar) → `z_b` | [encoders.py](perturbenergy/models/encoders.py) |
| `PertEncoder` `q_{phi_a}(z_a\|z_b,a)` | `z_b` **+ perturbation embedding** → 2-layer MLP → (mu, logvar) → `z_a` | [encoders.py](perturbenergy/models/encoders.py) |
| `Decoder` `p_theta(x\|z_b,z_a)` | `z_b + z_a` (or concat) → 1024 → 2000 | [decoder.py](perturbenergy/models/decoder.py) |
| `EnergyNet` `E_alpha(z_a, a)` | `-cos( MLP(z_a), u_a )` — **low energy = compatible** | [energy.py](perturbenergy/models/energy.py) |
| `PertEmbedding` | the perturbation table (see §3 on Norman) | [pert_embedding.py](perturbenergy/models/pert_embedding.py) |

`d_model = 512` is one width for both embedding tables, every MLP trunk and both latents.
Both embedding tables are owned by the top-level model and **shared** with the energy, so
`a` reaches the posterior and the prior through the same parameters.

Two energy variants are configurable: `mlp_cos` (default: a width-preserving 2-layer MLP
on `z_a`, then cosine with `u_a`) and `cos` (the cosine directly, no parameters of its
own). `energy.use_z_b: true` restores the full `E(z_a, z_b, a)` conditioning of Eq. 4.

## 2. Training

### What a training step is (`train.pairing`) — read this first

Perturbation response prediction is an **unpaired population** problem: the assay is
destructive, so the same cell is never seen before and after. There are two ways to write
a training step, and they are not close in quality:

* **`control` (default).** The basal encoder is fed a **control cell of the group's
  context** and the generator must produce the **perturbed population** — the task itself,
  and what the paper's Inference paragraph does. Because cells are unpaired, the
  reconstruction term is a distance between *populations*: the squared error of the
  condition means plus the MMD the paper uses. Batches come from `ConditionSampler`,
  shaped as `n_conditions` blocks of `set_size` cells.
* **`observed`.** The literal reading of the Eq. 7 ELBO: reconstruct the observed cell
  from its own `(x, c, a)`. It trains a better autoencoder and a **much** worse predictor:
  `z_b` already contains the whole cell, so the generator can ignore `z_a` entirely, and at
  inference — where `z_b` comes from a *control* cell — the output is close to that control.

The gap is not subtle. On Norman, `observed` reaches `R2_delta ≈ 0.09` and loses to a
mean-shift baseline; `control` reaches `R2_delta ≈ 0.70` after 1,500 steps. `observed` is
kept as the `obs_pairing` ablation because it is the reading the ELBO invites, and the gap
is the reason not to take it. This is also why the paper's reconstruction term is an MMD:
a per-cell MSE against an arbitrarily chosen partner cell would be noise.

### Schedules, `train.mode`:

* **`joint`** — encoders, generator and energy trained together from step 0.
* **`finetune`** — stage 1 trains the plain VAE (reconstruction + the two KL terms, energy
  off); stage 2 switches the energy on and continues.

Two ways to update the energy, `loss.energy_update`:

* **`alternating`** (default) — Algorithm 1 verbatim: phase A updates `alpha` on detached
  positives/negatives with `{theta, phi}` frozen; phase B updates `{theta, phi}` with
  `alpha` frozen.
* **`joint`** — one optimiser, the energy term folded into `L_ELBO`.

Two sources of negatives, `loss.negatives`:

* **`langevin`** (default, the paper) — K annealed Langevin steps from `p_0 = N(0,I)`.
* **`mismatch`** — the same `z_a` paired with **another perturbation's** embedding. Free,
  and for a similarity energy it is the contrast that matters: `z_a` must sit closer to
  its own perturbation than to any other.
* **`both`** — hinge against whichever of the two currently has the lower energy.

The objective is `recon + beta_b*KL(q_b||N(0,I)) + beta_a*KL(q_a||N(0,I)) + w*energy`,
with the KL weights deliberately small (`5e-5`). Reconstruction defaults to `mse+mmd`:
the paper approximates `E[log p_theta]` with an MMD, but pure MMD has no per-cell
gradient, so the MSE is kept alongside it. `loss.recon: mmd` reproduces the paper exactly.

> The energy is trained with a **hinge**, `max(0, margin + E(z+) - E(z-))`. Cosine
> energies live in `[-1,1]`, so a raw difference has no fixed point and drives
> `cos(z_a, u_a)` to 1 — which collapses every cell of a perturbation onto one point and
> flatters any probe of "how much perturbation information is in the latent space".
> `optim.use_margin: false` recovers the unbounded Eq. 12 form.

## 3. Data

All four datasets are reduced to ONE format (`perturbenergy/data/registry.py`):
`X` = log1p(CP10K) on **2,000 genes**, `obs = {context, pert, split}`, `uns = {control, …}`.
The 2,000 genes are PerturbDiff's shipped panel where one exists, otherwise the top-2,000
HVGs from `sc.pp.highly_variable_genes` — 2,000 either way, which is what every baseline
in the paper is fed.

Splits follow **PerturbDiff App. A.2.3** (which follows STATE):

| dataset | cells / perturbations / contexts | split |
|---|---|---|
| **Replogle** (Nadig CRISPRi) | 643,413 / 2,023 + control / 4 cell lines | hold out **hepg2**, leak 30% of its perturbations back into train |
| **Tahoe-100M** (drug) | 95.6M cells / **1,317 drug-at-a-dose** perturbations / 50 cell lines | hold out **5 cell lines chosen by PCA on the pseudobulk profiles**, leak 30% |
| **PBMC** (Parse cytokines) | 9.7M / 90 cytokines / 18 cell types × 12 donors | hold out **4 of 12 donors**, leak 30% (the model still conditions on cell type) |
| **Norman2019** (CRISPRa) | 106 single targets + 131 pairs, one cell line | see below |

**Norman needs its own protocol.** It has a single context (K562), so holding out a
context is degenerate. It does have something better: its perturbations are 106 single
CRISPRa targets and 131 **pairs** of them, and every pair's two targets also occur on
their own. So the split holds out 30% of the **pairs** (`unseen_combo`). Those
perturbations never occur in training, which is the generalisation the paper asks for —
and because the perturbation table is over **targets**, with a label embedded as the sum
of its targets, a held-out pair is still representable. A per-label lookup table could not
predict them at all; there would be no row. For the three single-target datasets the same
code is exactly the plain `nn.Embedding` table.

**A Tahoe perturbation is a drug AT A DOSE, not a drug.** The atlas screens **379 drugs**
(plus DMSO) at ~3 doses each, giving **1,344 (drug, sample) pairs**; PerturbDiff's
"#Perturbations 1,137" counts those — its wording is "1,100 small-molecule drug treatment
**and dosage combinations**". Pooling the doses into one condition is not harmless:
measured over 190 (cell line, drug) groups, the cosine between one dose's delta and
another's is **0.389 at the median** (38% of groups below 0.3), with the strongest dose's
delta 1.6x the weakest in norm. Different doses drive partly different transcriptional
programs, so pooling gives a target that matches no real experiment, dilutes the DE gene
set, and asks the model to fit three different answers from one `(c, a)` input. Labels are
therefore `Drug@k` (control stays `DMSO_TF`), and the drug name is kept in `obs["drug"]`.

**Tahoe shard layout.** Shards are ordered by sample, so a contiguous prefix covers very
few drugs — the first 419 held 11.8M cells but only 95. `--tahoe_stride N` spreads the
download over the whole atlas; `--tahoe_stride 1` (3,388 shards, ~310 GB) is the full one.

**Full Tahoe needs the memmap backend.** 95.6M cells x 2,000 genes of float32 is **765 GB**,
which no machine here keeps resident, so `--backend memmap` streams the matrix straight to
disk (`X.f32` + `obs.parquet` + `meta.json`) and `PerturbData` opens it with `np.memmap`.
A training step touches 1,024 rows (8 MB) and measured ~40 ms cold vs ~37 ms warm on this
box — the page cache holds the working set, so residency is not the bottleneck. Three
places had to stop materialising label strings for this to be viable (prep's obs, the
loader's vocabularies, the sampler's grouping); each is fine at 2M rows and fatal at 96M.
`scripts/probe_memmap.py` reports the shape, the measured step time, and the step budget
for a target number of passes over the conditions.

Control cells stay in `train` under every protocol: they are the unperturbed reference
that inference is fed, and PerturbDiff likewise counts them separately from its
train/valid/test totals. `--control_split context` is the stricter variant.

**PBMC's hold-out column is not its context column.** The model conditions on cell type
(18) while the split holds out donors (4 of 12), so at evaluation the control cells a
prediction starts from are drawn from the **held-out donors'** own unperturbed cells, not
from a training donor's. For Replogle and Tahoe the two columns coincide and this is a
no-op.

## 4. Metrics

Two groups, reported separately ([metrics.py](perturbenergy/evaluation/metrics.py)):

* **HVG group** (all 2,000 genes) — the suite from *Benchmarking virtual cell models for
  in-the-wild perturbation response* `[B]` and the cell-eval suite PerturbDiff reports
  `[P]`: PCC, R2, R2_delta, MAE, MSE, RMSE, PDCorr, LFCSpear, PCA-LFCSpear, LogFC-cosine,
  DirAgr, ES, DEPrec/DERec/DEOverlap, AUROC, AUPRC, energy distance, MMD, Sinkhorn,
  PCA-Sinkhorn, and the perturbation-discrimination scores PDS-l1/l2/cos (= RLogFC).
* **DEG group** — **MSE, RMSE, MAE, R2** (plus R2_delta and PCC) over the true **top-50 /
  100 / 200** differentially expressed genes. Errors over all 2,000 HVGs are dominated by
  the ~95% of genes a perturbation barely touches, so the two groups can rank models
  differently.

DE genes are always ranked on the **real** data (perturbed vs its matched control,
Wilcoxon rank-sum, the `scanpy.tl.rank_genes_groups` / cell-eval convention), never on a
prediction — so every method is scored on the same genes. `--de_method delta` ranks by
`|log fold change|` instead.

Reference predictors, run through the identical protocol: `control` (predict no change —
the honest zero point of every delta metric), `mean_shift` (control + the perturbation's
mean training delta), and, for Norman, `additive` (`delta(A_B) = delta(A) + delta(B)`, the
no-interaction hypothesis).

## 5. Running it

```bash
source env.sh                                    # conda env `squidiff`, paths, TMPDIR

# --- data (only Tahoe was already on disk) ---
python -m perturbenergy.data.download --what panels norman replogle
python -m perturbenergy.data.download --what pbmc                 # 53 GB -> ~800 GB
python -m perturbenergy.data.download --what tahoe --tahoe_stride 1 --workers 24
python -m perturbenergy.data.prepare  --dataset replogle
python -m perturbenergy.data.prepare  --dataset norman
python -m perturbenergy.data.prepare  --dataset tahoe --max_cells 1500000   # capped
python -m perturbenergy.data.prepare  --dataset tahoe --backend memmap --tag full \
       --cap_per_group 0 --cap_control 0                                    # full, 765 GB
python -m perturbenergy.data.prepare  --dataset pbmc  --max_cells 1500000

# --- one dataset, everything: baselines + 4 variants + report ---
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all.sh replogle

# --- full Tahoe: probe first, then use the step budget it reports ---
python scripts/probe_memmap.py --path $PE_PREPPED/tahoe_full
ONLY="noenergy energy_lang energy_mis" STEPS=95000 bash scripts/run_tahoe_full.sh

# --- or piecemeal ---
python scripts/train.py    --dataset replogle --run rep_energy --evaluate
python scripts/train.py    --dataset replogle --run rep_noenergy --set model.use_energy=false
python scripts/evaluate.py --dataset replogle --predictor perturbenergy \
       --ckpt $PE_MODELS_ROOT/rep_energy/model_best.pt
python scripts/report.py   --dataset replogle
```

Config is `configs/default.yaml` ← `configs/<dataset>.yaml` ← `--set a.b=c`. Every run
writes `model_best.pt`, `model_last.pt`, `metrics.csv`, `vocab.json`, `config_used.json`
and `eval.json` to `$PE_MODELS_ROOT/<run>` — **never** into the project directory. All
paths come from [`perturbenergy/paths.py`](perturbenergy/paths.py) and are environment
overridable (`PE_DATA_ROOT`, `PE_MODELS_ROOT`, `PE_PREPPED`, …).

## 6. Layout

```
perturbenergy/
  paths.py             every on-disk location, env-overridable
  config.py            YAML + dotted overrides
  data/
    registry.py        dataset specs: columns, control label, panel, split protocol
    download.py        HuggingFace / Zenodo fetchers
    h5utils.py         dense / CSR / CSC readers that never load a whole file
    prepare.py         raw -> the uniform prepped .h5ad, or a memmap directory
    splits.py          unseen_context / unseen_combo / unseen_pert / random
    loader.py          resident or memmapped batching + vocabularies
  models/
    layers.py encoders.py decoder.py energy.py pert_embedding.py perturbenergy.py
  training/
    losses.py          Eq. 8 / 12 / 16
    langevin.py        Eq. 13-15
    trainer.py         Algorithm 1 + the two schedules
  evaluation/
    deg.py             which genes are DE (real data only)
    metrics.py         the HVG group and the DEG group
    predictors.py      control / mean_shift / additive / perturbenergy
    runner.py          the protocol
scripts/  train.py evaluate.py report.py report_html.py probe_memmap.py
          run_all.sh reeval.sh run_tahoe_full.sh tahoe_full_pipeline.sh
configs/  default.yaml + one per dataset
```

## 7. Implementation choices worth knowing

1. **`recon: mse+mmd`** — the paper says MMD; pure MMD reconstructs a population with the
   right shape but individually poor cells. `loss.recon: mmd` is the literal version.
2. **Hinge energy, `margin: 0.2`** — see the note in §2.
3. **`energy_lr` 10x below `vae_lr`, plus `energy_l2_reg`** — standard EBM stability.
4. **`n_candidates: 1` during training** — the energy shapes `z_a` through the loss, not
   through the forward pass. Set it >1 to let the energy also *select* among candidates
   at training time (inference always draws `eval.m_samples` and keeps the lowest).
5. **A probe of "how much perturbation information is in `z_a`" is circular** — `z_a` is
   computed *from* `a`, so it must decode `a` well. The non-circular checks are the
   prediction metrics on held-out conditions, and the `noenergy` ablation.
