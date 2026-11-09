# Full SIDD-CC RAW training and gate evaluation

**Run date:** 2026-11-08 to 2026-11-09
**Status:** complete local experiment; candidate checkpoints are not promoted to production.

## Scope and provenance

This run used all 230 paired captures in the locally available SIDD-CC RAW-to-sRGB training split. It did not use SIDD-Full, Samsung S24, Zurich RAW-to-sRGB, MAI21, or the official 2024 MetaISP authors' weights. The trained 2024 model is the local paper-informed reproduction documented in [the reproduction note](metaisp-2024-reproduction.md); its 1,010,433 parameters follow the audited paper topology, but it is not an official author implementation or checkpoint. The [author publication page](https://czg1225.github.io/chenzigeng99/) lists the paper but did not expose a public 2024 code repository or weights.

The full-train teacher used a fixed 20+5 epoch schedule with no validation set or checkpoint selection. It ran on Apple MPS, seed 2026, patch size 128, and completed in 4,073 seconds. Its final checkpoint is `models/metaisp2024_siddcc_fulltrain.pt` with a TorchScript export at `models/metaisp2024_siddcc_fulltrain_teacher.ts`; the run record is `models/metaisp2024_siddcc_fulltrain.json`. A separate 29,980-parameter student was distilled for five epochs over the same 230 pairs (1,150 optimizer steps, MPS, seed 2028). It is recorded in `models/mod_isp_metaisp2024_fulltrain_distilled.pt` and `.json`. Neither model was selected using the official RAW test blocks.

The training inputs are pair visits with one random 128×128 crop per capture per epoch. Thus all 230 pairs contributed to optimization, but the run did not train over every pixel in each image. This is complete use of the available **SIDD-CC RAW training split**, not complete use of all SIDD or all datasets named in the proposal.

## RAW block preparation and split

The official SIDD-CC RAW test release contains 29 scenes × 32 paired blocks. Its RAW test MAT files do not include per-capture metadata. [`prepare_siddcc_raw_gate_test.py`](../prepare_siddcc_raw_gate_test.py) therefore infers camera family and CFA from the unique full-frame dimensions, then applies that camera family's median `AsShotNeutral` gains measured on the training split. These are explicit camera-level proxies, not the missing test-capture white-balance values. CFA phase is aligned to RGGB by applying the same leading-pixel crop to the RAW block and its RGB target; the remaining odd edge is cropped so Bayer dimensions are even. Do not treat these metrics as a metadata-complete RAW benchmark.

Scenes were stratified by camera family before processing: 14 scenes / 448 blocks for route calibration and 15 disjoint scenes / 480 blocks for final validation. The split and proxy details are in `work/siddcc_raw_gate_test/manifest.json`. The calibration fit uses tile features and tile-wise classical-vs-student error from the 230 SIDD-CC training pairs, selects the routing policy on the 448 calibration blocks, and evaluates it on the 480 validation blocks.

## Results

| Measurement | Result |
|---|---:|
| Final teacher | 1,010,433 parameters; 25 epochs; 230 training pairs |
| Distilled app student | 29,980 parameters; 5 epochs; 1,150 optimizer steps |
| Held-out RAW blocks | 480 across 15 scenes |
| Learned PSNR / SSIM | 23.9365 dB / 0.68361 |
| Learned Delta E / LPIPS | 10.0207 / 0.45291 |
| Gate-selected neural route | 100% on calibration and held-out blocks |
| Hybrid minus Learned PSNR | +0.00215 dB |
| Neural tile-call reduction | 0% |
| Mean warm Learned / Hybrid latency | 19.25 / 46.44 ms |
| Median warm Learned / Hybrid latency | 18.00 / 46.61 ms |

The gate selected 100% routing to meet its 0.5 dB calibration PSNR-loss budget. On held-out data it retained essentially identical quality while skipping no neural tiles; the hybrid orchestration was about 2.4× slower in this local timing. This experiment **does not meet the calibrated-gate computation-reduction objective**. Do not deploy or describe it as an efficiency win. The generated gate is explicitly marked `deployment_eligible: false` at `models/gate_siddcc_rawtest_fulltrain.json`.

The quality and timing summaries are in `work/tile_gate_siddcc_rawtest_fulltrain.json` and `work/tile_gate_siddcc_rawtest_runtime.json`. Timings are warm local inference/display-finishing measurements on macOS 15.7.7 arm64, over aligned 254×254 center crops from test blocks. They are not full-resolution upload latency, MAC count, energy, or mobile NPU measurements. The learned and hybrid metrics use the project's evaluation pipeline; the provided target is used only for scoring, never training.

## Reproduction commands

```sh
python scripts/stage_siddcc_full_bayer.py \
  --raw-data data/siddcc_rawsource/Raw/train/Data \
  --srgb-data data/siddcc_srgb/SRGB/train/Data \
  --out work/siddcc_bayer_all_train

.venv312/bin/python train_metaisp2024.py \
  --data work/siddcc_bayer_all_train \
  --final-full-training \
  --out models/metaisp2024_siddcc_fulltrain.pt \
  --epochs 20 --finetune-epochs 5 --patch 128 --seed 2026 \
  --device mps --lr 5e-4 --finetune-lr 2e-5

.venv312/bin/python train.py \
  --data work/siddcc_bayer_all_train \
  --teacher models/metaisp2024_siddcc_fulltrain_teacher.ts \
  --teacher-format torchscript \
  --out models/mod_isp_metaisp2024_fulltrain_distilled.pt \
  --epochs 5 --patch 128 --crops-per-image 1 --seed 2028

.venv312/bin/python prepare_siddcc_raw_gate_test.py

.venv312/bin/python calibrate_tile_quality_gate.py \
  --training-data work/siddcc_bayer_all_train \
  --calibration-data work/siddcc_raw_gate_test/calibration \
  --validation-data work/siddcc_raw_gate_test/validation \
  --checkpoint models/mod_isp_metaisp2024_fulltrain_distilled.pt \
  --patch-size 256 --max-psnr-drop 0.5 --predictor extra_trees \
  --trees 96 --min-leaf-tiles 20 --rank-within-image \
  --rank-fractions 0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.70 0.80 0.90 0.95 1 \
  --feather-pixels 0 2 4 \
  --out work/tile_gate_siddcc_rawtest_fulltrain.json \
  --gate-out models/gate_siddcc_rawtest_fulltrain.json

.venv312/bin/python verify_tile_quality_gate.py \
  --data work/siddcc_raw_gate_test/validation \
  --gate models/gate_siddcc_rawtest_fulltrain.json \
  --checkpoint models/mod_isp_metaisp2024_fulltrain_distilled.pt \
  --patch-size 256 --repeats 1 \
  --out work/tile_gate_siddcc_rawtest_runtime.json
```

Do not change route policy using the 480 validation blocks; they have already been evaluated and should remain a final held-out record for this run.
