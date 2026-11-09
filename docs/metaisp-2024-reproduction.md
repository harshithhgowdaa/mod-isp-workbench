# IJCAI 2024 MetaISP reproduction

## Source and scope

The requested 2024 model is **MetaISP: Efficient RAW-to-sRGB Mappings with Merely 1M Parameters**, published at IJCAI 2024. The paper describes MFEB, MCCB, MSCB, MRB, a four-level wavelet U-Net with widths 32/32/32/64, and 10 learned basis elements. The authors' paper page and available repository searches did not expose an official 2024 source repository or pretrained checkpoint. The public repository named `vccimaging/MetaISP` is a distinct 2023 model and is not used as the 2024 model here.

Accordingly, [`metaisp2024.py`](../metaisp2024.py) is a paper-informed PyTorch reproduction, not the authors' released implementation or weights. A 2026-11-08 architecture audit found that the earlier local experiment used incorrect U-Net widths (32/32/64/64 rather than 32/32/32/64), concatenated skips rather than the paper's per-pixel addition, and omitted the 2C correction projections. That earlier 1,095,553-parameter checkpoint and its metrics are superseded and must not be treated as the current paper-informed teacher. The corrected implementation uses the specified widths, additive skips, and K=2C MCCB/MSCB projections; it has 1,010,433 trainable parameters. It uses a 4-channel packed RGGB input and produces full-resolution RGB. The corrected run also uses the paper's two-stage loss family: stage 1 uses 0.25 Charbonnier + SSIM + VGG16 perceptual loss, and stage 2 uses 0.5 MSE + SSIM. The paper's published training datasets and compute schedule are not available locally, so the corrected model remains an adaptation and its metrics are not paper results.

## Training protocol

[`train_metaisp2024.py`](../train_metaisp2024.py) trains on the complete available SIDD-CC Bayer training split (166 paired captures), uses the separate 36-capture scene-disjoint validation split for checkpoint selection, and keeps the 28-capture calibration split out of optimizer updates. It trains the reproduction from random initialization; it does not claim to use an official 2024 teacher checkpoint. Training augmentation flips the 2×2 Bayer-cell grid, preserving RGGB phase while flipping all four packed planes spatially. The corrected run records its best checkpoint and TorchScript teacher export after each completed epoch. It can resume weights/history after interruption, but restarts the optimizer state. The earlier incorrect run used a clamped training prediction that could zero gradients; that behavior has been removed from the corrected loss path.

An initial attempt was stopped at epoch 8 after review found that single-pixel flips changed CFA phase. Its artifacts are preserved under `work/metaisp2024_bad_cfa_augmentation/` for audit only. A subsequent 20-epoch run and its dependent student candidate are also superseded: code review found that the network topology did not match the paper despite phase-preserving augmentation. The corrected topology was trained from random initialization. The corrected run completed the paper-inspired 20+5 schedule on 166 training pairs, with 36 scene-disjoint validation captures and 28 calibration captures excluded from training. The selected checkpoint is stage-two epoch 24 at 24.600 dB mean validation PSNR; results varied materially by crop/epoch, so this remains one-scene validation evidence. The run report and checkpoint are `models/metaisp2024_siddcc_paperfaithful.json` and `.pt` (1,010,433 parameters).

Run from the repository root:

```bash
.venv312/bin/python train_metaisp2024.py \
  --data data/siddcc_bayer/train \
  --validation-data data/siddcc_bayer/validation \
  --out models/metaisp2024_siddcc_paperfaithful.pt \
  --epochs 20 --finetune-epochs 5 --patch 128 --seed 2026 --device mps \
  --lr 5e-4 --finetune-lr 2e-5
```

Resume from the latest saved best checkpoint if needed:

```bash
.venv312/bin/python train_metaisp2024.py \
  --data data/siddcc_bayer/train \
  --validation-data data/siddcc_bayer/validation \
  --resume models/metaisp2024_siddcc_paperfaithful.pt \
  --out models/metaisp2024_siddcc_paperfaithful.pt \
  --epochs 20 --finetune-epochs 5 --patch 128 --seed 2026 --device mps \
  --lr 5e-4 --finetune-lr 2e-5
```

Do not use comparison or candidate results from the superseded topology as evidence for the corrected reproduction. The corrected reproduction versus current student comparison on identical 256×256 center crops measured PSNR 21.978/23.795 dB, SSIM 0.9205/0.8396, Delta E 11.096/10.221, and LPIPS 0.2037/0.2585. The teacher trades lower PSNR and higher Delta E for better SSIM and LPIPS; it should not be assumed superior.

`models/mod_isp_metaisp2024_distilled.pt` was distilled from the superseded topology and is not the corrected teacher's student. A fresh five-epoch candidate, `models/mod_isp_metaisp2024_paperfaithful_distilled.pt`, was distilled from the corrected TorchScript teacher over all 166 training pairs (one 128×128 crop per pair per epoch, 830 optimizer updates); epoch 2 was selected at validation L1 0.03769. On the same 36 app-path validation crops, baseline/candidate means were PSNR 23.813/24.052 dB, SSIM 0.8393/0.8458, Delta E 9.663/9.889, and LPIPS 0.2584/0.2588. PSNR and SSIM improve slightly, while Delta E worsens by 0.226 and LPIPS is essentially flat. This is a development candidate only and remains separate from `models/mod_isp.pt`.

To reproduce the student distillation run:

Compare the final teacher and the current student on the same validation crops before distillation:

```bash
.venv312/bin/python evaluate_metaisp2024.py \
  --data data/siddcc_bayer/validation \
  --teacher models/metaisp2024_siddcc_paperfaithful.pt \
  --student models/mod_isp.pt \
  --patch-size 256 \
  --out metaisp2024_student_comparison.json
```

```bash
.venv312/bin/python train.py \
  --data data/siddcc_bayer/train \
  --validation-data data/siddcc_bayer/validation \
  --init-checkpoint models/mod_isp.pt \
  --teacher models/metaisp2024_siddcc_paperfaithful_teacher.ts \
  --teacher-format torchscript \
  --out models/mod_isp_metaisp2024_paperfaithful_distilled.pt \
  --epochs 5 --patch 128 --crops-per-image 1 --validation-crops 5 --seed 2028
```

This is student distillation from a locally trained paper-informed reproduction, not author-provided 2024 weights. Gate calibration used MHC demosaicing and a separate 28-capture calibration scene. The selected ExtraTrees gate routed 99% of tiles; actual app-path inference on 36 validation captures lost 0.440 dB PSNR (within the 0.5 dB threshold), but reduced neural tile invocations by only 1% and ran slower (89.1 ms median Hybrid vs 38.5 ms Learned). A cheaper Ridge variant routed 98% but lost 1.204 dB. Neither gives a validated quality-safe computation reduction and neither is deployed. These repeated validation-scene experiments are development evidence, not an untouched test. See `tile_gate_paperfaithful_mhc_512_actual.json` and `tile_gate_paperfaithful_mhc_ridge_512.json`.

A separate MHC-aligned gate check used the live production checkpoint (`models/mod_isp.pt`) because the workbench was still loading a policy calibrated for the older classical renderer. On 36 validation center crops the loaded gate routed 77.56% of tiles, lost 1.168 dB PSNR, and measured 23.88 ms median Hybrid latency vs 18.41 ms Learned. Fresh calibration did not fix the trade-off: production-checkpoint 256px Ridge lost 1.864 dB at 96% routing; 512px ExtraTrees lost 0.552 dB at 99% routing and ran 91.6 ms vs 37.6 ms median. The selected MHC + ExtraTrees policy for the paper-informed student was the only candidate inside the PSNR-loss threshold, but it saved only 1% of neural calls and ran 89.1 ms vs 38.5 ms. These outputs are not evidence of a useful quality-safe compute reduction. Reproduction reports are `tile_gate_current_app_after_mhc.json`, `tile_gate_production_mhc_256.json`, `tile_gate_production_mhc_512_actual.json`, and `tile_gate_paperfaithful_mhc_512_actual.json`.

The requested full SIDD (about 24,000 images), the Samsung S24 data, ZRR, and MAI21 were not used. Only SIDD-CC was available locally; the disk has insufficient free space for the full releases, and no external or cloud storage is available. The Samsung repository supplied the generic-lite model initialization, not the full training data. Do not claim full-dataset training.

### Follow-up: all locally available SIDD-CC RAW training pairs

The earlier corrected reproduction and its validation-selected student above remain as recorded experiments. A later separate run used all 230 pairs in the SIDD-CC RAW training split for a fixed 20+5 schedule, with no validation or checkpoint selection, and distilled a separate five-epoch student from the resulting local paper-informed teacher. This does not replace the production checkpoint or the prior validation-selected reproduction. It supersedes the earlier count of 166 pairs only for this explicitly full-train experiment; it does not mean the roughly 24,000-image SIDD-Full, Samsung S24, Zurich, or MAI21 were used. See [the full-train report and independent RAW gate evaluation](siddcc-fulltrain-gate-evaluation.md) for exact run artifacts and the test-block white-balance proxy limitation.

## References

- [IJCAI 2024 paper](https://www.ijcai.org/proceedings/2024/76)
- [SIDD dataset](https://abdokamel.github.io/sidd/index.html)
- [Samsung Modular Neural ISP code and dataset notes](https://github.com/SamsungLabs/modular_neural_isp)
