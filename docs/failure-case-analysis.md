# Failure-case analysis

This report uses the current 29,980-parameter multi-crop Bayer checkpoint and all 36 captures from one physical scene-disjoint SIDD-CC validation scene. The scene was used for checkpoint selection, so these are validation measurements, not an untouched test or broad cross-camera benchmark. Full-frame rows are in `evaluation_siddcc_bayer_multicrop_full.json`; paired 256×256 center-crop rows are in `evaluation_siddcc_bayer_multicrop_256.json`.

## Full-frame comparison

LPIPS is an area-weighted average over 256×256 tiles (edge tiles reflected to 256×256). Means are:

| Path | PSNR ↑ dB | SSIM ↑ | Delta E 2000 ↓ | Tiled LPIPS ↓ |
|---|---:|---:|---:|---:|
| Classical | 17.4597 | 0.5961 | 14.8353 | 0.3650 |
| Learned | 27.1053 | 0.8966 | 8.0536 | 0.2250 |
| Hybrid, 100% neural route | 27.1053 | 0.8966 | 8.0536 | 0.2250 |

Learned beats Classical on PSNR for 34/36 captures, SSIM for 30/36, Delta E for 30/36, and LPIPS for 23/36. It is a substantial aggregate improvement, but not uniformly better on every quality measure or image.

## Residual failures

- `0006_NOISY_RAW_020.MAT`: PSNR falls 0.52 dB, SSIM falls 0.1062, Delta E rises 1.18, and LPIPS worsens 0.1546.
- `0006_NOISY_RAW_146.MAT`: PSNR falls 0.32 dB, SSIM falls 0.1064, Delta E rises 1.85, and LPIPS worsens 0.1531.
- `0030_NOISY_RAW_138.MAT`: PSNR improves 1.93 dB and SSIM 0.0647, but Delta E worsens 1.78.
- LPIPS remains worse than Classical on 13/36 captures even though the mean LPIPS improves.

The two `0006` samples belong to the S6 family, not N6. That family was a pronounced failure with the prior checkpoint; multi-crop training fixed its mean behavior but did not eliminate these two capture-specific regressions.

## Camera-family diagnostic

These groups come from one physical scene and have small samples. They are descriptive only, not evidence of cross-camera generalization.

| Family | n | Classical → Learned PSNR | Classical → Learned SSIM | Classical → Learned ΔE | Classical → Learned LPIPS |
|---|---:|---:|---:|---:|---:|
| G4 | 2 | 14.145→27.635 | 0.6116→0.9494 | 16.365→8.735 | 0.3674→0.1545 |
| GP | 6 | 15.672→29.110 | 0.4942→0.9185 | 18.265→6.870 | 0.4215→0.2066 |
| IP | 8 | 14.753→27.281 | 0.3891→0.9169 | 19.330→8.875 | 0.4791→0.2045 |
| N6 | 2 | 23.470→26.435 | 0.8704→0.8927 | 6.950→8.665 | 0.2490→0.2987 |
| S6 | 18 | 18.959→26.374 | 0.6898→0.8749 | 12.401→7.939 | 0.3081→0.2398 |

N6 now improves PSNR and SSIM but regresses Delta E and LPIPS in this validation scene. A two-sample family average is not enough to justify a camera-aware route. Another untouched physical scene is required.

## Previous-checkpoint comparison

On the same 36 full frames, the current model improves over the prior Bayer checkpoint by +2.294 dB mean PSNR and +0.0595 SSIM, lowers Delta E by 2.395 and tiled LPIPS by 0.0874. It wins on 34/36 PSNR, 36/36 SSIM, 33/36 Delta E, and 34/36 LPIPS comparisons. The paired center-crop comparison is reported separately in `evaluation_siddcc_bayer_multicrop_256.json`.

## Gate and compute

The recalibrated quality gate selects a 100% neural route on calibration data under its 0.5 dB PSNR-loss budget. The 95% route already loses 2.70 dB on calibration. On the held-out validation scene, the selected gate and all-neural path are identical. Mean warm latency was 50.45 ms all-neural and 48.54 ms gated for 256×256 crops; this difference is within timing noise. Do not claim a compute saving.

An earlier tile-level gate using 12 RAW-only statistics met the calibration PSNR budget at a 30.0% route, but lost 4.442 dB on validation. It is ineligible and not deployed; route fraction is not a measured latency or MAC reduction. See `tile_gate_evaluation.json`.

## LiteRT quantization

The current 47.1 KB full-INT8 export was compared with the current PyTorch checkpoint on 72 held-out tiles. Mean/median/p95 output MAE is 0.00625/0.00302/0.01211; maximum tile MAE is 0.05284 and mean input clipping is 0.00044%. CPU inference median/p95 is 0.315/0.345 ms for a 56×56 tile under TensorFlow Lite XNNPACK. These are desktop CPU timings, not end-to-end ISP or mobile NPU results. See `litert_verification_multicrop.json`.

## Next useful checks

1. Inspect the two `0006` failure captures and high-gain samples with a fixed display range; confirm alignment, CFA/WB metadata, and target color processing.
2. Freeze the current checkpoint and evaluate it on a new physical scene and camera-disjoint dataset.
3. Train and assess a gate against a held-out quality/compute Pareto frontier; keep it disabled for efficiency claims until real savings are measured.
