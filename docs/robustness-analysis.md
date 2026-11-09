# Held-out sRGB robustness analysis

## Protocol

`evaluate_srgb_stress.py` evaluates the SIDD-CC SRGB checkpoint on all 928 published test blocks. The model was selected using training-split validation only. Four deterministic transforms are applied to the noisy test inputs: additional Gaussian noise at σ=0.02 and σ=0.05 (in [0,1] RGB), 0.5× underexposure, and JPEG quality 30. Each transformed input and its denoised output are compared with the same published clean sRGB reference. No transformed test block is used for training, calibration, or checkpoint selection. Seed: 20261107.

These are synthetic RGB-domain stress tests. They are not RAW Bayer measurements, camera captures, or evidence of recovery from motion blur or arbitrary ISP artifacts.

## Exposure/noise fine-tune candidate (not promoted)

An optional training augmentation was added to `train_sidd_srgb.py`: on 75% of training patches it applies a random 0.35×–1.0× exposure in linear RGB plus signal-dependent Gaussian and read noise. The noise parameters are synthetic and are not calibrated to a phone sensor. A two-epoch fine-tune of the existing checkpoint used the original scene-separated split (194 train pairs from scenes 002–010; 36 validation pairs from scene 001), learning rate 1e-5, seed 2026, and MPS. The best clean validation PSNR was 36.717 dB, below the existing checkpoint's 37.317 dB.

The candidate was scored once on the published 928-block test set as an exploratory rejection check. It is not deployed. Compared with the current checkpoint it loses 0.954 dB clean-test PSNR and 0.0165 SSIM. It recovers more PSNR from 0.5× underexposure (+0.778 dB versus +0.398 dB) and slightly reduces the JPEG-30 PSNR regression (−0.430 versus −0.528 dB), but performs worse under both additional Gaussian-noise conditions:

| Held-out condition | Current checkpoint PSNR | Candidate PSNR | Candidate change |
|---|---:|---:|---:|
| Published noisy | 33.755 | 32.801 | −0.954 dB |
| Additional Gaussian σ=0.02 | 32.697 | 31.490 | −1.207 dB |
| Additional Gaussian σ=0.05 | 28.686 | 27.303 | −1.383 dB |
| 0.5× underexposure | 17.462 | 17.842 | +0.380 dB |
| JPEG quality 30 | 30.600 | 30.697 | +0.097 dB |

This trade-off is not acceptable for the default model. Keep `models/sidd_srgb_nafnet_lite.pt` as the active checkpoint. The candidate and its complete run report are `models/sidd_srgb_nafnet_lite_robust_candidate.pt` and `.json`; its stress metrics are `robustness_srgb_candidate.json`. Because this candidate was inspected on the published test blocks, that set should not be described as untouched for future candidate selection. The augmentation remains available for a new scene/camera-disjoint training and validation protocol.

## Results

| Held-out input condition | Input PSNR (dB) | Model PSNR (dB) | PSNR change | Input SSIM | Model SSIM | SSIM change |
|---|---:|---:|---:|---:|---:|---:|
| Published noisy | 26.119 | 33.755 | +7.636 | 0.4251 | 0.7757 | +0.3507 |
| Additional Gaussian σ=0.02 | 25.398 | 32.697 | +7.299 | 0.3740 | 0.7365 | +0.3625 |
| Additional Gaussian σ=0.05 | 23.003 | 28.686 | +5.683 | 0.2404 | 0.5341 | +0.2937 |
| 0.5× underexposure | 17.063 | 17.462 | +0.398 | 0.5196 | 0.6910 | +0.1714 |
| JPEG quality 30 | 31.127 | 30.600 | −0.528 | 0.7025 | 0.7107 | +0.0082 |

The denoiser handles the two additive-noise cases well and offers a small gain on underexposed inputs. It does not improve PSNR for JPEG artifacts, so the workbench should not imply universal restoration. Scores are arithmetic means over blocks; per-block values and environment are in [`robustness_srgb.json`](../robustness_srgb.json).

## Reproduction

```sh
python evaluate_srgb_stress.py \
  --dataset data/siddcc_srgb \
  --samsung-repo ../../work/modular_neural_isp \
  --checkpoint models/sidd_srgb_nafnet_lite.pt \
  --seed 20261107 \
  --out robustness_srgb.json
```
