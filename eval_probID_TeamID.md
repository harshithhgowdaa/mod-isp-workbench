# Comparative evaluation — Problem P2 / Team ID pending

> The blueprint identifies Problem P2. The assigned Team ID is not available in the workspace, so the report filename remains `eval_probID_TeamID.md` until it is known.

## Summary

MOD-ISP has separate Bayer RAW-to-sRGB and rendered-sRGB denoising models. Their metrics are reported separately. Results below come from local run artifacts; they are not official leaderboard scores or projected blueprint targets.

## Dataset use and split

All 230 SIDD-CC pairs in each modality are accounted for, but validation/calibration data are not used for gradient updates.

The current Bayer model uses three distinct random 128×128 crops per training pair per epoch for five epochs (2,490 optimizer steps); five deterministic center/corner crops per validation pair select the checkpoint. The sRGB model uses one random 256×256 crop per pair per epoch for 10 epochs. Thus every training pair is visited, but most source pixels are not sampled. See the [readiness audit](docs/hackathon-readiness-audit.md) for the full rubric gap analysis and dataset provenance.

| Modality | Training | Validation | Calibration | Test |
|---|---|---|---|---|
| Bayer RAW→sRGB | 166 pairs, eight physical scenes | 36 pairs, one held-out scene | 28 pairs, a separate scene | Not scored on official RAW test MATs because matching per-capture metadata is absent |
| Rendered sRGB | 194 pairs, nine physical scenes | 36 pairs, one held-out scene | — | 928 untouched SIDD-CC blocks |

The full SIDD training collection and Samsung S24, Zurich, and MIT-Adobe5K datasets were not used. Samsung generic-lite NAFNet weights initialized the separate sRGB denoiser. The Bayer student started from the previously MetaISP-2023-distilled checkpoint and received five epochs of multi-crop supervised SIDD-CC fine-tuning.

## Bayer RAW-to-sRGB comparison

36 scene-disjoint validation captures; results are reported both on 256×256 center crops and full frames. Center-crop validation loss informed checkpoint selection, so neither presentation is an untouched test.

Full-frame means across captures (LPIPS is an area-weighted average over reflected 256×256 tiles):

| Path | PSNR ↑ dB | SSIM ↑ | Delta E 2000 ↓ | Tiled LPIPS ↓ |
|---|---:|---:|---:|---:|
| Classical | 17.460 | 0.5961 | 14.835 | 0.3650 |
| Learned | 27.105 | 0.8966 | 8.054 | 0.2250 |
| Hybrid with calibrated gate | 27.105 | 0.8966 | 8.054 | 0.2250 |

Learned beats Classical on 34/36 PSNR, 30/36 SSIM, 30/36 Delta E, and 23/36 tiled LPIPS comparisons. The current full-frame report is `evaluation_siddcc_bayer_multicrop_full.json`; per-image regressions and camera-family details are in `docs/failure-case-analysis.md`. On this small one-scene split, N6 improves PSNR/SSIM but regresses Delta E/LPIPS (n=2). Hybrid routes 100% to neural and provides no measured compute savings.

The center-crop means below are retained as a separate checkpoint-selection diagnostic:

| Path | PSNR ↑ dB | SSIM ↑ | Delta E 2000 ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Classical | 15.693 | 0.6037 | 14.843 | 0.2478 |
| Learned | 23.813 | 0.8393 | 9.663 | 0.2584 |
| Hybrid calibrated | 23.813 | 0.8393 | 9.663 | 0.2584 |

Against the previous Learned checkpoint on these identical center crops, the current model wins 34/36 PSNR, 36/36 SSIM, 34/36 Delta E, and 36/36 LPIPS comparisons. It has 29,980 parameters and a 126,637-byte checkpoint. The full-frame result has a different LPIPS protocol and is reported above; both come from one validation scene used for checkpoint selection and do not establish multi-camera generalization.

An exploratory linear-RGB 3×3 color-matrix fit used 1,024 sampled pixels per each of the 166 Bayer training pairs. On the same 36 center-crop validation captures, the full matrix changed mean PSNR/SSIM/Delta E/tiled LPIPS from 20.339/0.7290/12.380/0.3462 to 20.680/0.7421/11.738/0.3560. Three metrics improve and LPIPS regresses. An interpolation sweep shows the tradeoff persists, and this validation scene previously informed model selection. The matrix is not deployed and is not an independent camera-calibration result. See `ccm_evaluation_siddcc.json` and `ccm_strength_sweep_siddcc.json`.

## Gate quality/compute check

Calibration used 28 pairs from a separate physical scene and allowed at most 0.5 dB mean PSNR loss against fully learned output. With the current checkpoint, the selected route fraction is 100%. On all 36 held-out pairs, the gate also routed 100% of tiles. Mean paired PSNR change was 0 dB; warm latency was 50.45 ms all-neural versus 48.54 ms calibrated per 256-pixel crop, within timing noise. Routing 95% already loses 2.70 dB on calibration. The variance gate does **not** reduce computation under this quality budget.

A RAW-statistics tile gate was also calibrated separately. It routed 30.0% of tiles to Learned with a 0.481 dB calibration PSNR drop, but on validation it routed 24.78% and lost 4.442 dB PSNR, 0.0736 SSIM, and 1.686 Delta E against all-Learned. It is explicitly marked ineligible and is not deployed. See `tile_gate_evaluation.json`.

## Rendered-sRGB benchmark

SIDD-CC benchmark split, all 928 blocks. Mean block scores:

| Input/model | PSNR ↑ dB | SSIM ↑ |
|---|---:|---:|
| Noisy input | 26.119 | 0.4251 |
| Samsung generic-lite initialization | 29.303 | 0.5738 |
| Fine-tuned sRGB model | 33.755 | 0.7757 |

The model uses 245,223 parameters and a 1.17 MB checkpoint. The strict validation split is physical-scene-disjoint: 194 train pairs and 36 validation pairs; epoch 10 was selected. The official test blocks were not used in training or selection. This is rendered-sRGB denoising, not Bayer RAW ISP performance.

### Held-out sRGB stress sweep

The same 928 clean references were used for deterministic synthetic perturbations to noisy test inputs:

| Input condition | Input PSNR | Model PSNR | Recovery | Input SSIM | Model SSIM |
|---|---:|---:|---:|---:|---:|
| Published noisy | 26.119 | 33.755 | +7.636 dB | 0.4251 | 0.7757 |
| Additional Gaussian σ=0.02 | 25.398 | 32.697 | +7.299 dB | 0.3740 | 0.7365 |
| Additional Gaussian σ=0.05 | 23.003 | 28.686 | +5.683 dB | 0.2404 | 0.5341 |
| 0.5× underexposure | 17.063 | 17.462 | +0.398 dB | 0.5196 | 0.6910 |
| JPEG quality 30 | 31.127 | 30.600 | −0.528 dB | 0.7025 | 0.7107 |

These are synthetic sRGB stress tests, not real-camera RAW or arbitrary blur claims. JPEG artifacts regress in PSNR.

## INT8 LiteRT

The current 47.1 KB export contains the Bayer tile network only. Representative data comprise 256 spatially distributed tiles from all 28 calibration captures, with calibrated input range −0.147 to 1.323. On 72 held-out tiles, output MAE mean/median/p95 are 0.00625 / 0.00302 / 0.01211, with maximum tile MAE 0.05284 and mean input clipping 0.00044%. Median/p95 CPU inference is 0.315/0.345 ms per 56×56 tile. The export excludes the gate, RGB model, LibRaw, tone/color finishing, and CNILUT. No phone or mobile NPU was available; these CPU figures do not show NPU compatibility or speedup.

## Rubric readiness and limits

- Evaluation results are measured, but the strong 928-block score is a separate sRGB task; Bayer evaluation is limited to one validation scene with camera-family and per-image regressions.
- Core local flows and bad-input responses work. The app is not deployed at the public URL required by the screenshot's top UI/UX band. A 4030×3022 iPhone SE DNG API request took 13.86 seconds, above its ~10-second response target; see `raw_dng_verification.json`.
- The README, architecture diagram, model design, ablation, failure-case, and readiness audit reflect current paths and evidence.
- The supplied screenshots show five visible criteria whose maxima total 100. The prose says Demo/Viva (25) plus five other criteria, while only four other rows are visible; displayed top-band ranges also conflict with several row maxima. No precise score can be inferred, and relative rank remains unknown.

## Artifacts

- `docs/hackathon-readiness-audit.md`
- `docs/ablation-study.md`
- `docs/failure-case-analysis.md`
- `evaluation_siddcc_bayer.json`
- `evaluation_siddcc_bayer_multicrop_full.json`
- `evaluation_siddcc_bayer_mps_smoke.json`
- `tile_gate_evaluation.json`
- `raw_dng_verification.json`
- `gate_validation_siddcc.json`
- `models/sidd_srgb_nafnet_lite.json`
- `robustness_srgb.json`
- `litert_verification_multicrop.json`
