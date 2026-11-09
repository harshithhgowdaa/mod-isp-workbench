# Ablation study and component comparisons

## Reporting rule

Results are separated by task and dataset. Bayer results include both 256×256 center crops and full-frame reconstructions for the 36 SIDD-CC captures in one physical scene. sRGB metrics use SIDD-CC benchmark blocks and a separate model. No statistical significance or cross-camera generalization is claimed. The RAW scene was disjoint from gradient training, but was used for checkpoint selection through center-crop validation loss; full-frame scores are validation evidence, not an untouched test.

**Current checkpoint:** the multi-crop student in `models/mod_isp.pt` was trained for five epochs from the prior 2023-MetaISP-distilled student. It used three distinct random 128×128 crops per pair per epoch (2,490 optimizer steps) and selected the checkpoint using five fixed validation crops per pair. Current results are listed below; they supersede the earlier one-crop values recorded in the first version of this ablation.

## Bayer three-path comparison

| Path | PSNR ↑ dB | SSIM ↑ | Delta E 2000 ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Classical | 15.693 | 0.6037 | 14.843 | 0.2478 |
| Prior Learned checkpoint | 20.479 | 0.7577 | 12.143 | 0.3377 |
| Current multi-crop Learned | 23.813 | 0.8393 | 9.663 | 0.2584 |
| Current Hybrid calibrated gate | 23.813 | 0.8393 | 9.663 | 0.2584 |

The current Learned path improves mean PSNR by 8.120 dB and SSIM by 0.2356 over Classical. Mean Delta E falls by 5.180; center-crop LPIPS rises by 0.0106 (worse). Compared with the prior Learned checkpoint, it wins 34/36 PSNR, 36/36 SSIM, 34/36 Delta E, and 36/36 LPIPS pairs. See `evaluation_siddcc_bayer_multicrop_256.json` and `docs/failure-case-analysis.md`.

On the same 36 center crops, the current model improves over the prior checkpoint by +3.334 dB PSNR and +0.0817 SSIM, lowering Delta E by 2.480 and LPIPS by 0.0793 on average. It is not proof that every metric beats Classical on every image.

## Full-frame Bayer comparison

The checkpoint was run on each entire aligned Bayer validation capture (36 images; no center crop). PSNR, SSIM, and Delta E are computed over full frames. To bound memory, LPIPS is the area-weighted mean over non-overlapping 256×256 tiles, with reflected padding for edge tiles; it is not the whole-frame single-forward LPIPS value.

| Path | PSNR ↑ dB | SSIM ↑ | Delta E 2000 ↓ | Tiled LPIPS ↓ | Median path latency |
|---|---:|---:|---:|---:|---:|
| Classical | 17.460 | 0.5961 | 14.835 | 0.3650 | 3.95 s |
| Current Learned Bayer student | 27.105 | 0.8966 | 8.054 | 0.2250 | 2.53 s |
| Current Hybrid calibrated gate | 27.105 | 0.8966 | 8.054 | 0.2250 | 1.62 s incremental composition |

Learned improves the full-frame mean by 9.646 dB PSNR, 0.3005 SSIM, reduces Delta E by 6.782, and reduces tiled LPIPS by 0.1400. It wins PSNR on 34/36 images, SSIM on 30/36, Delta E on 30/36, and LPIPS on 23/36. The calibrated gate routes 100%, so Hybrid equals Learned; its latency is composition after the learned output already exists and is not a standalone speedup. The full-resolution run's process high-water RSS was 3.68 GiB on macOS arm64. `evaluation_siddcc_bayer_multicrop_full.json` contains all per-image measurements.

### Batched Bayer inference follow-up

**MHC gate recheck (2026-11-08):** the classical Bayer demosaicer changed from bilinear to Malvar–He–Cutler, so the prior gate report no longer describes the live output. Reverification of the currently loaded production gate with MHC on 36 center crops routed 77.56%, lost 1.168 dB PSNR, and measured median warm latency 23.88 ms Hybrid vs 18.41 ms Learned. Fresh 512px ExtraTrees calibration for the production checkpoint lost 0.552 dB at 99% routing; 256px Ridge lost 1.864 dB at 96%. The separate 2024-inspired student had one policy within the PSNR bound (−0.440 dB) but only 1% fewer calls and over twice the latency. This reinforces that none of the tested policies delivers a quality-safe efficiency result.

The application now caches the Bayer checkpoint and batches same-shaped tiles on the available accelerator (256 tiles per batch on MPS, 128 on CPU). Compared with the prior scalar CPU tile loop, the first three full-resolution validation captures preserved PSNR, SSIM, Delta E, and tiled LPIPS exactly to the report precision. Learned-path latency on those same captures changed from 6.80/10.26/12.24 s to 3.09/2.47/2.65 s (median 10.26→2.65 s, 3.88× faster). The candidate output was also byte-identical to the former scalar path on a 256×256 crop. These are local Apple MPS measurements for three frames, not a new full-split latency result or mobile NPU benchmark. See `evaluation_siddcc_bayer_mps_smoke.json`.

## Gate ablation: quality-constrained routing

Gate calibration used 28 pairs from a physical scene excluded from training and validation. The criterion was minimum routed tiles while keeping mean calibration PSNR within 0.5 dB of the all-neural output.

| Measurement | Result |
|---|---:|
| Selected neural route fraction | 100% |
| Held-out mean routed tiles | 100% |
| All-neural mean warm latency | 50.45 ms |
| Calibrated mean warm latency | 48.54 ms |
| Paired quality change | 0.000 dB PSNR |

This is the older quality-constrained gate and remains an historical result: it selects every tile and yields no compute reduction. The 0.95 route candidate lost 2.70 dB on calibration data; lower fractions lost more. The app now uses a separate RAW-feature gate experiment described below. Routed tiles are not measured MACs. See `models/gate.json` and `gate_validation_multicrop.json`.

### Image-level RAW-statistics gate experiment

`evaluate_image_gate.py` fits a ridge regressor on RAW statistics from the 28-pair calibration scene to predict each 256-pixel center crop's Learned-minus-Classical PSNR. It selected a threshold that routed 39.3% of calibration crops to Learned while keeping calibration mean PSNR within 0.5 dB. On the untouched 36-pair validation scene, it routed only 22.2% to Learned (77.8% fewer neural image invocations in theory), but mean PSNR fell 2.035 dB versus all-Learned, outside the quality budget. SSIM fell 0.0229, Delta E was effectively unchanged (+0.0047), and LPIPS improved by 0.1180. Prediction correlation was 0.904, but benefit MAE remained 2.72 dB. This policy is not connected to the application: the single validation scene disproves the claimed quality-constrained efficiency. Features were measured on center crops, not full-resolution inputs, and no end-to-end latency savings were measured. Full model parameters, per-image predictions, and metrics are in `image_gate_evaluation.json`.

### Tile-level supervised quality gate experiment

`calibrate_tile_quality_gate.py` contains several historical gate experiments. The previous 12-feature tile-supervised gate in `models/gate_quality_tile.json` was not connected to the app and failed badly on validation. The current app uses the separately fitted 8-feature RAW-statistics Ridge model in `models/gate_quality_tile_runtimeaware.json`, originally calibrated on 166 training captures, tuned on 28 calibration captures, and evaluated on 36 captures from one validation scene. Its pre-MHC report (90.0% calibration routing, −0.483 dB calibration loss; 77.56% validation routing, −1.127 dB PSNR) is historical. After MHC replaced the classical renderer, actual selective inference on that same validation scene gave −1.168 dB PSNR, 77.56% routing, and median warm latency 23.88 ms Hybrid vs 18.41 ms Learned. The gate is explicitly `deployment_eligible: false` and shown as experimental in the app. See `tile_gate_current_app_after_mhc.json`, `models/gate_quality_tile_runtimeaware.json`, and `verify_tile_quality_gate.py` for the current app-path evidence.

## sRGB denoiser comparison

The strict physical-scene run used 194 pairs for training and 36 pairs from a disjoint physical scene for validation. On all 928 official SIDD-CC test blocks, noisy input measures 26.119 dB / 0.4251 SSIM, Samsung generic-lite initialization 29.303 / 0.5738, and the adapted model 33.755 / 0.7757. Test blocks were held out from optimization and checkpoint selection. This is a separate sRGB denoising task, not Bayer ISP evaluation.

## LiteRT and mobile status

The current 47.1 KB LiteRT artifact uses 256 spatially distributed representative tiles sampled across all 28 calibration captures. On 72 held-out tiles, mean/median/p95 PyTorch-to-INT8 output MAE are 0.00625 / 0.00302 / 0.01211; mean input clipping is 0.00044%. The maximum tile MAE is 0.05284. Median/p95 CPU inference time is 0.315/0.345 ms per 56×56 tile. The export covers the Bayer network only; it excludes the gate, RGB NAFNet, LibRaw, tone/color finishing, and CNILUT. No mobile NPU has been measured. See `litert_verification_multicrop.json`.

## Limitations

- Bayer full-frame evaluation covers one scene-disjoint validation scene across 36 captures; that scene informed checkpoint selection through center-crop validation loss, so it is not an untouched test and does not establish multi-camera generalization.
- On full frames, the current Bayer model improves mean LPIPS and Delta E over Classical, but some capture-level regressions remain; on center crops, mean LPIPS is slightly worse than Classical.
- The earlier calibrated gate routes 100% of tiles. The current local selective gate saves 22.4% of neural tile calls but, after MHC alignment, loses 1.168 dB PSNR and is slower than Learned (23.88 vs 18.41 ms median); it fails the 0.5 dB quality budget and is not deployment-eligible.
- sRGB evidence is separate from Bayer evidence and does not show how the Bayer model performs on arbitrary RAW cameras.
- No Android device, vendor delegate, public inference URL, or downstream machine-vision accuracy result is available.
