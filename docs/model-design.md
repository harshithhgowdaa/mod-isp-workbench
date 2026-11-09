# Model and implementation design

## Scope

MOD-ISP is an experimental local image signal processing workbench. It implements two distinct restoration domains:

1. A compact Bayer mosaic to RGB student (`bayer-v1`), with classical demosaicing and a tile-gated hybrid path.
2. A fine-tuned sRGB NAFNet denoiser for already rendered images, including camera RAW rendered to sRGB by LibRaw.

It is not a universal camera ISP, a camera-calibrated denoiser for arbitrary sensors, or a measured mobile-NPU deployment.

## Bayer model

`ModISP` accepts four packed RGGB planes at half resolution. It applies a 2×2 pixel-unshuffle and a fixed Haar-like sum/difference transform, followed by two convolutional encoder layers, channel attention, spatial attention, a decoder, and pixel-shuffle upsampling. A small pointwise color network receives the RGB estimate and its image-wide mean color context. The model has 29,980 trainable parameters in the shipped configuration.

Training uses registered input/target crops and an L1 reconstruction objective. An optional teacher adds a weighted L1 distillation term. The promoted student was distilled from the authors' 2023 MetaISP `real_D` checkpoint for 320 optimizer steps. The teacher adapter uses patch-local resized BGR context and explicit sensor metadata; this differs from the teacher's full-frame inference context and remains an adaptation.

The current Bayer candidate was fine-tuned for 10 epochs on all 166 RAW→sRGB training pairs (1,660 optimizer steps) from eight SIDD-CC physical scenes. It started from the student previously distilled from the alternative 2023 MetaISP teacher. Model selection used 36 pairs from a ninth physical scene; gate and LiteRT calibration use 28 pairs from a tenth. The split groups do not overlap. Both center-crop and full-frame metrics are available for the 36-pair validation scene. Since center-crop validation loss informed checkpoint selection, full-frame results are not an untouched test; they remain limited to one scene and do not establish multi-camera generalization.

## Classical and hybrid Bayer paths

The classical baseline normalizes and white-balances the mosaic, applies Malvar–He–Cutler (MHC) 5×5 Bayer demosaicing, then runs RGB finishing. Camera DNG rendering remains LibRaw-managed. The hybrid path is experimental and does not have a deployable quality-safe gate: the candidate-specific ExtraTrees policy routed 99% of tiles and lost 0.440 dB PSNR on the 36-capture validation scene, within the declared 0.5 dB bound, but invoked only 1% fewer neural tiles and was slower (89.1 ms median Hybrid vs 38.5 ms Learned). Its PSNR pass is not a compute or latency win. A cheaper Ridge gate routed 98% but lost 1.204 dB on the same validation scene. Neither is promoted. See `tile_gate_paperfaithful_mhc_512_actual.json` and `tile_gate_paperfaithful_mhc_ridge_512.json`.

An earlier ExtraTrees experiment with bilinear demosaicing and a different candidate is superseded. On the MHC path, the selected ExtraTrees policy routes 99% of tiles, loses 0.440 dB PSNR, and reduces tile calls by 1%; actual app-path median latency is 89.1 ms Hybrid versus 38.5 ms Learned. This passes only the PSNR-loss threshold, not the efficiency objective. The linear Ridge policy saves two percentage points of tile calls but loses 1.204 dB. Both use the same repeatedly consulted 36-capture development-validation scene and are not independent final tests. A tile-call percentage is not a measured MAC, energy, or end-to-end latency saving.

## sRGB path

The sRGB denoiser is a 245,223-parameter Samsung generic-lite NAFNet adaptation. The current run trained for 10 epochs on 194 SIDD-CC pairs from nine physical scenes and selected epoch 10 using 36 pairs from a tenth scene. On those validation crops, mean PSNR rose from 34.41 dB for the upstream initialization to 37.32 dB. On the untouched 928-block SIDD-CC test set, mean PSNR/SSIM are 33.755 dB/0.7757 (initialization: 29.303 dB/0.5738; noisy input: 26.119 dB/0.4251). This is an sRGB-domain denoising experiment, not the Bayer model and not an official leaderboard result.

For camera RAW, LibRaw performs camera-aware demosaic and sRGB rendering. The sRGB denoiser runs after that render. The Bayer student is not used on arbitrary camera RAW because camera-specific paired training and validation are unavailable.

## Execution and evaluation

`app.py` exposes local upload and processing APIs. It enforces a 30 MB upload cap and a 16-megapixel processing limit; oversized rendered RGB inputs are resized by the app while oversized RAW/NPY inputs fail to preserve mosaic handling. Where a same-size aligned reference is supplied, the evaluator records PSNR, SSIM, CIEDE2000 Delta E, and LPIPS where available. Per-run latency and memory figures are local to the recorded machine and process; memory accounting has documented exclusions.

Training/evaluation data, local virtual environments, and the original 2023 teacher weights are deliberately excluded from this repository. The code and scripts are complete; reproduce experiments only with separately acquired, authorized data and teacher checkpoints.

## Known design limits

- SIDD-CC provides only ten physical scene groups for the RAW training split; Bayer full-frame validation covers 36 captures from one scene, which informed checkpoint selection and is not broad camera generalization.
- Bayer learned and hybrid paths have substantial perceptual-metric regressions on some held-out patches.
- Relative gate calibration does not guarantee quality on unseen scenes.
- The sRGB and Bayer domains have separate weights and color assumptions.
- LiteRT tile equivalence has a severe outlier on one held-out tile; the mean alone hides it.
- No phone/NPU runtime, broad multi-camera benchmark, or downstream machine-vision benefit has been measured.
