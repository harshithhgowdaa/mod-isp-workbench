# Three-minute MOD-ISP demo and viva runbook

This runbook is aligned to the supplied rubric's Demo/Viva top band. The public URL hosts browser-side inference for unseen sRGB JPEG/PNG/WebP. Camera RAW and Bayer NPY still use the local Python app. The public path is not a mobile latency claim.

## Three-minute flow

| Time | Action | What to say |
|---|---|---|
| 0:00–0:20 | Open the live workbench and state the task. | “MOD-ISP compares classical, learned, and hybrid processing for degraded images. Camera RAW and rendered RGB use separate paths because their input domains differ.” |
| 0:20–0:55 | Run **Run held-out SIDD-CC example**. Compare Classical, Learned, and Hybrid; point to the matched reference and its PSNR/SSIM. | “This is one fixed published SIDD-CC sRGB test block, so these scores are paired measurements. It demonstrates the rendered-sRGB denoiser; it is not Bayer RAW-to-RGB evidence.” |
| 0:55–1:35 | On the public URL, upload the judge's unseen JPEG/PNG/WebP. Compare the browser-side Classical, Learned and Hybrid paths; explain why unpaired metrics are blank. For RAW, switch to the local app. | “The public sRGB upload stays in this browser and is resized to 1024 px on its long edge. Metrics require a matching reference. Camera DNG is handled locally by LibRaw with embedded camera metadata, then the sRGB model operates on that render.” |
| 1:35–2:10 | Explain the implemented stages and compact models. | “The Bayer student has 29,980 parameters and reconstructs SIDD-CC Bayer input to rendered sRGB. The separate Samsung NAFNet adaptation has 245,223 parameters and denoises rendered sRGB. The app keeps those domains separate. On full-frame Bayer validation, the current model averages 27.105 dB PSNR / 0.8966 SSIM; the prior model averaged 24.811 / 0.8372 on the same scene.” |
| 2:10–2:35 | Show the failure case and limits. | “Some Bayer validation captures regress. Our experimental RAW-statistics gate skips 22.4% of neural tiles after MHC alignment but loses 1.168 dB PSNR and is slower than Learned on median warm crop latency, so it fails the 0.5 dB quality budget. The 47.1 KB INT8 LiteRT file is a tile-model export, not an end-to-end mobile ISP.” |
| 2:35–3:00 | Summarize evidence and invite questions. | “The sRGB model averages 33.755 dB PSNR and 0.7757 SSIM over 928 SIDD-CC test blocks. Bayer evaluation uses a single validation scene; mobile NPU and native-phone browser performance are not measured.” |

## Evidence to keep visible

- The fixed benchmark example uses a 256×256 SIDD-CC test pair and computes its metrics against the aligned reference. The separate public browser path handles unseen sRGB uploads but has no reference metrics by default.
- The separate sRGB benchmark averages 928 blocks. It is not comparable to the Bayer validation task.
- The Bayer validation covers 36 captures from one held-out physical scene. Five deterministic crops per pair were used for checkpoint selection, so do not call it an untouched test. The current full-frame results are 27.105/0.8966 PSNR/SSIM mean; full paired metrics are in the readiness audit.
- The current training uses SIDD-CC. Samsung generic-lite weights initialized the sRGB model; Samsung S24, Zurich and MIT-Adobe5K datasets were not used.
- An iPhone SE DNG took 13.82 seconds end to end for 4030×3022 on Apple MPS. It exceeds the rubric's approximately 10-second target and is not a mobile measurement. LibRaw camera-rendered the DNG; this path uses the separate sRGB denoiser rather than the Bayer student.
- An unseen 6,000×4,000 JPEG took 18.40 seconds end to end after resizing to 16 MP; its Learned path took 11.94 seconds. The fixed bundled demo returned in 1.53 seconds after app restart. The separate browser-side path processed one generated 384×256 image in 1.42 seconds on the development workstation and one 1024×683 image in 1.64 seconds warm. Neither result is a phone benchmark or a general latency guarantee.

## Team rehearsal

Before presenting, assign a named speaker to each section and have every team member explain their own contribution. Each person should be able to answer:

1. What is the input representation and which code path processes it?
2. What does the model predict, what are its training targets, and what data split selected it?
3. How is the classical baseline constructed, and which stages are delegated to LibRaw for camera RAW?
4. How are PSNR, SSIM, Delta E and LPIPS measured, and when are scores unavailable?
5. What is the N6 failure case, why is the gate not an efficiency claim, and what does the LiteRT export exclude?

## Readiness gate before using this runbook at judging

- Verify the deployed browser-side inference on an actual phone, including first-use model/runtime download, unsupported-file handling, and a reference-free run; laptop and production smoke checks have passed.
- Measure cold and warm response time on named hardware against the rubric's ~10-second target; the current 1.64 s generated-image result is a warm desktop-browser smoke check only.
- Keep a camera RAW example ready in the local Python app; public RAW processing is not implemented.
- Rehearse to three minutes and ensure every member can explain the code, choices, metrics, and failure case.

Until those checks pass, present the script as a rehearsal guide, not evidence that the Demo/Viva top band is satisfied.
