# Vercel workbench

The public page combines a measured SIDD-CC sRGB benchmark comparison with browser-side inference for a judge's unseen rendered-RGB image. JPEG, PNG, and WebP image pixels stay in the browser. The static site uses a fixed-tile ONNX export of the fine-tuned sRGB NAFNet with ONNX Runtime Web/WASM. It does not send uploads to a server.

This public path is limited to sRGB. Images above 1024 px on the long edge are reduced before processing. Camera RAW and normalized Bayer NPY remain available through the local Python app. The Hosted Classical path applies exposure/tone mapping, a lightweight bilateral filter, and sharpening. Learned uses the sRGB NAFNet plus the local RGB tone finish. Hybrid blends the two by tile-level high-pass score after the full Learned path has run; this is a visual hybrid comparison and does not save compute. PSNR, SSIM, Delta E, and LPIPS are shown only for the aligned benchmark sample because arbitrary uploads have no ground truth.

## Build

The fixed example assets are derived from the held-out SIDD-CC test block at static/demo/. Regenerate the comparison and the browser model from the trained checkpoint:

```sh
.venv312/bin/python scripts/build_static_showcase_assets.py
.venv312/bin/python export_srgb_onnx.py --out static/showcase/nafnet-srgb.onnx
python3 scripts/build_vercel_site.py
```

The exporter requires PyTorch, the ONNX package, the checkpoint, and the pinned Samsung NAFNet source under third_party/modular_neural_isp. The ONNX Runtime Web 1.31.1 WASM files are vendored under static/vendor/ort with their MIT license. The adapted model's upstream CC BY-NC-SA 4.0 terms and provenance are published in static/showcase/MODEL-NOTICE.md and MODEL-LICENSE.md.

The fixed sample is one 256×256 aligned test block (scene 0164, block 1 of 928). Its per-path scores are measured after the local workbench finishing path. The separate 928-block aggregate belongs to the sRGB denoiser evaluation; neither result is a Bayer RAW score. See static/demo/README.md for sample provenance.

## Deployment

`vercel.json` serves dist/ as a static site. scripts/build_vercel_site.py checks for the example data, ONNX model, browser runtime, and license files. The production alias is [mod-isp-workbench.vercel.app](https://mod-isp-workbench.vercel.app/). The browser inference update was deployed on 2026-11-08. Production HTML, ONNX, WASM, and license assets returned HTTP 200, and a generated 384×256 JPEG processed to all three paths in 1.42 s on the development workstation's in-app browser (first model use; not a phone benchmark). A 390px responsive viewport had no horizontal overflow. The site code uses only static model/runtime GETs; no image upload endpoint is configured.
