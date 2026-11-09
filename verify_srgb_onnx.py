"""Numerically compare a fixed-tile ONNX export with the PyTorch checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent
SAMSUNG = ROOT / "third_party" / "modular_neural_isp"
sys.path.insert(0, str(SAMSUNG))
from denoising.nafnet_arch import NAFNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/sidd_srgb_nafnet_lite.pt")
    parser.add_argument("--onnx", default="static/showcase/nafnet-srgb.onnx")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="onnx_verification_srgb.json")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = NAFNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    rng = np.random.default_rng(args.seed)
    sample = rng.random((1, 3, 256, 256), dtype=np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["restored_srgb"], {"srgb": sample})[0]
    with torch.inference_mode():
        expected = model(torch.from_numpy(sample)).cpu().numpy()
    if actual.shape != expected.shape:
        raise SystemExit(f"output shape mismatch: ONNX {actual.shape}, PyTorch {expected.shape}")
    if not np.isfinite(actual).all():
        raise SystemExit("ONNX output contains non-finite values")

    difference = np.abs(actual - expected)
    report = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "onnxruntime": ort.__version__,
        "providers": session.get_providers(),
        "seed": args.seed,
        "input_shape": list(sample.shape),
        "output_shape": list(actual.shape),
        "mean_absolute_error": float(difference.mean()),
        "p99_absolute_error": float(np.quantile(difference, 0.99)),
        "max_absolute_error": float(difference.max()),
        "finite": True,
        "tolerance": {"mean_absolute_error_max": 1e-4, "max_absolute_error_max": 2e-3},
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["mean_absolute_error"] > 1e-4 or report["max_absolute_error"] > 2e-3:
        raise SystemExit("ONNX/PyTorch parity exceeds the configured tolerance")


if __name__ == "__main__":
    main()
