"""Export the SIDD-CC sRGB NAFNet checkpoint for browser-side inference."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
SAMSUNG = ROOT / "third_party" / "modular_neural_isp"
sys.path.insert(0, str(SAMSUNG))
from denoising.nafnet_arch import NAFNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/sidd_srgb_nafnet_lite.pt")
    parser.add_argument("--out", default="static/showcase/nafnet-srgb.onnx")
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()
    if args.tile < 16 or args.tile % 16:
        parser.error("--tile must be a positive multiple of 16")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = NAFNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    sample = torch.zeros((1, 3, args.tile, args.tile), dtype=torch.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample,
            out,
            input_names=["srgb"],
            output_names=["restored_srgb"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    import onnx

    graph = onnx.load(out)
    onnx.checker.check_model(graph)
    print(f"exported {out} ({out.stat().st_size / 1024:.1f} KiB), fixed input [1,3,{args.tile},{args.tile}]")


if __name__ == "__main__":
    main()
