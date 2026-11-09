"""Measure sRGB denoiser recovery under deterministic held-out corruptions.

The SIDD-CC test benchmark remains untouched by training and calibration. This
script adds synthetic corruptions to its noisy blocks and compares the same
fine-tuned sRGB model with each corrupted input against the published clean
reference. It does not represent Bayer RAW performance or real camera tests.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from skimage.metrics import structural_similarity

from train_sidd_srgb import import_naf


def load_blocks(root: Path):
    base = root / "SRGB/test"
    gt = loadmat(base / "BenchmarkGtBlocksSrgb.mat")["data"]
    noisy = loadmat(base / "BenchmarkNoisyBlocksSrgb.mat")["data"]
    if gt.shape != noisy.shape or gt.ndim != 5:
        raise ValueError(f"Unexpected SIDD-CC block shapes: noisy={noisy.shape}, gt={gt.shape}")
    if gt.shape[-1] != 3 and gt.shape[2] == 3:
        gt, noisy = gt.transpose(0, 1, 3, 4, 2), noisy.transpose(0, 1, 3, 4, 2)
    noisy = noisy.reshape(-1, *noisy.shape[-3:])
    gt = gt.reshape(-1, *gt.shape[-3:])
    return noisy.astype(np.float32) / 255.0, gt.astype(np.float32) / 255.0


def corruptions(noisy: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(noisy.shape, dtype=np.float32)
    # Each transform is generated from the same held-out noisy blocks; none is
    # used for model fitting or checkpoint selection.
    jpeg = np.empty_like(noisy)
    for i, block in enumerate(noisy):
        buf = io.BytesIO()
        Image.fromarray(np.uint8(np.clip(block, 0, 1) * 255)).save(buf, format="JPEG", quality=30)
        buf.seek(0)
        jpeg[i] = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return {
        "published_noisy": noisy,
        "additional_gaussian_sigma_0.02": np.clip(noisy + 0.02 * normal, 0, 1),
        "additional_gaussian_sigma_0.05": np.clip(noisy + 0.05 * normal, 0, 1),
        "underexposure_0.5x": np.clip(noisy * 0.5, 0, 1),
        "jpeg_quality_30": jpeg,
    }


def aggregate(src, pred, gt):
    psnr_src, psnr_pred, ssim_src, ssim_pred = [], [], [], []
    for x, y, target in zip(src, pred, gt):
        psnr_src.append(-10 * np.log10(max(float(np.mean((x - target) ** 2)), 1e-12)))
        psnr_pred.append(-10 * np.log10(max(float(np.mean((y - target) ** 2)), 1e-12)))
        ssim_src.append(float(structural_similarity(target, x, channel_axis=-1, data_range=1.0)))
        ssim_pred.append(float(structural_similarity(target, y, channel_axis=-1, data_range=1.0)))
    return {
        "mean_input_psnr_db": float(np.mean(psnr_src)),
        "mean_model_psnr_db": float(np.mean(psnr_pred)),
        "mean_psnr_recovery_db": float(np.mean(np.asarray(psnr_pred) - np.asarray(psnr_src))),
        "median_psnr_recovery_db": float(np.median(np.asarray(psnr_pred) - np.asarray(psnr_src))),
        "mean_input_ssim": float(np.mean(ssim_src)),
        "mean_model_ssim": float(np.mean(ssim_pred)),
        "mean_ssim_recovery": float(np.mean(np.asarray(ssim_pred) - np.asarray(ssim_src))),
        "blocks": len(src),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/siddcc_srgb"))
    p.add_argument("--samsung-repo", type=Path, default=Path("../../work/modular_neural_isp"))
    p.add_argument("--checkpoint", type=Path, default=Path("models/sidd_srgb_nafnet_lite.pt"))
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("robustness_srgb.json"))
    a = p.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg = import_naf(a.samsung_repo)
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    noisy, gt = load_blocks(a.dataset)
    variants = corruptions(noisy, a.seed)
    rows = {}
    for name, images in variants.items():
        outputs = []
        for start in range(0, len(images), a.batch_size):
            batch = images[start:start + a.batch_size].transpose(0, 3, 1, 2)
            inp = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            with torch.inference_mode():
                pred = model(inp).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            outputs.extend(pred)
        rows[name] = aggregate(images, np.asarray(outputs), gt)
        print(name, json.dumps(rows[name]), flush=True)
    report = {
        "task": "SIDD-CC sRGB robustness sweep with synthetic held-out corruptions",
        "dataset": str(a.dataset.resolve()), "checkpoint": str(a.checkpoint.resolve()),
        "architecture_config": cfg, "device": str(device), "environment": platform.platform(),
        "seed": a.seed, "batch_size": a.batch_size,
        "note": "All corruptions are deterministic synthetic transforms of the untouched published noisy test blocks. Results describe sRGB denoising, not RAW Bayer ISP or real-world degradation. The test set was not used for training, calibration, or checkpoint selection.",
        "results": rows,
    }
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print("saved", a.out.resolve())


if __name__ == "__main__":
    main()
