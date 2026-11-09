"""Fine-tune Samsung Modular ISP's Lite NAFNet denoiser on SIDD-CC sRGB pairs.

This is explicitly an sRGB-domain denoising adaptation, not the Samsung raw-RGB
ISP pipeline. It uses scene-separated validation and the public SIDD-CC benchmark
blocks for final evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat


def import_naf(repo: Path):
    sys.path.insert(0, str(repo.resolve()))
    from denoising.nafnet_arch import NAFNet
    cfg = json.loads((repo / "denoising/configs/lite.json").read_text())
    return NAFNet(**cfg), cfg


def image_pairs(root: Path):
    pairs = []
    for target in sorted(root.rglob("*_GT_SRGB_*.PNG")):
        noisy_name = target.name.replace("_GT_SRGB_", "_NOISY_SRGB_")
        noisy = target.with_name(noisy_name)
        if noisy.is_file():
            # SIDD folder names are instance_scene-camera-...; all instances
            # with the same second token depict the same physical scene.
            fields = target.parent.name.split("_")
            if len(fields) < 2 or not fields[1].isdigit():
                raise ValueError(f"Cannot read SIDD scene group from {target.parent.name!r}")
            scene_group = fields[1]
            pairs.append((scene_group, noisy, target))
    if not pairs:
        raise FileNotFoundError(f"No paired SIDD-CC SRGB PNGs found under {root}")
    return pairs


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def sample_crop(noisy: np.ndarray, clean: np.ndarray, size: int, rng: random.Random):
    h, w = noisy.shape[:2]
    y = rng.randrange(max(1, h - size + 1))
    x = rng.randrange(max(1, w - size + 1))
    a, b = noisy[y:y + size, x:x + size], clean[y:y + size, x:x + size]
    if a.shape[0] != size or a.shape[1] != size:
        y = max(0, h - size); x = max(0, w - size)
        a, b = noisy[y:y + size, x:x + size], clean[y:y + size, x:x + size]
    if rng.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    if rng.random() < 0.5:
        a, b = a[::-1], b[::-1]
    if rng.random() < 0.5:
        a, b = np.swapaxes(a, 0, 1), np.swapaxes(b, 0, 1)
    return np.ascontiguousarray(a.transpose(2, 0, 1)), np.ascontiguousarray(b.transpose(2, 0, 1))


def augment_exposure_noise(x: torch.Tensor, probability: float = 0.75) -> torch.Tensor:
    """Simulate lower exposure plus shot/read noise on sRGB training patches.

    Targets remain the original aligned clean images. This is deliberately a
    synthetic robustness augmentation, not a physical camera noise model.
    """
    if probability <= 0:
        return x
    batch = x.shape[0]
    apply = (torch.rand(batch, 1, 1, 1, device=x.device) < probability)
    exposure = torch.empty(batch, 1, 1, 1, device=x.device).uniform_(0.35, 1.0)
    shot_level = torch.empty(batch, 1, 1, 1, device=x.device).uniform_(0.00002, 0.00030)
    read_level = torch.empty(batch, 1, 1, 1, device=x.device).uniform_(0.0001, 0.0015)
    linear = torch.where(x <= 0.04045, x / 12.92,
                         ((x + 0.055) / 1.055).clamp_min(0).pow(2.4))
    signal = linear * exposure
    degraded = signal + torch.randn_like(signal) * torch.sqrt(signal.clamp_min(0) * shot_level)
    degraded = degraded + torch.randn_like(signal) * read_level
    degraded = degraded.clamp(0, 1)
    srgb = torch.where(degraded <= 0.0031308, 12.92 * degraded,
                       1.055 * degraded.clamp_min(0).pow(1 / 2.4) - 0.055)
    return torch.where(apply, srgb, x).clamp(0, 1)


def infer_patch(model, arr: np.ndarray, device):
    x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x).clamp(0, 1)
    return y[0].permute(1, 2, 0).cpu().numpy()


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def validate(model, pairs, device, patch=256, limit=None):
    rng = random.Random(1729)
    rows = []
    for _, noisy_path, gt_path in pairs[:limit]:
        noisy, gt = read_rgb(noisy_path), read_rgb(gt_path)
        h, w = noisy.shape[:2]
        y = min(max((h - patch) // 2, 0), max(h - patch, 0))
        x = min(max((w - patch) // 2, 0), max(w - patch, 0))
        noisy, gt = noisy[y:y + patch, x:x + patch], gt[y:y + patch, x:x + patch]
        pred = infer_patch(model, noisy, device)
        rows.append({"image": noisy_path.name, "noisy_psnr_db": psnr(noisy, gt), "model_psnr_db": psnr(pred, gt)})
    return {"n_patches": len(rows), "mean_noisy_psnr_db": float(np.mean([r["noisy_psnr_db"] for r in rows])),
            "mean_model_psnr_db": float(np.mean([r["model_psnr_db"] for r in rows])), "items": rows}


def benchmark(model, root: Path, device, batch_size=4):
    base = root / "SRGB/test"
    gt_path, noisy_path = base / "BenchmarkGtBlocksSrgb.mat", base / "BenchmarkNoisyBlocksSrgb.mat"
    if not gt_path.exists() or not noisy_path.exists():
        return {"status": "unavailable", "reason": "SIDD-CC SRGB benchmark block files were not downloaded"}
    gt_mat, noisy_mat = loadmat(gt_path), loadmat(noisy_path)
    gt = next(v for k, v in gt_mat.items() if not k.startswith("__") and isinstance(v, np.ndarray))
    noisy = next(v for k, v in noisy_mat.items() if not k.startswith("__") and isinstance(v, np.ndarray))
    # The published MATLAB array is [instance, block, row, col, channel].
    if gt.ndim != 5 or noisy.shape != gt.shape:
        return {"status": "error", "reason": f"Unexpected benchmark shapes: noisy={noisy.shape}, gt={gt.shape}"}
    if gt.shape[-1] != 3 and gt.shape[2] == 3:
        gt, noisy = gt.transpose(0, 1, 3, 4, 2), noisy.transpose(0, 1, 3, 4, 2)
    gt = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)
    noisy = noisy.astype(np.float32) / 255.0 if noisy.max() > 1.0 else noisy.astype(np.float32)
    rows = []
    model.eval()
    with torch.no_grad():
        flat_x, flat_y = noisy.reshape(-1, *noisy.shape[-3:]), gt.reshape(-1, *gt.shape[-3:])
        for start in range(0, len(flat_x), batch_size):
            batch = np.stack([flat_x[i].transpose(2, 0, 1) for i in range(start, min(start + batch_size, len(flat_x)))])
            inp = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            pred = model(inp).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            for j, out in enumerate(pred):
                i = start + j
                ref, src = flat_y[i], flat_x[i]
                rows.append({"instance_index": int(i // noisy.shape[1]), "block_index": int(i % noisy.shape[1]),
                             "noisy_psnr_db": psnr(src, ref), "model_psnr_db": psnr(out, ref),
                             "noisy_ssim": float(__import__('skimage.metrics', fromlist=['structural_similarity']).structural_similarity(ref, src, channel_axis=-1, data_range=1.0)),
                             "model_ssim": float(__import__('skimage.metrics', fromlist=['structural_similarity']).structural_similarity(ref, out, channel_axis=-1, data_range=1.0))})
    return {"status": "ok", "blocks": len(rows), "instances": int(noisy.shape[0]),
            "mean_block_psnr_db_noisy": float(np.mean([r["noisy_psnr_db"] for r in rows])),
            "mean_block_psnr_db_model": float(np.mean([r["model_psnr_db"] for r in rows])),
            "mean_block_ssim_noisy": float(np.mean([r["noisy_ssim"] for r in rows])),
            "mean_block_ssim_model": float(np.mean([r["model_ssim"] for r in rows])),
            "items": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/siddcc_srgb"))
    p.add_argument("--samsung-repo", type=Path, default=Path("../../work/modular_neural_isp"))
    p.add_argument("--init", type=Path, default=Path("../../work/modular_neural_isp/denoising/models/generic_lite.pth"))
    p.add_argument("--out", type=Path, default=Path("models/sidd_srgb_nafnet_lite.pt"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--resume", type=Path,
                   help="Continue fine-tuning a saved NAFNet checkpoint rather than the generic-lite initialization")
    p.add_argument("--augment-exposure-noise", action="store_true",
                   help="Apply synthetic lower-exposure and shot/read noise augmentation to training inputs")
    p.add_argument("--augmentation-probability", type=float, default=0.75)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device))
    pairs = image_pairs(args.dataset / "SRGB/train/Data")
    scenes = sorted({s for s, _, _ in pairs})
    rng = random.Random(args.seed); rng.shuffle(scenes)
    val_scenes = set(scenes[:max(1, round(0.1 * len(scenes)))])
    train_pairs = [x for x in pairs if x[0] not in val_scenes]
    val_pairs = [x for x in pairs if x[0] in val_scenes]
    if not 0 <= args.augmentation_probability <= 1:
        p.error("--augmentation-probability must be between 0 and 1")
    model, cfg = import_naf(args.samsung_repo)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("config") != cfg:
            p.error("Resume checkpoint architecture does not match the Samsung NAFNet configuration")
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        state = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.to(device)
    baseline = validate(model, val_pairs, device)
    baseline_benchmark = benchmark(model, args.dataset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train(); history=[]; started=time.time(); sample_rng=random.Random(args.seed)
    best_val = -float("inf"); best_epoch = 0; best_state = None
    train_instances = sorted({x[1].parent.name for x in train_pairs})
    validation_instances = sorted({x[1].parent.name for x in val_pairs})
    if {x[0] for x in train_pairs} & {x[0] for x in val_pairs}:
        raise RuntimeError("Physical SIDD scene groups overlap between train and validation")
    print(f"device={device}; train={len(train_pairs)} pairs/{len(train_instances)} instances/{len(scenes)-len(val_scenes)} physical scenes; validation={len(val_pairs)} pairs/{len(validation_instances)} instances/{len(val_scenes)} physical scenes", flush=True)
    for epoch in range(args.epochs):
        order=list(train_pairs); sample_rng.shuffle(order); losses=[]
        for start in range(0, len(order), args.batch_size):
            xs, ys=[], []
            for _, noisy_path, gt_path in order[start:start+args.batch_size]:
                x,y=sample_crop(read_rgb(noisy_path),read_rgb(gt_path),args.patch,sample_rng);xs.append(x);ys.append(y)
            x=torch.from_numpy(np.stack(xs)).to(device); y=torch.from_numpy(np.stack(ys)).to(device)
            if args.augment_exposure_noise:
                x = augment_exposure_noise(x, args.augmentation_probability)
            optimizer.zero_grad(set_to_none=True); pred=model(x); loss=F.l1_loss(pred,y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01);optimizer.step();losses.append(float(loss.detach().cpu()))
        val=validate(model,val_pairs,device)
        row={"epoch":epoch+1,"train_l1":float(np.mean(losses)),"val_psnr_db":val["mean_model_psnr_db"]};history.append(row)
        if row["val_psnr_db"] > best_val:
            best_val = row["val_psnr_db"]; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row),flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    report={"task":"SIDD-CC SRGB image denoising adaptation of Samsung Modular Neural ISP generic-lite NAFNet",
      "dataset":"MhdMohammadi/SIDD-CC SRGB train; 230 pairs from 115 scene instances across 10 physical scenes",
      "split_unit":"physical SIDD scene number (second token in scene-instance directory name)",
      "train_pairs":len(train_pairs),"train_scene_groups":sorted({x[0] for x in train_pairs}),"train_scene_instances":train_instances,
      "validation_pairs":len(val_pairs),"validation_scene_groups":sorted(val_scenes),"validation_scene_instances":validation_instances,
      "random_seed":args.seed,"device":str(device),"epochs":args.epochs,"best_epoch":best_epoch,"batch_size":args.batch_size,"patch":args.patch,"learning_rate":args.lr,
      "initial_checkpoint":str(args.resume or args.init),
      "training_augmentation":{"enabled":args.augment_exposure_noise,
          "name":"synthetic lower-exposure plus shot/read noise" if args.augment_exposure_noise else None,
          "probability":args.augmentation_probability if args.augment_exposure_noise else 0.0,
          "noise_model":"linear-domain signal-dependent Gaussian plus read noise; synthetic only" if args.augment_exposure_noise else None},
      "model_config":cfg,"parameters":sum(x.numel() for x in model.parameters()),
      "baseline_validation":baseline,"final_validation":validate(model,val_pairs,device),"training_history":history,
      "elapsed_seconds":time.time()-started,"note":"sRGB-domain adaptation only; not raw-RGB Bayer ISP processing. SIDD-CC benchmark test blocks are evaluated separately and were not used in training."}
    report["initial_checkpoint_benchmark_test"] = baseline_benchmark
    report["benchmark_test"]=benchmark(model,args.dataset,device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.cpu().state_dict(),"config":cfg,"task":"SIDD-CC sRGB denoising",
                "best_epoch":int(best_epoch),"epochs":int(args.epochs),"seed":int(args.seed),"device":str(device)},args.out)
    out_json=args.out.with_suffix(".json");out_json.write_text(json.dumps(report,indent=2))
    print(f"Saved checkpoint {args.out} and report {out_json}",flush=True)


if __name__ == "__main__":
    main()
