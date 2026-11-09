"""Train the local paper-informed MetaISP-2024 reproduction on paired Bayer data.

This does not load the authors' unpublished/unavailable checkpoint. The saved
weights are a SIDD-CC-trained reproduction and can serve as a teacher only
after the accompanying validation and parity checks are reviewed.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from metaisp2024 import MetaISP2024
from isp import pack
from train import pairs, read_pair


def ssim_loss(pred, target):
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    var_x = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_x * mu_x
    var_y = F.avg_pool2d(target * target, 7, 1, 3) - mu_y * mu_y
    cov = F.avg_pool2d(pred * target, 7, 1, 3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
        (mu_x.square() + mu_y.square() + 0.01 ** 2) *
        (var_x + var_y + 0.03 ** 2)).clamp_min(1e-8)
    return 1 - score.mean()


class VGGPerceptualLoss(torch.nn.Module):
    """Frozen ImageNet VGG16 feature loss used in MetaISP's first stage."""
    def __init__(self):
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.capture_layers = {3, 8, 15, 22}

    def forward(self, prediction, target):
        x = (prediction - self.mean) / self.std
        y = (target - self.mean) / self.std
        total = x.new_zeros(())
        target_features = {}
        with torch.no_grad():
            for index, layer in enumerate(self.features):
                y = layer(y)
                if index in self.capture_layers:
                    target_features[index] = y
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.capture_layers:
                total = total + F.l1_loss(x, target_features[index])
        return total


def crop_pair(mosaic, target, patch, rng):
    h, w = mosaic.shape
    if min(h, w) < patch:
        raise ValueError(f'Pair is smaller than crop size {patch}: {mosaic.shape}')
    y = 2 * rng.randrange((h - patch) // 2 + 1)
    x = 2 * rng.randrange((w - patch) // 2 + 1)
    raw = mosaic[y:y + patch, x:x + patch]
    rgb = target[y:y + patch, x:x + patch]
    if rng.random() < 0.5:
        raw, rgb = flip_bayer_cells(raw, axis=1), rgb[:, ::-1]
    if rng.random() < 0.5:
        raw, rgb = flip_bayer_cells(raw, axis=0), rgb[::-1]
    return np.ascontiguousarray(raw), np.ascontiguousarray(rgb)


def flip_bayer_cells(mosaic, axis):
    """Flip the 2x2-cell grid while preserving each RGGB CFA phase plane."""
    h, w = mosaic.shape
    if h % 2 or w % 2 or axis not in (0, 1):
        raise ValueError('Bayer augmentation requires an even mosaic and axis 0 or 1.')
    cells = mosaic.reshape(h // 2, 2, w // 2, 2)
    cells = np.flip(cells, axis=0 if axis == 0 else 2)
    return np.ascontiguousarray(cells.reshape(h, w))


def tensor_pair(mosaic, target, device):
    x = torch.from_numpy(pack(mosaic).transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    y = torch.from_numpy(target.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    return x, y


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float('inf') if mse <= 0 else -10 * np.log10(mse)


def evaluate(model, data_root, patch, device):
    model.eval()
    scores, losses = [], []
    with torch.inference_mode():
        for src, target_path in pairs(Path(data_root)):
            mosaic, target = read_pair(src, target_path)
            h, w = mosaic.shape
            y = max(0, ((h - patch) // 4) * 2)
            x = max(0, ((w - patch) // 4) * 2)
            raw = mosaic[y:y + patch, x:x + patch]
            ref = target[y:y + patch, x:x + patch]
            inp, truth = tensor_pair(raw, ref, device)
            pred = model(inp).clamp(0, 1)
            scores.append(psnr(pred[0].permute(1, 2, 0).cpu().numpy(), ref))
            losses.append(float(F.l1_loss(pred, truth).cpu()))
    model.train()
    return {'pairs': len(scores), 'mean_psnr_db': float(np.mean(scores)),
            'mean_l1': float(np.mean(losses)), 'per_pair_psnr_db': scores}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/siddcc_bayer/train')
    p.add_argument('--validation-data', default='data/siddcc_bayer/validation')
    p.add_argument('--out', default='models/metaisp2024_siddcc.pt')
    p.add_argument('--epochs', type=int, default=80,
                   help='MetaISP stage-one epochs (paper schedule uses 80).')
    p.add_argument('--finetune-epochs', type=int, default=5,
                   help='Stage-two epochs (paper schedule uses 5).')
    p.add_argument('--patch', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--finetune-lr', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--resume', type=Path, help='Continue from a saved paper-reproduction checkpoint; optimizer state restarts.')
    p.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    p.add_argument('--final-full-training', action='store_true',
                   help='Train the fixed schedule on every pair in the supplied training root without validation or epoch selection.')
    args = p.parse_args()
    if args.patch < 64 or args.patch % 32 or args.epochs < 1 or args.finetune_epochs < 0:
        p.error('--patch must be >=64 and divisible by 32; epochs must be >=1; finetune epochs must be >=0')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('mps' if args.device == 'auto' and torch.backends.mps.is_available()
                          else ('cpu' if args.device == 'auto' else args.device))
    train_files = pairs(Path(args.data))
    val_files = [] if args.final_full_training else pairs(Path(args.validation_data))
    train_ids = {a.stem.split('_')[0] for a, _ in train_files}
    val_ids = {a.stem.split('_')[0] for a, _ in val_files}
    if train_ids & val_ids:
        raise ValueError('Train and validation instances overlap.')
    model = MetaISP2024()
    start_epoch = 0
    history = []
    best_psnr, best_epoch, best_state = (None, 0, None) if args.final_full_training else (-float('inf'), 0, None)
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if saved.get('architecture') != 'metaisp2024-paper-reproduction-v2':
            p.error('Resume checkpoint architecture is not the local MetaISP-2024 reproduction')
        model.load_state_dict(saved.get('last_state_dict', saved['state_dict']), strict=True)
        previous_report = args.resume.with_suffix('.json')
        if previous_report.exists():
            previous = json.loads(previous_report.read_text())
            start_epoch = int(previous.get('epochs_completed', saved.get('epochs_completed', 0)))
            history = list(previous.get('history', []))
            best_psnr = None if args.final_full_training else float(previous.get('best_validation_psnr_db', -float('inf')))
            best_epoch = int(previous.get('best_epoch', saved.get('best_epoch', start_epoch)))
        else:
            start_epoch = int(saved.get('epochs_completed', 0))
            best_epoch = int(saved.get('best_epoch', start_epoch))
        best_state = {k: v.detach().cpu().clone() for k, v in saved['state_dict'].items()}
    model = model.to(device)
    perceptual = VGGPerceptualLoss().to(device).eval()
    rng = random.Random(args.seed)
    start = time.perf_counter()
    print(f"device={device}; train_pairs={len(train_files)}; validation_pairs={len(val_files)}; "
          f"parameters={sum(v.numel() for v in model.parameters())}; "
          f"stages={args.epochs}+{args.finetune_epochs}", flush=True)
    global_epoch = start_epoch
    stages = [('stage1', args.epochs, args.lr),
              ('stage2', args.finetune_epochs, args.finetune_lr)]
    for stage_name, stage_epochs, learning_rate in stages:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for stage_epoch in range(stage_epochs):
            epoch_number = sum(n for _, n, _ in stages[:stages.index((stage_name, stage_epochs, learning_rate))]) + stage_epoch + 1
            if epoch_number <= start_epoch:
                continue
            if stage_name == 'stage1' and stage_epoch == 50:
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate * 0.5
            order = list(train_files); rng.shuffle(order)
            epoch_losses = []
            model.train()
            for source, target_path in order:
                mosaic, target = read_pair(source, target_path)
                crop, ref = crop_pair(mosaic, target, args.patch, rng)
                x, y = tensor_pair(crop, ref, device)
                # Keep the training prediction unclamped. Early model outputs
                # can be outside display range; clamping a fully negative
                # prediction would zero the gradient for every output pixel.
                pred = model(x)
                if stage_name == 'stage1':
                    charbonnier = torch.sqrt((pred - y).square() + 1e-6).mean()
                    loss = 0.25 * charbonnier + ssim_loss(pred, y) + perceptual(pred, y)
                else:
                    loss = 0.5 * F.mse_loss(pred, y) + ssim_loss(pred, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            global_epoch = epoch_number
            if args.final_full_training:
                validation = {'pairs': 0, 'mean_psnr_db': None, 'mean_l1': None,
                              'per_pair_psnr_db': []}
                row = {'epoch': global_epoch, 'stage': stage_name, 'stage_epoch': stage_epoch + 1,
                       'train_loss': float(np.mean(epoch_losses)),
                       'selection': 'fixed final schedule; no validation used'}
            else:
                validation = evaluate(model, args.validation_data, args.patch, device)
                row = {'epoch': global_epoch, 'stage': stage_name, 'stage_epoch': stage_epoch + 1,
                       'train_loss': float(np.mean(epoch_losses)), **validation}
            history.append(row)
            if args.final_full_training:
                best_epoch = global_epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif validation['mean_psnr_db'] > best_psnr:
                best_psnr, best_epoch = validation['mean_psnr_db'], global_epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(json.dumps({k: v for k, v in row.items() if k != 'per_pair_psnr_db'}), flush=True)
            if best_state is not None:
                last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(best_state)
                save_checkpoint(args.out, model, args, device, global_epoch, best_epoch, history,
                                best_psnr, time.perf_counter() - start, val_files, last_state)
                model.load_state_dict({k: v.detach().to(device) for k, v in last_state.items()})
    if args.final_full_training:
        print(f"saved {args.out}; fixed_final_epoch={best_epoch}; validation=not_used", flush=True)
    else:
        print(f"saved {args.out}; best_epoch={best_epoch}; val_psnr={best_psnr:.4f}", flush=True)


def save_checkpoint(out, model, args, device, epoch, best_epoch, history, best_psnr, elapsed, val_files,
                    last_state):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.to('cpu').eval()
    validation_dataset = None if args.final_full_training else str(Path(args.validation_data).resolve())
    payload = {'architecture': 'metaisp2024-paper-reproduction-v2', 'state_dict': cpu_model.state_dict(),
               'width': 32, 'basis_count': 10, 'epochs_completed': epoch, 'best_epoch': best_epoch,
               'best_validation_psnr_db': best_psnr,
               'last_state_dict': last_state,
               'dataset': str(Path(args.data).resolve()), 'train_pairs': len(pairs(Path(args.data))),
               'validation_dataset': validation_dataset,
               'validation_pairs': len(val_files), 'seed': args.seed, 'patch': args.patch,
               'checkpoint_selection': 'fixed final schedule; no validation used' if args.final_full_training else 'best validation PSNR',
               'augmentation': 'random crop; horizontal/vertical flips of 2x2 Bayer cells preserve RGGB phase',
               'optimizer': 'Adam', 'learning_rate': args.lr, 'finetune_learning_rate': args.finetune_lr,
               'stage1_epochs': args.epochs, 'stage2_epochs': args.finetune_epochs,
               'training_losses': {'stage1': '0.25 * Charbonnier + SSIM + VGG16 perceptual',
                                   'stage2': '0.5 * MSE + SSIM'},
               'training_device': str(device),
               'teacher_origin': 'local paper-informed reproduction; no official 2024 checkpoint available'}
    torch.save(payload, out)
    report = {'architecture': payload['architecture'], 'parameters': sum(v.numel() for v in cpu_model.parameters()),
              'dataset': payload['dataset'], 'train_pairs': payload['train_pairs'],
              'validation_dataset': payload['validation_dataset'], 'validation_pairs': payload['validation_pairs'],
              'epochs_completed': epoch, 'best_epoch': best_epoch, 'best_validation_psnr_db': best_psnr,
              'checkpoint_selection': payload['checkpoint_selection'],
              'seed': args.seed, 'patch': args.patch, 'training_device': str(device),
              'augmentation': payload['augmentation'],
              'optimizer': payload['optimizer'], 'learning_rate': args.lr,
              'finetune_learning_rate': args.finetune_lr,
              'stage1_epochs': args.epochs, 'stage2_epochs': args.finetune_epochs,
              'training_losses': payload['training_losses'],
              'history': history, 'elapsed_seconds': elapsed, 'checkpoint_bytes': out.stat().st_size,
              'teacher_origin': payload['teacher_origin'], 'scripted_teacher_export': None}
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    try:
        scripted = torch.jit.script(cpu_model)
        script_path = out.with_name(out.stem + '_teacher.ts')
        scripted.save(str(script_path))
        report['scripted_teacher_export'] = str(script_path)
        out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    except Exception as exc:
        report['scripted_export_error'] = f'{type(exc).__name__}: {exc}'
        out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    model.to(device).train()


if __name__ == '__main__':
    main()
