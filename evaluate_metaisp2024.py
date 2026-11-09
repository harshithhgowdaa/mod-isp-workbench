"""Compare the local MetaISP-2024 reproduction and MOD-ISP on paired Bayer data.

The default evaluation is a deterministic, even-aligned center crop per capture
so the teacher and student are compared on exactly the same pixels. The report
is a local SIDD-CC measurement, not the IJCAI paper benchmark.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from app import metrics
from metaisp2024 import MetaISP2024
from model import load_model
from train import pairs, read_pair
from isp import pack


def crop(mosaic, target, size):
    if not size:
        return mosaic, target
    size = min(size, *mosaic.shape)
    size -= size % 2
    y = max(0, ((mosaic.shape[0] - size) // 2) // 2 * 2)
    x = max(0, ((mosaic.shape[1] - size) // 2) // 2 * 2)
    return mosaic[y:y + size, x:x + size], target[y:y + size, x:x + size]


def as_tensor(mosaic, device):
    packed = pack(mosaic).transpose(2, 0, 1).copy()
    return torch.from_numpy(packed).unsqueeze(0).to(device)


def as_image(output):
    rgb = output[0].detach().float().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255), 'RGB')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/siddcc_bayer/validation')
    p.add_argument('--teacher', default='models/metaisp2024_siddcc_reproduction.pt')
    p.add_argument('--student', default='models/mod_isp_siddcc.pt')
    p.add_argument('--patch-size', type=int, default=256)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', default='metaisp2024_student_comparison.json')
    args = p.parse_args()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    payload = torch.load(args.teacher, map_location='cpu', weights_only=True)
    teacher = MetaISP2024().eval()
    teacher.load_state_dict(payload['state_dict'], strict=True)
    teacher = teacher.to(device)
    student, student_meta = load_model(args.student)
    student = student.to(device).eval()
    data = pairs(Path(args.data))
    if args.limit:
        data = data[:args.limit]
    rows = []
    started = time.perf_counter()
    for src, target_path in data:
        mosaic, target = read_pair(src, target_path)
        mosaic, target = crop(mosaic, target, args.patch_size)
        inp = as_tensor(mosaic, device)
        with torch.inference_mode():
            teacher_image = as_image(teacher(inp).clamp(0, 1))
            student_image = as_image(student(inp).clamp(0, 1))
        ref = Image.fromarray(np.uint8(np.clip(target, 0, 1) * 255), 'RGB')
        row = {'image': src.name, 'teacher': metrics(teacher_image, ref),
               'student': metrics(student_image, ref)}
        row['teacher_minus_student'] = {
            key: (float(row['teacher'][key]) - float(row['student'][key])
                  if isinstance(row['teacher'].get(key), (int, float)) and
                     isinstance(row['student'].get(key), (int, float)) else None)
            for key in ('psnr', 'ssim', 'delta_e', 'lpips')}
        rows.append(row)
        print(f"evaluated {src.name}: teacher PSNR {row['teacher']['psnr']}, "
              f"student PSNR {row['student']['psnr']}", flush=True)
    summary = {}
    for pipeline in ('teacher', 'student'):
        summary[pipeline] = {
            metric: float(np.mean([row[pipeline][metric] for row in rows
                                   if isinstance(row[pipeline].get(metric), (int, float))]))
            if any(isinstance(row[pipeline].get(metric), (int, float)) for row in rows) else None
            for metric in ('psnr', 'ssim', 'delta_e', 'lpips')}
    report = {
        'architecture': payload['architecture'],
        'teacher_checkpoint': str(Path(args.teacher).resolve()),
        'teacher_origin': payload.get('teacher_origin'),
        'student_checkpoint': str(Path(args.student).resolve()),
        'student_metadata': student_meta,
        'dataset': str(Path(args.data).resolve()),
        'captures': len(rows), 'crop_size': args.patch_size or 'full image',
        'metrics': summary,
        'mean_teacher_minus_student': {
            metric: (summary['teacher'][metric] - summary['student'][metric]
                     if summary['teacher'][metric] is not None and summary['student'][metric] is not None else None)
            for metric in ('psnr', 'ssim', 'delta_e', 'lpips')},
        'environment': platform.platform(), 'device': str(device),
        'elapsed_seconds': time.perf_counter() - started,
        'scope_note': 'Scene-disjoint SIDD-CC validation captures; deterministic center crops; not the IJCAI benchmark.',
        'per_capture': rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'captures': len(rows), 'metrics': summary,
                      'mean_teacher_minus_student': report['mean_teacher_minus_student']}, indent=2))


if __name__ == '__main__':
    main()
