"""Compare a metadata-based camera matrix against the existing Bayer baseline.

This offline experiment mirrors the SIDD authors' MATLAB reference ordering:
white-balance, demosaic RGB, normalize ColorMatrix2 rows, invert to XYZ, clip,
convert XYZ to sRGB. No live application code is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

import app
from isp import demosaic
from train import pairs, read_pair


XYZ_D65_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=np.float64)


def camera_to_srgb(camera_rgb: np.ndarray, color_matrix2: np.ndarray) -> np.ndarray:
    """Use the SIDD reference pipeline's ColorMatrix2 -> XYZ -> sRGB mapping."""
    # DNG stores a row-major vector for a matrix whose MATLAB reference form
    # explicitly transposes after reshape (run_pipeline.m).
    xyz_to_camera = np.asarray(color_matrix2, dtype=np.float64).reshape(3, 3).T
    if not np.isfinite(xyz_to_camera).all() or np.linalg.matrix_rank(xyz_to_camera) != 3:
        raise ValueError('ColorMatrix2 is not a finite full-rank 3×3 matrix.')
    row_sums = xyz_to_camera.sum(axis=1, keepdims=True)
    if np.any(np.abs(row_sums) < 1e-8):
        raise ValueError('ColorMatrix2 has a near-zero row sum.')
    xyz_to_camera = xyz_to_camera / row_sums
    camera_to_xyz = np.linalg.inv(xyz_to_camera)
    xyz = np.asarray(camera_rgb, dtype=np.float64) @ camera_to_xyz.T
    xyz = np.clip(xyz, 0.0, 1.0)
    xyz_to_srgb = XYZ_D65_TO_SRGB.copy()
    xyz_to_srgb /= xyz_to_srgb.sum(axis=1, keepdims=True)
    return np.clip(xyz @ xyz_to_srgb.T, 0.0, 1.0)


def load_native_pair(source: Path, target: Path):
    raw = np.asarray(loadmat(source, variable_names=['data'])['data'], dtype=np.float32)
    meta_path = source.parent.parent / 'metadata' / f'{source.stem}.MAT'
    if not meta_path.is_file():
        meta_path = source.with_name(source.name.replace('_NOISY_RAW_', '_METADATA_RAW_'))
    metadata = loadmat(meta_path)
    cfa = str(np.asarray(metadata['cfa_str']).squeeze()).lower()
    shifts = {'rggb': (0, 0), 'grbg': (0, 1), 'gbrg': (1, 0), 'bggr': (1, 1)}
    if cfa not in shifts:
        raise ValueError(f'Unsupported SIDD CFA {cfa!r} in {meta_path}')
    dy, dx = shifts[cfa]
    raw = raw[dy:, dx:]
    gt = np.asarray(Image.open(target).convert('RGB'), dtype=np.float32) / 255.0
    if dy or dx:
        gt = gt[dy:, dx:]
    h = min(raw.shape[0], gt.shape[0]) & ~1
    w = min(raw.shape[1], gt.shape[1]) & ~1
    raw, gt = raw[:h, :w], gt[:h, :w]
    matrix = np.asarray(metadata.get('ColorMatrix2', []), dtype=np.float64).reshape(-1)
    neutral = np.asarray(metadata.get('AsShotNeutral', []), dtype=np.float64).reshape(-1)
    if matrix.size != 9 or neutral.size != 3:
        raise ValueError(f'Missing ColorMatrix2 or AsShotNeutral in {meta_path}')
    return raw, gt, neutral, matrix


def center_crop(mosaic, target, size):
    h, w = mosaic.shape
    size = min(size, h, w) & ~1
    y = max(0, ((h - size) // 2) // 2 * 2)
    x = max(0, ((w - size) // 2) // 2 * 2)
    return mosaic[y:y + size, x:x + size], target[y:y + size, x:x + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/siddcc_bayer/validation'))
    parser.add_argument('--checkpoint', type=Path, default=Path('models/mod_isp.pt'))
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--limit', type=int, default=0,
                        help='Optionally evaluate only the first N pairs as a smoke check')
    parser.add_argument('--out', type=Path, default=Path('camera_ccm_siddcc.json'))
    args = parser.parse_args()
    app.CHECKPOINT = args.checkpoint
    records = []
    input_pairs = pairs(args.data)
    if args.limit:
        input_pairs = input_pairs[:args.limit]
    for source, target in input_pairs:
        native, gt, neutral, matrix = load_native_pair(source, target)
        wb_mosaic, _ = read_pair(source, target)
        native, gt = center_crop(native, gt, args.patch_size)
        wb_mosaic, _ = center_crop(wb_mosaic, gt, args.patch_size)

        classical = app.classical_bayer(wb_mosaic, 'human', 1.0)
        linear_camera_rgb = demosaic(wb_mosaic)
        srgb = camera_to_srgb(linear_camera_rgb, matrix)
        # Match the SIDD authors' reference sequence after XYZ->sRGB: gamma,
        # then their published tone curve. This remains an offline reference.
        tone_curve = loadmat(Path(__file__).parent / 'work' / 'sidd_simple_camera_pipeline' /
                             'python' / 'tone_curve.mat')['tc'].squeeze()
        gamma = np.power(np.clip(srgb, 0.0, 1.0), 1.0 / 2.2)
        tone_idx = np.clip(np.round(gamma * (len(tone_curve) - 1)).astype(int),
                           0, len(tone_curve) - 1)
        rendered = np.asarray(tone_curve[tone_idx], dtype=np.float32)
        ccm_image = Image.fromarray(np.uint8(np.clip(rendered, 0, 1) * 255), 'RGB')
        reference = Image.fromarray(np.uint8(np.clip(gt, 0, 1) * 255), 'RGB')
        classical_metrics = app.metrics(classical, reference)
        ccm_metrics = app.metrics(ccm_image, reference)
        records.append({'image': source.name, 'cfa': str(np.asarray(
            loadmat(source.parent.parent / 'metadata' / f'{source.stem}.MAT')['cfa_str']).squeeze()),
            'classical': classical_metrics, 'metadata_ccm': ccm_metrics})
        print(f"{source.name}: base={classical_metrics['psnr']}dB "
              f"ccm={ccm_metrics['psnr']}dB ΔE={ccm_metrics['delta_e']}", flush=True)

    metric_names = ('psnr', 'ssim', 'delta_e', 'lpips')
    summary = {}
    for pipeline in ('classical', 'metadata_ccm'):
        summary[pipeline] = {
            metric: float(np.mean([r[pipeline][metric] for r in records
                                   if r[pipeline].get(metric) is not None]))
            for metric in metric_names
        }
    report = {
        'status': 'offline metadata-based CCM evaluation; not deployed',
        'dataset': str(args.data.resolve()), 'checkpoint': str(args.checkpoint.resolve()),
        'patch_size': args.patch_size, 'images': len(records), 'summary': summary,
        'difference_metadata_ccm_minus_classical': {
            key: summary['metadata_ccm'][key] - summary['classical'][key]
            for key in metric_names
        },
        'protocol': 'SIDD RAW loader white-balances the CFA-aligned mosaic; bilinear demosaic; '
                    'normalize ColorMatrix2 rows, invert camera-to-XYZ, clip XYZ, then use the '
                    'SIDD reference pipeline normalized XYZ-to-sRGB matrix. Compare to existing app classical output.',
        'limitations': ['Uses the SIDD reference simplification of ColorMatrix2 only; no ForwardMatrix interpolation or DNG chromatic adaptation.',
                        'One scene-disjoint validation scene; not an independent test.',
                        'The transform is evaluated as a classical RGB rendering stage, not as a post-transform on the learned model.',
                        'No live application path is changed by this experiment.'],
        'per_image': records,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({'summary': summary,
                      'difference_metadata_ccm_minus_classical': report['difference_metadata_ccm_minus_classical'],
                      'out': str(args.out.resolve())}, indent=2))


if __name__ == '__main__':
    main()
