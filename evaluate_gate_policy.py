"""Evaluate a frozen RAW-tile benefit model at one fixed route fraction."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

import app
from calibrate_tile_quality_gate import image_rows, render_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', required=True, help='Frozen regression coefficients from calibration.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--calibration-data', required=True)
    parser.add_argument('--validation-data', required=True)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--route-fraction', type=float,
                        help='Fixed per-image rank fraction. Omit to calibrate a score threshold on calibration-data.')
    parser.add_argument('--max-psnr-drop', type=float, default=0.5)
    parser.add_argument('--feather-pixels', type=int, nargs='+', default=[0],
                        help='Candidate boundary widths for threshold calibration; fixed-fraction mode uses the first value.')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.route_fraction is not None and not 0 <= args.route_fraction <= 1:
        parser.error('Route fraction must be 0..1.')
    if args.max_psnr_drop < 0 or any(v < 0 for v in args.feather_pixels):
        parser.error('Quality budget and feather widths must be non-negative.')
    started = time.perf_counter()
    app.CHECKPOINT = Path(args.checkpoint)
    gate = json.loads(Path(args.gate).read_text())
    mean = np.asarray(gate['feature_mean'], dtype=np.float64)
    scale = np.asarray(gate['feature_scale'], dtype=np.float64)
    coef = np.asarray(gate['coefficients'], dtype=np.float64)
    split_rows = {split: image_rows(root, args.patch_size) for split, root in
                  [('calibration', args.calibration_data), ('validation', args.validation_data)]}
    def predictions(row):
        features = np.asarray([tile[0] for tile in row['features']], dtype=np.float64)
        design = np.column_stack([np.ones(len(features)), np.clip((features - mean) / scale, -5, 5)])
        return design @ coef

    selected = {'route_fraction': args.route_fraction, 'feather_pixels': args.feather_pixels[0]}
    if args.route_fraction is None:
        calibration = split_rows['calibration']
        baseline = float(np.mean([psnr(row['learned'], row['reference']) for row in calibration]))
        all_scores = np.concatenate([predictions(row) for row in calibration])
        thresholds = np.unique(np.quantile(all_scores, np.linspace(0, 1, 101)))
        thresholds = np.concatenate((thresholds, [np.nextafter(float(all_scores.max()), np.inf)]))
        candidates = []
        for threshold in thresholds:
            for feather in sorted(set(args.feather_pixels)):
                scores, routed = [], []
                for row in calibration:
                    output, count = render_gate(row, predictions(row), threshold=float(threshold),
                                                feather_pixels=feather)
                    scores.append(psnr(output, row['reference']))
                    routed.append(count / len(row['features']))
                mean_score = float(np.mean(scores))
                if mean_score >= baseline - args.max_psnr_drop:
                    candidates.append({'threshold_db': float(threshold), 'feather_pixels': feather,
                        'mean_psnr_db': mean_score, 'route_percent': 100 * float(np.mean(routed)),
                        'quality_drop_db': baseline - mean_score})
        if not candidates:
            raise RuntimeError('No score threshold meets the calibration PSNR budget.')
        chosen = min(candidates, key=lambda r: (r['route_percent'], r['feather_pixels'], -r['mean_psnr_db']))
        selected = {'route_fraction': None, **chosen, 'candidate_count': len(candidates),
                    'calibration_baseline_psnr_db': baseline, 'max_psnr_drop_db': args.max_psnr_drop}

    reports = {}
    for split, rows in split_rows.items():
        per_image = []
        for row in rows:
            scores = predictions(row)
            output, count = render_gate(row, scores,
                route_fraction=selected['route_fraction'],
                threshold=selected.get('threshold_db'),
                feather_pixels=selected['feather_pixels'])
            ref = Image.fromarray(np.uint8(np.clip(row['reference'], 0, 1) * 255), 'RGB')
            learned = Image.fromarray(np.uint8(np.clip(row['learned'], 0, 1) * 255), 'RGB')
            gated = Image.fromarray(np.uint8(np.clip(output, 0, 1) * 255), 'RGB')
            full_metrics = app.metrics(learned, ref)
            gate_metrics = app.metrics(gated, ref)
            per_image.append({
                'image': row['image'], 'tile_count': len(row['features']),
                'routed_percent': 100 * count / max(len(row['features']), 1),
                'learned': full_metrics, 'gated': gate_metrics,
                'psnr_delta_db': float(gate_metrics['psnr']) - float(full_metrics['psnr']),
            })
            print(f"{split}: {row['image']} route={100 * count / len(row['features']):.1f}% "
                  f"PSNR delta={per_image[-1]['psnr_delta_db']:.3f} dB", flush=True)
        mean_routed = float(np.mean([r['routed_percent'] for r in per_image]))
        summary = {}
        for path in ('learned', 'gated'):
            summary[path] = {metric: float(np.mean([r[path][metric] for r in per_image
                if isinstance(r[path].get(metric), (float, int))]))
                if any(isinstance(r[path].get(metric), (float, int)) for r in per_image) else None
                for metric in ('psnr', 'ssim', 'delta_e', 'lpips')}
        reports[split] = {'captures': len(per_image), 'mean_routed_percent': mean_routed,
            'neural_tile_call_reduction_percent': 100 - mean_routed,
            'learned': summary['learned'], 'gated': summary['gated'],
            'metric_delta': {m: (summary['gated'][m] - summary['learned'][m]
                                  if summary['gated'][m] is not None else None)
                             for m in ('psnr', 'ssim', 'delta_e', 'lpips')},
            'per_image': per_image}
    report = {'status': 'frozen_policy_diagnostic_not_deployed',
        'gate': str(Path(args.gate).resolve()), 'checkpoint': str(app.CHECKPOINT.resolve()),
        'selected_policy': selected,
        'patch_size': args.patch_size, 'splits': reports,
        'environment': platform.platform(), 'elapsed_seconds': time.perf_counter() - started,
        'limitations': ['The validation split has already been inspected in earlier gate experiments.',
            'Neural tile-call reduction is not a measured latency, MAC, or energy reduction.',
            'Calibration and validation use aligned center crops, not full-resolution RAW uploads.']}
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({split: {k: v for k, v in data.items() if k not in ('per_image',)}
                      for split, data in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
