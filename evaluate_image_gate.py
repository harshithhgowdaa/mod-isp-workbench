"""Fit an image-level Bayer compute gate on one scene and audit it on another.

This is an experiment, not the production tile gate. It predicts the measured
per-image PSNR benefit of Learned over Classical from RAW-only image features,
chooses its threshold on the calibration scene, then reports the untouched
validation scene without refitting.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from train import pairs, read_pair

FEATURE_NAMES = [
    'mosaic_median', 'mosaic_std', 'highpass_median_abs',
    'local_highpass_p10', 'local_highpass_median', 'mosaic_p99',
    'fraction_above_0_98', 'fraction_below_0', 'mosaic_p90_minus_p10',
]
METRICS = ('psnr', 'ssim', 'delta_e', 'lpips')


def features(mosaic, patch_size=256):
    h, w = mosaic.shape
    size = min(patch_size, h, w)
    size -= size % 2
    y = max(0, ((h - size) // 2) // 2 * 2)
    x = max(0, ((w - size) // 2) // 2 * 2)
    a = np.asarray(mosaic[y:y + size, x:x + size], dtype=np.float32)
    highpass = np.abs(a - gaussian_filter(a, sigma=1.0))
    sampled = highpass[::4, ::4]
    bh = sampled.shape[0] // 16 * 16
    bw = sampled.shape[1] // 16 * 16
    blocks = sampled[:bh, :bw].reshape(bh // 16, 16, bw // 16, 16)
    local_mad_proxy = np.median(blocks.transpose(0, 2, 1, 3).reshape(bh // 16, bw // 16, -1), axis=-1)
    return np.asarray([
        np.median(a), np.std(a), np.median(highpass), np.percentile(local_mad_proxy, 10),
        np.median(local_mad_proxy), np.percentile(a, 99), np.mean(a > .98), np.mean(a < 0),
        np.percentile(a, 90) - np.percentile(a, 10),
    ], dtype=np.float64)


def load_scores(path):
    result = {}
    for row in json.loads(Path(path).read_text())['results']:
        result.setdefault(row['image'], {})[row['pipeline']] = row
    return result


def collect(data_dir, scores, patch_size):
    rows = []
    for src, target in pairs(Path(data_dir)):
        if src.name not in scores or not {'classical', 'learned'} <= scores[src.name].keys():
            raise ValueError(f'Missing Classical/Learned evaluation rows for {src.name}')
        mosaic, _ = read_pair(src, target)
        rows.append({'image': src.name, 'features': features(mosaic, patch_size),
                     'scores': scores[src.name]})
    if not rows:
        raise ValueError(f'No paired Bayer samples in {data_dir}')
    return rows


def means(rows, selected):
    result = {}
    for metric in METRICS:
        vals = []
        for row, use_learned in zip(rows, selected):
            value = row['scores']['learned' if use_learned else 'classical'].get(metric)
            if value is not None:
                vals.append(float(value))
        result[metric] = float(np.mean(vals)) if vals else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration-data', required=True)
    parser.add_argument('--validation-data', required=True)
    parser.add_argument('--calibration-evaluation', required=True)
    parser.add_argument('--validation-evaluation', required=True)
    parser.add_argument('--out', default='image_gate_evaluation.json')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--max-psnr-drop', type=float, default=.5)
    parser.add_argument('--ridge', type=float, default=10.0)
    args = parser.parse_args()
    if args.max_psnr_drop < 0 or args.ridge < 0:
        parser.error('--max-psnr-drop and --ridge must be non-negative')

    started = time.perf_counter()
    calibration = collect(args.calibration_data, load_scores(args.calibration_evaluation), args.patch_size)
    validation = collect(args.validation_data, load_scores(args.validation_evaluation), args.patch_size)
    x_train = np.stack([r['features'] for r in calibration])
    x_test = np.stack([r['features'] for r in validation])
    y_train = np.array([r['scores']['learned']['psnr'] - r['scores']['classical']['psnr'] for r in calibration])
    y_test = np.array([r['scores']['learned']['psnr'] - r['scores']['classical']['psnr'] for r in validation])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-10] = 1
    z_train = np.clip((x_train - mean) / scale, -5, 5)
    z_test = np.clip((x_test - mean) / scale, -5, 5)
    design_train = np.column_stack([np.ones(len(z_train)), z_train])
    design_test = np.column_stack([np.ones(len(z_test)), z_test])
    regularizer = np.eye(design_train.shape[1]) * args.ridge
    regularizer[0, 0] = 0
    coefficients = np.linalg.solve(design_train.T @ design_train + regularizer,
                                   design_train.T @ y_train)
    prediction_train = design_train @ coefficients
    prediction_test = design_test @ coefficients

    baseline = means(calibration, np.ones(len(calibration), dtype=bool))['psnr']
    candidates = []
    for threshold in sorted(set(prediction_train)):
        selected = prediction_train >= threshold
        score = means(calibration, selected)['psnr']
        drop = baseline - score
        if drop <= args.max_psnr_drop:
            candidates.append({'threshold': float(threshold), 'routed_percent': float(100 * selected.mean()),
                               'mean_psnr': score, 'quality_drop_db': drop})
    if not candidates:
        raise RuntimeError('No threshold satisfies the configured quality budget.')
    selected_policy = min(candidates, key=lambda row: (row['routed_percent'], -row['mean_psnr']))
    threshold = selected_policy['threshold']
    selected_validation = prediction_test >= threshold
    full_validation = means(validation, np.ones(len(validation), dtype=bool))
    policy_validation = means(validation, selected_validation)
    report = {
        'status': 'experimental_only_not_deployed',
        'policy': 'ridge regression predicts per-image Learned-minus-Classical PSNR; route Learned when predicted benefit meets calibration threshold, otherwise route Classical',
        'feature_names': FEATURE_NAMES,
        'fit': {'calibration_pairs': len(calibration), 'calibration_scene_split': str(args.calibration_data),
                'ridge_l2': args.ridge, 'coefficients': coefficients.tolist(),
                'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
                'threshold_db': threshold, 'max_calibration_psnr_drop_db': args.max_psnr_drop,
                'calibration_policy': selected_policy},
        'validation': {'pairs': len(validation), 'scene_split': str(args.validation_data),
                       'neural_image_percent': float(100 * selected_validation.mean()),
                       'theoretical_neural_invocations_avoided_percent': float(100 * (~selected_validation).mean()),
                       'all_learned_metrics_mean': full_validation,
                       'routed_metrics_mean': policy_validation,
                       'metric_change_vs_all_learned': {k: (policy_validation[k] - full_validation[k]
                           if policy_validation[k] is not None and full_validation[k] is not None else None) for k in METRICS},
                       'predicted_benefit_correlation': float(np.corrcoef(prediction_test, y_test)[0, 1]),
                       'predicted_benefit_mae_db': float(np.mean(np.abs(prediction_test - y_test))),
                       'per_image': [{'image': r['image'], 'predicted_benefit_db': float(p),
                                      'actual_benefit_db': float(y), 'route_learned': bool(s)}
                                     for r, p, y, s in zip(validation, prediction_test, y_test, selected_validation)]},
        'runtime_seconds': time.perf_counter() - started,
        'limitations': ['One calibration scene and one validation scene only.',
                        'Features are computed on 256-pixel center crops, not full-resolution images.',
                        'The feature policy met the declared calibration budget but failed that 0.5 dB budget on validation.',
                        'Avoided neural image invocation percentage is theoretical; end-to-end gate latency and memory were not benchmarked.',
                        'This policy is not connected to the application.'],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'status': report['status'], 'fit': report['fit'],
                      'validation': {k: v for k, v in report['validation'].items() if k != 'per_image'},
                      'runtime_seconds': report['runtime_seconds'],
                      'limitations': report['limitations']}, indent=2))


if __name__ == '__main__':
    main()
