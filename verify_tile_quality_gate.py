"""Evaluate actual selective Bayer inference against full learned inference."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

import app
from train import pairs, read_pair


def center_crop(mosaic, target, size):
    if not size:
        return mosaic, target
    h, w = mosaic.shape
    size = min(size, h, w) & ~1
    y = max(0, ((h - size) // 2) // 2 * 2)
    x = max(0, ((w - size) // 2) // 2 * 2)
    return mosaic[y:y + size, x:x + size], target[y:y + size, x:x + size]


def target_image(target):
    return Image.fromarray(np.uint8(np.clip(target, 0, 1) * 255), 'RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/siddcc_bayer/validation')
    parser.add_argument('--gate', default='models/gate_quality_tile_trainingfit.json')
    parser.add_argument('--checkpoint', default='models/mod_isp.pt')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--split-description', default='Held-out Bayer evaluation split; see dataset manifest for scene grouping.')
    parser.add_argument('--out', default='tile_gate_runtime_validation.json')
    args = parser.parse_args()
    gate = json.loads(Path(args.gate).read_text())
    if gate.get('deployment_eligible') is not False:
        raise ValueError('This verifier expects an explicitly experimental gate artifact.')
    app.CHECKPOINT = Path(args.checkpoint)
    rows = []
    for source, target_path in pairs(Path(args.data)):
        mosaic, target = center_crop(*read_pair(source, target_path), args.patch_size)
        reference = target_image(target)
        classical = app.classical_bayer(mosaic, 'human', 1)

        # Warm model/runtime paths before measuring; the first accelerator call
        # is reported separately nowhere and cannot skew one path alone.
        app.learned(mosaic, hybrid=False)
        app.learned(mosaic, hybrid=True, gate_config=gate, base_output=classical)
        learned_times, hybrid_times = [], []
        learned_output = hybrid_output = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            learned_raw, _ = app.learned(mosaic, hybrid=False)
            learned_output = app.finish_rgb(learned_raw, 'human', 1, apply_exposure=False)
            learned_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            hybrid_output, routed = app.learned(mosaic, hybrid=True, gate_config=gate,
                                                base_output=classical)
            hybrid_times.append((time.perf_counter() - start) * 1000)

        learned_metrics = app.metrics(learned_output, reference)
        hybrid_metrics = app.metrics(hybrid_output, reference)
        rows.append({
            'image': source.name,
            'routed_percent': routed,
            'learned_latency_median_ms': statistics.median(learned_times),
            'hybrid_latency_median_ms': statistics.median(hybrid_times),
            'learned': learned_metrics,
            'hybrid': hybrid_metrics,
            'delta': {name: hybrid_metrics[name] - learned_metrics[name]
                      for name in ('psnr', 'ssim', 'delta_e', 'lpips')
                      if isinstance(hybrid_metrics.get(name), (int, float))
                      and isinstance(learned_metrics.get(name), (int, float))},
        })
        print(f'checked {source.name}: {routed:.1f}% routed', flush=True)

    metric_names = ('psnr', 'ssim', 'delta_e', 'lpips')
    means = {path: {metric: float(np.mean([r[path][metric] for r in rows
                                           if isinstance(r[path].get(metric), (int, float))]))
                    for metric in metric_names}
             for path in ('learned', 'hybrid')}
    deltas = {metric: means['hybrid'][metric] - means['learned'][metric]
              for metric in metric_names}
    quality_budget = float(gate.get('max_calibration_psnr_drop_db', 0.5))
    quality_passed = deltas['psnr'] >= -quality_budget
    learned_times = [r['learned_latency_median_ms'] for r in rows]
    hybrid_times = [r['hybrid_latency_median_ms'] for r in rows]
    report = {
        'status': ('experimental_quality_budget_passed' if quality_passed
                   else 'experimental_quality_budget_failed'),
        'dataset': str(Path(args.data).resolve()),
        'split': args.split_description,
        'capture_count': len(rows),
        'patch_size': args.patch_size,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'gate': gate,
        'all_learned_metrics': means['learned'],
        'hybrid_metrics': means['hybrid'],
        'hybrid_minus_learned': deltas,
        'mean_route_percent': float(np.mean([r['routed_percent'] for r in rows])),
        'neural_tile_invocation_reduction_percent': 100 - float(np.mean([r['routed_percent'] for r in rows])),
        'mean_warm_latency_ms': {'learned': float(np.mean(learned_times)),
                                 'hybrid': float(np.mean(hybrid_times))},
        'median_warm_latency_ms': {'learned': float(np.median(learned_times)),
                                   'hybrid': float(np.median(hybrid_times))},
        'environment': platform.platform(),
        'timing_scope': 'Warm local inference and display finishing; classical baseline is precomputed for both paths.',
        'limitations': [
            'Scores and times use even-aligned center crops, not full-resolution RAW uploads.',
            'Tile-call counts do not measure MACs or energy; these host timings are not mobile NPU results.',
            'The gate is experimental and must not be described as an efficiency win unless both quality and measured runtime/computation improve.',
            (f'The policy exceeded its {quality_budget:.3g} dB PSNR loss budget on validation.'
             if not quality_passed else
             f'The policy stayed within its {quality_budget:.3g} dB PSNR loss budget on this evaluation split; this does not establish generalization.'),
        ],
        'per_image': rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in report.items() if key != 'per_image'}, indent=2))


if __name__ == '__main__':
    main()
