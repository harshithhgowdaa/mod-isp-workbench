"""Evaluate aligned Bayer-to-sRGB pairs with the classical, learned, and hybrid ISP paths."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image

from train import pairs, read_pair
from model import load_model
import app


def center_crop(mosaic, target, size):
    if not size: return mosaic, target
    h, w = mosaic.shape
    size = min(size, h, w)
    size -= size % 2
    y = max(0, ((h - size) // 2) // 2 * 2)
    x = max(0, ((w - size) // 2) // 2 * 2)
    return mosaic[y:y+size, x:x+size], target[y:y+size, x:x+size]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', default='models/mod_isp.pt')
    p.add_argument('--gate', default='models/gate.json')
    p.add_argument('--patch-size', type=int, default=0, help='Evaluate center crops; 0 evaluates full images')
    p.add_argument('--limit', type=int, default=0, help='Optional pair limit for smoke checks')
    p.add_argument('--out', default='evaluation.json')
    args = p.parse_args()
    app.CHECKPOINT = Path(args.checkpoint)
    model, metadata = load_model(app.CHECKPOINT)
    gate_path = Path(args.gate)
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {'mode':'relative_rank','route_fraction':0.5}
    rows = []
    data_pairs = pairs(Path(args.data))
    if args.limit: data_pairs = data_pairs[:args.limit]
    for source, target in data_pairs:
        mosaic, reference = read_pair(source, target)
        mosaic, reference = center_crop(mosaic, reference, args.patch_size)
        ref = Image.fromarray(np.uint8(np.clip(reference, 0, 1) * 255), 'RGB')
        start = time.perf_counter(); classical = app.classical_bayer(mosaic, 'human', 1)
        classical_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); learned, _ = app.learned(mosaic, hybrid=False)
        learned_ms = (time.perf_counter() - start) * 1000
        hybrid_start = time.perf_counter()
        hybrid, routed = app.blend_bayer_paths(classical, learned, mosaic, gate_config=gate)
        hybrid_ms = (time.perf_counter() - hybrid_start) * 1000
        scored = {}
        for pipeline, output, elapsed, route in [
            ('classical', classical, classical_ms, None),
            ('learned', learned, learned_ms, None),
            ('hybrid', hybrid, hybrid_ms, routed),
        ]:
            # Avoid a duplicate full-frame metric pass when the calibrated
            # Hybrid happens to equal Learned (for example, a 100% route).
            if pipeline == 'hybrid' and np.array_equal(np.asarray(hybrid), np.asarray(learned)):
                measured = scored['learned']
            else:
                measured = app.metrics(output, ref)
            scored[pipeline] = measured
            rows.append({'image': source.name, 'pipeline': pipeline,
                         'latency_ms': round(elapsed, 2), 'routed_percent': route,
                         'reused_learned_inference': pipeline == 'hybrid',
                         **measured})
        print(f'evaluated {source.name}', flush=True)
    report = {'dataset': str(Path(args.data).resolve()), 'checkpoint': str(app.CHECKPOINT.resolve()),
              'model': metadata, 'parameters': sum(x.numel() for x in model.parameters()),
              'checkpoint_bytes': app.CHECKPOINT.stat().st_size, 'environment': platform.platform(),
              'patch_size': args.patch_size or 'full image', 'gate': str(gate_path.resolve()) if gate_path.exists() else None,
              'latency_note': 'Hybrid latency is incremental image composition from the measured Classical/Learned outputs; standalone routed inference is measured by verify_gate.py.',
              'peak_rss_raw': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'peak_rss_note': 'Process high-water mark; bytes on macOS, KiB on Linux; includes Python and metrics libraries.',
              'results': rows}
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Wrote {len(rows)} measurements for {len(data_pairs)} Bayer/RGB pairs to {args.out}')


if __name__ == '__main__': main()
