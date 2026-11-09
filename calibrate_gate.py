"""Choose a Bayer neural-tile budget on a scene-disjoint calibration split."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from train import pairs, read_pair
import app


def center_crop(mosaic, target, size):
    if not size: return mosaic, target
    h, w = mosaic.shape
    size = min(size, h, w); size -= size % 2
    y = max(0, ((h - size) // 2) // 2 * 2)
    x = max(0, ((w - size) // 2) // 2 * 2)
    return mosaic[y:y+size, x:x+size], target[y:y+size, x:x+size]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='models/gate.json')
    p.add_argument('--checkpoint', default='models/mod_isp.pt')
    p.add_argument('--max-psnr-drop', type=float, default=0.5)
    p.add_argument('--patch-size', type=int, default=256)
    a = p.parse_args()
    if a.max_psnr_drop < 0: p.error('--max-psnr-drop must be non-negative')
    app.CHECKPOINT = Path(a.checkpoint)
    samples = []
    for source, target in pairs(Path(a.data)):
        m, t = read_pair(source, target)
        m,t=center_crop(m,t,a.patch_size)
        classical=app.classical_bayer(m,'human',1)
        learned,_=app.learned(m,hybrid=False)
        samples.append((source.name,m,t,classical,learned))
    if not samples: raise ValueError('Calibration split is empty')
    results = []
    for fraction in [1.0, .95, .9, .85, .8, .75, .7, .6, .5, .4, .2, 0.0]:
        scores, rates = [], []
        for _, mosaic, target, classical, learned in samples:
            output, routed = app.blend_bayer_paths(classical,learned,mosaic,
                gate_config={'mode':'relative_rank','route_fraction':fraction})
            predicted = np.asarray(output, dtype=np.float32) / 255
            reference = np.asarray(Image.fromarray(np.uint8(np.clip(target, 0, 1) * 255), 'RGB'), dtype=np.float32) / 255
            mse = float(np.mean((predicted - reference) ** 2))
            scores.append(float('inf') if mse == 0 else float(-10 * np.log10(mse)))
            rates.append(routed)
        row = {'route_fraction': fraction, 'mean_psnr_db': float(np.mean(scores)),
               'mean_routed_percent': float(np.mean(rates))}
        results.append(row); print(row, flush=True)
    baseline = results[0]['mean_psnr_db']
    eligible = [r for r in results if r['mean_psnr_db'] >= baseline - a.max_psnr_drop]
    best = min(eligible, key=lambda r: (r['mean_routed_percent'], -r['mean_psnr_db']))
    report = {**best, 'mode': 'relative_rank', 'checkpoint': str(Path(a.checkpoint).resolve()),
              'dataset': str(Path(a.data).resolve()), 'patch_size': a.patch_size,
              'criterion': 'Minimize selected neural tiles while keeping calibration mean PSNR within budget of the fully Learned output. Hybrid output composes selected Learned regions with the Classical output. Tiles are ranked by Bayer variance / (mean intensity squared + 1e-6).',
              'max_psnr_drop_db': a.max_psnr_drop, 'baseline_psnr_db': baseline,
              'warning': 'Calibration is scene-disjoint from training but quality/speed are not guaranteed on unseen scenes; routed tile percentage is not measured MAC reduction.',
              'candidates': results}
    Path(a.out).write_text(json.dumps(report, indent=2))
    print('Saved', a.out)


if __name__ == '__main__': main()
