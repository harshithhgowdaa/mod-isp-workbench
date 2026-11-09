"""Fit a small post-restoration CCM on training scenes and score it on a held-out scene."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import app
from train import pairs, read_pair


def center_crop(mosaic, target, size):
    h, w = mosaic.shape
    size = min(size, h, w) & ~1
    y = max(0, ((h - size) // 2) // 2 * 2)
    x = max(0, ((w - size) // 2) // 2 * 2)
    return mosaic[y:y + size, x:x + size], target[y:y + size, x:x + size]


def srgb_to_linear(value):
    value = np.asarray(value, dtype=np.float32)
    return np.where(value <= .04045, value / 12.92,
                    ((value + .055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    value = np.clip(value, 0, 1)
    return np.where(value <= .0031308, 12.92 * value,
                    1.055 * np.power(value, 1 / 2.4) - .055)


def learned_output(mosaic):
    image, _ = app.learned(mosaic, hybrid=False)
    return app.finish_rgb(image, 'human', 1, apply_exposure=False, sharpen=True)


def fit_matrix(data_dir, checkpoint, patch, pixels_per_image, ridge, seed):
    app.CHECKPOINT = Path(checkpoint)
    rng = np.random.default_rng(seed)
    source_pixels, target_pixels = [], []
    count = 0
    for src, tgt in pairs(Path(data_dir)):
        mosaic, target = read_pair(src, tgt)
        mosaic, target = center_crop(mosaic, target, patch)
        output = np.asarray(learned_output(mosaic), dtype=np.float32) / 255
        source = srgb_to_linear(output).reshape(-1, 3)
        truth = srgb_to_linear(np.asarray(Image.fromarray(
            np.uint8(np.clip(target, 0, 1) * 255), 'RGB'), dtype=np.float32) / 255).reshape(-1, 3)
        indices = rng.choice(len(source), min(pixels_per_image, len(source)), replace=False)
        source_pixels.append(source[indices])
        target_pixels.append(truth[indices])
        count += 1
        print(f'fit sample {src.name}', flush=True)
    x = np.concatenate(source_pixels).astype(np.float64)
    y = np.concatenate(target_pixels).astype(np.float64)
    gram = x.T @ x
    identity = np.eye(3, dtype=np.float64)
    matrix = np.linalg.solve(gram + ridge * identity, x.T @ y + ridge * identity)
    return matrix, count


def apply_matrix(image, matrix):
    source = np.asarray(image, dtype=np.float32) / 255
    linear = srgb_to_linear(source)
    corrected = linear @ np.asarray(matrix, dtype=np.float32)
    encoded = np.uint8(np.clip(linear_to_srgb(corrected), 0, 1) * 255)
    return Image.fromarray(encoded, 'RGB')


def evaluate(data_dir, checkpoint, patch, matrix):
    app.CHECKPOINT = Path(checkpoint)
    rows = []
    for src, tgt in pairs(Path(data_dir)):
        mosaic, target = read_pair(src, tgt)
        mosaic, target = center_crop(mosaic, target, patch)
        before = learned_output(mosaic)
        after = apply_matrix(before, matrix)
        reference = Image.fromarray(np.uint8(np.clip(target, 0, 1) * 255), 'RGB')
        before_scores = app.metrics(before, reference)
        after_scores = app.metrics(after, reference)
        rows.append({'image': src.name, 'before': before_scores, 'after': after_scores})
        print(f'scored {src.name}', flush=True)
    metrics = ('psnr', 'ssim', 'delta_e', 'lpips')
    summary = {}
    for name in ('before', 'after'):
        summary[name] = {metric: float(np.mean([row[name][metric] for row in rows
                                                if row[name].get(metric) is not None]))
                         if any(row[name].get(metric) is not None for row in rows) else None
                         for metric in metrics}
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--validation-data', required=True)
    parser.add_argument('--checkpoint', default='models/mod_isp_siddcc.pt')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--pixels-per-image', type=int, default=1024)
    parser.add_argument('--ridge', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--out', default='ccm_evaluation_siddcc.json')
    args = parser.parse_args()
    if args.patch_size < 2 or args.pixels_per_image < 1 or args.ridge < 0:
        parser.error('patch-size/pixels-per-image must be positive and ridge non-negative')
    matrix, train_count = fit_matrix(args.train_data, args.checkpoint, args.patch_size,
                                     args.pixels_per_image, args.ridge, args.seed)
    summary, rows = evaluate(args.validation_data, args.checkpoint, args.patch_size, matrix)
    report = {'status': 'training-fit CCM; validation-only evaluation',
              'checkpoint': str(Path(args.checkpoint).resolve()),
              'training_dataset': str(Path(args.train_data).resolve()),
              'validation_dataset': str(Path(args.validation_data).resolve()),
              'training_images': train_count, 'validation_images': len(rows),
              'sampled_pixels_per_training_image': args.pixels_per_image,
              'ridge_to_identity': args.ridge, 'seed': args.seed,
              'operation': 'linearize learned sRGB; apply fitted 3x3 matrix; encode sRGB',
              'matrix_applied_on_right_to_linear_rgb_row_vector': matrix.tolist(), 'metrics': summary,
              'metric_delta_after_minus_before': {key: (summary['after'][key] - summary['before'][key]
                  if summary['after'][key] is not None else None) for key in summary['before']},
              'per_image': rows,
              'limitations': ['One validation physical scene; checkpoint was previously selected using its center-crop loss.',
                              'This is a post-restoration color calibration experiment, not camera sensor calibration.',
                              'Do not deploy unless all quality metrics and additional scene/camera tests support it.']}
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'matrix': report['matrix_applied_on_right_to_linear_rgb_row_vector'], 'metrics': summary,
                      'metric_delta_after_minus_before': report['metric_delta_after_minus_before']}, indent=2))


if __name__ == '__main__':
    main()
