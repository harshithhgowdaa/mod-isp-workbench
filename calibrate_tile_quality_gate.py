"""Fit a RAW-statistics tile gate on training scenes, calibrate on one scene, audit another."""
import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

import app
from tile_quality_gate import FEATURE_NAMES, tile_features
from train import pairs, read_pair


def crop_pair(mosaic, target, size):
    if not size:
        return mosaic, target
    h, w = mosaic.shape
    size = min(size, h, w) & ~1
    y = max(0, ((h - size) // 2) // 2 * 2)
    x = max(0, ((w - size) // 2) // 2 * 2)
    return mosaic[y:y + size, x:x + size], target[y:y + size, x:x + size]


def image_rows(root, patch, classical_affine=None):
    rows = []
    for source, target_path in pairs(Path(root)):
        mosaic, target = read_pair(source, target_path)
        mosaic, target = crop_pair(mosaic, target, patch)
        classical = np.asarray(app.classical_bayer(mosaic, 'human', 1), dtype=np.float32) / 255
        classical = correct_classical(classical, classical_affine)
        learned_raw, _ = app.learned(mosaic, hybrid=False)
        learned = app.finish_rgb(learned_raw, 'human', 1, apply_exposure=False)
        learned = np.asarray(learned, dtype=np.float32) / 255
        reference = np.asarray(Image.fromarray(np.uint8(np.clip(target, 0, 1) * 255), 'RGB'), dtype=np.float32) / 255
        features = tile_features(mosaic)
        local_targets = []
        for _, y, x in features:
            y1, x1 = min(y + 56, mosaic.shape[0]), min(x + 56, mosaic.shape[1])
            c_mse = float(np.mean((classical[y:y1, x:x1] - reference[y:y1, x:x1]) ** 2))
            n_mse = float(np.mean((learned[y:y1, x:x1] - reference[y:y1, x:x1]) ** 2))
            local_targets.append(10 * np.log10((c_mse + 1e-10) / (n_mse + 1e-10)))
        rows.append({'image': source.name, 'features': features,
                     'classical': classical, 'learned': learned, 'reference': reference,
                     'local_gain_db': np.asarray(local_targets, dtype=np.float64)})
        print(f'prepared {source.name}', flush=True)
    if not rows:
        raise ValueError(f'No paired data under {root}')
    return rows


def fit_classical_affine(data_root, patch, pixels_per_capture=2048, seed=2026):
    """Fit a global sRGB affine correction using training pairs only."""
    rng = np.random.default_rng(seed)
    features, targets = [], []
    for source, target_path in pairs(Path(data_root)):
        mosaic, target = read_pair(source, target_path)
        mosaic, target = crop_pair(mosaic, target, patch)
        classical = np.asarray(app.classical_bayer(mosaic, 'human', 1), dtype=np.float32) / 255
        reference = np.asarray(target, dtype=np.float32)
        n = min(pixels_per_capture, classical.shape[0] * classical.shape[1])
        indexes = rng.choice(classical.shape[0] * classical.shape[1], n, replace=False)
        x = classical.reshape(-1, 3)[indexes]
        y = reference.reshape(-1, 3)[indexes]
        # Exclude clipping endpoints, which otherwise dominate least squares
        # while providing little information about color mapping.
        keep = np.all((x > 2 / 255) & (x < 253 / 255), axis=1)
        if np.any(keep):
            x, y = x[keep], y[keep]
            features.append(np.column_stack((x, np.ones(len(x), dtype=np.float32))))
            targets.append(y)
        print(f'color-fit {source.name}', flush=True)
    if not features:
        raise ValueError('No unclipped training pixels available for classical color calibration')
    x = np.concatenate(features).astype(np.float64)
    y = np.concatenate(targets).astype(np.float64)
    # Tiny Tikhonov penalty stabilizes nearly collinear channel samples while
    # leaving the fitted intercept unregularized.
    gram = x.T @ x
    regularizer = np.eye(4, dtype=np.float64) * 1e-5
    regularizer[-1, -1] = 0
    affine = np.linalg.solve(gram + regularizer, x.T @ y)
    return affine.tolist(), {'pairs': len(features), 'pixels': int(len(x)),
                             'fit': 'global affine RGB correction in rendered sRGB domain',
                             'seed': seed, 'pixels_per_capture_max': pixels_per_capture}


def correct_classical(classical, affine):
    if affine is None:
        return classical
    matrix = np.asarray(affine, dtype=np.float32)
    if matrix.shape != (4, 3) or not np.isfinite(matrix).all():
        raise ValueError('classical affine transform must be a finite 4x3 matrix')
    flat = classical.reshape(-1, 3)
    design = np.column_stack((flat, np.ones(len(flat), dtype=np.float32)))
    return np.clip((design @ matrix).reshape(classical.shape), 0, 1)


def render_gate(row, scores, threshold=None, route_fraction=None, feather_pixels=0):
    h, w = row['reference'].shape[:2]
    scale=8
    mask = np.zeros((h,w),dtype=bool)
    if route_fraction is None:
        chosen=np.asarray(scores) >= float(threshold)
    else:
        requested=int(np.ceil(len(scores)*float(route_fraction)))
        order=np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
        chosen=np.zeros(len(scores),dtype=bool);chosen[order[:requested]]=True
    count = 0
    for route, (_, y, x) in zip(chosen, row['features']):
        if route:
            mask[y:min(y+56,h),x:min(x+56,w)] = True
            count += 1
    feather_pixels=max(0,int(feather_pixels))
    if feather_pixels:
        inside=distance_transform_edt(mask)
        outside=distance_transform_edt(~mask)
        signed=inside-outside
        alpha=np.clip(.5+signed/(2*feather_pixels),0,1).astype(np.float32)[...,None]
    else:
        alpha=mask.astype(np.float32)[...,None]
    output = row['classical'] * (1 - alpha) + row['learned'] * alpha
    return output, count


def psnr(output, reference):
    mse = float(np.mean((output - reference) ** 2))
    return float('inf') if mse == 0 else -10 * np.log10(mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', required=True,
                        help='Training split used to fit the RAW-only tile benefit predictor.')
    parser.add_argument('--calibration-data', required=True)
    parser.add_argument('--validation-data', required=True)
    parser.add_argument('--checkpoint', default='models/mod_isp_siddcc.pt')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--max-psnr-drop', type=float, default=.5)
    parser.add_argument('--ridge', type=float, default=10.0)
    parser.add_argument('--predictor', choices=('ridge', 'extra_trees'), default='ridge',
                        help='RAW-feature tile-benefit model; ExtraTrees captures nonlinear interactions.')
    parser.add_argument('--trees', type=int, default=96,
                        help='Number of trees when --predictor extra_trees is selected.')
    parser.add_argument('--min-leaf-tiles', type=int, default=20,
                        help='Minimum tiles per ExtraTrees leaf to limit scene-specific overfit.')
    parser.add_argument('--rank-within-image', action='store_true',
                        help='Choose top predicted-benefit tiles per image, avoiding score-offset drift across captures.')
    parser.add_argument('--rank-fractions', type=float, nargs='+',
                        help='Optional exact route fractions for rank mode; defaults to 0..1 at 1%% steps.')
    parser.add_argument('--feather-pixels', type=int, nargs='+', default=[0, 2, 4, 8],
                        help='Candidate transition half-widths. Zero uses exact tile mask; calibration selects the narrowest quality-safe option.')
    parser.add_argument('--fit-classical-affine', action='store_true',
                        help='Fit one global rendered-sRGB affine correction on training pairs, then use it for calibration and validation baseline tiles.')
    parser.add_argument('--classical-affine-file', type=Path,
                        help='Reuse a previously fit classical affine from a gate/report JSON instead of refitting it.')
    parser.add_argument('--out', default='tile_gate_evaluation.json')
    parser.add_argument('--gate-out', default='models/gate_quality_tile.json')
    args = parser.parse_args()
    if args.max_psnr_drop < 0 or args.ridge < 0:
        parser.error('quality budget and ridge penalty must be non-negative')
    started = time.perf_counter()
    app.CHECKPOINT = Path(args.checkpoint)
    if args.fit_classical_affine and args.classical_affine_file:
        parser.error('Choose --fit-classical-affine or --classical-affine-file, not both')
    classical_affine = None
    color_fit = None
    if args.fit_classical_affine:
        classical_affine, color_fit = fit_classical_affine(args.training_data, args.patch_size)
    elif args.classical_affine_file:
        payload = json.loads(args.classical_affine_file.read_text())
        payload = payload.get('gate', payload)
        classical_affine = payload.get('classical_affine_srgb')
        if classical_affine is None:
            parser.error(f'No classical_affine_srgb in {args.classical_affine_file}')
        color_fit = {'source': str(args.classical_affine_file.resolve()),
                     'fit': 'reused training-only affine transform'}
    training = image_rows(args.training_data, args.patch_size, classical_affine)
    training_rows = len(training)

    x_train = np.asarray([tile[0] for row in training for tile in row['features']], dtype=np.float64)
    y_train = np.concatenate([row['local_gain_db'] for row in training])
    # The regression fit needs only these tile-level arrays. Drop full RGB
    # outputs for the training split before materializing the larger held-out
    # images so high-resolution calibration runs stay bounded in memory.
    del training
    gc.collect()
    calibration = image_rows(args.calibration_data, args.patch_size, classical_affine)
    validation = image_rows(args.validation_data, args.patch_size, classical_affine)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-10] = 1
    trees_payload = None
    if args.predictor == 'extra_trees':
        from sklearn.ensemble import ExtraTreesRegressor
        predictor = ExtraTreesRegressor(
            n_estimators=args.trees, max_features=0.8, min_samples_leaf=args.min_leaf_tiles,
            max_depth=12, random_state=2026, n_jobs=-1,
        ).fit(x_train, y_train)
        trees_payload = []
        for estimator in predictor.estimators_:
            tree = estimator.tree_
            trees_payload.append({
                'children_left': tree.children_left.tolist(),
                'children_right': tree.children_right.tolist(),
                'feature': tree.feature.tolist(),
                'threshold': tree.threshold.tolist(),
                'value': tree.value[:, 0, 0].tolist(),
            })
        def predict_features(x):
            return predictor.predict(x)
    else:
        z_train = np.clip((x_train - mean) / scale, -5, 5)
        design_train = np.column_stack([np.ones(len(z_train)), z_train])
        regularizer = np.eye(design_train.shape[1]) * args.ridge
        regularizer[0, 0] = 0
        coef = np.linalg.solve(design_train.T @ design_train + regularizer,
                               design_train.T @ y_train)
        def predict_features(x):
            z = np.clip((x - mean) / scale, -5, 5)
            return np.column_stack([np.ones(len(z)), z]) @ coef
    for row in calibration + validation:
        x = np.asarray([tile[0] for tile in row['features']], dtype=np.float64)
        row['predictions'] = predict_features(x)

    baseline = np.mean([psnr(row['learned'], row['reference']) for row in calibration])
    all_predictions = np.concatenate([r['predictions'] for r in calibration])
    candidates = []
    if args.rank_within_image:
        fractions=args.rank_fractions if args.rank_fractions else np.linspace(0,1,101)
        if any(not 0 <= fraction <= 1 for fraction in fractions):
            parser.error('Every --rank-fractions value must be between 0 and 1')
        policies=[('route_fraction',float(fraction)) for fraction in sorted(set(fractions))]
    else:
        thresholds = np.unique(np.quantile(all_predictions, np.linspace(0, 1, 101)))
        thresholds = np.concatenate((thresholds, [np.nextafter(float(all_predictions.max()), np.inf)]))
        policies=[('threshold_db',float(threshold)) for threshold in thresholds]
    feather_options=sorted(set(max(0,int(v)) for v in args.feather_pixels))
    if not feather_options:
        parser.error('At least one --feather-pixels candidate is required')
    for policy_name, policy_value in policies:
        for feather in feather_options:
            image_scores, routes = [], []
            for row in calibration:
                output, count = render_gate(row, row['predictions'],
                    threshold=policy_value if policy_name=='threshold_db' else None,
                    route_fraction=policy_value if policy_name=='route_fraction' else None,
                    feather_pixels=feather)
                image_scores.append(psnr(output, row['reference']))
                routes.append(count / len(row['features']))
            mean_score = float(np.mean(image_scores))
            route_percent = 100 * sum(routes) / len(routes)
            if mean_score >= baseline - args.max_psnr_drop:
                candidates.append({policy_name: policy_value, 'feather_pixels': feather,
                                   'mean_psnr_db': mean_score,
                                   'route_percent': route_percent,
                                   'quality_drop_db': baseline - mean_score})
    if not candidates:
        raise RuntimeError('No tile-gate threshold meets the calibration quality budget.')
    selected = min(candidates, key=lambda r: (r['route_percent'], r['feather_pixels'], -r['mean_psnr_db']))
    mode = ('raw_feature_extra_trees_rank' if args.rank_within_image else 'raw_feature_extra_trees') \
        if args.predictor == 'extra_trees' else \
        ('raw_feature_ridge_rank' if args.rank_within_image else 'raw_feature_ridge')
    gate = {'mode': mode, 'tile_size': 56,
            'feature_names': list(FEATURE_NAMES), 'feature_mean': mean.tolist(),
            'feature_scale': scale.tolist(),
            'max_calibration_psnr_drop_db': args.max_psnr_drop,
            'training_dataset': str(Path(args.training_data).resolve()),
            'calibration_dataset': str(Path(args.calibration_data).resolve()),
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'patch_size': args.patch_size, 'feather_pixels': selected['feather_pixels']}
    if classical_affine is not None:
        gate['classical_affine_srgb'] = classical_affine
    if args.predictor == 'extra_trees':
        gate['predictor'] = {'type': 'ExtraTreesRegressor', 'trees': args.trees,
                             'max_depth': 12, 'min_samples_leaf': args.min_leaf_tiles,
                             'random_state': 2026,
                             'feature_importances': predictor.feature_importances_.tolist()}
        gate['trees'] = trees_payload
    else:
        gate['coefficients'] = coef.tolist()
    if args.rank_within_image:
        gate['route_fraction']=selected['route_fraction']
    else:
        gate['threshold_db']=selected['threshold_db']
    validation_rows = []
    for row in validation:
        output, count = render_gate(row, row['predictions'],
            threshold=gate.get('threshold_db'), route_fraction=gate.get('route_fraction'),
            feather_pixels=gate['feather_pixels'])
        learned_metrics = app.metrics(Image.fromarray(np.uint8(np.clip(row['learned'], 0, 1) * 255)),
                                      Image.fromarray(np.uint8(np.clip(row['reference'], 0, 1) * 255)))
        gate_metrics = app.metrics(Image.fromarray(np.uint8(np.clip(output, 0, 1) * 255)),
                                   Image.fromarray(np.uint8(np.clip(row['reference'], 0, 1) * 255)))
        validation_rows.append({'image': row['image'], 'routed_percent': 100 * count / len(row['features']),
                                'all_learned': learned_metrics, 'gated': gate_metrics})
        print(f'audited {row["image"]}', flush=True)
    metric_names = ('psnr', 'ssim', 'delta_e', 'lpips')
    validation_summary = {}
    for name in ('all_learned', 'gated'):
        validation_summary[name] = {
            metric: float(np.mean([r[name][metric] for r in validation_rows if r[name].get(metric) is not None]))
            if any(r[name].get(metric) is not None for r in validation_rows) else None
            for metric in metric_names}
    report = {'status': 'experimental_not_deployed', 'gate': gate,
              'fit': {'rows': training_rows, 'tiles': int(len(y_train)),
                      'dataset': str(Path(args.training_data).resolve()),
                      'classical_color_fit': color_fit,
                      'predictor': args.predictor,
                      'predictor_config': ({'trees': args.trees,
                                            'max_depth': 12,
                                            'min_samples_leaf': args.min_leaf_tiles,
                                            'random_state': 2026}
                                           if args.predictor == 'extra_trees' else
                                           {'ridge': args.ridge})},
              'calibration': {'all_learned_mean_psnr_db': baseline, 'selected_policy': selected,
                              'candidate_count': len(candidates),
                              'rows': len(calibration),
                              'tiles': int(sum(len(r['features']) for r in calibration))},
              'validation': {'rows': len(validation_rows),
                             'mean_routed_percent': float(np.mean([r['routed_percent'] for r in validation_rows])),
                             'all_learned_metrics': validation_summary['all_learned'],
                             'gated_metrics': validation_summary['gated'],
                             'metric_delta': {m: (validation_summary['gated'][m] - validation_summary['all_learned'][m]
                                                   if validation_summary['gated'][m] is not None else None)
                                              for m in metric_names},
                             'per_image': validation_rows},
              'environment': platform.platform(), 'runtime_seconds': time.perf_counter() - started,
              'limitations': [
                  f"Calibration uses {len(calibration)} captures/blocks; validation uses {len(validation_rows)} captures/blocks. "
                  'Scene grouping must be interpreted from the supplied dataset manifest.',
                  f"Calibration and validation use even-aligned center crops up to {args.patch_size} pixels."
                  if args.patch_size else
                  'Calibration and validation use full aligned captures.',
                  'The training split fits the tile-benefit predictor; the calibration split selects the route policy; the validation split is not used for selection.',
                  'The gate remains experimental until a quality-safe reduction in measured target-device computation or latency is demonstrated.']}
    psnr_delta = validation_summary['gated']['psnr'] - validation_summary['all_learned']['psnr']
    gate['validation_mean_psnr_delta_db'] = psnr_delta
    gate['validation_route_percent'] = report['validation']['mean_routed_percent']
    gate['validation_psnr_budget_passed'] = bool(psnr_delta >= -args.max_psnr_drop)
    # This research script measures route quality but does not establish a
    # target-device MAC/energy or standalone end-to-end latency reduction.
    gate['deployment_eligible'] = False
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(gate, indent=2, allow_nan=False))
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'calibration': report['calibration'], 'validation': {
        k: v for k, v in report['validation'].items() if k != 'per_image'},
        'runtime_seconds': report['runtime_seconds']}, indent=2))


if __name__ == '__main__':
    main()
