"""Stage SIDD-CC RAW test blocks for a frozen, scene-disjoint gate check.

The RAW test release omits per-frame metadata. This script maps sensor family
from the published frame dimensions and applies camera-level median WB gains
estimated from the SIDD-CC training metadata. Those gains are an explicit proxy,
not the missing test-capture AsShotNeutral values.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat, whosmat


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / 'data/siddcc_rawsource/Raw'
SRGB_TEST = ROOT / 'data/siddcc_srgb/SRGB/test'
OUT = ROOT / 'work/siddcc_raw_gate_test'
CAMERA_BY_SHAPE = {}
CFA_SHIFTS = {'rggb': (0, 0), 'grbg': (0, 1), 'gbrg': (1, 0), 'bggr': (1, 1)}


def camera_profiles():
    cameras = defaultdict(lambda: {'shapes': set(), 'cfas': set(), 'gains': []})
    for folder in (RAW_ROOT / 'train/Data').iterdir():
        if not folder.is_dir():
            continue
        camera = folder.name.split('_')[2]
        raw_path = next(folder.glob('*_NOISY_RAW_*.MAT'))
        metadata_path = next(folder.glob('*_METADATA_RAW_*.MAT'))
        dims = whosmat(raw_path)[0][1]
        metadata = loadmat(metadata_path)
        cfa = str(np.asarray(metadata['cfa_str']).squeeze()).lower()
        neutral = np.asarray(metadata['AsShotNeutral'], dtype=np.float64).reshape(-1)
        if cfa not in CFA_SHIFTS or neutral.size != 3 or np.any(neutral <= 0):
            raise ValueError(f'Invalid sensor metadata in {metadata_path}')
        gains = [neutral[1] / neutral[0], 1.0, 1.0, neutral[1] / neutral[2]]
        cameras[camera]['shapes'].add(tuple(dims))
        cameras[camera]['cfas'].add(cfa)
        cameras[camera]['gains'].append(gains)
    shape_map = {}
    for camera, record in cameras.items():
        if len(record['shapes']) != 1 or len(record['cfas']) != 1:
            raise ValueError(f'Camera profile is not invariant for {camera}: {record}')
        shape = next(iter(record['shapes']))
        if shape in shape_map:
            raise ValueError(f'Ambiguous frame dimensions {shape}: {camera}, {shape_map[shape]}')
        shape_map[shape] = camera
        record['cfa'] = next(iter(record['cfas']))
        record['median_wb_rggb'] = np.median(record['gains'], axis=0).tolist()
        del record['gains'], record['shapes'], record['cfas']
    return cameras, shape_map


def load_scene_rows(path):
    rows = path.read_text().splitlines()
    result = []
    for row in rows[1:]:
        scene_id, image_dir, shape_text = row.split('\t')
        result.append({'scene_id': scene_id, 'image_dir': image_dir,
                       'shape': tuple(map(int, shape_text.split('x')))})
    return result


def stratified_scene_split(scenes, count=14, seed=20261108):
    grouped = defaultdict(list)
    for scene in scenes:
        grouped[scene['camera']].append(scene)
    total = len(scenes)
    quota = {camera: count * len(rows) / total for camera, rows in grouped.items()}
    allocation = {camera: int(value) for camera, value in quota.items()}
    leftover = count - sum(allocation.values())
    for camera in sorted(grouped, key=lambda c: (-(quota[c] - allocation[c]), c))[:leftover]:
        allocation[camera] += 1
    rng = random.Random(seed)
    calibrate, validation = [], []
    for camera, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row['scene_id'])
        rng.shuffle(rows)
        calibrate.extend(rows[:allocation[camera]])
        validation.extend(rows[allocation[camera]:])
    return ({row['scene_id'] for row in calibrate},
            {row['scene_id'] for row in validation}, allocation)


def main():
    profiles, shape_map = camera_profiles()
    scenes = load_scene_rows(RAW_ROOT / 'test/test_scenes_info.txt')
    for scene in scenes:
        if scene['shape'] not in shape_map:
            raise ValueError(f'No training camera profile for test sensor shape {scene["shape"]}')
        scene['camera'] = shape_map[scene['shape']]
        scene['cfa'] = profiles[scene['camera']]['cfa']
        scene['wb_rggb'] = profiles[scene['camera']]['median_wb_rggb']
    calibration_ids, validation_ids, allocation = stratified_scene_split(scenes)
    assert len(calibration_ids) == 14 and len(validation_ids) == 15
    assert not (calibration_ids & validation_ids)

    noisy = loadmat(RAW_ROOT / 'test/BenchmarkNoisyBlocksRaw.mat')['data']
    target = loadmat(SRGB_TEST / 'BenchmarkGtBlocksSrgb.mat')['data']
    if noisy.shape[:2] != (len(scenes), 32) or target.shape[:2] != noisy.shape[:2]:
        raise ValueError(f'Unexpected benchmark shapes: RAW {noisy.shape}, sRGB {target.shape}')
    manifest = {'dataset': 'SIDD-CC official RAW test blocks; SIDD terms apply',
                'scenes_total': len(scenes), 'blocks_per_scene': 32,
                'calibration_scene_ids': sorted(calibration_ids),
                'validation_scene_ids': sorted(validation_ids),
                'camera_stratified_calibration_counts': allocation,
                'white_balance': 'camera-family median AsShotNeutral gains from train metadata; test-frame metadata absent',
                'cfa': 'camera-family CFA pattern from train metadata; inferred by unique full-frame dimensions',
                'alignment': 'CFA rephase to RGGB; same leading-pixel crop applied to RGB target; odd residual extent cropped to even',
                'seed': 20261108}
    for name, selected in [('calibration', calibration_ids), ('validation', validation_ids)]:
        root = OUT / name
        (root / 'input').mkdir(parents=True, exist_ok=True)
        (root / 'target').mkdir(parents=True, exist_ok=True)
        for scene_index, scene in enumerate(scenes):
            if scene['scene_id'] not in selected:
                continue
            sy, sx = CFA_SHIFTS[scene['cfa']]
            gains = np.asarray(scene['wb_rggb'], dtype=np.float32)
            for block_index in range(32):
                mosaic = np.asarray(noisy[scene_index, block_index], dtype=np.float32)[sy:, sx:]
                ref = np.asarray(target[scene_index, block_index], dtype=np.uint8)[sy:, sx:]
                h, w = mosaic.shape
                h -= h % 2; w -= w % 2
                mosaic = mosaic[:h, :w].copy()
                ref = ref[:h, :w].copy()
                for gain, (y, x) in zip(gains, [(0, 0), (0, 1), (1, 0), (1, 1)]):
                    mosaic[y::2, x::2] *= gain
                stem = f'test{scene["scene_id"]}_block{block_index:02d}'
                np.save(root / 'input' / f'{stem}.npy', mosaic, allow_pickle=False)
                Image.fromarray(ref, mode='RGB').save(root / 'target' / f'{stem}.png')
    manifest['camera_profiles'] = {k: {'cfa': v['cfa'], 'median_wb_rggb': v['median_wb_rggb']}
                                   for k, v in profiles.items()}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'calibration_pairs': len(calibration_ids) * 32,
                      'validation_pairs': len(validation_ids) * 32,
                      'calibration_scenes': manifest['calibration_scene_ids'],
                      'validation_scenes': manifest['validation_scene_ids'],
                      'camera_counts': allocation}, indent=2))


if __name__ == '__main__':
    main()
