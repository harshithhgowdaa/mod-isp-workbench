"""Extract one fixed SIDD-CC held-out sRGB block for the live judge demo.

The benchmark MAT files are read one at a time to keep peak memory bounded.
The block is a reproducible demo sample only; it is never used for training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('data/siddcc_srgb'))
    parser.add_argument('--out', type=Path, default=Path('static/demo'))
    args = parser.parse_args()
    test_dir = args.dataset / 'SRGB' / 'test'
    scene_info = test_dir / 'test_scenes_info.txt'
    noisy_mat = test_dir / 'BenchmarkNoisyBlocksSrgb.mat'
    gt_mat = test_dir / 'BenchmarkGtBlocksSrgb.mat'
    for path in (scene_info, noisy_mat, gt_mat):
        if not path.is_file():
            raise FileNotFoundError(f'Missing SIDD-CC test file: {path}')
    scenes = [line.split('\t')[0] for line in scene_info.read_text().splitlines()[1:] if line.strip()]
    scene_id, block_index = '0164', 0
    if scene_id not in scenes:
        raise ValueError(f'SIDD-CC test scene {scene_id} was not found in {scene_info}.')
    scene_index = scenes.index(scene_id)

    args.out.mkdir(parents=True, exist_ok=True)
    stems = (f'siddcc_scene{scene_id}_block01_input.png', f'siddcc_scene{scene_id}_block01_reference.png')
    for mat_path, output_name in zip((noisy_mat, gt_mat), stems):
        data = loadmat(mat_path, variable_names=['data'])['data']
        if data.ndim != 5 or data.shape[2:] != (256, 256, 3):
            raise ValueError(f'Unexpected SIDD-CC test array shape in {mat_path}: {data.shape}')
        if not 0 <= scene_index < data.shape[0] or not 0 <= block_index < data.shape[1]:
            raise ValueError(f'Indices are outside the SIDD-CC test array bounds: {data.shape[:2]}')
        block = data[scene_index, block_index]
        if block.dtype != np.uint8:
            block = np.uint8(np.clip(block, 0, 255))
        Image.fromarray(block, 'RGB').save(args.out / output_name, optimize=True)
        del data, block
        print(f'Wrote {args.out / output_name}')
    print(f'Pair provenance: SIDD-CC SRGB/test; scene {scenes[scene_index]}; block {block_index+1}/32.')


if __name__ == '__main__':
    main()
