"""RAW-only local statistics used by the experimental Bayer quality gate."""
import numpy as np

FEATURE_NAMES = (
    'mean', 'std', 'relative_variance', 'negative_fraction',
    'above_white_fraction', 'mean_absolute_deviation',
    'gradient_x_abs', 'gradient_y_abs',
)


def tile_features(mosaic, tile=56):
    """Return one finite feature vector per tile, in row-major spatial order."""
    mosaic=np.asarray(mosaic,dtype=np.float32)
    h,w=mosaic.shape
    # Pad partial edge tiles by reflection, then calculate tile statistics in
    # batched NumPy operations. This avoids Python percentile/median calls per
    # tile, which dominated megapixel-frame routing time.
    rows=(h+tile-1)//tile;cols=(w+tile-1)//tile
    padded=np.pad(mosaic,((0,rows*tile-h),(0,cols*tile-w)),mode='reflect')
    blocks=padded.reshape(rows,tile,cols,tile).transpose(0,2,1,3).reshape(rows*cols,tile,tile)
    flat=blocks.reshape(rows*cols,-1)
    mean=flat.mean(axis=1);variance=flat.var(axis=1);std=np.sqrt(variance)
    mean_abs_deviation=np.abs(flat-mean[:,None]).mean(axis=1)
    gx=np.abs(np.diff(blocks,axis=2)).mean(axis=(1,2))
    gy=np.abs(np.diff(blocks,axis=1)).mean(axis=(1,2))
    values=np.column_stack((mean,std,variance/(mean*mean+1e-6),
        (flat<0).mean(axis=1),(flat>1).mean(axis=1),
        mean_abs_deviation,gx,gy))
    return [(values[i].tolist(),(i//cols)*tile,(i%cols)*tile) for i in range(rows*cols)]


def predict_benefit(mosaic, gate, tile=56):
    """Predict learned-minus-classical local PSNR gain from frozen RAW features."""
    rows = tile_features(mosaic, tile)
    x = np.asarray([r[0] for r in rows], dtype=np.float64)
    if gate.get('mode') in ('raw_feature_extra_trees', 'raw_feature_extra_trees_rank'):
        predictions = np.zeros(len(x), dtype=np.float64)
        for tree in gate['trees']:
            left = np.asarray(tree['children_left'], dtype=np.int32)
            right = np.asarray(tree['children_right'], dtype=np.int32)
            feature = np.asarray(tree['feature'], dtype=np.int32)
            threshold = np.asarray(tree['threshold'], dtype=np.float64)
            value = np.asarray(tree['value'], dtype=np.float64)
            nodes = np.zeros(len(x), dtype=np.int32)
            while True:
                f = feature[nodes]
                active = f >= 0
                if not np.any(active):
                    break
                indexes = np.flatnonzero(active)
                go_left = x[indexes, f[indexes]] <= threshold[nodes[indexes]]
                nodes[indexes] = np.where(go_left, left[nodes[indexes]], right[nodes[indexes]])
            predictions += value[nodes] / len(gate['trees'])
        return [(float(score), row[1], row[2]) for score, row in zip(predictions, rows)]
    mean = np.asarray(gate['feature_mean'], dtype=np.float64)
    scale = np.asarray(gate['feature_scale'], dtype=np.float64)
    coef = np.asarray(gate['coefficients'], dtype=np.float64)
    z = np.clip((x - mean) / scale, -5, 5)
    design = np.column_stack([np.ones(len(z)), z])
    return [(float(score), row[1], row[2]) for score, row in zip(design @ coef, rows)]


def selection_mask(mosaic, gate, tile=56):
    """Select predicted-beneficial neural tiles under a calibrated threshold."""
    h, w = mosaic.shape
    predictions = predict_benefit(mosaic, gate, tile)
    threshold = float(gate['threshold_db'])
    chosen = [(score, y, x) for score, y, x in predictions if score >= threshold]
    mask = np.zeros((h, w), dtype=np.float32)
    for _, y, x in chosen:
        mask[y:min(y + tile, h), x:min(x + tile, w)] = 1
    return mask, len(chosen), len(predictions)
