import numpy as np

from .common import fit_volume
from .dki2d_fast import fit_dki2d_fast_voxel


def _skew(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1)
    return float(np.mean(((x - m) / s) ** 3)) if s > 0 else 0.0


def fit_dki2d_fast_plus_voxel(sig, b):
    s0, d, k, rmse = fit_dki2d_fast_voxel(sig, b)
    skew = _skew(sig)
    fused = float(0.5 * (skew + k))
    return float(s0), float(d), float(k), float(skew), float(fused), float(rmse)


def model_dki2d_fast_plus(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix,
                      'DKI fast+', fit_dki2d_fast_plus_voxel,
                      ['s0', 'd', 'k', 'skew', 'fused_skew_k', 'residual'],
                      workers=workers, **kwargs)
