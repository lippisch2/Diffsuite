import numpy as np
from .common import fit_volume


def fit_dki2d_fast_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    good = np.isfinite(sig) & np.isfinite(b)
    if good.sum() < 3:
        return 0.0, 0.0, 0.0, 0.0
    y = np.log(sig[good])
    b2 = b[good] ** 2
    X = np.column_stack([np.ones(good.sum()), -b[good], 0.5 * b2])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    s0 = float(np.exp(beta[0]))
    d = float(np.clip(beta[1], 0.0, 0.05))
    k = float(np.clip((3.0 * beta[2]) / max(d * d, 1e-12), 0.0, 10.0))
    bg = b[good]
    pred = s0 * np.exp(np.clip(-bg * d + (bg ** 2) * (d ** 2) * k / 6.0, -80, 80))
    rmse = float(np.sqrt(np.mean((sig[good] - pred) ** 2)))
    return s0, d, k, rmse


def fit_dki2d_fast_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, '2D DKI fast', fit_dki2d_fast_voxel, ['s0', 'd', 'k', 'residual'], workers=workers, **kwargs)