import numpy as np
from scipy.optimize import least_squares
from .common import fit_volume
from .adc import fit_adc_voxel
from .mittag_leffler import mittag_leffler


def ctrw_signal(b, s0, d, alpha, beta):
    return s0 * np.array([mittag_leffler(alpha, -( (bb * d) ** beta )) for bb in b])


def fit_ctrw_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    s0 = float(sig[np.argmin(b)])
    _, adc0, _ = fit_adc_voxel(sig, b)
    x0 = np.array([s0, max(adc0, 1e-5), 0.5, 0.9], float)
    def fun(p):
        return ctrw_signal(b, *p) - sig
    try:
        res = least_squares(
            fun, x0,
            bounds=([1e-8, 1e-8, 1e-8, 1e-8], [np.inf, 0.05, 1.0, 1.0]),
            method='trf',
        )
        p = res.x
    except Exception:
        p = x0
    pred = ctrw_signal(b, *p)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(p[0]), float(p[1]), float(p[2]), float(p[3]), rmse


def fit_ctrw_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix,
                      'CTRW', fit_ctrw_voxel, ['s0', 'd', 'alpha', 'beta', 'residual'],
                      workers=workers, **kwargs)
