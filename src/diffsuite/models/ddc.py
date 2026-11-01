import numpy as np
from scipy.optimize import least_squares
from .common import fit_volume
from .adc import fit_adc_voxel


def ddc_signal(b, s0, ddc, alpha):
    return s0 * np.exp(-np.power(np.maximum(b * ddc, 0.0), alpha))


def fit_ddc_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    s0 = float(sig[np.argmin(b)])
    _, adc0, _ = fit_adc_voxel(sig, b)
    x0 = np.array([s0, max(adc0, 1e-5), 0.8], float)
    def fun(p):
        return ddc_signal(b, *p) - sig
    try:
        res = least_squares(fun, x0, bounds=([1e-8, 0.0, 0.05], [np.inf, 0.05, 1.0]), method='trf')
        p = res.x
    except Exception:
        p = x0
    pred = ddc_signal(b, *p)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(p[0]), float(p[1]), float(p[2]), rmse


def fit_ddc_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, 'DDC', fit_ddc_voxel, ['s0', 'ddc', 'alpha', 'residual'], workers=workers, **kwargs)
