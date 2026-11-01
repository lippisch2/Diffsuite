import numpy as np
from scipy.optimize import least_squares
from .common import fit_volume
from .adc import fit_adc_voxel


def ivim_signal(b, s0, f, dstar, d):
    return s0 * (f * np.exp(-b * dstar) + (1.0 - f) * np.exp(-b * d))


def fit_ivim_full_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    b = np.asarray(b, float)
    s0 = float(sig[np.argmin(b)])
    _, adc0, _ = fit_adc_voxel(sig, b)
    x0 = np.array([max(s0, 1e-8), 0.1, 0.01, max(adc0, 1e-5)], float)

    def fun(p):
        return ivim_signal(b, *p) - sig

    # FireVoxel Model 18 bounds: Dp ∈ [0, 1], Dt ∈ [0, 0.01], f ∈ [0, 1]
    try:
        res = least_squares(fun, x0, bounds=([1e-8, 0.0, 1e-4, 0.0], [np.inf, 1.0, 1.0, 0.01]), method='trf')
        p = res.x
    except Exception:
        p = x0

    pred = ivim_signal(b, *p)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(p[0]), float(p[1]), float(p[2]), float(p[3]), rmse


def fit_ivim_full_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(
        dataobj,
        mask,
        uniq_b,
        groups,
        affine,
        header,
        out_prefix,
        'IVIM full',
        fit_ivim_full_voxel,
        ['s0', 'f', 'dstar', 'd', 'residual'],
        workers=workers,
        **kwargs,
    )