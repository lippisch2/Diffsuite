import numpy as np
from scipy.optimize import least_squares
from .common import fit_volume
from .adc import fit_adc_voxel


def froc_signal(b, s0, d, beta, mu):
    # Reduced b-value-only form of the fractional-order-calculus model:
    # S(b)/S0 = exp[ -D * mu^(2(beta-1)) * b^beta ]
    return s0 * np.exp(-d * np.power(mu, 2.0 * (beta - 1.0)) * np.power(b, beta))


def fit_froc_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    s0 = float(sig[np.argmin(b)])
    _, adc0, _ = fit_adc_voxel(sig, b)
    x0 = np.array([s0, max(adc0, 1e-5), 0.9, 50.0], float)
    def fun(p):
        return froc_signal(b, *p) - sig
    try:
        res = least_squares(
            fun, x0,
            bounds=([1e-8, 1e-8, 1e-8, 1e-3], [np.inf, 0.05, 1.0, np.inf]),
            method='trf',
        )
        p = res.x
    except Exception:
        p = x0
    pred = froc_signal(b, *p)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(p[0]), float(p[1]), float(p[2]), float(p[3]), rmse


def fit_froc_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix,
                      'FROC', fit_froc_voxel, ['s0', 'd', 'beta', 'mu', 'residual'],
                      workers=workers, **kwargs)
