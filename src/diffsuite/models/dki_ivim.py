import numpy as np
from scipy.optimize import least_squares
from .common import fit_volume
from .adc import fit_adc_voxel


def dki_ivim_signal(b, s0, f, dstar, d, k):
    # IVIM perfusion component (fast decay) + DKI diffusion component (kurtosis)
    perf = f * np.exp(-b * dstar)
    diff = (1.0 - f) * np.exp(np.clip(-b * d + (b ** 2) * (d ** 2) * k / 6.0, -80, 80))
    return s0 * (perf + diff)


def fit_dki_ivim_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    b = np.asarray(b, float)
    s0 = float(sig[np.argmin(b)])
    _, adc0, _ = fit_adc_voxel(sig, b)

    idx_high = b > 200
    if idx_high.sum() >= 2:
        y = np.log(sig[idx_high])
        X = np.column_stack([np.ones(idx_high.sum()), -b[idx_high]])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        d0 = float(np.clip(beta[1], 1e-5, 0.01))
    else:
        d0 = float(np.clip(adc0, 1e-5, 0.01))

    x0 = np.array([max(s0, 1e-8), 0.1, 0.01, d0, 0.7], float)

    def fun(p):
        return dki_ivim_signal(b, *p) - sig

    try:
        res = least_squares(
            fun, x0,
            bounds=([1e-8, 0.0, 1e-4, 0.0, 0.0], [np.inf, 1.0, 0.2, 0.05, 5.0]),
            method='trf'
        )
        p = res.x
    except Exception:
        p = x0

    pred = dki_ivim_signal(b, *p)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), rmse


def fit_dki_ivim_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(
        dataobj, mask, uniq_b, groups, affine, header, out_prefix,
        'DKI-IVIM',
        fit_dki_ivim_voxel,
        ['s0', 'f', 'dstar', 'd', 'k', 'residual'],
        workers=workers,
        **kwargs,
    )
