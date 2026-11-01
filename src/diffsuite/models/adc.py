import numpy as np
from scipy.optimize import least_squares
from .common import fit_volume


def mono_signal(b, s0, adc):
    return s0 * np.exp(-b * adc)


def fit_adc_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    good = np.isfinite(sig) & np.isfinite(b)
    if good.sum() < 2:
        return 0.0, 0.0, 0.0
    y = np.log(sig[good])
    X = np.column_stack([np.ones(good.sum()), -b[good]])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    s0 = float(np.exp(beta[0]))
    adc = float(max(beta[1], 0.0))
    pred = mono_signal(b[good], s0, adc)
    rmse = float(np.sqrt(np.mean((sig[good] - pred) ** 2)))
    return s0, adc, rmse


def fit_adc_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, 'ADC', fit_adc_voxel, ['s0', 'adc', 'residual'], workers=workers, **kwargs)
