import numpy as np
from scipy.optimize import least_squares
from .common import fit_volume
from .adc import fit_adc_voxel
from .dki2d_fast import fit_dki2d_fast_voxel


def dki2d_signal(b, s0, d, k):
    expo = -b * d + (b ** 2) * (d ** 2) * k / 6.0
    return s0 * np.exp(np.clip(expo, -80, 80))


def fit_dki2d_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    s0 = float(sig[np.argmin(b)])

    # Robust seed from the analytic log-linear DKI estimator (deterministic,
    # never trapped). For high D*K the kurtosis term can dominate at large b and
    # make the signal non-monotonic, where a blind scipy least_squares from the
    # crude ADC init frequently converges to the wrong branch (see the full-vs-
    # fast D correlation on real data). Start from the analytic solution instead.
    try:
        f_s0, f_d, f_k, _ = fit_dki2d_fast_voxel(sig, b)
        if f_s0 > 0 and f_d > 0:
            x0 = np.array([f_s0, f_d, min(max(f_k, 0.0), 5.0)], float)
            analytic = np.array([f_s0, f_d, min(max(f_k, 0.0), 5.0)], float)
        else:
            raise ValueError
    except Exception:
        _, adc0, _ = fit_adc_voxel(sig, b)
        x0 = np.array([s0, max(adc0, 1e-5), 0.7], float)
        analytic = None

    def fun(p):
        return dki2d_signal(b, *p) - sig

    try:
        res = least_squares(fun, x0, bounds=([1e-8, 0.0, 0.0], [np.inf, 0.05, 5.0]), method='trf')
        p = res.x
    except Exception:
        p = x0

    # Keep the better of the nonlinear-fitted and analytic solutions by RMSE.
    candidates = [p] + ([analytic] if analytic is not None else [])
    best = p
    best_rmse = float(np.sqrt(np.mean((sig - dki2d_signal(b, *p)) ** 2)))
    for cand in candidates[1:]:
        r = float(np.sqrt(np.mean((sig - dki2d_signal(b, *cand)) ** 2)))
        if r < best_rmse:
            best = cand
            best_rmse = r

    pred = dki2d_signal(b, *best)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(best[0]), float(best[1]), float(best[2]), rmse


def fit_dki2d_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, '2D DKI', fit_dki2d_voxel, ['s0', 'd', 'k', 'residual'], workers=workers, **kwargs)