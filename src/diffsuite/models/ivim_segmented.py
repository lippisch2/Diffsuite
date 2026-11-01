import numpy as np
from scipy.optimize import least_squares
from .common import fit_volume
from .adc import fit_adc_voxel


def ivim_signal(b, s0, f, dstar, d):
    return s0 * (f * np.exp(-b * dstar) + (1.0 - f) * np.exp(-b * d))


def fit_ivim_segmented_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    b = np.asarray(b, float)
    s0 = float(sig[np.argmin(b)])

    # ── Stage 1: high-b monoexponential fit (b > 200) → Dt, then f ──
    # Per FireVoxel Model 17 (Sigmund et al.): for b >> 1/Dp the perfusion
    # term vanishes, so S ≈ S0 (1-f) exp(-b Dt). Fit log S = log(S0(1-f)) - b Dt
    # to recover Dt; intercept Si gives f = (S0 - Si)/S0.
    idx_high = b > 200
    if idx_high.sum() >= 3:
        y = np.log(sig[idx_high])
        X = np.column_stack([np.ones(idx_high.sum()), -b[idx_high]])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        si = float(np.exp(beta[0]))          # Si = S0 (1 - f)
        d0 = float(np.clip(beta[1], 1e-5, 0.01))
        f0 = float(np.clip(1.0 - si / max(s0, 1e-8), 0.0, 1.0))
        f0 = max(f0, 1e-3)
    else:
        _, adc0, _ = fit_adc_voxel(sig, b)
        d0 = float(np.clip(adc0, 1e-5, 0.01))
        f0 = 0.1

    # ── Stage 2: fix Dt and f, fit only Dp on the full dataset ──
    def fun_dp(p):
        dstar = p[0]
        return ivim_signal(b, s0, f0, dstar, d0) - sig

    try:
        res = least_squares(fun_dp, [0.01], bounds=([1e-4], [1.0]), method='trf')
        dstar = float(res.x[0])
    except Exception:
        dstar = 0.01

    pred = ivim_signal(b, s0, f0, dstar, d0)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return s0, f0, dstar, d0, rmse


def fit_ivim_segmented_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(
        dataobj,
        mask,
        uniq_b,
        groups,
        affine,
        header,
        out_prefix,
        'IVIM segmented',
        fit_ivim_segmented_voxel,
        ['s0', 'f', 'dstar', 'd', 'residual'],
        workers=workers,
        **kwargs,
    )