"""
DKI parameter estimation via secant-based ADC slope analysis.

Estimates D and K from the linear relationship between the apparent diffusion
coefficient (ADC) computed between consecutive b-value pairs and their
midpoint b-values:

    ADC_secant(i) ≈ D − (D²·K/3) · b_mid(i)

where b_mid(i) = (b_i + b_{i+1}) / 2. A linear regression of ADC vs b_mid
yields the intercept (= D) and slope (= −D²·K/3), from which K = −3·slope/D².

This is algebraically equivalent to the log-quadratic / WLLS closed-form DKI
estimator (Masutani & Aoki 2014, bodymritools getAdcAkc), differing only in
weighting. The value is as a fast screening / QC tool, not a new estimator.

Model:  S(b) = S0 * exp(−b·D + b²·D²·K/6)

Requires ≥3 distinct b-values. b0 is NOT required (S0 estimated from minimum-b).
"""

import numpy as np
from .common import fit_volume

D_MAX = 0.05
K_MAX = 10.0
EPS = 1e-12


def fit_akc_voxel(sig, b):
    """Fit DKI parameters from secant ADC vs b_mid regression.

    Parameters
    ----------
    sig : array (Nb,)
        Signal amplitudes at each b-value (group-averaged).
    b : array (Nb,)
        Corresponding b-values (sorted, distinct).

    Returns
    -------
    s0, d, k, residual : float
        Estimated S0, diffusivity D, kurtosis K, RMSE residual.
    """
    sig = np.clip(np.asarray(sig, float), EPS, None)
    good = np.isfinite(sig) & np.isfinite(b) & (sig > 0)
    if good.sum() < 3:
        return 0.0, 0.0, 0.0, 0.0

    s = sig[good]
    bv = b[good]
    n = len(bv)

    # S0 from minimum-b volume
    s0 = float(s[0])

    if n < 3:
        return float(s0), 0.0, 0.0, 0.0

    # Secant ADC between consecutive b-value pairs
    ln_s = np.log(np.maximum(s, EPS))
    dln = -np.diff(ln_s)
    db = np.diff(bv)
    adc_sec = dln / np.maximum(db, EPS)
    b_mid = 0.5 * (bv[:-1] + bv[1:])

    # Weighted linear regression: ADC = D + slope * b_mid
    # Weights = 1/Var(ADC_secant) ≈ b_i * S_i² (lower b, higher signal → higher weight)
    w = bv[:-1] * np.maximum(s[:-1], EPS) ** 2
    w_sum = w.sum()
    if w_sum < EPS:
        return float(s0), 0.0, 0.0, 0.0

    wb_mid = (w * b_mid).sum() / w_sum
    w_adc = (w * adc_sec).sum() / w_sum
    w_bmid2 = (w * b_mid * b_mid).sum() / w_sum
    w_bmid_adc = (w * b_mid * adc_sec).sum() / w_sum

    denom = w_bmid2 - wb_mid * wb_mid
    if abs(denom) < EPS:
        return float(s0), 0.0, 0.0, 0.0

    slope = (w_bmid_adc - wb_mid * w_adc) / denom
    d = float(np.clip(w_adc - slope * wb_mid, 0.0, D_MAX))

    if d < EPS:
        k = 0.0
    else:
        k = float(np.clip(-3.0 * slope / (d * d), 0.0, K_MAX))

    # Residual RMSE
    pred_adc = d + slope * b_mid
    rmse = float(np.sqrt(np.mean((adc_sec - pred_adc) ** 2)))

    return float(s0), d, k, rmse


def akc_extrapolate_signal(b, s0, d, k, b_max):
    """AKC extrapolation via b_mid ADC.

    b_mid = (b_max + b_target) / 2
    ADC(b_mid) = D − (D²K/3) · b_mid
    S(b_target) = S0 · exp(−b_target · ADC(b_mid))
    """
    b = np.asarray(b, float)
    b_mid = (b_max + b) / 2.0
    adc_bmid = d - (d * d * k / 3.0) * b_mid
    return s0 * np.exp(np.clip(-b * adc_bmid, -80, 80))


def fit_akc_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    """Fit DKI via secant ADC slope (AKC-ADC(i) approximation)."""
    return fit_volume(
        dataobj, mask, uniq_b, groups, affine, header, out_prefix,
        'AKC-ADC', fit_akc_voxel,
        ['s0', 'd', 'k', 'residual'],
        workers=workers, **kwargs,
    )
