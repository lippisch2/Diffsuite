"""
DKI parameter estimation via the Alternative Iteration Schema (AIS).

Implements the fast DKI fitting schema of:
    Yan X., Zhou M., Ying L., Liu W., Yang G., Wu D., Zhou Y.,
    Peterson B.S., Xu D. (2014)
    "A fast schema for parameter estimation in diffusion kurtosis imaging",
    Computerized Medical Imaging and Graphics 38:469-480.

Three variants share one vectorized core:
    dki_ais_u  — UAIS,  unconstrained alternating iteration
    dki_ais_c  — CAIS,  physical constraints D>=0, K>=0, K<=3/(b_max*D)
    dki_ais_sc — SCAIS, CAIS + a temporary 3x3x3 Gaussian-smoothed copy of the
                 current D map used inside EVERY K update (paper section 2.4)

Model:  S(b) = S0 * exp(-b*D + b^2 * D^2 * K / 6)

Unlike the other models this family is fitted volume-level with numpy
broadcasting (the point of AIS is speed), not voxel-by-voxel through
common.fit_volume. Preprocessing (group-mean per b-value, mask ANDing with the
optional ROI) replicates fit_volume exactly; outputs are saved identically as
{out_prefix}_{param}.nii.gz and returned as {param: 3D ndarray}.

Deviations from the paper:
- No true b0 in many clinical inputs: initialization uses a joint one-shot
  log-linear least squares (fitted intercept) instead of the paper's measured
  baseline + Jensen 3-point init — the latter amplifies S0 estimation error
  through K0 = 6c/D0² and diverges. The alternating-update machinery itself is
  faithful to Eqs 5-6.
- `workers` is accepted for signature compatibility but unused (numpy already
  vectorizes the work); progress is reported per iteration, not per slice.
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

from .common import save_nii

# Project-wide parameter conventions (same clipping as the other DKI models).
D_MAX = 0.05
K_MAX = 10.0
EPS = 1e-12

MAX_ITER = 30
TOL = 1e-4


def _group_means(dataobj, groups):
    """Replicate common.fit_volume preprocessing: mean signal per b-group."""
    x, y, z, t = dataobj.shape
    pav = np.zeros((x, y, z, len(groups)), np.float32)
    for k in range(z):
        sl = np.asarray(dataobj[:, :, k, :], np.float32)
        for i, idx in enumerate(groups):
            pav[:, :, k, i] = np.mean(sl[..., idx], axis=-1)
    return pav


def _init_loglinear(sig, b):
    """
    Robust joint initialization: one-shot log-linear DKI least squares
    (same linearization as dki2d_fast): ln S = beta0 - b*D + b^2*beta2,
    with beta2 = D^2*K/3 -> K = 3*beta2/D^2.

    Used instead of the paper's literal 3-point init because we have no
    measured baseline: the 3-point K0 divides by D0^2 and amplifies any S0
    estimation error into divergence. Returns s0, d, k arrays.
    """
    n = len(b)
    sb, sb2, sb3, sb4 = float(b.sum()), float((b ** 2).sum()), float((b ** 3).sum()), float((b ** 4).sum())
    G = np.array([
        [n, -sb, sb2 / 2],
        [-sb, sb2, -sb3 / 2],
        [sb2 / 2, -sb3 / 2, sb4 / 4],
    ], float)
    y = np.log(np.maximum(sig, EPS))
    rhs = np.stack([y.sum(axis=1),
                    -(y * b).sum(axis=1),
                    0.5 * ((y * b * b)).sum(axis=1)], axis=1)
    beta = rhs @ np.linalg.inv(G).T
    s0 = np.exp(beta[:, 0])
    d = beta[:, 1]
    k = 3.0 * beta[:, 2] / np.maximum(d * d, EPS)
    return s0, d, k


def _clip_d(d):
    return np.clip(d, 0.0, D_MAX)


def _clip_k(k, d, bmax, constrained):
    if not constrained:
        return k
    upper = np.minimum(K_MAX, 3.0 / (bmax * np.maximum(d, EPS)))
    return np.clip(k, 0.0, upper)


def _ais_core(sig, b, variant, progress_callback=None, smooth_hook=None):
    """
    Vectorized AIS over an (Nvox, Nb) group-averaged signal array.

    variant: 'u' | 'c' | 'sc'.  For 'sc', smooth_hook(vector)->vector applies the
    paper's temporary Gaussian-smoothed ADC copy at map level each iteration.
    Returns s0, d, k, residual arrays shaped (Nvox,).
    """
    constrained = variant in ("c", "sc")
    bmax = float(b.max())
    w = sig ** 2                      # paper weights wn = S(bn)^2
    lnsig = np.log(np.maximum(sig, EPS))

    # ── Initialization ────────────────────────────────────────────────────
    s0, d_init, k_init = _init_loglinear(sig, b)
    d = _clip_d(d_init)
    k = np.clip(k_init, 0.0, K_MAX)
    # Paper equations operate on ln(S/S0); S0 is fixed after init.
    lnrel = np.log(np.maximum(sig, EPS)) - np.log(np.maximum(s0, EPS))[:, None]

    total = MAX_ITER
    if progress_callback is not None:
        progress_callback(1, total)

    swb2 = np.maximum((w * b * b).sum(axis=1), EPS)      # denominator of Eq 5
    swb4 = np.maximum((w * b ** 4).sum(axis=1), EPS)     # denominator of Eq 6

    # ── Alternating closed-form updates (Eqs 5-6) until convergence ──────
    for it in range(MAX_ITER):
        if progress_callback is not None and 1 < it + 1 < total:
            progress_callback(1 + int(((it + 1) / MAX_ITER) * (total - 2)), total)

        # Update D (Eq 5): old quadratic term frozen (Taylor approximation).
        # The quadratic term of the DKI log-model is b^2*D^2*K/6 -> d*d*k.
        quad = (b * b / 6.0) * (d * d * k)[:, None]
        d_new = (w * b * (-lnrel + quad)).sum(axis=1) / swb2
        d_new = _clip_d(d_new)

        # Update K (Eq 6): SCAIS substitutes the smoothed temporary copy here
        d_for_k = smooth_hook(d_new) if smooth_hook is not None else d_new
        num = (w * (6.0 * b * b) * (lnrel + b * d_for_k[:, None])).sum(axis=1)
        k_new = num / (swb4 * np.maximum(d_for_k ** 2, EPS))
        k_new = _clip_k(k_new, d_new, bmax, constrained)

        delta = max(float(np.abs(d_new - d).max()), float(np.abs(k_new - k).max()))
        d, k = d_new, k_new
        if delta < TOL:
            break

    if progress_callback is not None:
        progress_callback(total, total)

    # ── Residual: RMSE of the DKI prediction vs the input signal ─────────
    expo = np.clip(-b * d[:, None] + (b ** 2) * (d ** 2 * k)[:, None] / 6.0, -80, 80)
    pred = s0[:, None] * np.exp(expo)
    residual = np.sqrt(np.mean((sig - pred) ** 2, axis=1))
    return (s0.astype(np.float32), d.astype(np.float32),
            k.astype(np.float32), residual.astype(np.float32))


def _fit_dki_ais_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix,
                        variant, progress_callback=None, roi_mask=None,
                        smooth_sigma=1.0, **kwargs):
    x, y, z, t = dataobj.shape
    m = np.asarray(mask, bool)
    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, bool)
        if roi_mask.shape == (x, y, z):
            m = m & roi_mask
    b = np.asarray(uniq_b, float)
    pav = _group_means(dataobj, groups)

    maps = {name: np.zeros((x, y, z), np.float32) for name in ("s0", "d", "k", "residual")}
    idxs = np.where(m.reshape(-1))[0]
    if idxs.size == 0:
        for name in maps:
            save_nii(Path(f"{out_prefix}_{name}.nii.gz"), maps[name], affine, header)
        return maps

    sig = pav.reshape(-1, len(groups))[idxs]          # (Nvox, Nb)

    smooth_hook = None
    if variant == "sc":
        def smooth_hook(vec):
            vol = np.zeros(x * y * z, np.float32)
            vol[idxs] = vec
            vol = gaussian_filter(vol.reshape(x, y, z), sigma=smooth_sigma, mode="nearest")
            return vol.reshape(-1)[idxs]

    s0v, dv, kv, rv = _ais_core(sig, b, variant,
                                progress_callback=progress_callback,
                                smooth_hook=smooth_hook)

    def to_map(v):
        out = np.zeros(x * y * z, np.float32)
        out[idxs] = v
        return out.reshape(x, y, z)

    maps["s0"] = to_map(s0v)
    maps["d"] = to_map(dv)
    maps["k"] = to_map(kv)
    maps["residual"] = to_map(rv)

    for name in maps:
        save_nii(Path(f"{out_prefix}_{name}.nii.gz"), maps[name], affine, header)
    return maps


def fit_dki_ais_u_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    """UAIS: unconstrained alternative iteration schema (Yan et al. 2014)."""
    return _fit_dki_ais_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix,
                               "u", **kwargs)


def fit_dki_ais_c_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    """CAIS: UAIS + constraints D>=0, K>=0, K<=3/(b_max*D)."""
    return _fit_dki_ais_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix,
                               "c", **kwargs)


def fit_dki_ais_sc_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    """SCAIS: CAIS + per-iteration 3x3x3 Gaussian-smoothed temporary ADC copy."""
    return _fit_dki_ais_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix,
                               "sc", **kwargs)
