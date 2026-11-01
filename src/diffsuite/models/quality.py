"""Automatic residual-map quality assessment for fitted DWI maps.

Every model produces a per-voxel `residual` map: the RMSE between the observed
and predicted signal, in raw signal-intensity units. Because those units vary
with scanner / weighting (T2, proton density, receiver gain), quality is judged
on a *normalized* (unitless) error.

Normalization base: median foreground S0 if available, else a robust residual
scale. This keeps the assessment intensity-independent across acquisitions.

Thresholds are model-family-aware: models with more free parameters / structural
under-determination tolerate a higher residual before being flagged, and some
emit explanatory notes (e.g. sparse low-b sampling for IVIM).
"""

import numpy as np

# A voxel is "unexplained" when its relative error exceeds the family threshold.
_FAMILY_EXCEEDS = {
    "adc": 0.05, "eadc": 0.05, "dki2d": 0.05, "dki2d_fast": 0.05,
    "akc_adci": 0.05,
    "dki_ivim": 0.15, "ivim_seg": 0.15, "ivim_full": 0.15, "ddvd": 0.15,
    "ddc": 0.15, "dki2d_fast+": 0.15,
    "dki_ais_u": 0.15, "dki_ais_c": 0.15, "dki_ais_sc": 0.15,
    "ctrw": 0.20, "froc": 0.20,
}

# Overall verdict from the median relative error.
_GOOD = 0.10
_FAIR = 0.20


def _family_exceeded(model_key):
    return _FAMILY_EXCEEDS.get(model_key, 0.15)


def _notes(model_key, uniq_b, frac_bad):
    """Human-readable hints based on model + b-sampling."""
    notes = []
    if model_key in ("ivim_full", "ivim_seg", "dki_ivim"):
        ub = np.asarray(uniq_b, float)
        n_low = int(np.sum((ub > 0) & (ub < 200)))
        if n_low < 3:
            notes.append(
                f"Few low-b acquisitions ({n_low} b<200): perfusion parameters "
                "f / D* are weakly constrained and may be unstable.")
        if len(ub) < 6:
            notes.append(
                f"Only {len(ub)} b-values: IVIM is under-determined; "
                "interpret f / D* with caution.")
    if model_key in ("ctrw", "froc") and frac_bad > 0.2:
        notes.append("Experimental model with heavy parameters; high residual "
                     "may reflect the reduced/approximate signal form.")
    return notes


def assess_fit(maps, model_key, uniq_b, roi_mask=None):
    """Return a quality-assessment dict from a fitted `maps` dict.

    Parameters
    ----------
    maps : dict[str, np.ndarray]
        Output maps from a fit (must contain 'residual').
    model_key : str
        Registry key of the model that produced the fit.
    uniq_b : list/array
        Unique (rounded) b-values used for the fit.
    roi_mask : np.ndarray or None
        If given, report ROI-restricted stats too.

    Returns
    -------
    dict with an overall verdict, a unitless 0..1 score, residual statistics,
    fraction of unexplained voxels, worst slices, and notes.
    """
    residual = np.asarray(maps.get("residual"), float)
    if residual is None or residual.size == 0:
        return {"overall": "unknown", "notes": ["No residual map available."]}

    # Foreground = fitted voxels (residual > 0). fit_volume writes residual only
    # inside the fit mask (AND-ed with ROI), so residual==0 marks background.
    fg = residual > 0.0
    if int(fg.sum()) == 0:
        return {"overall": "unknown",
                "notes": ["No fitted voxels found in the residual map."]}

    # Normalization base: median foreground S0 (intensity scale), else fall back
    # to twice the median foreground residual (a crude proxy for signal scale).
    norm = 1.0
    try:
        s0 = np.asarray(maps.get("s0"), float)
        if s0 is not None and s0.shape == residual.shape and (s0 > 0).any():
            norm = float(np.median(s0[s0 > 0]))
        else:
            norm = 2.0 * float(np.median(residual[fg]))
    except Exception:
        norm = 2.0 * float(np.median(residual[fg]))
    if not np.isfinite(norm) or norm <= 0:
        # Last resort: absolute residual median.
        norm = float(np.median(residual[fg])) or 1.0

    rel_vals = residual[fg] / norm
    med_rel = float(np.median(rel_vals))
    mean_rel = float(np.mean(rel_vals))
    p95_rel = float(np.percentile(rel_vals, 95))

    exceeded = _family_exceeded(model_key)
    bad = rel_vals > exceeded
    frac_bad = float(bad.mean()) if bad.size else 0.0

    if med_rel < _GOOD:
        overall = "good"
    elif med_rel < _FAIR:
        overall = "fair"
    else:
        overall = "poor"

    score = float(np.clip(1.0 - med_rel, 0.0, 1.0))

    # Worst slices (1-based numbering matching the GUI "Slice k/Z").
    slice_means = []
    for k in range(residual.shape[2]):
        sfg = residual[:, :, k][fg[:, :, k]]
        if sfg.size:
            slice_means.append((k + 1, float(np.mean(sfg)) / norm))
    slice_means.sort(key=lambda kv: -kv[1])
    worst = [{"slice": s, "mean_relative": round(r, 3)}
             for s, r in slice_means[:5]]

    result = {
        "overall": overall,
        "score": round(score, 3),
        "norm_base": round(norm, 4),
        "frac_unexplained": round(frac_bad, 4),
        "threshold": exceeded,
        "residual": {
            "mean": round(mean_rel, 4),
            "median": round(med_rel, 4),
            "p95": round(p95_rel, 4),
        },
        "worst_slices": worst,
        "notes": _notes(model_key, uniq_b, frac_bad),
    }

    # ROI-restricted stats when a ROI mask is supplied.
    if roi_mask is not None:
        try:
            rim = np.asarray(roi_mask, bool)
            if rim.shape == residual.shape:
                roifg = fg & rim
                if int(roifg.sum()) > 0:
                    roi_rel = residual[roifg] / norm
                    result["roi"] = {
                        "voxels": int(roifg.sum()),
                        "median": round(float(np.median(roi_rel)), 4),
                        "p95": round(float(np.percentile(roi_rel, 95)), 4),
                        "frac_unexplained": round(
                            float((roi_rel > exceeded).mean()), 4),
                    }
        except Exception:
            pass

    return result
