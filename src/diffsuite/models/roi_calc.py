"""Quick semi-auto ROI calculation for DiffSuite.

The user measures mean signal intensity within an ROI on each b-value map in
any external viewer, enters the (b, signal) pairs, picks a model, and DiffSuite
fits that model's parameters to the scalar ROI mean values.

This is file-independent: it operates purely on the entered numbers and reuses
the exact same voxel fitters as the full-volume fits, so parameters are
consistent with a per-voxel fit for the same underlying signal.

DDVD is a special case: it is not a least-squares fit but a direct signal
difference. In scalar (ROI-mean) mode there is no spatial smoothing or
registration (the ROI mean is already averaged and there is no image geometry),
so only the analytic formulas are applied.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np

from .adc import fit_adc_voxel
from .dki2d import fit_dki2d_voxel
from .ddc import fit_ddc_voxel
from .ivim_full import fit_ivim_full_voxel
from .ivim_segmented import fit_ivim_segmented_voxel
from .dki_ivim import fit_dki_ivim_voxel


# Param label -> display + unit, in output order.
ROI_CALC_MODELS: Dict[str, Dict] = {
    "adc": {
        "label": "ADC",
        "formula": "S(b)=S₀·e^(−b·ADC)",
        "requirement": "≥2 b-values",
        "params": [
            ("S0", "signal units"),
            ("ADC", "mm²/s"),
            ("residual (RMSE)", "signal units"),
        ],
    },
    "dki2d": {
        "label": "2D DKI",
        "formula": "S(b)=S₀·e^(−bD+b²D²K/6)",
        "requirement": "≥3 distinct b-values",
        "params": [
            ("S0", "signal units"),
            ("D", "mm²/s"),
            ("K", "dimensionless"),
            ("residual (RMSE)", "signal units"),
        ],
    },
    "ddc": {
        "label": "DDC",
        "formula": "S(b)=S₀·e^(−(b·DDC)^α)",
        "requirement": "≥3 b-values incl. b0",
        "params": [
            ("S0", "signal units"),
            ("DDC", "mm²/s"),
            ("α", "dimensionless"),
            ("residual (RMSE)", "signal units"),
        ],
    },
    "ivim_full": {
        "label": "IVIM full",
        "formula": "S(b)=S₀[(1−f)e^(−bD)+f·e^(−bD*)]",
        "requirement": "≥4 b-values, ≥2 b in (0,200)",
        "params": [
            ("S0", "signal units"),
            ("f", "dimensionless (0–1)"),
            ("D*", "mm²/s"),
            ("D", "mm²/s"),
            ("residual (RMSE)", "signal units"),
        ],
    },
    "ivim_seg": {
        "label": "IVIM segmented",
        "formula": "S(b)=S₀[(1−f)e^(−bD)+f·e^(−bD*)]",
        "requirement": "≥4 b-values, ≥2 b in (0,200), ≥3 b>200",
        "params": [
            ("S0", "signal units"),
            ("f", "dimensionless (0–1)"),
            ("D*", "mm²/s"),
            ("D", "mm²/s"),
            ("residual (RMSE)", "signal units"),
        ],
    },
    "dki_ivim": {
        "label": "DKI-IVIM",
        "formula": "S(b)=S₀[f·e^(−bD*)+(1−f)e^(−bD+b²D²K/6)]",
        "requirement": "≥5 b-values, ≥2 b in (0,200)",
        "params": [
            ("S0", "signal units"),
            ("f", "dimensionless (0–1)"),
            ("D*", "mm²/s"),
            ("D", "mm²/s"),
            ("K", "dimensionless"),
            ("residual (RMSE)", "signal units"),
        ],
    },
    "ddvd": {
        "label": "DDVD",
        "formula": "DDVD=S(b0)−S(b_low); nddvd=DDVD/S(b0)·100",
        "requirement": "b0 + ≥1 b in [10,50] s/mm²",
        "params": [
            ("s0", "signal units"),
            ("ddvd", "signal units"),
            ("nddvd", "%"),
        ],
    },
}


# model key -> (voxel fitter, tuple of param labels in return order)
_ROI_FITTERS: Dict[str, Tuple[Callable, Tuple[str, ...]]] = {
    "adc": (fit_adc_voxel, ("S0", "ADC", "residual (RMSE)")),
    "dki2d": (fit_dki2d_voxel, ("S0", "D", "K", "residual (RMSE)")),
    "ddc": (fit_ddc_voxel, ("S0", "DDC", "α", "residual (RMSE)")),
    "ivim_full": (fit_ivim_full_voxel, ("S0", "f", "D*", "D", "residual (RMSE)")),
    "ivim_seg": (fit_ivim_segmented_voxel, ("S0", "f", "D*", "D", "residual (RMSE)")),
    "dki_ivim": (fit_dki_ivim_voxel, ("S0", "f", "D*", "D", "K", "residual (RMSE)")),
}


def _check_requirements(model_key: str, bvals: np.ndarray) -> List[str]:
    """Return human-readable reasons the b-value scheme is insufficient (empty if ok)."""
    uniq = np.unique(np.round(bvals).astype(int))
    n = len(bvals)
    n_low = int(np.sum((bvals > 0) & (bvals < 200)))
    n_high = int(np.sum(bvals > 200))
    n_distinct = len(uniq)
    has_b0 = bool(np.any(bvals == 0))

    if model_key == "adc":
        return [] if n >= 2 else ["need ≥2 b-values"]
    if model_key == "dki2d":
        return [] if n >= 3 and n_distinct >= 3 else ["need ≥3 distinct b-values"]
    if model_key == "ddc":
        parts = []
        if n < 3:
            parts.append("need ≥3 b-values")
        if not has_b0:
            parts.append("need a b0 (b=0)")
        return parts
    if model_key in ("ivim_full", "dki_ivim"):
        need = 5 if model_key == "dki_ivim" else 4
        parts = []
        if n < need:
            parts.append(f"need ≥{need} b-values")
        if n_low < 2:
            parts.append("need ≥2 b in (0,200)")
        return parts
    if model_key == "ivim_seg":
        parts = []
        if n < 4:
            parts.append("need ≥4 b-values")
        if n_low < 2:
            parts.append("need ≥2 b in (0,200)")
        if n_high < 3:
            parts.append("need ≥3 b>200")
        return parts
    if model_key == "ddvd":
        low = [bb for bb in uniq if 10 <= bb <= 50]
        parts = []
        if not has_b0:
            parts.append("need a b0 (b=0)")
        if not low:
            parts.append("need ≥1 b in [10,50] s/mm²")
        return parts
    return ["unsupported model"]


def _calc_ddvd(signals: np.ndarray, bvals: np.ndarray):
    """Analytic scalar DDVD from ROI-mean signals (no smoothing / registration)."""
    uniq = np.unique(np.round(bvals).astype(int))
    low = [bb for bb in uniq if 10 <= bb <= 50]
    b0_val = int(uniq[0])  # b0 is the minimum b
    s0 = float(np.mean(signals[bvals == b0_val])) if np.any(bvals == b0_val) else float(signals[np.argmin(bvals)])

    low_list = sorted(low)
    s_low1 = float(np.mean(signals[bvals == low_list[0]])) if np.any(bvals == low_list[0]) else 0.0
    ddvd = s0 - s_low1
    nddvd = (ddvd / s0 * 100.0) if s0 > 0 else 0.0
    out = {"s0": s0, "ddvd": ddvd, "nddvd": nddvd}
    if len(low_list) >= 2:
        s_low2 = float(np.mean(signals[bvals == low_list[1]])) if np.any(bvals == low_list[1]) else 0.0
        out["ddvd_diff"] = s0 - 2.0 * s_low1 + s_low2
    return out


def calc_roi(model_key: str, bvals, signals):
    """Fit a curated model to scalar ROI-mean signals.

    Args:
        model_key: registry key (adc, dki2d, ddc, ivim_full, ivim_seg, dki_ivim, ddvd).
        bvals: b-values matching the entered signals.
        signals: ROI-mean signal at each b-value (same length as bvals).

    Returns:
        dict {model_key, label, params: [{name, value, unit}...], residual, notes}.
        raises ValueError on unsupported model, length mismatch, or unmet requirements.
    """
    if model_key not in ROI_CALC_MODELS:
        raise ValueError(f"Unsupported model '{model_key}' for ROI calculation")

    bvals = np.asarray([float(x) for x in bvals], float)
    signals = np.asarray([float(x) for x in signals], float)
    if bvals.ndim != 1 or signals.ndim != 1:
        raise ValueError("b-values and signals must be 1-D lists")
    if len(bvals) != len(signals):
        raise ValueError(f"b-values ({len(bvals)}) and signals ({len(signals)}) must be the same length")
    if len(signals) < 2:
        raise ValueError("Need at least 2 data points")

    reasons = _check_requirements(model_key, bvals)
    if reasons:
        raise ValueError("Insufficient b-value scheme: " + "; ".join(reasons))

    if model_key == "ddvd":
        vals = _calc_ddvd(signals, bvals)
        units = {"s0": "signal units", "ddvd": "signal units",
                 "nddvd": "%", "ddvd_diff": "signal units"}
        params = [{"name": k, "value": round(float(v), 6), "unit": units.get(k, "")}
                  for k, v in vals.items()]
        residual = 0.0
    else:
        fitter, labels = _ROI_FITTERS[model_key]
        units = dict((n, u) for n, u in ROI_CALC_MODELS[model_key]["params"])
        try:
            result = fitter(signals, bvals)
        except Exception as e:  # fit fitters guard internally, but be safe
            raise ValueError(f"Calculation failed: {e}")
        params = []
        for name, val in zip(labels, result):
            is_res = name == "residual (RMSE)"
            params.append({
                "name": name,
                "value": round(float(val), 6) if not is_res else round(float(val), 4),
                "unit": units.get(name, ""),
            })
        residual = float(result[-1])

    return {
        "model_key": model_key,
        "label": ROI_CALC_MODELS[model_key]["label"],
        "formula": ROI_CALC_MODELS[model_key]["formula"],
        "params": params,
        "residual": round(residual, 4),
    }
