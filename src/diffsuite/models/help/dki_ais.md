# DKI AIS — Alternative Iteration Schema (UAIS / CAIS / SCAIS)

Fast DKI parameter estimation after Yan et al. 2014.

## Equation

```
S(b) = S0 * exp(-b*D + b^2 * D^2 * K / 6)
```

Where:
- `S0` — baseline signal at b=0
- `D` — apparent diffusion coefficient (mm²/s if b is in s/mm²)
- `K` — apparent kurtosis coefficient (dimensionless)

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `s0` | Baseline signal | > 0 |
| `d` | Diffusion coefficient | 0 – 0.05 mm²/s |
| `k` | Kurtosis excess | 0 – 3 (tissue) |
| `residual` | RMSE of fit | ≥ 0 |

## Variants

| Key | Name | Description |
|-----|------|-------------|
| `dki_ais_u` | UAIS | Unconstrained alternating iteration |
| `dki_ais_c` | CAIS | Adds constraints per iteration: `D ≥ 0`, `K ≥ 0`, `K ≤ 3/(b_max·D)` |
| `dki_ais_sc` | SCAIS | CAIS + temporary 3×3×3 Gaussian-smoothed ADC copy used inside every K update |

## Algorithm

1. **Init**: joint one-shot log-linear least squares (same linearization as
   `dki2d_fast`) for S0/D/K. (The paper's literal 3-point init assumes a
   measured baseline and diverges when S0 must be estimated — see module docstring.)
2. **Iterate until convergence** (|Δ|<1e-4 or 30 iterations):
   - Update D by weighted linear least squares with the old quadratic term frozen
   - Apply constraints (C/SC)
   - (SC only) smooth a *temporary copy* of the current D map (3×3×3 Gaussian)
   - Update K analytically using the new D (SC: smoothed copy)
   - Apply constraints
3. **Residual**: RMSE of the final DKI prediction vs the group-averaged signal.

Weights follow the paper: `wn = S(bn)²`.

## B-value Requirements

- **Minimum:** 3 distinct b-values
- **b0 required:** No (S0 estimated as log-linear intercept when absent)

## Implementation notes

- Volume-level vectorized numpy (all voxels simultaneously) instead of the
  per-voxel `fit_volume` loop — this is what makes AIS fast. Preprocessing,
  ROI handling, output filenames and returned maps match the other models.
- `workers` is accepted but unused; progress is reported per iteration.
- Reference: Yan X, Zhou M, Ying L, Liu W, Yang G, Wu D, Zhou Y, Peterson BS,
  Xu D. A fast schema for parameter estimation in diffusion kurtosis imaging.
  Comput Med Imaging Graph. 2014;38(6):469-480.
