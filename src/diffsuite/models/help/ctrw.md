# CTRW — Continuous-Time Random Walk (Experimental)

## Equation

```
S(b) / S0 = E_α[ -(b·D)^β ]
```

Where:
- `E_α` — one-parameter Mittag-Leffler function of order `α`
- `D` — anomalous diffusion coefficient (mm²/s)
- `α` — temporal diffusion heterogeneity index
- `β` — spatial diffusion heterogeneity index

When `α=β=1` the model reduces to the monoexponential/Gaussian case.
CTRW generalizes Brownian motion to anomalous diffusion with both temporal
(α) and spatial (β) fractional orders.

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `D` | Anomalous diffusion coefficient | 0 – 0.05 mm²/s |
| `α` | Temporal diffusion heterogeneity index | 0 – 1 |
| `β` | Spatial diffusion heterogeneity index | 0 – 1 |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 5 b-values
- **b_max:** ≥ 2000 s/mm²
- **b0 required:** Not strictly (S0 init from min-b volume)

These dense, high-b acquisitions are needed to resolve the fractional orders.

## Fitting Method

Voxel-wise nonlinear least squares (`scipy.optimize.least_squares`, TRF),
initialized with an ADC log-linear estimate. The Mittag-Leffler function is
evaluated by its convergent series (`models/mittag_leffler.py`).

> **EXPERIMENTAL model** — listed under the "Experimental" category. The
> Mittag-Leffler series is truncated for speed, so values at very high b / low β
> may be approximate. Interpret with caution.

## Output Files

```
{stem}_ctrw_s0.nii.gz
{stem}_ctrw_d.nii.gz
{stem}_ctrw_alpha.nii.gz
{stem}_ctrw_beta.nii.gz
{stem}_ctrw_residual.nii.gz
```
