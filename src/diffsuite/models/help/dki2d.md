# 2D DKI — Diffusion Kurtosis Imaging

## Equation

```
S(b) = S0 * exp(-b*D + (b² * D² * K) / 6)
```

Where:
- `S(b)` — signal at b-value `b`
- `S0` — signal at b=0
- `D` — apparent diffusion coefficient (mm²/s)
- `K` — kurtosis coefficient (dimensionless, measures non-Gaussianity)

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `D` | Diffusion coefficient | 0 – 0.05 mm²/s |
| `K` | Kurtosis (non-Gaussianity) | 0 – 5 |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 3 b-values
- **b0 required:** No (b0 optional)
- **Distinct b-values:** At least 3 distinct values

## Fitting Method

Nonlinear least squares (`scipy.optimize.least_squares`, trust-region reflective):

1. Initial S0 from minimum-b volume
2. Initial D from ADC linear fit
3. Initial K = 0.7
4. Bounds: S0 ∈ [1e-8, ∞], D ∈ [0, 0.05], K ∈ [0, 5]

## Output Files

```
{stem}_dki2d_s0.nii.gz
{stem}_dki2d_d.nii.gz
{stem}_dki2d_k.nii.gz
{stem}_dki2d_residual.nii.gz
```

## Notes

- More accurate than DKI fast (full nonlinear optimization)
- Slower than linearized version
- Kurtosis K > 0 indicates restricted diffusion (e.g., cellular structures)
- Works without b0, unlike DDC/IVIM
