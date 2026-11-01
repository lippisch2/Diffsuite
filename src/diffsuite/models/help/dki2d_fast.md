# 2D DKI fast — Linearized Diffusion Kurtosis

## Equation

```
S(b) = S0 * exp(-b*D + (b² * D² * K) / 6)
```

Same as full 2D DKI, but fitted via linearization.

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `D` | Diffusion coefficient | 0 – 0.05 mm²/s |
| `K` | Kurtosis | 0 – 10 |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 3 b-values
- **b0 required:** No
- **Distinct b-values:** At least 3 distinct values

## Fitting Method

Linearized via log transform:

```
ln(S) = ln(S0) - b*D + 0.5 * b² * (D² * K / 3)
```

Solved with ordinary least squares. K derived from the quadratic coefficient:

```
K = 3 * beta₂ / D²
```

## Output Files

```
{stem}_dki2d_fast_s0.nii.gz
{stem}_dki2d_fast_d.nii.gz
{stem}_dki2d_fast_k.nii.gz
{stem}_dki2d_fast_residual.nii.gz
```

## Notes

- ~10x faster than full nonlinear DKI
- Slightly less accurate, good for quick screening
- K clipped to [0, 10]
- Recommended default for large datasets
