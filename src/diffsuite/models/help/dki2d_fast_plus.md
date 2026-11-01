# 2D DKI fast+ — Linearized DKI with Skewness

## Equation

```
S(b) = S0 * exp(-b*D + (b² * D² * K) / 6)
```

Extends DKI fast by adding a statistical skewness metric of the signal.

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `D` | Diffusion coefficient | 0 – 0.05 mm²/s |
| `K` | Kurtosis | 0 – 10 |
| `skew` | Signal skewness (3rd moment) | any real |
| `fused_skew_k` | 0.5 * (skew + K) | any real |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 3 b-values
- **b0 required:** No
- **Distinct b-values:** At least 3 distinct values

## Fitting Method

1. Run DKI fast to get S0, D, K
2. Compute signal skewness: `skew = mean(((x - μ)/σ)³)`
3. Fused metric: `fused = 0.5 * (skew + K)`

## Output Files

```
{stem}_dki2d_fast_plus_s0.nii.gz
{stem}_dki2d_fast_plus_d.nii.gz
{stem}_dki2d_fast_plus_k.nii.gz
{stem}_dki2d_fast_plus_skew.nii.gz
{stem}_dki2d_fast_plus_fused_skew_k.nii.gz
{stem}_dki2d_fast_plus_residual.nii.gz
```

## Notes

- Adds signal distribution asymmetry information
- Useful for detecting heterogeneous tissue
- `fused_skew_k` combines kurtosis (non-Gaussianity) with raw skewness
