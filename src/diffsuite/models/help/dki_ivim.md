# DKI-IVIM — Combined Kurtosis + Perfusion Model

## Equation

```
S(b) = S0 * [ f * exp(-b * D*) + (1 - f) * exp(-b*D + (b² * D² * K) / 6) ]
```

Combines:
- **IVIM perfusion term:** `f * exp(-b * D*)` — microvascular contribution
- **DKI diffusion term:** `(1-f) * exp(-b*D + b²*D²*K/6)` — tissue with kurtosis

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `f` | Perfusion fraction | 0 – 1 |
| `D*` | Pseudo-diffusion (perfusion) | 1e-4 – 0.2 mm²/s |
| `D` | Diffusion coefficient | 0 – 0.05 mm²/s |
| `K` | Kurtosis (non-Gaussianity) | 0 – 5 |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 5 b-values
- **b0 required:** No (S0 is estimated from the minimum-b volume — same convention as IVIM segmented)
- **High b-values:** At least 2 b-values > 200 s/mm²
- **Low/intermediate b-values:** At least 2 b-values with 0 < b < 200 s/mm² (to constrain the fast perfusion f / D*). Blocked (forceable) when unmet.

## Fitting Method

Nonlinear least squares (`scipy.optimize.least_squares`):

1. Initial S0 from minimum-b volume
2. Initial D from high-b linear fit (b > 200)
3. Initial guesses: f=0.1, D*=0.01, D from step 2, K=0.7
4. Bounds:
   - S0 ∈ [1e-8, ∞], f ∈ [0, 1], D* ∈ [1e-4, 0.2], D ∈ [0, 0.05], K ∈ [0, 5]

## Output Files

```
{stem}_dki_ivim_s0.nii.gz
{stem}_dki_ivim_f.nii.gz
{stem}_dki_ivim_dstar.nii.gz
{stem}_dki_ivim_d.nii.gz
{stem}_dki_ivim_k.nii.gz
{stem}_dki_ivim_residual.nii.gz
```

## Notes

- Most comprehensive model: captures both perfusion and non-Gaussian diffusion
- Requires the most b-values (5+) for stable fitting
- Use when you have rich b-value sampling and want both perfusion + kurtosis
- Higher risk of overfitting with sparse data

> **⚠ S0 bias when b_min > 0:** The initial S0 is taken from the minimum-b volume
> (`S0 ≈ S(b_min)`), which under-estimates the true S0 when no b=0 volume is
> present. S0 is refined as a free parameter during the nonlinear fit, but a true
> b=0 volume is recommended for accurate perfusion/kurtosis separation.
