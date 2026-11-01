# IVIM Full — Intravoxel Incoherent Motion (single-stage fit)

## Equation

```
S(b) = S0 * [ f * exp(-b * D*) + (1 - f) * exp(-b * D) ]
```

Same equation as IVIM segmented, but fitted in a single nonlinear optimization.

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `f` | Perfusion fraction | 0 – 1 |
| `D*` | Pseudo-diffusion (perfusion) | 1e-4 – 1.0 mm²/s |
| `D` | Diffusion coefficient | 0 – 0.01 mm²/s |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 4 b-values
- **b0 required:** No (S0 is estimated from the minimum-b volume)
- **High b-values:** At least 2 b-values > 200 s/mm² (for the slow diffusion component)
- **Low/intermediate b-values:** At least 2 b-values with 0 < b < 200 s/mm² (to constrain the fast perfusion component). Without these, `f` and `D*` become unidentifiable and the fit returns spuriously high perfusion fractions. The GUI blocks IVIM when this requirement is unmet (forceable).

## Fitting Method

**Full nonlinear least squares** (FireVoxel Model 18):
1. Initial S0 from minimum-b volume
2. Initial D from ADC linear fit
3. Initial guesses: f=0.1, D*=0.01, D from step 2
4. Single `least_squares` call with bounds:
    - S0 ∈ [1e-8, ∞], f ∈ [0, 1], D* ∈ [1e-4, 1.0], D ∈ [0, 0.01]

## Output Files

```
{stem}_ivim_full_s0.nii.gz
{stem}_ivim_full_f.nii.gz
{stem}_ivim_full_dstar.nii.gz
{stem}_ivim_full_d.nii.gz
{stem}_ivim_full_residual.nii.gz
```

## Notes

- More prone to local minima than segmented version
- Use when b-value sampling is dense and well-distributed
- For sparse data, prefer IVIM segmented

> **⚠ S0 bias when b_min > 0:** The initial S0 is taken from the minimum-b volume
> (`S0 ≈ S(b_min)`), which under-estimates the true S0 when no b=0 volume is
> present. The full nonlinear fit refines S0 as a free parameter, so the impact is
> smaller than in the segmented model, but a poor initial guess can still steer the
> optimizer toward local minima. Acquire a true b=0 volume for best results.
