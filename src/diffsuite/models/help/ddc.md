# DDC — Stretched Exponential (Distributed Diffusion Coefficient)

## Equation

```
S(b) = S0 * exp(-(b * DDC)^α)
```

Where:
- `α` (alpha) — stretching exponent (0 < α ≤ 1, Kohlrausch-Williams-Watts)
  - α = 1 → monoexponential (single compartment)
  - α < 1 → multi-compartment / heterogeneous diffusion

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `DDC` | Distributed diffusion coefficient | 0 – 0.05 mm²/s |
| `α` | Stretching exponent | 0.05 – 1.0 |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 3 b-values
- **b0 required:** **Yes** (without b0, S0 and α are degenerate/correlated)
- **Distinct b-values:** At least 3 distinct values

## Fitting Method

Nonlinear least squares (`scipy.optimize.least_squares`):

1. Initial S0 from minimum-b volume
2. Initial DDC from ADC linear fit
3. Initial α = 0.8
4. Bounds: S0 ∈ [1e-8, ∞], DDC ∈ [0, 0.05], α ∈ [0.05, 1.0]

> **Note:** α is physically constrained to (0, 1] for the stretched-exponential
> (Kohlrausch-Williams-Watts) model. α > 1 ("compressed exponential") implies
> signal decay faster than monoexponential, which is not physically meaningful for
> water diffusion in tissue (Kastrup et al., MRM 2003).

## Output Files

```
{stem}_ddc_s0.nii.gz
{stem}_ddc_ddc.nii.gz
{stem}_ddc_alpha.nii.gz
{stem}_ddc_residual.nii.gz
```

## Notes

- Models tissue heterogeneity better than monoexponential
- Requires b0 to avoid parameter degeneracy
- α near 1 → tissue behaves like single compartment
