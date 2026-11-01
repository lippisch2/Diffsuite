# FROC — Fractional Order Calculus (Experimental)

## Equation

```
S(b) / S0 = exp[ -D · μ^(2(β−1)) · b^β ]
```

Where:
- `D` — diffusion coefficient (mm²/s)
- `β` — fractional order derivative in space (0 < β ≤ 1)
- `μ` — spatial constant (µm)

This is the **reduced, b-value-only** form. The full fractional-order-calculus
expression includes the Stejskal-Tanner gradient-timing constants
`(γ·G_d·δ)^(2β)·(Δ − (2β−1)/(2β+1)·δ)`; those collapse into the measured
b-value, so only the fitted parameters `S0, D, β, μ` remain. At `β=1` the
model reduces exactly to the monoexponential `S(b)=S0·exp(−b·D)`.

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `D` | Diffusion coefficient | 0 – 0.05 mm²/s |
| `β` | Fractional order (space) | 0 – 1 |
| `μ` | Spatial constant | > 0 |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 5 b-values
- **b_max:** ≥ 2000 s/mm²
- **b0 required:** Not strictly (S0 init from min-b volume)

## Fitting Method

Voxel-wise nonlinear least squares (`scipy.optimize.least_squares`, TRF),
initialized with an ADC log-linear estimate, with `μ > 0` and `0 < β ≤ 1` bounds.

> **EXPERIMENTAL model** — listed under the "Experimental" category. The
> reduced form ignores explicit gradient-timing (γ, G_d, δ, Δ) inputs, which the
> full model would require. Interpret with caution.

## Output Files

```
{stem}_froc_s0.nii.gz
{stem}_froc_d.nii.gz
{stem}_froc_beta.nii.gz
{stem}_froc_mu.nii.gz
{stem}_froc_residual.nii.gz
```
