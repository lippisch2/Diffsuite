# IVIM Segmented — Intravoxel Incoherent Motion (two-stage fit)

## Equation

```
S(b) = S0 * [ f * exp(-b * D*) + (1 - f) * exp(-b * D) ]
```

Where:
- `f` — perfusion fraction (vascular volume fraction)
- `D*` — pseudo-diffusion coefficient (perfusion, mm²/s)
- `D` — true diffusion coefficient (mm²/s)

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
- **b0 required:** No (S0 is estimated from the minimum-b volume — per FireVoxel Model 17)
- **High b-values:** At least 3 b-values > 200 s/mm² (hard requirement — segmented fit uses them to derive Dt and f)
- **Low/intermediate b-values:** At least 2 b-values with 0 < b < 200 s/mm² (to constrain the fast perfusion component f / D*). Blocked (forceable) when unmet.

## Fitting Method

**Two-stage segmented approach** (FireVoxel Model 17, Sigmund et al.):
1. High-b monoexponential fit (b > 200): `log S = log(S0(1-f)) - b·Dt` → recovers Dt; intercept Si gives `f = (S0 - Si)/S0`
2. Fix Dt and f, then fit **only D\*** on the full dataset via nonlinear least squares
3. Bounds: D* ∈ [1e-4, 1.0], D ∈ [0, 0.01], f ∈ [0, 1]

## Output Files

```
{stem}_ivim_seg_s0.nii.gz
{stem}_ivim_seg_f.nii.gz
{stem}_ivim_seg_dstar.nii.gz
{stem}_ivim_seg_d.nii.gz
{stem}_ivim_seg_residual.nii.gz
```

## Notes

- "Segmented" = uses high-b values to initialize D before full fit
- More stable than full IVIM when b-value sampling is sparse
- D* represents microvascular perfusion (fast decay at low b)
- D represents tissue water diffusion (slow decay at high b)

> **⚠ S0 bias when b_min > 0:** S0 is taken from the minimum-b volume, i.e.
> `S0 ≈ S(b_min)`. This is exact only when a b=0 volume is present. When the
> lowest b-value is > 0 (e.g. b_min = 50), S(b_min) < S0 because of perfusion +
> diffusion decay, so S0 is under-estimated. Since the perfusion fraction is
> derived as `f = (S0 - Si)/S0`, this bias propagates into `f`. For accurate
> IVIM, acquire a true b=0 volume.
