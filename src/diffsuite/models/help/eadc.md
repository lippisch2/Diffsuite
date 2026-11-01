# eADC — Exponential Apparent Diffusion Coefficient

## Equation

```
eADC = exp(-ADC * b_ref)     with  b_ref = b_max
```

Where:
- `ADC` — apparent diffusion coefficient (mm²/s), from the monoexponential fit
- `b_ref` — reference b-value, taken as the highest b-value in the acquisition

eADC is the **fraction of signal remaining** at the reference b-value. It is
dimensionless (typically 0–1) and shows the **opposite contrast** to ADC: it is
higher in tissue with restricted diffusion (more residual signal at high b).

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal (fit intercept at b=0) | > 0 |
| `ADC` | Apparent diffusion coefficient (same as the ADC model) | 0 – 0.05 mm²/s |
| `eADC` | Fraction of signal remaining at b_ref | 0 – 1 |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 2 b-values (same as ADC)
- **b0 required:** No (b0 optional, helps S0 estimation)
- **Distinct b-values:** At least 2 distinct values

## Fitting Method

Identical to the ADC model — linearized via natural log and solved with OLS —
then `eADC` is computed from the fitted `ADC` at the maximum b-value.

## Output Files

```
{stem}_eadc_s0.nii.gz
{stem}_eadc_adc.nii.gz
{stem}_eadc_eadc.nii.gz
{stem}_eadc_residual.nii.gz
```

## Notes

- Belongs to the same **ADC** family in the GUI (grouped with ADC, ≥2 b-values)
- eADC is redundant with ADC (a deterministic transform of it), but clinicians
  often prefer it because restricted diffusion appears bright, opposite to ADC
- Choosing `b_ref = b_max` gives the full dynamic range; other conventions (e.g.
  a fixed 1000 s/mm²) exist in the literature
