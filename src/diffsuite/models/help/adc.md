# ADC — Monoexponential Model

## Equation

```
S(b) = S0 * exp(-b * ADC)
```

Where:
- `S(b)` — signal at b-value `b`
- `S0` — signal at b=0 (proton density)
- `ADC` — apparent diffusion coefficient (mm²/s)

## Parameters

| Param | Description | Typical Range |
|-------|-------------|---------------|
| `S0` | Baseline signal | > 0 |
| `ADC` | Apparent diffusion coefficient | 0 – 0.05 mm²/s |
| `residual` | RMSE of fit | ≥ 0 |

## B-value Requirements

- **Minimum:** 2 b-values
- **b0 required:** No (b0 optional, helps S0 estimation)
- **Distinct b-values:** At least 2 distinct values

## Fitting Method

Linearized via natural log transform:

```
ln(S) = ln(S0) - b * ADC
```

Solved with ordinary least squares (`np.linalg.lstsq`).

## Output Files

```
{stem}_adc_s0.nii.gz
{stem}_adc_adc.nii.gz
{stem}_adc_residual.nii.gz
```

## Notes

- Fastest model, good baseline for monoexponential decay
- When b0 is absent, S0 is taken from the minimum-b volume
- ADC is clipped to [0, 0.05] mm²/s to avoid non-physical values
