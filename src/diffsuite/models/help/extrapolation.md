# Extrapolation — Synthetic Signal at Target b-value

## Purpose

Generate a synthetic DWI volume at a higher b-value using a previously fitted model.

## How it works

After fitting any model, you can extrapolate the signal to a target b-value `b_target`:

| Source Model | Extrapolation Formula |
|--------------|----------------------|
| `adc` | `S0 * exp(-b_target * ADC)` |
| `dki2d`, `dki2d_fast`, `dki2d_fast+` | `S0 * exp(-b_target*D + b_target²*D²*K/6)` |
| `ddc` | `S0 * exp(-(b_target * DDC)^α)` |
| `ivim_seg`, `ivim_full` | `S0 * [f*exp(-b_target*D*) + (1-f)*exp(-b_target*D)]` |
| `dki_ivim` | `S0 * [f*exp(-b_target*D*) + (1-f)*exp(-b_target*D + b_target²*D²*K/6)]` |

## Requirements

- Must fit the source model **first** in the same session
- The fitted parameter maps are cached in memory
- Target b-value must be higher than the maximum fitted b-value (extrapolation, not interpolation)

## Output File

```
{stem}_synthetic_b{TARGET}_{model}.nii.gz
```

Example: `dwi_synthetic_b2000_dki2d.nii.gz`

## Usage in CLI

```
Enter model number: 9  (Extrapolate)
Choose source fitted model:
  1 - ADC
  2 - 2D DKI
  ...
Enter target higher b-value: 2000
```

## Notes

- Useful for simulating high-b acquisitions without scanning
- Accuracy depends on how well the source model fits the data
- Extrapolating far beyond the fitted range increases uncertainty
