# AKC-ADC(i) — Secant-based DKI approximation

## Model

S(b) = S₀ · exp(−b·D + b²·D²·K/6)

Same DKI signal model as dki2d/dki2d_fast/AIS, but estimated via a different (algebraically equivalent) route.

## Method

1. Compute secant ADC between consecutive b-value pairs:
   ```
   ADC_secant(i) = −[ln S(b_{i+1}) − ln S(b_i)] / [b_{i+1} − b_i]
   ```
2. Assign each secant ADC to its midpoint b-value:
   ```
   b_mid(i) = (b_i + b_{i+1}) / 2
   ```
3. Weighted linear regression of ADC vs b_mid:
   ```
   ADC(b_mid) ≈ D − (D²·K/3) · b_mid
   ```
   - Intercept → D
   - Slope → K = −3·slope/D²

## Parameters

| Output | Description |
|--------|-------------|
| S₀ | Signal at b=0 (estimated from minimum-b volume) |
| D | Mean diffusivity (mm²/s) |
| K | Mean kurtosis (dimensionless) |
| residual | RMSE of the secant ADC linear fit |

## Requirements

- ≥3 distinct b-values
- b₀ NOT required (S₀ from minimum-b volume)

## Relationship to other DKI models

This is **mathematically equivalent** to the log-quadratic / WLLS closed-form DKI estimator (Masutani & Aoki 2014, bodymritools `getAdcAkc.py`). The value lies in:

- **Transparency**: the secant → regression pipeline is easy to inspect and debug
- **Fast screening**: useful as a QC check before running more expensive fits
- **Bias/variance characterization**: the slope analysis reveals b-value sensitivity

## References

- Masutani Y, Aoki S. (2014) "Rapid and Simple Measurement of Diffusion Kurtosis Imaging Using Routine MRI." *MRMS* 13(1):47-53.
- Koh D-M. (2015) "Diffusion-weighted MRI for the detection of prostate cancer." *JMRI* 42:265-276.
