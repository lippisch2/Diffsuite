# Automatic Quality Assessment (residual-map analysis)

After every fit, DiffSuite analyzes the **residual map** and attaches a
human-readable quality verdict to the job. It is shown as a "Quality
assessment" panel in the Electron Results step.

## What it measures

Each voxel's `residual` is the RMSE between the observed signal and the fitted
model prediction, in raw signal-intensity units. Because those units vary with
scanner / weighting (T2, proton density, gain), the assessment uses a
**normalized** (unitless) relative error:

```
relative_error(voxel) = residual(voxel) / norm
```

`norm` = median foreground S0 when the model has an S0 map, else a robust proxy
for the signal scale.

## Outputs (the `quality` object on the job)

| Field | Meaning |
|-------|---------|
| `overall` | `good` / `fair` / `poor` / `unknown` verdict |
| `score` | 0..1 unitless quality (1 = best) |
| `frac_unexplained` | fraction of fitted voxels whose relative error exceeds the family threshold |
| `threshold` | family-specific unexplained-voxel threshold |
| `residual` | mean / median / p95 of the relative residual |
| `worst_slices` | top-5 slices by mean relative residual |
| `notes` | model- and b-sampling-specific hints |
| `roi` | (when an ROI is active) ROI-restricted stats |

## Verdict thresholds

- median rel. error < 0.10 → **good**
- < 0.20 → **fair**
- otherwise → **poor**

Per-voxel "unexplained" thresholds are family-aware: simpler models (ADC, DKI)
use a stricter 0.05, high-parameter models (IVIM, DKI-IVIM) 0.15, experimental
models (CTRW, FROC) 0.20. These live in `_FAMILY_EXCEEDS` in `quality.py` and
are deliberately easy to tune.

## Notes / hints

The assessor adds model-specific advice, e.g. it flags when an IVIM family model
runs with few low-b acquisitions (`f` / `D*` weakly constrained) or few total
b-values (under-determined), matching the tightened IVIM b-value requirements.

## Implementation

- Module: `src/diffsuite/models/quality.py` → `assess_fit(maps, model_key, uniq_b, roi_mask=None)`
- Wired into `api.py::_run_fit_job`; the result rides along the existing
  `GET /api/fit/{job_id}` poll (job field `quality`).
- Pure numpy; no file writes; runs in-process on the already-computed maps.
