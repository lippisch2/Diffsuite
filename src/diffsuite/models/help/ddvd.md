# DDVD — Diffusion-Derived Vessel Density

Ni CX et al. (2025) BMC Med Imaging 25:79.

## Signal model

DDVD is not a parametric signal model — it is a direct signal difference
metric that quantifies microvasculature density via IVIM pseudo-diffusion.

### DDVD (basic)

    DDVD = S(b0) − S(b_low)   [au/pixel]

where b_low ∈ (10, 50] s/mm². Lower b_low → more IVIM-sensitive.

### DDVD_diff (diffusion-corrected)

    DDVD_diff = [S(b0) − S(b_low1)] − [S(b_low1) − S(b_low2)]
              = S(b0) − 2·S(b_low1) + S(b_low2)

Second-order difference that removes the linear tissue diffusion component.

### nDDVD (normalized)

    nDDVD = [S(b0) − S(b_low)] / S(b0) × 100   [%]

Fractional signal drop, independent of absolute signal intensity.

## Preprocessing

Each b-volume is smoothed with a 3×3×3 uniform box kernel before
differencing (replicates the paper's ROI-mean paradigm as a sliding window).

## Optional motion correction (registration)

DDVD assumes the b0 and low-b volumes are spatially aligned. If the patient
moves between those acquisitions, the differencing produces spurious
vessel-density signal at tissue/fluid boundaries. With **register=True**,
each low-b volume is registered to the b0 volume before differencing using
SimpleITK **rigid** registration (3 rotations + 3 translations — the robust
standard for within-exam motion; chosen over affine because affine can settle
into local minima on smooth DWI volumes). This is exposed in the GUI as a
"Register b0 ↔ low-b (motion correction)" checkbox shown only for the DDVD
model.

Requires `SimpleITK>=2.0`. If it is not installed, the fit falls back to
unregistered volumes and prints a warning — it never fails.

When registration is on, an additional **resample** control (default on) chooses
whether each registered low-b volume is resampled onto the b0 grid (`True`) or
kept on its native grid (`False`). For a single 4D DWI the grids already match,
so the two differ only in how the rigid transform is applied; keeping the
native grid avoids any interpolation onto a different lattice.

## b-value requirements

- **b=0** (b ≤ 10 s/mm²) — required
- **One or more b in (10, 50] s/mm²** — required
- Optimal: b=0, 10, 20 (Ni et al. protocol)
- Warning issued when b_low > 30 (tissue diffusion contamination increases)

## Outputs

| Map | Description | Units |
|-----|-------------|-------|
| s0 | Smoothed b=0 signal | a.u. |
| ddvd | S(b0) − S(b_low) | a.u./pixel |
| nddvd | Fractional signal drop | % |
| ddvd_diff | Diffusion-corrected (if 2+ low-b) | a.u./pixel |
| residual | 0 (analytical) | — |

## Clinical context

DDVD is a perfusion biomarker. Higher DDVD indicates greater microvasculature
density. Validated for glioma IDH genotyping (IDH-mutant lower DDVD than
IDH-wildtype) and Ki-67 proliferation correlation.
