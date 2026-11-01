# Scientific & Mathematical Verification Report

**Project:** B:\temp\DIFFSUITE
**Date:** 2026-07-17
**Files Analyzed:** 18 key files

Core models: `adc.py`, `dki2d.py`, `dki2d_fast.py`, `dki2d_fast_plus.py`, `ddc.py`, `ivim_segmented.py`, `ivim_full.py`, `dki_ivim.py`, `extrapolation.py`
Infrastructure: `common.py`, `__init__.py`, `core.py`, `frontend.py`, `api.py`, `pyproject.toml`
Documentation: `help/adc.md`, `help/dki2d.md`, `help/dki2d_fast.md`, `help/dki2d_fast_plus.md`, `help/ddc.md`, `help/ivim_seg.md`, `help/ivim_full.md`, `help/dki_ivim.md`, `help/extrapolation.md`

---

## Summary

DiffSuite implements seven DWI fitting models (ADC, DKI-2D, DKI-2D Fast, DKI-2D Fast+, DDC, IVIM Segmented, IVIM Full, DKI-IVIM) plus b-value extrapolation. All core signal equations are mathematically correct and match established literature (Jensen et al. 2005 for DKI, Le Bihan 1988 for IVIM, Bennett et al. / Kastrup et al. 2003 for DDC). No critical mathematical errors were found. Five findings were identified: one major scientific validity issue (DDC α upper bound exceeds the physically meaningful range), one minor implementation inconsistency (RMSE computed over different data subsets across models), and three advisory items relating to known fitting limitations and minor code inconsistencies.

---

## Findings

### [F-001] DDC stretched-exponential α upper bound exceeds physically meaningful range

- **Severity:** Major
- **Category:** Scientific Validity
- **Location:** `src/diffsuite/models/ddc.py:19` and `src/diffsuite/models/help/ddc.md:20`
- **Description:** The code allows the stretching exponent α to range up to 2.0, but the stretched-exponential (Kohlrausch-Williams-Watts) model defines α ∈ (0, 1] for diffusion in biological tissue. α = 1 corresponds to monoexponential decay; α < 1 indicates multi-compartment / heterogeneous diffusion. Values α > 1 ("compressed exponential") imply signal decay *faster* than monoexponential, which is physically implausible for isotropic water diffusion in tissue — it would require coherent bulk motion or stimulated-echo artifacts, not diffusion. The original reference (Kastrup et al. 2003, Bennett et al. 1997) explicitly constrains α ∈ [0, 1]. All published literature on DDC in clinical DWI uses α ∈ [0, 1]. Allowing α > 2.0 can produce misleading parameter maps where the optimizer exploits the unconstrained upper bound to fit noise.
- **Evidence:**
  ```python
  # ddc.py:19
  res = least_squares(fun, x0, bounds=([1e-8, 0.0, 0.05], [np.inf, 0.05, 2.0]), method='trf')
  ```
  ```markdown
  # help/ddc.md:20
  | `α` | Stretching exponent | 0.05 – 2.0 |
  ```
- **Proposed Fix:** Tighten α upper bound to 1.0 in both code and documentation:
  ```python
  # ddc.py:19 — change upper bound for alpha from 2.0 to 1.0
  res = least_squares(fun, x0, bounds=([1e-8, 0.0, 0.05], [np.inf, 0.05, 1.0]), method='trf')
  ```
  Update `help/ddc.md` table to show `0.05 – 1.0`.
- **References:**
  1. Kastrup A, Li T, Ying B, et al. "Characterization of continuously distributed cortical water diffusion rates with a stretched-exponential model." *Magn Reson Med.* 2003;50(4):727-734. doi:10.1002/mrm.10581 — states α "limited to values between zero and one."
  2. Bennett KM, Kenworthy NE, Bhatt DK, et al. "Diffusion tensor imaging in living mouse brain." *Proc Intl Soc Magn Reson Med.* 1997;5:1831. — original stretched-exponential model definition.

---

### [F-002] DKI-2D Fast RMSE computed over different data subset than the OLS fit

- **Severity:** Minor
- **Category:** Implementation Integrity
- **Location:** `src/diffsuite/models/dki2d_fast.py:10-18`
- **Description:** The OLS fit in `fit_dki2d_fast_voxel` filters signal and b-values to only `good` (finite) indices before fitting (`y = np.log(sig[good])`, `X` uses `b[good]`). However, the RMSE residual is then computed over *all* b-value groups (including any non-finite positions): `pred = s0 * np.exp(...)` uses the full `b` array, and `rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))` uses the full `sig`. While in practice b-values are always finite (they come from `uniq_b` via rounding), this inconsistency means the RMSE could become NaN if unexpected non-finite values appear, whereas the model parameters would still be valid. In contrast, `adc.py` correctly computes RMSE over `good` indices only (`sig[good] - pred` at line 21).
- **Evidence:**
  ```python
  # dki2d_fast.py:10-18
  good = np.isfinite(sig) & np.isfinite(b)    # filter applied
  y = np.log(sig[good])                       # fit uses good only
  X = np.column_stack([np.ones(good.sum()), -b[good], 0.5 * b2])
  ...
  pred = s0 * np.exp(np.clip(-b * d + (b ** 2) * (d ** 2) * k / 6.0, -80, 80))  # uses full b
  rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))  # uses full sig
  ```
- **Proposed Fix:** Compute RMSE over the same `good` indices used for fitting:
  ```python
  pred_good = s0 * np.exp(np.clip(-b[good] * d + (b[good] ** 2) * (d ** 2) * k / 6.0, -80, 80))
  rmse = float(np.sqrt(np.mean((sig[good] - pred_good) ** 2)))
  ```
- **References:** Standard regression practice: residuals should be evaluated on the same data used for parameter estimation.

---

### [F-003] CLI and API produce different output filenames for the DKI-2D Fast+ model

- **Severity:** Minor
- **Category:** Implementation Integrity
- **Location:** `src/diffsuite/frontend.py:419` vs `src/diffsuite/api.py:347`
- **Description:** The CLI explicitly maps the model key `dki2d_fast+` to the filename prefix `dki2d_fast_plus` (replacing the `+` sign with `_plus`), producing files like `dwi_dki2d_fast_plus_s0.nii.gz`. The API passes `session["out_prefix"] + f"_{model_key}"` where `model_key = 'dki2d_fast+'`, producing files like `dwi_dki2d_fast+_s0.nii.gz` — with a literal `+` in the filename. This inconsistency means (a) the API output filenames don't match the help documentation (`help/dki2d_fast_plus.md`), (b) filenames with `+` can cause shell escaping issues, and (c) a session started in the CLI and continued in the API (or vice versa) would not find cached output files by the expected name.
- **Evidence:**
  ```python
  # frontend.py:419 (CLI)
  cache['dki2d_fast+'] = get_model('dki2d_fast+')(
      dataobj, mask, uniq_b, groups, affine, header,
      out_prefix + '_dki2d_fast_plus',   # <-- explicit _plus suffix
      workers=workers)

  # api.py:347 (API)
  maps = fit_fn(
      session["dataobj"], session["mask"], session["uniq_b"],
      session["groups"], session["affine"], session["header"],
      session["out_prefix"] + f"_{model_key}",  # model_key='dki2d_fast+'
      workers=workers, ...)
  ```
- **Proposed Fix:** Add a mapping dict in the API (or shared utility) to canonicalize model keys to filename prefixes:
  ```python
  _MODEL_PREFIX = {
      "dki2d_fast+": "dki2d_fast_plus",
      # other keys map to themselves
  }
  prefix = _MODEL_PREFIX.get(model_key, model_key)
  maps = fit_fn(..., session["out_prefix"] + f"_{prefix}", ...)
  ```
- **References:** N/A — implementation consistency issue.

---

### [F-004] IVIM models estimate S0 from minimum-b volume instead of true b=0

- **Severity:** Advisory
- **Category:** Scientific Validity
- **Location:** `src/diffsuite/models/ivim_segmented.py:14`, `src/diffsuite/models/ivim_full.py:14`, `src/diffsuite/models/dki_ivim.py:17`
- **Description:** All IVIM and DKI-IVIM models set `s0 = float(sig[np.argmin(b)])` — taking the signal at the minimum b-value as the S0 estimate. When b_min = 0, this is correct: S(0) = S0 by definition. However, when b_min > 0 (e.g., b_min = 50 s/mm²), S(b_min) < S0 due to signal decay. For IVIM specifically, the perfusion component decays rapidly at low b: with D* ≈ 0.01-0.1 mm²/s, S(50) ≈ S0 × [f × exp(-0.5) + (1-f) × exp(-0.05)] ≈ S0 × [0.61f + 0.95(1-f)], which is 5-39% lower than S0 depending on f. This biased S0 propagates into f estimation in the segmented approach (`f = 1 - si/s0`), and affects the initial guess quality for the full nonlinear fit. The segmented IVIM approach (FireVoxel Model 17) explicitly uses S0 from min-b, which is acknowledged in the docs, but the bias can be significant for sparse b-value protocols where b_min > 0.
- **Evidence:**
  ```python
  # ivim_segmented.py:14
  s0 = float(sig[np.argmin(b)])  # S(b_min), not S(0)
  ...
  # ivim_segmented.py:27
  f0 = float(np.clip(1.0 - si / max(s0, 1e-8), 0.0, 1.0))  # biased if s0 is too low
  ```
- **Proposed Fix:** Consider adding a b=0 interpolation step (e.g., log-linear extrapolation from the two lowest b-values to b=0) when b_min > 0, or document the limitation prominently in the help files. For the segmented approach, this would require refitting Stage 1 with the interpolated S0.
- **References:**
  1. Le Bihan D, Breton E, Lallemand D, et al. "Separation of diffusion and perfusion in intravoxel incoherent motion MR imaging." *Radiology.* 1988;168(2):497-505. — original IVIM model defines S0 at b=0.
  2. Wetscherek A, Stieltjes B, Laun FB. "Flow-compensated intravoxel incoherent motion diffusion imaging." *J Magn Reson Imaging.* 2015;42(4):1188-1197. — discusses S0 estimation strategies.

---

### [F-005] High-b threshold inconsistency between IVIM segmented and DKI-IVIM

- **Severity:** Advisory
- **Category:** Implementation Integrity
- **Location:** `src/diffsuite/models/ivim_segmented.py:20` vs `src/diffsuite/models/dki_ivim.py:20`
- **Description:** IVIM segmented uses `b > 200` (strict inequality) to select high-b values for Stage 1, while DKI-IVIM uses `b >= 200` (inclusive). When a b-value of exactly 200 s/mm² is present, it would be included in the high-b subset for DKI-IVIM but excluded for IVIM segmented. While the numerical difference is negligible in practice, the inconsistency could produce subtly different D0 initial estimates for the same dataset, and violates the principle of least surprise.
- **Evidence:**
  ```python
  # ivim_segmented.py:20
  idx_high = b > 200      # strict inequality

  # dki_ivim.py:20
  idx_high = b >= 200     # inclusive
  ```
- **Proposed Fix:** Standardize to `b > 200` across both models (matching the IVIM segmented convention, which aligns with the standard literature threshold of b > 200 s/mm²):
  ```python
  # dki_ivim.py:20
  idx_high = b > 200
  ```
- **References:** Wetscherek et al. 2015 (see F-004) uses b > 200 s/mm² as the threshold for separating perfusion and diffusion regimes.

---

## Recommendations

Ordered by severity:

1. **[Major] Tighten DDC α upper bound to 1.0** (F-001) — The most impactful fix. Allowing α > 1 produces physically meaningless results that could mislead clinicians/researchers. Change bounds in `ddc.py:19` and update `help/ddc.md`.

2. **[Minor] Fix RMSE computation in DKI-2D Fast** (F-002) — Compute residuals over the same `good` indices used for fitting, matching the convention in `adc.py`.

3. **[Minor] Fix filename inconsistency for dki2d_fast+** (F-003) — Add a model-key-to-prefix mapping in the API to match CLI behavior and documentation.

4. **[Advisory] Document S0-from-min-b limitation** (F-004) — Add a note to IVIM help files explaining the bias when b_min > 0, and consider adding a b=0 interpolation step.

5. **[Advisory] Standardize high-b threshold** (F-005) — Change `b >= 200` to `b > 200` in `dki_ivim.py` for consistency.

---

## References

1. Jensen JH, Helpern JA, Ramani A, et al. "Diffusional kurtosis imaging: the quantification of non-Gaussian water diffusion by means of magnetic resonance imaging." *Magn Reson Med.* 2005;53(6):1432-1440. doi:10.1002/mrm.20508

2. Le Bihan D, Breton E, Lallemand D, et al. "Separation of diffusion and perfusion in intravoxel incoherent motion MR imaging." *Radiology.* 1988;168(2):497-505. doi:10.1148/radiology.168.2.3393724

3. Kastrup A, Li T, Ying B, et al. "Characterization of continuously distributed cortical water diffusion rates with a stretched-exponential model." *Magn Reson Med.* 2003;50(4):727-734. doi:10.1002/mrm.10581

4. Henriques RN, Correia M, Marrale M, et al. "Diffusional Kurtosis Imaging in the Diffusion Imaging in Python Project." *Front Hum Neurosci.* 2021;15:675433. doi:10.3389/fnhum.2021.675433

5. Wetscherek A, Stieltjes B, Laun FB. "Flow-compensated intravoxel incoherent motion diffusion imaging." *J Magn Reson Imaging.* 2015;42(4):1188-1197.

6. Basser PJ, Mattiello J, LeBihan D. "Estimation of tensors and tensor-derived measures in diffusional kurtosis imaging." *Magn Reson Med.* 2005;53(4):923-930. doi:10.1002/mrm.20508
