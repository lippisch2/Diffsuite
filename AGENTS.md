# DiffSuite

## Project Overview

Python package for DWI (Diffusion Weighted Imaging) fitting and b-value extrapolation. Medical imaging / neuroimaging tool for fitting diffusion models to 4D DWI NIfTI volumes.

## Development Setup

```bash
pip install -e .
```

Package uses `src` layout (`src/diffsuite/`). Entry point: `diffsuite` → `diffsuite.cli:main`.

## Running

```bash
diffsuite path/to/dwi.nii.gz
diffsuite path/to/dicom_folder --bvals "50 800 1400"
diffsuite dwi.nii.gz --open-roi --roi-map dki2d_fast
```

- Input: NIfTI (`.nii`/`.nii.gz`) or DICOM folder/file (requires `dcm2niix` in PATH)
- `--bvals` overrides/provides b-values if no `.bval` sidecar exists
- Interactive CLI: prompts for model selection and worker count

## Project Structure

- `src/diffsuite/`
  - `cli.py` — thin entry wrapper → `frontend.main()`
  - `frontend.py` — CLI logic, DICOM→NIfTI, interactive model selection, `compute_mask()`, `group_by_bvals()`, `propose_models()`
  - `core.py` — `run_fit()` dispatch helper
  - `api.py` — FastAPI backend (REST endpoints wrapping `frontend`/`core`/`models` for the Electron app)
  - `models/` — fitting models
    - `__init__.py` — `MODEL_REGISTRY` dict maps string key → fit function; `get_model(key)` returns a `fit_*_volume` function
    - `common.py` — shared `fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, name, func, out_names, workers=1, chunk_size=128, progress_callback=None, roi_mask=None)`. Iterates z-slices; calls `progress_callback(done, total)` per slice; ANDs `roi_mask` (bool `(x,y,z)`) into `mask`. `round_bvals()`, `save_nii()`
    - `adc.py`, `dki2d.py`, `dki2d_fast.py`, `dki2d_fast_plus.py` (named `model_dki2d_fast_plus`), `ddc.py`, `ivim_segmented.py`, `ivim_full.py`, `dki_ivim.py` — model implementations. Each `fit_*_volume(...)` ends with `workers=1, **kwargs` and forwards `**kwargs` (incl. `progress_callback`, `roi_mask`) to `fit_volume`
    - `dki_ais.py` — AIS family (`dki_ais_u`/`dki_ais_c`/`dki_ais_sc`, Yan et al. 2014). EXCEPTION to the fit_volume pattern: volume-level vectorized numpy, replicates preprocessing/ROI/output-saving itself; `workers` accepted but unused, progress per iteration. See `help/dki_ais.md`
    - `akc_adci.py` — AKC-ADC(i): secant-based DKI approximation via ADC-vs-b_mid regression. Standard `fit_volume` pattern. Same model as dki2d/dki2d_fast but different (algebraically equivalent) estimation route. See `help/akc_adci.md`
    - `ddvd.py` — Diffusion-Derived Vessel Density (Ni et al. 2025). Volume-level vectorized numpy (like dki_ais). Direct signal difference between b=0 and low-b, with 3×3×3 uniform smoothing. Outputs: s0, ddvd, nddvd, ddvd_diff* (if 2+ low-b), residual. See `help/ddvd.md`
    - `froc.py` — FROC fractional-order calculus (experimental), reduced b-value-only form `S=S0·exp(−D·μ^(2β−2)·b^β)`, outputs s0,d,beta,mu,residual. `fit_volume` pattern. See `help/froc.md`
    - `ctrw.py` — CTRW continuous-time random walk (experimental), `S=S0·E_α(−(bD)^β)` using the `mittag_leffler.py` helper, outputs s0,d,alpha,beta,residual. `fit_volume` pattern. See `help/ctrw.md`
    - `mittag_leffler.py` — one-parameter Mittag-Leffler `E_α(z)` for real z≤0 via convergent series (used by CTRW)
    - `quality.py` — `assess_fit(maps, model_key, uniq_b, roi_mask=None)` automatic residual-map quality assessment: S0-normalized relative error per voxel, family-aware thresholds, overall good/fair/poor verdict + 0..1 score, worst slices, and b-sampling hints. Attached to the job as `quality` in `api.py`.
    - `roi_calc.py` — quick semi-auto ROI calculator: `calc_roi(model_key, bvals, signals)` fits a curated subset of models (adc, dki2d, ddc, ivim_full, ivim_seg, dki_ivim, ddvd) to scalar ROI-mean signals entered by the user (file-independent, no volume output). Reuses the exact voxel fitters, so parameters match a per-voxel fit. DDVD is analytic (no smoothing/registration in scalar mode). Exposed as `ROI_CALC_MODELS`.
    - `extrapolation.py` — `extrapolate_signal(model_key, maps, b_target, **kwargs)` synthetic signal at target b (accepts `**kwargs` so a stray `workers`/extra kwarg never breaks it); dispatches all DKI-type keys incl. the AIS family
    - `experimental/` — OOP stub examples (unused by registry)
    - `help/` — per-model markdown docs (equation, params, b-value requirements)
- `niviewer.py` — standalone matplotlib ROI viewer at project root (launched via `--open-roi`)
- `requirements-api.txt` — API deps: `fastapi`, `uvicorn[standard]`, `python-multipart`
- `old/` — legacy/deprecated scripts (do not use)

## Electron GUI (`electron-app/`)

Desktop GUI built with Electron + React + TypeScript, talking to the FastAPI backend (HTTP on `127.0.0.1:18923`).

### Structure
```
electron-app/
├── package.json          # JS deps + scripts (dev, build, build:win)
├── electron.vite.config.ts   # MUST be named electron.vite.config.ts (NOT vite.config.ts)
├── electron/main.ts      # Electron main process — spawns `python -m uvicorn diffsuite.api:app --port 18923`, polls /api/status, IPC handlers
├── electron/preload.ts   # contextBridge: openFileDialog, openExternalViewer, apiBase (http://127.0.0.1:18923)
├── src/App.tsx           # Single-page wizard: 1 Input → 2 Model → 3 Fit → 4 Results; also applies the theme (data-theme on <html>) + cycles themes via the discreet header dot (default → Gruvbox → Cappuccin), persisted to localStorage `diffsuite-theme`
├── src/api/client.ts     # Fetch wrapper for all /api/* endpoints + referenceUrl/roiMaskUrl/setRoi/clearRoi
├── src/types/index.ts    # SessionMeta, ModelsResponse, JobStatus, RoiState, RoiResponse, ...
├── src/components/
│   ├── FileInput, ModelSelector   # wizard steps 1-2; FileInput also has a third "Fitted map" box that opens an existing .nii.gz map directly into NiftiViewer (with enableHistogramSelection) without fitting; ModelSelector has an "Allow forcing blocked models" toggle and a "Formula & principle" column (from /api/models formula/principle fields) — see below
│   ├── RoiCalcPanel.tsx   # quick semi-auto ROI calculator on the Input step (file-independent): model dropdown, dynamic (b, signal) rows, Calculate → inline numeric result table via /api/roi-calc
│   ├── FittingView.tsx  # step 3; "Limit to ROI" toggle → inline NiftiViewer for ellipse draw
│   ├── ResultsPanel.tsx # step 4; table of output maps + "Map legend" (per-param descriptions from PARAM_LEGEND) + "View" (NiftiViewer, enableHistogramSelection) + multi-b Extrapolate
│   ├── NiftiViewer.tsx  # canvas NIfTI viewer (gray/turbo/jet), interactive ellipse ROI (draw/move/resize), stats + histogram; supports roiPath + roiWindowing props; optional multi-slice voxel selection (lasso/brush) + histogram study
│   └── ErrorBoundary.tsx# catches render errors → visible message instead of blank screen
├── src/lib/
│   └── histogram.ts     # in-browser histogram + Gaussian fit + stats (scipy-matching skew/kurtosis/percentiles) for the histogram study
└── electron-builder.yml # win: nsis target; files: dist-electron/{renderer,main,preload}/**, package.json
```

> NOTE: The standalone `BValueInput.tsx` component was REMOVED. b-values are detected at
> load time and sent to `/api/models` automatically from `App.tsx` (step 1 → step 2).
> The Model tab still DISPLAYS the detected b-values; a manual override field appears
> there ONLY when no b-values were detected (no `.bval` sidecar).

### Force-fit of blocked models
- `/api/models` returns each rejected entry with a registry `key` (see `REJECTED_MODEL_KEYS`
  in `api.py`) so the GUI can offer a forced fit. "Extrapolate" has NO key — it is not a
  fittable volume model and must never be forced through `/api/fit`.
- `/api/models` also attaches `formula` + `principle` strings to every applicable AND
  rejected entry, sourced from `MODEL_FORMULAS` in `api.py` (single source of truth — add
  new models there). The Model tab renders these in the "Formula & principle" column.
- ModelSelector has an "Allow forcing blocked models" checkbox (default OFF, resets to OFF
  on every new session). When ON, forceable rejected rows become selectable; clicking one
  shows an inline confirmation (reason + risk), then proceeds like a normal selection.
- App.tsx tracks `forcedReason` for the chosen model; FittingView shows an amber warning
  banner when set. `/api/fit` itself never validates requirements — per-voxel fits are
  try/except-guarded, so forced fits degrade gracefully instead of crashing.

### Build / dependency notes (learned the hard way)
- Pin `electron-vite@^2.3.0` + `vite@^5.4` + `@vitejs/plugin-react@^4.3.4`. (electron-vite@5 + vite@6 had a broken SSR build; electron-vite@2 + vite@6 also conflicts.)
- The config file MUST be `electron.vite.config.ts` — electron-vite explicitly forbids `vite.config.ts`.
- Output dirs: `main` → `dist-electron/main/main.js`, `preload` → `dist-electron/preload/preload.js`, `renderer` → `dist-electron/renderer`. `package.json` `"main"` = `dist-electron/main/main.js`. `main.ts` references `../preload/preload.js` and `../renderer/index.html`.
- `npm run build:win` = `electron-vite build && electron-builder --win`. Output: `dist-electron/dist/DiffSuite-{version}-setup.exe`.
- Build step `rcedit` can fail with "Unable to commit changes" if a previous `DiffSuite.exe` is still running — kill stale processes first.
- API backend requirements: `pip install -r requirements-api.txt`.

### Running the GUI (dev)
```bash
pip install -r requirements-api.txt
cd electron-app && npm install
npm run dev          # Starts Vite + Electron; main.ts spawns the Python backend automatically
```

> **Lightweight layout**: the repo intentionally does NOT commit `node_modules`, `dist-electron`,
> or their Windows installer output — these are regenerable build artifacts (`npm install` / `npm run build`
> / `npm run build:win` recreate them). Source + configs are only ~1 MB. After a fresh clone / copy, run
> `npm install` first, then `npm run dev` (dev rebuilds the renderer/main bundles automatically). See the
> root `.gitignore` for what is kept out of version control.

### Building the EXE (Windows)
```bash
cd electron-app && npm install
npm run build:win   # Output: dist-electron/dist/DiffSuite-{version}-setup.exe (NSIS installer)
```

The Electron main process launches `python -m uvicorn diffsuite.api:app --port 18923`,
polls `/api/status` until ready, then opens the renderer. The Python subprocess is killed on app quit.

### FastAPI backend endpoints (`src/diffsuite/api.py`)
- `GET  /api/status`, `GET /api/health` — health/version
- `POST /api/load`  `{path, bvals?}` → session metadata (session_id, shape, bvals, stem, out_prefix). Stores dataobj, mask, **ref (b0 reference)**, roi_mask=None in SESSIONS.
- `POST /api/load/upload` (multipart) — same for uploaded file
- `POST /api/models` `{bvals[]}` → applicable/rejected models (rejected entries include a `key` when forceable)
- `POST /api/fit` `{session_id, model, workers}` → job_id (runs in BackgroundTasks)
- `GET  /api/fit/{job_id}` → `{status, progress_pct, current_slice, total_slices, outputs?, roi_voxels?, error?}`
- `GET  /api/results/{job_id}` → outputs
- `GET  /api/reference/{session_id}` → b0 reference NIfTI (served for ROI drawing)
- `POST /api/roi` `{session_id, slice, cx, cy, a, b, mode}` → rasterizes ellipse into 3D mask (mode=`extrude`|`single`), stores in session; `POST /api/roi/clear` resets
- `GET  /api/roi/mask/{session_id}` → serve the stored 3D ROI mask as a NIfTI (used by the viewer for ROI-based colormap windowing)
- `GET  /api/nifti?path=<abs>` → serve NIfTI (octet-stream) for the viewer
- `POST /api/extrapolate` `{session_id, model_key, b_target}` → synthetic map at target b (call once per target b-value; the GUI loops over selected targets)
- `GET  /api/roi-calc/models` → curated models + requirements for the standalone ROI calculator
- `POST /api/roi-calc` `{model_key, bvals[], signals[]}` → fit a curated model to scalar ROI-mean signals (stateless — no file/session required). Returns `{label, formula, params:[{name,value,unit}], residual}`
- CORS is enabled (`allow_origins=["*"]`) so the renderer can call any endpoint cross-origin.

### ROI feature
- The wizard Fit step has a "Limit to ROI" checkbox. When enabled, an inline `NiftiViewer` shows the b0 reference; the user draws an interactive ellipse (drag to draw, drag body to move, drag yellow handle to resize). "Apply ROI" calls `POST /api/roi`. A "Single slice only" checkbox chooses `single` vs default `extrude` across slices.
- `fit_volume` ANDs the ROI mask into the fit mask, so only ROI voxels are fitted; outputs elsewhere are background (0).

### Extrapolation UI (multi-b)
- In ResultsPanel (step 4), the user picks a fitted model, then toggles one or MORE target b-values (chips from the file's detected b-values plus a suggested `3000`, plus a free-form "Add custom b"). "Extrapolate" calls `/api/extrapolate` once per selected target and lists each result; clicking "View b=…" opens it in `NiftiViewer`.
- A "Window colormap by ROI" checkbox (when viewing a map) fetches `/api/roi/mask/{session_id}` and windows the colormap over ROI foreground only.

### Voxel selection + histogram study (in the map viewer)
- Any fitted map opened via "View" mounts `NiftiViewer` with `enableHistogramSelection`, enabling a selection toolbar (Lasso / Brush / Clear / Run Histogram Study).
- Selection is a **3D voxel mask** (`selectionMask`/`selectionRef`, sized `nx*ny*nz`), so voxels accumulate across **any number of slices** (navigate with the wheel/slider while selected). The selection tint (teal) overlays each slice; `selectionRef` is the source of truth and `setSelectionMask` mirrors it for rendering.
- **Lasso**: start by clicking ON the image, drag to trace the freehand boundary, and **double-click** to close → `fillPolygonOnSlice` (scanline point-in-polygon) writes into the 3D mask for that z only. Releasing the mouse does NOT close a lasso. **Brush**: `stampBrush` stamps a circular disc (radius in voxels, default 5) while dragging on the current slice. **Clear** resets the whole mask. The histogram-results panel renders as a flow sibling below the viewer grid (not clipped by the fixed-height grid).
- "Run Histogram Study" collects all selected voxel values (finite + `>0`, mirroring the reference `prepare_adc_values`) into `src/lib/histogram.ts::runHistogramStudy`, then draws a histogram + **Gaussian fit** (mean/sigma MLE, counts=N·binWidth·pdf) on a canvas with a stats table (N, mean, median, P10/P50/P75/P90, min/max, std, skewness, excess kurtosis).
- Statistical formulas deliberately match **scipy** (`bias=False`): skewness `g1=√(n(n-1))/(n-2)·M3/M2^1.5`, excess kurtosis `G2=1/((n-2)(n-3))·((n²−1)·M4/M2²−3(n−1)²)` with `Mk=Σ(x−x̄)^k/n`. Use the Slicer `adc_computation.py` (`Slicer_extensions/ADC_histogram`) as the reference for these metrics.
- All computation is **in-browser**; no backend/Python changes.

### Key facts for agents
- The Electron app does NOT modify any Python model code — it calls the same `frontend.load_input`,
  `propose_models`, `core.run_fit`, and `models.extrapolation` functions via HTTP.
- NIfTI is parsed in-browser with `nifti-reader-js` (ESM import, NOT `require`). Pass a raw `ArrayBuffer`
  to `isNIFTI`/`readHeader`/`readImage` (Chromium's `DataView` rejects TypedArrays). `isNIFTI()` returns
  false for gzipped files, so also accept `isCompressed()` and call `decompress()` first. `readImage(header, data)`
  requires BOTH args. Files served by `/api/nifti` and `/api/reference` are gzip `.nii.gz`.
- The embedded viewer (`NiftiViewer.tsx`) is canvas-based using `nifti-reader-js`; `api.niftiUrl()` is only
  for local file paths — http(s) URLs (e.g. `referenceUrl`) are fetched directly.
- "Open Externally" button spawns `python niviewer.py <path>` (same as CLI `--open-roi`).

## Model Registry

Add new models in `src/diffsuite/models/__init__.py`:
```python
MODEL_REGISTRY = { "name": fit_function, ... }
```
Each `fit_*_volume` function must accept `(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs)`
and forward `**kwargs` to `fit_volume`. Return `dict {param_name: 3D np.ndarray}`.

### Model registry entries (current)

| # | Key | Outputs | b-value reqs | Pattern |
|---|-----|---------|-------------|---------|
| 1 | `adc` | S0, ADC, residual | ≥2 b-values | fit_volume (voxel) |
| 1b | `eadc` | S0, ADC, eADC, residual | ≥2 b-values (grouped with ADC) | fit_volume (voxel) |
| 2 | `dki2d` | S0, D, K, residual | ≥3 distinct b | fit_volume (voxel) |
| 3 | `dki2d_fast` | S0, D, K, residual | ≥3 distinct b | fit_volume (voxel) |
| 4 | `dki2d_fast+` | S0, D, K, skew, fused, residual | ≥3 distinct b | fit_volume (voxel) |
| 5 | `ddc` | S0, DDC, α, residual | ≥3 b + b0 | fit_volume (voxel) |
| 6 | `ivim_seg` | S0, f, D*, D, residual | ≥4 b, ≥3 b>200, ≥2 b<200 | fit_volume (voxel) |
| 7 | `ivim_full` | S0, f, D*, D, residual | ≥4 b, ≥2 b>200, ≥2 b<200 | fit_volume (voxel) |
| 8 | `dki_ivim` | S0, f, D*, D, K, residual | ≥5 b, ≥2 b>200, ≥2 b<200 | fit_volume (voxel) |
| 10 | `dki_ais_u` | S0, D, K, residual | ≥3 distinct b | volume-level numpy |
| 11 | `dki_ais_c` | S0, D, K, residual | ≥3 distinct b | volume-level numpy |
| 12 | `dki_ais_sc` | S0, D, K, residual | ≥3 distinct b | volume-level numpy |
| 13 | `akc_adci` | S0, D, K, residual | ≥3 distinct b | fit_volume (voxel) |
| 14 | `ddvd` | s0, ddvd, nddvd, ddvd_diff*, residual | b0 + ≥1 b in [10,50] | volume-level numpy |
| 15 | `ctrw` | S0, D, α, β, residual | ≥5 b + b_max≥2000 (Experimental) | fit_volume (voxel) |
| 16 | `froc` | S0, D, β, μ, residual | ≥5 b + b_max≥2000 (Experimental) | fit_volume (voxel) |
| 9 | `extrap` | synthetic volume | — | extrapolation only |

## Conventions

- Fitting iterates slice-by-slice (z-axis), voxel-by-voxel inside `common.fit_volume`
- b-values are rounded to integers via `round_bvals()` before grouping into `uniq_b` / `groups`
- Output maps: `{stem}_{model}_{param}.nii.gz` (e.g. `dwi_dki2d_d.nii.gz`)
- `S0` estimated from minimum-b volume; `ADC` initial guess from log-linear fit
- `ProcessPoolExecutor` parallelization controlled by `workers` arg (1 = serial)
- Progress is reported via `progress_callback(done, total)` so the GUI can show a real progress bar

## External Dependencies

- `dcm2niix` — must be in PATH for DICOM input
- `SimpleITK` — optional fallback NIfTI loader if nibabel fails
- Core: numpy, nibabel, scipy, pydicom, rich, tqdm, matplotlib

## Notes

- No test suite, CI, or linting configured
- `base.py` / `exp_fit.py` / `dki_fit.py` OOP classes are unused legacy stubs
- Interactive `input()` prompts make the CLI non-batch-friendly
- The DKI-IVIM combined model (`dki_ivim.py`): `S0·[f·exp(-b·D*) + (1-f)·exp(-b·D + b²·D²·K/6)]`,
  params `S0,f,D*,D,K,residual`; requires ≥5 b-values + b0 + ≥2 b>200 + ≥2 b<200.
