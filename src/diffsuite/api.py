"""
FastAPI backend for DiffSuite.

Wraps the existing CLI fitting pipeline (frontend.load_input, propose_models,
core.run_fit) into REST endpoints that the Electron app consumes.

Run with:
    python -m uvicorn diffsuite.api:app --host 127.0.0.1 --port 18923

Or via the Electron app which spawns this automatically.
"""

import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .frontend import load_input, propose_models, compute_mask, group_by_bvals, run_qsiprep
from .core import run_fit
from .models.extrapolation import extrapolate_signal
from .models import get_model, model_file_prefix
from .models.common import round_bvals, save_nii
from .models.quality import assess_fit
from .models.roi_calc import calc_roi, ROI_CALC_MODELS

app = FastAPI(title="DiffSuite API", version="0.2.1")

# Allow the Electron renderer (different origin) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session + job state ────────────────────────────────────────────
# session_id -> dict with: nifti_path, dataobj, mask, uniq_b, groups, affine,
#                          header, shape, bvals, out_prefix, cache
SESSIONS: Dict[str, Dict[str, Any]] = {}

# job_id -> dict with: status, progress_pct, current_slice, total_slices,
#                       model, error
JOBS: Dict[str, Dict[str, Any]] = {}

# Global lock for session dict mutations (reads are safe, writes protected)
_SESSION_LOCK = threading.Lock()


# ── Request schemas ──────────────────────────────────────────────────────────
class LoadByPath(BaseModel):
    path: str
    bvals: Optional[str] = None
    denoise: Optional[bool] = False


class ModelsQuery(BaseModel):
    bvals: List[float]


class FitRequest(BaseModel):
    session_id: str
    model: str
    workers: int = 1
    selected_bvals: Optional[List[float]] = None
    options: Optional[dict] = None


class ExtrapolateRequest(BaseModel):
    session_id: str
    model_key: str
    b_target: float


class RoiRequest(BaseModel):
    session_id: str
    slice: int
    cx: float
    cy: float
    a: float
    b: float
    mode: str = "extrude"  # "extrude" | "single"


class SessionOnly(BaseModel):
    session_id: str


class RoiCalcRequest(BaseModel):
    model_key: str
    bvals: List[float]
    signals: List[float]


# ── Status endpoint ───────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    import sys
    return {"python": sys.version.split()[0], "version": "0.2.1", "ok": True}


# ── Load endpoint (JSON path-based) ──────────────────────────────────────────
@app.post("/api/load")
def load_by_path(body: LoadByPath):
    """
    Load a NIfTI or DICOM input from a local file path.
    Returns session metadata: shape, b-values, source, stem, etc.
    """
    input_path = body.path
    manual_bvals = body.bvals
    manual = None
    if manual_bvals:
        manual = np.asarray([float(x) for x in manual_bvals.replace(",", " ").split() if x.strip()], float)

    try:
        (nifti_path, source, img, dataobj, bvals, bvecs,
         bval_path, bvec_path, affine, header, shape) = load_input(input_path, manual_bvals=manual)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # QSIprep denoising + motion correction
    if body.denoise:
        try:
            preproc_nifti, preproc_bval, preproc_bvec = run_qsiprep(
                nifti_path, bval_path, bvec_path
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"QSIprep failed: {e}")
        try:
            (nifti_path, source, img, dataobj, bvals, bvecs,
             bval_path, bvec_path, affine, header, shape) = load_input(
                preproc_nifti, manual_bvals=manual
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load QSIprep output: {e}")
        if preproc_bval and preproc_bval.exists():
            bvals = np.loadtxt(str(preproc_bval)).astype(float).reshape(-1)
        if preproc_bvec and preproc_bvec.exists():
            raw = np.loadtxt(str(preproc_bvec)).astype(float)
            bvecs = raw.T if raw.ndim == 2 and raw.shape[0] == 3 else raw

    # Compute mask + reference (same as CLI does)
    mask, ref = compute_mask(dataobj, bvals)
    uniq_b, groups = group_by_bvals(bvals)

    stem = Path(str(nifti_path)).stem
    if str(nifti_path).endswith(".nii.gz"):
        stem = Path(str(nifti_path)[:-7]).stem
    out_prefix = str(Path(nifti_path).parent / stem)

    session_id = str(uuid.uuid4())
    with _SESSION_LOCK:
        SESSIONS[session_id] = {
            "nifti_path": str(nifti_path),
            "dataobj": dataobj,
            "mask": mask,
            "ref": ref,
            "roi_mask": None,
            "uniq_b": uniq_b,
            "groups": groups,
            "affine": affine,
            "header": header,
            "shape": shape,
            "bvals": bvals.tolist(),
            "bvecs": bvecs.tolist() if bvecs is not None else None,
            "source": source,
            "out_prefix": out_prefix,
            "cache": {},
        }

    return {
        "session_id": session_id,
        "nifti_path": str(nifti_path),
        "source": source,
        "shape": list(shape),
        "bvals": round_bvals(bvals).tolist(),
        "bvec_count": len(bvecs) if bvecs is not None else 0,
        "stem": stem,
        "out_prefix": out_prefix,
    }


# ── Load endpoint (multipart file upload) ────────────────────────────────────
@app.post("/api/load/upload")
async def load_by_file(file: UploadFile = File(...), bvals: Optional[str] = None, denoise: Optional[str] = None):
    """
    Upload a NIfTI or DICOM file (multipart form-data).
    Returns the same session metadata as /api/load.
    """
    import tempfile

    suffix = Path(file.filename).suffix
    if suffix not in (".nii", ".nii.gz", ".gz", ".dcm"):
        suffix = ".nii.gz"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.flush()
    tmp.close()

    manual = None
    if bvals:
        manual = np.asarray([float(x) for x in bvals.replace(",", " ").split() if x.strip()], float)

    try:
        (nifti_path, source, img, dataobj, bvals_arr, bvecs,
         bval_path, bvec_path, affine, header, shape) = load_input(tmp.name, manual_bvals=manual)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # QSIprep denoising + motion correction
    do_denoise = denoise and denoise.lower() in ("true", "1", "yes")
    if do_denoise:
        try:
            preproc_nifti, preproc_bval, preproc_bvec = run_qsiprep(
                nifti_path, bval_path, bvec_path
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"QSIprep failed: {e}")
        try:
            (nifti_path, source, img, dataobj, bvals_arr, bvecs,
             bval_path, bvec_path, affine, header, shape) = load_input(
                preproc_nifti, manual_bvals=manual
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load QSIprep output: {e}")
        if preproc_bval and preproc_bval.exists():
            bvals_arr = np.loadtxt(str(preproc_bval)).astype(float).reshape(-1)
        if preproc_bvec and preproc_bvec.exists():
            raw = np.loadtxt(str(preproc_bvec)).astype(float)
            bvecs = raw.T if raw.ndim == 2 and raw.shape[0] == 3 else raw

    mask, ref = compute_mask(dataobj, bvals_arr)
    uniq_b, groups = group_by_bvals(bvals_arr)

    stem = Path(str(nifti_path)).stem
    if str(nifti_path).endswith(".nii.gz"):
        stem = Path(str(nifti_path)[:-7]).stem
    out_prefix = str(Path(nifti_path).parent / stem)

    session_id = str(uuid.uuid4())
    with _SESSION_LOCK:
        SESSIONS[session_id] = {
            "nifti_path": str(nifti_path),
            "dataobj": dataobj,
            "mask": mask,
            "ref": ref,
            "roi_mask": None,
            "uniq_b": uniq_b,
            "groups": groups,
            "affine": affine,
            "header": header,
            "shape": shape,
            "bvals": bvals_arr.tolist(),
            "bvecs": bvecs.tolist() if bvecs is not None else None,
            "source": source,
            "out_prefix": out_prefix,
            "cache": {},
        }

    return {
        "session_id": session_id,
        "nifti_path": str(nifti_path),
        "source": source,
        "shape": list(shape),
        "bvals": round_bvals(bvals_arr).tolist(),
        "bvec_count": len(bvecs) if bvecs is not None else 0,
        "stem": stem,
        "out_prefix": out_prefix,
    }


# ── Reference (b0) serving endpoint ────────────────────────────────────────
@app.get("/api/reference/{session_id}")
def reference(session_id: str):
    """Serve a b0 reference volume (mean of minimum-b volumes) for ROI drawing."""
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    out_dir = Path(session["out_prefix"]).parent
    stem = Path(session["out_prefix"]).name
    ref_path = out_dir / f"{stem}_reference.nii.gz"
    if not ref_path.exists():
        save_nii(ref_path, session["ref"], session["affine"], session["header"])
    return FileResponse(str(ref_path), media_type="application/octet-stream", filename=ref_path.name)


# ── ROI mask endpoint ─────────────────────────────────────────────────────────
@app.post("/api/roi")
def set_roi(req: RoiRequest):
    """Build a 3D ROI mask from an ellipse drawn on a slice and store it on the session."""
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    x, y, z, t = session["shape"]
    if req.slice < 0 or req.slice >= z:
        raise HTTPException(status_code=400, detail="Slice out of range")
    if req.a <= 0 or req.b <= 0:
        raise HTTPException(status_code=400, detail="Ellipse radii must be positive")
    if req.mode not in ("extrude", "single"):
        raise HTTPException(status_code=400, detail="mode must be 'extrude' or 'single'")

    ellipse = np.zeros((x, y), bool)
    # mgrid axes must match the ellipse array's (x, y) layout; using
    # np.mgrid[0:y, 0:x] produced transposed coordinates and crashed for
    # any non-square FOV.
    xx, yy = np.mgrid[0:x, 0:y]
    ellipse[(xx - req.cx) ** 2 / (req.a ** 2) + (yy - req.cy) ** 2 / (req.b ** 2) <= 1.0] = True

    mask3d = np.zeros((x, y, z), bool)
    if req.mode == "extrude":
        for k in range(z):
            mask3d[:, :, k] = ellipse
    else:
        mask3d[:, :, req.slice] = ellipse

    session["roi_mask"] = mask3d
    n = int(mask3d.sum())
    return {"ok": True, "voxels": n, "mode": req.mode, "slice": req.slice}


@app.post("/api/roi/clear")
def clear_roi(body: SessionOnly):
    session = SESSIONS.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session["roi_mask"] = None
    return {"ok": True}


@app.get("/api/roi/mask/{session_id}")
def roi_mask(session_id: str):
    """Serve the stored 3D ROI mask (bool volume) as a NIfTI for windowing."""
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    roi = session.get("roi_mask")
    if roi is None:
        raise HTTPException(status_code=404, detail="No ROI defined for session")
    out_dir = Path(session["out_prefix"]).parent
    stem = Path(session["out_prefix"]).name
    roi_path = out_dir / f"{stem}_roi_mask.nii.gz"
    save_nii(roi_path, roi.astype(np.float32), session["affine"], session["header"])
    return FileResponse(str(roi_path), media_type="application/octet-stream", filename=roi_path.name)


# ── Models endpoint ──────────────────────────────────────────────────────────
# Rejected model labels -> registry keys, so the GUI can offer a forced fit.
# "Extrapolate" is intentionally absent: it is not a fittable volume model and
# forcing it through /api/fit would break (extrapolate_signal has a different
# signature than the fit_*_volume functions).
REJECTED_MODEL_KEYS = {
    "ADC": "adc",
    "eADC": "eadc",
    "2D DKI": "dki2d",
    "2D DKI fast": "dki2d_fast",
    "2D DKI fast+": "dki2d_fast+",
    "DDC": "ddc",
    "IVIM segmented": "ivim_seg",
    "IVIM full": "ivim_full",
    "DKI-IVIM": "dki_ivim",
    "DKI AIS-U": "dki_ais_u",
    "DKI AIS-C": "dki_ais_c",
    "DKI AIS-SC": "dki_ais_sc",
    "AKC-ADC(i)": "akc_adci",
    "DDVD": "ddvd",
    "CTRW": "ctrw",
    "FROC": "froc",
}

# Fitting formula + one-line principle for every registry model, surfaced in the
# GUI Model tab (step 2) next to the outputs.
MODEL_FORMULAS = {
    "adc": {
        "formula": "S(b)=S₀·e^(−b·ADC)",
        "principle": "Monoexponential Gaussian diffusion decay; ADC is the log-slope.",
    },
    "eadc": {
        "formula": "eADC = e^(−ADC·b_ref),  b_ref = b_max",
        "principle": "Fraction of signal remaining at the highest b-value; dimensionless (0–1), contrast opposite to ADC (higher in restricted diffusion).",
    },
    "dki2d": {
        "formula": "S(b)=S₀·e^(−bD+b²D²K/6)",
        "principle": "DKI — kurtosis K captures non-Gaussian restriction; constrained nonlinear least squares (TRF).",
    },
    "dki2d_fast": {
        "formula": "S(b)=S₀·e^(−bD+b²D²K/6)",
        "principle": "Same DKI model solved in closed form by a single log-linear least squares.",
    },
    "dki2d_fast+": {
        "formula": "S(b)=S₀·e^(−bD+b²D²K/6)",
        "principle": "Fast log-linear DKI extended with intravoxel skew and fused maps.",
    },
    "ddc": {
        "formula": "S(b)=S₀·e^(−(b·DDC)^α)",
        "principle": "Stretched exponential; exponent α∈(0,1] encodes intravoxel heterogeneity (α=1 → monoexponential).",
    },
    "ivim_seg": {
        "formula": "S(b)=S₀[(1−f)e^(−bD)+f·e^(−bD*)]",
        "principle": "Biexponential IVIM, segmented: D from high-b points, then f/D* from low-b.",
    },
    "ivim_full": {
        "formula": "S(b)=S₀[(1−f)e^(−bD)+f·e^(−bD*)]",
        "principle": "Biexponential IVIM, all 4 parameters fitted simultaneously.",
    },
    "dki_ivim": {
        "formula": "S(b)=S₀[f·e^(−bD*)+(1−f)e^(−bD+b²D²K/6)]",
        "principle": "Combined perfusion (IVIM) + kurtosis model; needs ≥5 b-values.",
    },
    "dki_ais_u": {
        "formula": "S(b)=S₀·e^(−bD+b²D²K/6)",
        "principle": "Yan 2014 AIS (UAIS): alternate closed-form linear updates of D and K until convergence; unconstrained, vectorized.",
    },
    "dki_ais_c": {
        "formula": "S(b)=S₀·e^(−bD+b²D²K/6)",
        "principle": "AIS with physical constraints each iteration: D≥0, K≥0, K≤3/(b_max·D) (CAIS).",
    },
    "dki_ais_sc": {
        "formula": "S(b)=S₀·e^(−bD+b²D²K/6)",
        "principle": "CAIS plus a temporary 3×3×3 Gaussian-smoothed ADC copy inside each K update for noise robustness (SCAIS).",
    },
    "akc_adci": {
        "formula": "ADC(b_mid) ≈ D − (D²K/3)·b_mid",
        "principle": "Secant ADC between consecutive b-values regressed against midpoint b; algebraically equivalent to log-quadratic WLLS (screening/QC tool).",
    },
    "ddvd": {
        "formula": "DDVD = S(b0) − S(b_low)  [au/pixel]",
        "principle": "Low-b signal drop dominated by IVIM pseudo-diffusion from microvasculature. "
                     "3×3×3 uniform-smoothed before differencing. DDVD_diff subtracts diffusion component.",
    },
    "ctrw": {
        "formula": "S(b)/S₀ = E_α[−(b·D)^β]",
        "principle": "Continuous-time random walk: Mittag-Leffler E_α; D anomalous diffusion, α temporal  heterogeneity index, β spatial heterogeneity index. EXPERIMENTAL.",
    },
    "froc": {
        "formula": "S(b)/S₀ = exp[−D·μ^(2β−2)·b^β]",
        "principle": "Fractional-order calculus (reduced, b-value-only): D diffusion, β fractional spatial order, μ spatial constant. β=1 → monoexponential. EXPERIMENTAL.",
    },
    "extrap": {
        "formula": "Ŝ(b_target) from fitted params",
        "principle": "Model-predicted synthetic image at an arbitrary target b-value.",
    },
}

# Model family grouping for the GUI dropdown.
MODEL_GROUPS = {
    "adc": "ADC",
    "eadc": "ADC",
    "dki2d": "Kurtosis (DKI)",
    "dki2d_fast": "Kurtosis (DKI)",
    "dki2d_fast+": "Kurtosis (DKI)",
    "dki_ais_u": "Kurtosis (DKI)",
    "dki_ais_c": "Kurtosis (DKI)",
    "dki_ais_sc": "Kurtosis (DKI)",
    "akc_adci": "Kurtosis (DKI)",
    "ivim_seg": "IVIM",
    "ivim_full": "IVIM",
    "dki_ivim": "IVIM",
    "ddc": "DDC",
    "ddvd": "DDVD",
    "ctrw": "Experimental",
    "froc": "Experimental",
    "extrap": "Extrapolation",
}


@app.post("/api/models")
def models(query: ModelsQuery):
    """Return applicable + rejected models for a given b-value list.

    Rejected entries carry their registry `key` when the model could be
    force-fitted despite unmet requirements (see REJECTED_MODEL_KEYS);
    entries without a key can never be forced.
    """
    bvals = np.asarray(query.bvals, float)
    models, rejected, uniq = propose_models(bvals)
    applicable = []
    for k in sorted(models):
        if k == 0:
            continue
        code = models[k][1]
        if code == "exit":
            continue
        info = MODEL_FORMULAS.get(code, {})
        applicable.append({
            "number": k,
            "label": models[k][0],
            "key": code,
            "outputs": models[k][2],
            "group": models[k][3] if len(models[k]) > 3 else MODEL_GROUPS.get(code, "Other"),
            "formula": info.get("formula"),
            "principle": info.get("principle"),
        })
    return {
        "applicable": applicable,
        "rejected": [
            {
                "label": name,
                "reason": reason,
                "key": REJECTED_MODEL_KEYS.get(name),
                "group": MODEL_GROUPS.get(REJECTED_MODEL_KEYS.get(name), "Other"),
                "formula": MODEL_FORMULAS.get(REJECTED_MODEL_KEYS.get(name), {}).get("formula"),
                "principle": MODEL_FORMULAS.get(REJECTED_MODEL_KEYS.get(name), {}).get("principle"),
            }
            for name, reason in rejected.items()
        ],
        "uniq_b": uniq.tolist(),
    }


# ── Fit endpoint ─────────────────────────────────────────────────────────────
def _run_fit_job(job_id: str, session_id: str, model_key: str, workers: int,
                 selected_bvals: Optional[List[float]] = None,
                 options: Optional[dict] = None):
    """Background worker: runs fit_volume and updates job progress."""
    session = SESSIONS.get(session_id)
    if session is None:
        JOBS[job_id].update({"status": "error", "error": "Session not found"})
        return

    total_z = session["shape"][2]
    JOBS[job_id].update({
        "status": "running",
        "progress_pct": 0.0,
        "current_slice": 0,
        "total_slices": total_z,
    })

    def _progress(done, total):
        JOBS[job_id]["current_slice"] = done
        JOBS[job_id]["progress_pct"] = round(100.0 * done / total, 1)

    try:
        fit_fn = get_model(model_key)
        roi = session.get("roi_mask")

        uniq_b = session["uniq_b"]
        groups = session["groups"]

        if selected_bvals:
            sel = set(round(b) for b in selected_bvals)
            mask = np.isin(uniq_b, list(sel))
            uniq_b = uniq_b[mask]
            groups = [g for g, m in zip(groups, mask) if m]
            if len(uniq_b) < 2:
                JOBS[job_id].update({"status": "error", "error": "Need at least 2 b-values for fitting."})
                return

        # Use a filesystem-safe prefix so filenames match the CLI + docs
        # (e.g. registry key "dki2d_fast+" -> "dki2d_fast_plus").
        file_key = model_file_prefix(model_key)
        maps = fit_fn(
            session["dataobj"], session["mask"], uniq_b,
            groups, session["affine"], session["header"],
            session["out_prefix"] + f"_{file_key}", workers=workers,
            progress_callback=_progress,
            roi_mask=roi,
            **(options or {}),
        )
        session["cache"][model_key] = maps
        # Automatic residual-map quality assessment (surfaces how well the
        # model explained the signal, normalized so it is intensity-independent).
        quality = assess_fit(maps, model_key, uniq_b, roi_mask=roi)
        # Collect output paths
        out_dir = Path(session["out_prefix"]).parent
        stem = Path(session["out_prefix"]).name
        outputs = {}
        for param in maps:
            p = out_dir / f"{stem}_{file_key}_{param}.nii.gz"
            outputs[param] = str(p)
        JOBS[job_id].update({
            "status": "done",
            "outputs": outputs,
            "progress_pct": 100.0,
            "current_slice": total_z,
            "roi_voxels": int(roi.sum()) if roi is not None else None,
            "quality": quality,
        })
    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})


@app.post("/api/fit")
def fit(req: FitRequest, background: BackgroundTasks):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress_pct": 0.0, "current_slice": 0, "total_slices": 0}
    background.add_task(_run_fit_job, job_id, req.session_id, req.model, req.workers, req.selected_bvals, req.options)
    return {"job_id": job_id}


@app.get("/api/fit/{job_id}")
def fit_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── Results endpoint ──────────────────────────────────────────────────────────
@app.get("/api/results/{job_id}")
def results(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        return {"status": job["status"], "outputs": {}}
    return {"status": "done", "outputs": job.get("outputs", {})}


# ── NIfTI serving endpoint ───────────────────────────────────────────────────
@app.get("/api/nifti")
def serve_nifti(path: str):
    """Serve a NIfTI file (absolute path) as octet-stream for the viewer."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)


# ── Extrapolation endpoint ───────────────────────────────────────────────────
@app.post("/api/extrapolate")
def extrapolate(req: ExtrapolateRequest):
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    cache = session["cache"].get(req.model_key)
    if cache is None:
        raise HTTPException(status_code=400, detail=f"Model '{req.model_key}' not fitted yet.")
    try:
        syn = extrapolate_signal(req.model_key, cache, req.b_target, uniq_b=session["uniq_b"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    syn = np.asarray(syn, np.float32) * session["mask"].astype(np.float32)
    out_dir = Path(session["out_prefix"]).parent
    stem = Path(session["out_prefix"]).name
    out = out_dir / f"{stem}_synthetic_b{int(round(req.b_target))}_{model_file_prefix(req.model_key)}.nii.gz"
    from .models.common import save_nii
    save_nii(out, syn, session["affine"], session["header"])
    return {"path": str(out), "param": "synthetic"}


# ── ROI semi-auto calculation endpoints ──────────────────────────────────────
@app.get("/api/roi-calc/models")
def roi_calc_models():
    """Return the curated models available for the ROI calculator (file-independent)."""
    items = []
    for key, info in ROI_CALC_MODELS.items():
        items.append({
            "key": key,
            "label": info["label"],
            "formula": info["formula"],
            "requirement": info["requirement"],
            "params": [p[0] for p in info["params"]],
        })
    return {"models": items}


@app.post("/api/roi-calc")
def roi_calc(req: RoiCalcRequest):
    """Fit a curated model to scalar ROI-mean signals. No session/file required.

    Body: {model_key, bvals: [number], signals: [number]} where signals are the
    ROI-mean signal intensities entered by the user at each b-value.
    Returns the fitted parameters as numbers (no volume output).
    """
    try:
        result = calc_roi(req.model_key, req.bvals, req.signals)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


# ── Health check for Electron startup ────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True}
