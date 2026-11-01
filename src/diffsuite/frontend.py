from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import nibabel as nib
import pydicom

try:
    import SimpleITK as sitk
    HAS_SITK = True
except Exception:
    sitk = None
    HAS_SITK = False

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .models import get_model, model_file_prefix
from .models.extrapolation import extrapolate_signal
from .models.common import round_bvals, save_nii as _save_nii

console = Console()


def strip_nii_ext(name):
    s = str(name)
    if s.endswith('.nii.gz'):
        return s[:-7]
    if s.endswith('.nii'):
        return s[:-4]
    return s


def is_nifti(path):
    s = str(path).lower()
    return s.endswith('.nii') or s.endswith('.nii.gz')


def looks_like_dicom(path):
    path = Path(path)
    if path.is_dir() or path.suffix.lower() == '.dcm':
        return True
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True)
        return True
    except Exception:
        return False


def find_sidecars(nifti_path):
    base = strip_nii_ext(str(nifti_path))
    bval = Path(base + '.bval')
    bvec = Path(base + '.bvec')
    return bval if bval.exists() else None, bvec if bvec.exists() else None


def dcm_series_info(dicom_input):
    dicom_input = Path(dicom_input)
    files = [dicom_input] if dicom_input.is_file() else sorted([p for p in dicom_input.rglob('*') if p.is_file()])
    rows, seen = [], set()
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        sid = getattr(ds, 'SeriesInstanceUID', None)
        if not sid or sid in seen:
            continue
        seen.add(sid)
        num = getattr(ds, 'SeriesNumber', '')
        desc = getattr(ds, 'SeriesDescription', '')
        rows.append((str(num), str(desc), sid))
    return rows


def collect_series_files(dicom_input, series_uid):
    dicom_input = Path(dicom_input)
    files = [dicom_input] if dicom_input.is_file() else sorted([p for p in dicom_input.rglob('*') if p.is_file()])
    picked = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, 'SeriesInstanceUID', None) == series_uid:
            picked.append(f)
    return picked


def scan_series_representatives(dicom_input):
    dicom_input = Path(dicom_input)
    files = [dicom_input] if dicom_input.is_file() else sorted([p for p in dicom_input.rglob('*') if p.is_file()])
    seen, series = {}, []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True, specific_tags=['SeriesInstanceUID', 'SeriesNumber', 'SeriesDescription'])
        except Exception:
            continue
        sid = getattr(ds, 'SeriesInstanceUID', None)
        if not sid or sid in seen:
            continue
        seen[sid] = True
        series.append((str(getattr(ds, 'SeriesNumber', '')), str(getattr(ds, 'SeriesDescription', '')), sid, f))
    return series


def run_dcm2niix(dicom_input):
    dicom_input = Path(dicom_input)
    outdir = dicom_input if dicom_input.is_dir() else dicom_input.parent
    if shutil.which('dcm2niix') is None:
        raise RuntimeError('dcm2niix not found in PATH.')

    series = scan_series_representatives(dicom_input)
    if not series:
        raise RuntimeError('No DICOM series found.')

    series.sort(key=lambda x: (x[0].isdigit(), int(x[0]) if x[0].isdigit() else 10**9, x[1]))
    console.print('[bold cyan]Found DICOM series:[/bold cyan]')
    for i, (num, desc, _, _) in enumerate(series, 1):
        label = f'{num} - {desc}'.strip(' -')
        console.print(f'{i}. {label}')

    choice = input('Type the series number shown on the right (blank = first): ').strip()
    if choice:
        wanted = choice
        idx = next((i for i, (num, _, _, _) in enumerate(series) if num == wanted), None)
        if idx is None:
            if choice.isdigit() and 1 <= int(choice) <= len(series):
                idx = int(choice) - 1
            else:
                raise RuntimeError('Invalid series selection.')
    else:
        idx = 0

    series_uid = series[idx][2]
    picked = collect_series_files(dicom_input, series_uid)
    if not picked:
        raise RuntimeError('Selected series has no readable DICOM files.')

    with tempfile.TemporaryDirectory() as td:
        series_dir = Path(td) / 'series'
        series_dir.mkdir(parents=True, exist_ok=True)
        for f in picked:
            shutil.copy2(f, series_dir / f.name)
        cmd = ['dcm2niix', '-i', 'n', '-z', 'y', '-b', 'y', '-o', str(outdir), str(series_dir)]
        result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or 'unknown dcm2niix error'
        raise RuntimeError(f'dcm2niix failed with exit code {result.returncode}: {msg}')

    candidates = sorted([p for p in outdir.glob('*.nii*') if is_nifti(p)], key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError('No NIfTI produced by dcm2niix.')
    return candidates[0]


def parse_bvals_manual(s):
    vals = [float(x) for x in str(s).replace(',', ' ').split() if x.strip()]
    if len(vals) < 2:
        raise RuntimeError('Need at least 2 b-values.')
    return np.asarray(vals, dtype=float)


def load_image_any(nifti_path):
    try:
        img = nib.load(str(nifti_path), mmap=True)
        return 'nibabel', img, img.dataobj, img.affine, img.header
    except Exception:
        if not HAS_SITK:
            raise RuntimeError(f'nibabel failed to load "{nifti_path.name}" and SimpleITK is not available. Ensure the file is a valid NIfTI.')
        try:
            sitk_img = sitk.ReadImage(str(nifti_path))
        except RuntimeError:
            raise RuntimeError(f'Cannot read "{nifti_path.name}" — this file is not a supported NIfTI or DICOM format. Use .nii / .nii.gz for DWI data.')
        arr = sitk.GetArrayFromImage(sitk_img)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        affine = np.eye(4)
        header = nib.Nifti1Header()
        header.set_data_shape(arr.shape)
        return 'simpleitk', None, arr, affine, header


def load_input(input_path, manual_bvals=None):
    input_path = Path(input_path)
    if is_nifti(input_path):
        nifti_path = input_path
    elif looks_like_dicom(input_path):
        console.print('[bold cyan]DICOM input detected -> selecting series...[/bold cyan]')
        nifti_path = run_dcm2niix(input_path)
        console.print(f'[green]Converted NIfTI:[/green] {nifti_path}')
    else:
        raise RuntimeError('Input must be a DICOM folder/file (.dcm) or a NIfTI file (.nii / .nii.gz). PNG, JPG and other image formats are not supported.')

    source, img, dataobj, affine, header = load_image_any(nifti_path)
    side_bval, side_bvec = find_sidecars(nifti_path)

    if manual_bvals is not None:
        bvals = manual_bvals
    else:
        if side_bval is None:
            stem = strip_nii_ext(nifti_path.name)
            raise RuntimeError(f'Missing .bval sidecar for {stem}. Use --bvals "50 800 1400" or place a matching .bval file next to the NIfTI.')
        bvals = np.loadtxt(str(side_bval)).astype(float).reshape(-1)

    bvecs = None
    if side_bvec is not None:
        raw = np.loadtxt(str(side_bvec)).astype(float)
        if raw.ndim == 2:
            if raw.shape[0] == 3:
                bvecs = raw.T
            elif raw.shape[1] == 3:
                bvecs = raw

    shape = dataobj.shape if hasattr(dataobj, 'shape') else img.shape
    if len(shape) != 4:
        raise RuntimeError('DWI must be 4D.')
    if shape[3] != len(bvals):
        raise RuntimeError(f'Volume count ({shape[3]}) does not match number of b-values ({len(bvals)}).')
    if bvecs is not None and bvecs.shape[0] != len(bvals):
        console.print('[yellow]Warning:[/yellow] bvec count mismatch; ignoring bvecs.')
        bvecs = None

    return nifti_path, source, img, dataobj, np.asarray(bvals, float), bvecs, side_bval, side_bvec, affine, header, shape


def save_nii(path, arr, affine, header):
    _save_nii(path, arr, affine, header)
    console.print(f'[green]Saved:[/green] {path}')


def group_by_bvals(bvals):
    rb = round_bvals(bvals)
    uniq = np.unique(rb)
    groups = [np.where(rb == u)[0] for u in uniq]
    return uniq, groups


def create_bids_staging(nifti_path, bval_path, bvec_path):
    """Create a temporary BIDS directory from a single DWI NIfTI."""
    tmpdir = Path(tempfile.mkdtemp(prefix="diffsuite_bids_"))
    sub_dwi = tmpdir / "sub-01" / "dwi"
    sub_dwi.mkdir(parents=True)
    shutil.copy2(nifti_path, sub_dwi / "sub-01_dwi.nii.gz")
    if bval_path:
        shutil.copy2(bval_path, sub_dwi / "sub-01_dwi.bval")
    if bvec_path:
        shutil.copy2(bvec_path, sub_dwi / "sub-01_dwi.bvec")
    desc = {"Name": "DiffSuite Input", "BIDSVersion": "1.8.0"}
    (tmpdir / "dataset_description.json").write_text(json.dumps(desc))
    return tmpdir


QSIPREP_IMAGE = "docker.io/pennlinc/qsiprep:latest"


def run_qsiprep(nifti_path, bval_path, bvec_path):
    """Run QSIprep via podman, return (preproc_nifti, preproc_bval, preproc_bvec)."""
    bids_dir = create_bids_staging(nifti_path, bval_path, bvec_path)
    out_dir = Path(tempfile.mkdtemp(prefix="diffsuite_qsiprep_"))

    check = subprocess.run(
        ["podman", "image", "exists", QSIPREP_IMAGE],
        capture_output=True,
    )
    if check.returncode != 0:
        console.print("[cyan]Pulling QSIprep image (first run only)...[/cyan]")
        pull = subprocess.run(["podman", "pull", QSIPREP_IMAGE])
        if pull.returncode != 0:
            raise RuntimeError("Failed to pull QSIprep image.")

    cmd = [
        "podman", "run", "--rm",
        "-v", f"{bids_dir}:/data:ro",
        "-v", f"{out_dir}:/out",
        QSIPREP_IMAGE,
        "/data", "/out", "participant",
        "--participant-label", "01",
        "--denoise-method", "mpca",
        "--anat-modality", "none",
        "--output-spaces", "ACPC",
        "--nthreads", "4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"QSIprep failed:\n{result.stderr[-1000:]}")

    preproc_dwi = list(out_dir.rglob("*_desc-preproc_dwi.nii.gz"))
    if not preproc_dwi:
        raise RuntimeError("QSIprep produced no preprocessed DWI output")
    dwi_path = preproc_dwi[0]
    bval_candidates = list(out_dir.rglob("*_desc-preproc_dwi.bval"))
    bvec_candidates = list(out_dir.rglob("*_desc-preproc_dwi.bvec"))
    preproc_bval = bval_candidates[0] if bval_candidates else None
    preproc_bvec = bvec_candidates[0] if bvec_candidates else None
    return dwi_path, preproc_bval, preproc_bvec


def compute_mask(dataobj, bvals):
    rb = round_bvals(bvals)
    b0_idx = np.where(rb == np.min(rb))[0]
    x, y, z, t = dataobj.shape
    ref = np.zeros((x, y, z), np.float32)
    for k in range(z):
        sl = np.asarray(dataobj[:, :, k, :], dtype=np.float32)
        if len(b0_idx):
            ref[:, :, k] = np.mean(sl[:, :, b0_idx], axis=-1)
        else:
            ref[:, :, k] = np.max(sl, axis=-1)
    positive = ref[ref > 0]
    thr = max(10.0, np.percentile(positive, 20) * 0.25) if positive.size else 10.0
    return (ref > thr), ref


def propose_models(bvals):
    rb = round_bvals(bvals)
    uniq = np.unique(rb)
    has_b0 = np.any(rb == 0)
    n = len(rb)
    n_high = np.sum(rb > 200)
    # Intermediate/low b-values (0 < b < 200) that sample the fast perfusion
    # decay; needed to constrain the IVIM f / D* parameters.
    n_low = np.sum((rb > 0) & (rb < 200))
    n_distinct = len(uniq)
    models = {0: ('Exit', 'exit', '', 'Other')}
    rejected = {}

    # ── ADC (2 params: S0, ADC) ─────────────────────────────────────
    if n >= 2:
        models[1] = ('ADC [≥2 b-values]', 'adc', 'S0, ADC, residual', 'ADC')
        models[1.5] = ('eADC [≥2 b-values]', 'eadc', 'S0, ADC, eADC, residual', 'ADC')
    else:
        reason = 'Need ≥2 b-values.'
        rejected['ADC'] = reason
        rejected['eADC'] = reason

    # ── DKI (3 params: S0, D, K) — b0 NOT required ────────────────
    if n >= 3 and n_distinct >= 3:
        models[2] = ('2D DKI [≥3 b-values]', 'dki2d', 'S0, D, K, residual', 'Kurtosis (DKI)')
        models[3] = ('2D DKI fast [≥3 b-values]', 'dki2d_fast', 'S0, D, K, residual', 'Kurtosis (DKI)')
        models[4] = ('2D DKI fast+ [≥3 b-values]', 'dki2d_fast+', 'S0, D, K, skew, fused, residual', 'Kurtosis (DKI)')
    else:
        reason = 'Need ≥3 distinct b-values.'
        for label in ('2D DKI', '2D DKI fast', '2D DKI fast+'):
            rejected[label] = reason

    # ── DDC / stretched exponential (3 params: S0, DDC, α) ────────
    # α controls curvature; without b0, S0 and α are highly correlated → degeneracy
    if n >= 3 and has_b0:
        models[5] = ('DDC [≥3 b-values, b0]', 'ddc', 'S0, DDC, α, residual', 'DDC')
    else:
        parts = []
        if n < 3: parts.append('≥3 b-values')
        if not has_b0: parts.append('b0 required')
        rejected['DDC'] = 'Requires: ' + ', '.join(parts) + '.'

    # ── IVIM (4 params: S0, f, D*, D) — b0 NOT required ───────────
    # FireVoxel Model 17/18: S0 is an output (estimated from min-b volume),
    # b0 is not a prerequisite. Segmented fit (Model 17) needs ≥3 b>200 to
    # derive Dt & f; full fit (Model 18) needs ≥2 b>200. Both also need ≥2
    # intermediate b-values (0<b<200) to actually constrain the fast perfusion
    # component f / D* — without them the fit is degenerate (spurious f).
    ivim_parts = []
    if n < 4: ivim_parts.append('≥4 b-values')
    if n_high < 2: ivim_parts.append('≥2 b>200')
    if n_low < 2: ivim_parts.append('≥2 b in (0,200)')
    ivim_ok = n >= 4 and n_high >= 2 and n_low >= 2
    seg_parts = []
    if n < 4: seg_parts.append('≥4 b-values')
    if n_high < 3: seg_parts.append('≥3 b>200')
    if n_low < 2: seg_parts.append('≥2 b in (0,200)')
    seg_ok = n >= 4 and n_high >= 3 and n_low >= 2
    if seg_ok:
        models[6] = ('IVIM segmented [≥4 b, ≥3 b>200, ≥2 b<200]', 'ivim_seg', 'S0, f, D*, D, residual', 'IVIM')
    else:
        rejected['IVIM segmented'] = 'Requires: ' + ', '.join(seg_parts) + '.'
    if ivim_ok:
        models[7] = ('IVIM full [≥4 b, ≥2 b>200, ≥2 b<200]', 'ivim_full', 'S0, f, D*, D, residual', 'IVIM')
    else:
        rejected['IVIM full'] = 'Requires: ' + ', '.join(ivim_parts) + '.'

    # ── DKI-IVIM (5 params: S0, f, D*, D, K) — b0 NOT required ─────
    dki_ivim_parts = []
    if n < 5: dki_ivim_parts.append('≥5 b-values')
    if n_high < 2: dki_ivim_parts.append('≥2 b>200')
    if n_low < 2: dki_ivim_parts.append('≥2 b in (0,200)')
    dki_ivim_ok = n >= 5 and n_high >= 2 and n_low >= 2
    if dki_ivim_ok:
        models[8] = ('DKI-IVIM [≥5 b, ≥2 b>200, ≥2 b<200]', 'dki_ivim', 'S0, f, D*, D, K, residual', 'IVIM')
    else:
        parts = []
        if n < 5: parts.append('≥5 b-values')
        if n_high < 2: parts.append('≥2 b>200')
        if n_low < 2: parts.append('≥2 b in (0,200)')
        rejected['DKI-IVIM'] = 'Requires: ' + ', '.join(parts) + '.'

    # ── DKI AIS family (Yan et al. 2014) — b0 NOT required ─────────
    # Fast alternating-iteration schema; same >=3 distinct b-values need as DKI.
    if n >= 3 and n_distinct >= 3:
        models[10] = ('DKI AIS-U [≥3 b-values]', 'dki_ais_u', 'S0, D, K, residual', 'Kurtosis (DKI)')
        models[11] = ('DKI AIS-C [≥3 b-values]', 'dki_ais_c', 'S0, D, K, residual', 'Kurtosis (DKI)')
        models[12] = ('DKI AIS-SC [≥3 b-values]', 'dki_ais_sc', 'S0, D, K, residual', 'Kurtosis (DKI)')
    else:
        reason = 'Need ≥3 distinct b-values.'
        for label in ('DKI AIS-U', 'DKI AIS-C', 'DKI AIS-SC'):
            rejected[label] = reason

    # ── AKC-ADC(i) — secant-based DKI approximation ─────────────────
    # Same ≥3 distinct b-values need as DKI; mathematically equivalent to
    # log-quadratic WLLS, value is as a fast screening / QC tool.
    if n >= 3 and n_distinct >= 3:
        models[13] = ('AKC-ADC(i) [≥3 b-values]', 'akc_adci', 'S0, D, K, residual', 'Kurtosis (DKI)')
    else:
        rejected['AKC-ADC(i)'] = 'Need ≥3 distinct b-values.'

    # ── DDVD — diffusion-derived vessel density (Ni et al. 2025) ────
    # Needs b=0 + at least one b in (10, 50] s/mm².
    has_b0_ddvd = np.any(rb <= 10)
    low_b_ddvd = [bb for bb in uniq if 10 <= bb <= 50]
    if has_b0_ddvd and len(low_b_ddvd) > 0:
        n_low = len(low_b_ddvd)
        extra = ', ddvd_diff' if n_low >= 2 else ''
        models[14] = (f'DDVD (Diffusion Derived Vessel Density) [b0 + low-b ({n_low} avail.)]',
                       'ddvd', f's0, ddvd, nddvd{extra}, residual', 'DDVD')
    else:
        parts_ddvd = []
        if not has_b0_ddvd:
            parts_ddvd.append('b0 required')
        if len(low_b_ddvd) == 0:
            parts_ddvd.append('≥1 b in [10,50] s/mm²')
        rejected['DDVD'] = 'Requires: ' + ', '.join(parts_ddvd) + '.'

    # ── Experimental: FROC / CTRW (fractional anomalous-diffusion models) ─────
    # Need a dense, high-b acquisition: >=5 b-values with the max >= 2000 s/mm^2.
    exp_parts = []
    if n < 5: exp_parts.append('≥5 b-values')
    if len(uniq) == 0 or np.max(uniq) < 2000:
        exp_parts.append('b_max ≥ 2000 s/mm²')
    exp_ok = n >= 5 and len(uniq) > 0 and np.max(uniq) >= 2000
    if exp_ok:
        models[20] = ('CTRW [≥5 b, b_max≥2000]', 'ctrw', 'S0, D, α, β, residual', 'Experimental')
        models[21] = ('FROC [≥5 b, b_max≥2000]', 'froc', 'S0, D, β, μ, residual', 'Experimental')
    else:
        reason = 'Requires: ' + ', '.join(exp_parts) + '.'
        rejected['CTRW'] = reason
        rejected['FROC'] = reason

    # ── Extrapolation ──────────────────────────────────────────────
    has_any_model = any(k != 0 for k in models)
    if has_any_model:
        models[9] = ('Extrapolate to higher b-value', 'extrap', 'Synthetic volume at target b', 'Extrapolation')
    else:
        rejected['Extrapolate'] = 'No fitted model available yet.'

    return models, rejected, uniq


def open_roi_viewer(path):
    viewer = Path(__file__).resolve().parent.parent / 'niviewer.py'
    if not viewer.exists():
        console.print('[yellow]ROI viewer hook skipped:[/yellow] niviewer.py not found at project root.')
        return
    try:
        subprocess.Popen([sys.executable, str(viewer), str(path)])
        console.print(f'[green]ROI viewer opened:[/green] {path}')
    except Exception as e:
        console.print(f'[yellow]Could not open ROI viewer:[/yellow] {e}')


def main():
    parser = argparse.ArgumentParser(prog='diffsuite')
    parser.add_argument('input', help='NIfTI or DICOM input')
    parser.add_argument('--bvals', default=None, help='Manual b-values separated by spaces')
    parser.add_argument('--open-roi', action='store_true', help='Open ROI viewer after processing')
    parser.add_argument('--roi-map', default='dki2d_fast', help='Map to open in ROI viewer')
    args = parser.parse_args()

    manual = parse_bvals_manual(args.bvals) if args.bvals else None
    nifti_path, source, img, dataobj, bvals, bvecs, bval_path, bvec_path, affine, header, shape = load_input(args.input, manual_bvals=manual)

    console.print(Panel.fit(Text('DiffSuite', style='bold white'), border_style='cyan'))
    console.print(f'[bold cyan]Input:[/bold cyan] {nifti_path}')
    console.print(f'[bold cyan]Load method:[/bold cyan] {source}')
    console.print(f'[bold cyan]bval source:[/bold cyan] {"manual command line" if manual is not None else bval_path}')
    console.print(f'[bold cyan]Detected b-values:[/bold cyan] {round_bvals(bvals).tolist()}')

    mask, ref = compute_mask(dataobj, bvals)
    stem = Path(strip_nii_ext(nifti_path.name)).name
    out_prefix = str(nifti_path.parent / stem)

    save_nii(Path(f'{out_prefix}_mask.nii.gz'), mask.astype(np.float32), affine, header)
    save_nii(Path(f'{out_prefix}_reference.nii.gz'), ref.astype(np.float32), affine, header)

    models, rejected, _ = propose_models(bvals)
    console.print('[bold cyan]Applicable models:[/bold cyan]')
    for k in sorted(models):
        console.print(f'[cyan]{k}[/cyan] - {models[k][0]}')
        if models[k][2]:
            console.print(f' [dim]outputs:[/dim] {models[k][2]}')

    if rejected:
        console.print('[bold yellow]Rejected models:[/bold yellow]')
        for name, reason in rejected.items():
            console.print(f'[yellow]- {name}:[/yellow] [dim]{reason}[/dim]')

    cache = {}
    uniq_b, groups = group_by_bvals(bvals)

    while True:
        try:
            choice = int(input('Enter model number: ').strip())
        except Exception:
            console.print('[red]Invalid input.[/red]')
            continue

        if choice not in models:
            console.print('[red]Unknown choice.[/red]')
            continue

        code = models[choice][1]
        if code == 'exit':
            break

        try:
            if code == 'adc':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['adc'] = get_model('adc')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_adc', workers=workers)
            elif code == 'dki2d':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['dki2d'] = get_model('dki2d')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_dki2d', workers=workers)
            elif code == 'dki2d_fast':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['dki2d_fast'] = get_model('dki2d_fast')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_dki2d_fast', workers=workers)
            elif code == 'dki2d_fast+':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['dki2d_fast+'] = get_model('dki2d_fast+')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_dki2d_fast_plus', workers=workers)
            elif code == 'ddc':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['ddc'] = get_model('ddc')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_ddc', workers=workers)
            elif code == 'ivim_seg':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['ivim_seg'] = get_model('ivim_seg')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_ivim_seg', workers=workers)
            elif code == 'ivim_full':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['ivim_full'] = get_model('ivim_full')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_ivim_full', workers=workers)
            elif code == 'dki_ivim':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['dki_ivim'] = get_model('dki_ivim')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_dki_ivim', workers=workers)
            elif code == 'akc_adci':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['akc_adci'] = get_model('akc_adci')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_akc_adci', workers=workers)
            elif code == 'ddvd':
                workers = int(input('Number of workers (1 = serial): ').strip())
                cache['ddvd'] = get_model('ddvd')(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_ddvd', workers=workers)
            elif code == 'extrap':
                console.print('[bold cyan]Choose source fitted model:[/bold cyan]')
                options = [('adc', 'ADC'), ('dki2d', '2D DKI'), ('dki2d_fast', '2D DKI fast'), ('dki2d_fast+', '2D DKI fast+'), ('ddc', 'DDC'), ('ivim_seg', 'IVIM segmented'), ('ivim_full', 'IVIM full'), ('dki_ivim', 'DKI-IVIM'), ('dki_ais_c', 'DKI AIS-C'), ('akc_adci', 'AKC-ADC(i)')]
                for i, (_, lab) in enumerate(options, 1):
                    console.print(f'{i} - {lab}')
                sub = int(input('Model number: ').strip())
                if sub < 1 or sub > len(options):
                    console.print('[red]Invalid choice.[/red]')
                    continue
                key, lab = options[sub - 1]
                if key not in cache:
                    console.print(f'[yellow]{lab} must be fitted first.[/yellow]')
                    continue
                b_target = float(input('Enter target higher b-value: ').strip())
                syn = extrapolate_signal(key, cache[key], b_target, uniq_b=uniq_b)
                syn = np.asarray(syn, np.float32) * mask.astype(np.float32)
                out = Path(f'{out_prefix}_synthetic_b{int(round(b_target))}_{model_file_prefix(key)}.nii.gz')
                save_nii(out, syn, affine, header)
                if args.open_roi:
                    open_roi_viewer(out)
        except Exception as e:
            console.print(f'[red]Error:[/red] {e}')

    if args.open_roi:
        roi_candidates = {
            'dki2d_fast': Path(f'{out_prefix}_dki2d_fast_d.nii.gz'),
            'dki2d_fast+': Path(f'{out_prefix}_dki2d_fast_plus_d.nii.gz'),
            'dki2d': Path(f'{out_prefix}_dki2d_d.nii.gz'),
            'adc': Path(f'{out_prefix}_adc_adc.nii.gz'),
            'ivim_seg': Path(f'{out_prefix}_ivim_seg_d.nii.gz'),
            'ivim_full': Path(f'{out_prefix}_ivim_full_d.nii.gz'),
            'dki_ivim': Path(f'{out_prefix}_dki_ivim_d.nii.gz'),
            'ddc': Path(f'{out_prefix}_ddc_d.nii.gz'),
            'dki_ais_c': Path(f'{out_prefix}_dki_ais_c_d.nii.gz'),
            'akc_adci': Path(f'{out_prefix}_akc_adci_d.nii.gz'),
            'ddvd': Path(f'{out_prefix}_ddvd_ddvd.nii.gz'),
        }
        target = roi_candidates.get(args.roi_map.lower())
        if target is not None and target.exists():
            open_roi_viewer(target)
        else:
            console.print('[yellow]ROI viewer hook requested but target map not found.[/yellow]')


if __name__ == '__main__':
    main()
