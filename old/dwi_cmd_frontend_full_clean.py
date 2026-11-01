#!/usr/bin/env python3
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib
import pydicom
from scipy.optimize import least_squares

try:
    import SimpleITK as sitk
    HAS_SITK = True
except Exception:
    sitk = None
    HAS_SITK = False


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
    if path.is_dir():
        return True
    if path.suffix.lower() == '.dcm':
        return True
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True)
        return True
    except Exception:
        return False


def run_dcm2niix(dicom_input):
    dicom_input = Path(dicom_input)
    outdir = dicom_input if dicom_input.is_dir() else dicom_input.parent
    if shutil.which('dcm2niix') is None:
        raise RuntimeError('dcm2niix not found in PATH.')
    before = {p.resolve() for p in outdir.glob('*.nii*')}
    subprocess.run(['dcm2niix', '-z', 'y', '-b', 'y', '-o', str(outdir), str(dicom_input)], check=True)
    after = [p for p in outdir.glob('*.nii*') if p.resolve() not in before and is_nifti(p)]
    if not after:
        after = sorted([p for p in outdir.glob('*.nii*') if is_nifti(p)], key=lambda x: x.stat().st_mtime, reverse=True)
    if not after:
        raise RuntimeError('No NIfTI produced by dcm2niix.')
    return after[0]


def find_sidecars(nifti_path):
    base = strip_nii_ext(str(nifti_path))
    bval = Path(base + '.bval')
    bvec = Path(base + '.bvec')
    return bval if bval.exists() else None, bvec if bvec.exists() else None


def parse_bvals_manual(s):
    s = s.strip().replace(',', ' ')
    vals = [float(x) for x in s.split() if x.strip()]
    if len(vals) < 2:
        raise RuntimeError('Need at least 2 b-values.')
    return np.asarray(vals, dtype=float)


def load_image_any(nifti_path):
    try:
        img = nib.load(str(nifti_path), mmap=True)
        return 'nibabel', img, img.dataobj, img.affine, img.header
    except Exception as e:
        if not HAS_SITK:
            raise RuntimeError(f'nibabel failed and SimpleITK unavailable: {e}')
        sitk_img = sitk.ReadImage(str(nifti_path))
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
        print('DICOM input detected -> converting with dcm2niix...')
        nifti_path = run_dcm2niix(input_path)
        print(f'Converted NIfTI: {nifti_path}')
    else:
        raise RuntimeError('Input must be a DICOM folder/file or a .nii/.nii.gz file.')

    source, img, dataobj, affine, header = load_image_any(nifti_path)

    side_bval, side_bvec = find_sidecars(nifti_path)
    if manual_bvals is not None:
        bvals = manual_bvals
    else:
        if side_bval is None:
            raise RuntimeError('Missing .bval sidecar. Use --bvals "50 800 1400" or a matching .bval file.')
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
        print('Warning: bvec count mismatch; ignoring bvecs.')
        bvecs = None

    return nifti_path, source, img, dataobj, np.asarray(bvals, float), bvecs, side_bval, side_bvec, affine, header, shape


def round_bvals(bvals):
    return np.round(bvals).astype(int)


def group_by_bvals(bvals):
    rb = round_bvals(bvals)
    uniq = np.unique(rb)
    groups = [np.where(rb == u)[0] for u in uniq]
    return uniq, groups


def save_nii(path, arr, affine, header):
    arr = np.asarray(arr, dtype=np.float32)
    hdr = header.copy()
    hdr.set_data_shape(arr.shape)
    nib.save(nib.Nifti1Image(arr, affine, hdr), str(path))
    print(f'Saved: {path}')


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


def mono_signal(b, s0, adc):
    return s0 * np.exp(-b * adc)


def ddc_signal(b, s0, ddc, alpha):
    return s0 * np.exp(-np.power(np.maximum(b * ddc, 0.0), alpha))


def dki2d_signal(b, s0, d, k):
    expo = -b * d + (b ** 2) * (d ** 2) * k / 6.0
    return s0 * np.exp(np.clip(expo, -80, 80))


def ivim_signal(b, s0, f, dstar, d):
    return s0 * (f * np.exp(-b * dstar) + (1 - f) * np.exp(-b * d))


def fit_adc_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    good = np.isfinite(sig) & np.isfinite(b)
    if good.sum() < 2:
        return 0.0, 0.0, 0.0
    y = np.log(sig[good])
    X = np.column_stack([np.ones(good.sum()), -b[good]])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    s0 = float(np.exp(beta[0]))
    adc = float(np.clip(beta[1], 0.0, 0.05))
    pred = mono_signal(b[good], s0, adc)
    rmse = float(np.sqrt(np.mean((sig[good] - pred) ** 2)))
    return s0, adc, rmse


def fit_ddc_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    s0 = float(sig[np.argmin(b)])
    _, adc0, _ = fit_adc_voxel(sig, b)
    x0 = np.array([s0, max(adc0, 1e-5), 0.8], float)
    def fun(p):
        return ddc_signal(b, *p) - sig
    try:
        res = least_squares(fun, x0, bounds=([1e-8, 0.0, 0.05], [np.inf, 0.05, 2.0]), method='trf')
        p = res.x
    except Exception:
        p = x0
    pred = ddc_signal(b, *p)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(p[0]), float(p[1]), float(p[2]), rmse


def fit_dki2d_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    s0 = float(sig[np.argmin(b)])
    _, adc0, _ = fit_adc_voxel(sig, b)
    x0 = np.array([s0, max(adc0, 1e-5), 0.7], float)
    def fun(p):
        return dki2d_signal(b, *p) - sig
    try:
        res = least_squares(fun, x0, bounds=([1e-8, 0.0, 0.0], [np.inf, 0.05, 5.0]), method='trf')
        p = res.x
    except Exception:
        p = x0
    pred = dki2d_signal(b, *p)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(p[0]), float(p[1]), float(p[2]), rmse


def fit_ivim_seg_voxel(sig, b, split_b=200.0):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    hi = b >= split_b
    if hi.sum() < 2:
        raise RuntimeError('Not enough high-b points.')
    y = np.log(sig[hi])
    X = np.column_stack([np.ones(hi.sum()), -b[hi]])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A = float(np.exp(beta[0]))
    d = float(np.clip(beta[1], 0.0, 0.02))
    s0 = float(sig[np.argmin(b)])
    f0 = float(np.clip(1.0 - A / max(s0, 1e-8), 0.0, 1.0))
    dstar0 = max(d * 10.0, 0.005)
    def fun(p):
        return ivim_signal(b, s0, p[0], p[1], d) - sig
    try:
        res = least_squares(fun, np.array([f0, dstar0]), bounds=([0.0, d + 1e-6], [1.0, 0.3]), method='trf')
        f, dstar = res.x
    except Exception:
        f, dstar = f0, dstar0
    pred = ivim_signal(b, s0, f, dstar, d)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(s0), float(f), float(dstar), float(d), rmse


def fit_ivim_full_voxel(sig, b):
    sig = np.clip(np.asarray(sig, float), 1e-8, None)
    s0 = float(sig[np.argmin(b)])
    _, adc0, _ = fit_adc_voxel(sig, b)
    x0 = np.array([s0, 0.15, max(adc0, 1e-5) * 10.0, max(adc0, 1e-5)], float)
    def fun(p):
        if p[2] < p[3]:
            return 1e3 * np.ones_like(sig)
        return ivim_signal(b, *p) - sig
    try:
        res = least_squares(fun, x0, bounds=([1e-8, 0.0, 0.0, 0.0], [np.inf, 1.0, 0.3, 0.05]), method='trf')
        p = res.x
    except Exception:
        p = x0
    pred = ivim_signal(b, *p)
    rmse = float(np.sqrt(np.mean((sig - pred) ** 2)))
    return float(p[0]), float(p[1]), float(p[2]), float(p[3]), rmse


def fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, name, func, out_names):
    x, y, z, t = dataobj.shape
    maps = {k: np.zeros((x, y, z), np.float32) for k in out_names}
    b = np.asarray(uniq_b, float)
    for k in range(z):
        sl = np.asarray(dataobj[:, :, k, :], np.float32)
        pav = np.zeros((x, y, len(groups)), np.float32)
        for i, idx in enumerate(groups):
            pav[..., i] = np.mean(sl[..., idx], axis=-1)
        vox = pav.reshape(-1, pav.shape[-1])
        m = mask[:, :, k].reshape(-1)
        for i in np.where(m)[0]:
            try:
                vals = func(vox[i], b)
                for key, val in zip(out_names, vals):
                    maps[key].reshape(-1, z)[i, k] = val
            except Exception:
                pass
        print(f'{name} slice {k+1}/{z}', end='')
    print()
    for key, arr in maps.items():
        save_nii(Path(f'{out_prefix}_{key}.nii.gz'), arr, affine, header)
    return maps


def propose_models(bvals):
    rb = round_bvals(bvals)
    uniq = np.unique(rb)
    has_b0 = np.any(rb == 0)
    n = len(rb)
    models = {}
    rejected = {}
    models[0] = ('Exit', 'exit', '')
    if n >= 2:
        models[1] = ('ADC / monoexponential', 'adc', 's0, adc, residual')
    else:
        rejected['ADC / monoexponential'] = 'Need at least 2 volumes.'
    if n >= 3:
        models[2] = ('2D DKI scalar (kurtosis + real diffusion)', 'dki2d', 's0, d, k, residual')
    else:
        rejected['2D DKI scalar (kurtosis + real diffusion)'] = 'Need at least 3 volumes.'
    if has_b0 and n >= 4 and len(uniq) >= 3:
        models[3] = ('DDC / stretched exponential', 'ddc', 's0, ddc, alpha, residual')
    else:
        rejected['DDC / stretched exponential'] = 'Rejected because this dataset has no b=0 and too few volumes for a stable stretched-exponential fit.'
    if has_b0 and n >= 5 and len(uniq) >= 4:
        models[4] = ('IVIM segmented', 'ivim_seg', 's0, f, dstar, d, residual')
    else:
        rejected['IVIM segmented'] = 'Rejected because segmented IVIM needs enough b-values, including b=0, to separate perfusion and diffusion.'
    if has_b0 and n >= 6 and len(uniq) >= 4:
        models[5] = ('IVIM full biexponential', 'ivim_full', 's0, f, dstar, d, residual')
    else:
        rejected['IVIM full biexponential'] = 'Rejected because the full biexponential IVIM model is overparameterized for only 3 volumes and no b=0.'
    if 1 in models:
        models[6] = ('Extrapolate higher b-value', 'extrap', 'synthetic volume')
    else:
        rejected['Extrapolate higher b-value'] = 'Rejected because no fitted model is available.'
    return models, rejected, uniq


def main():
    args = sys.argv[1:]
    if not args:
        raise RuntimeError('Usage: python dwi_cmd_frontend.py <input> [--bvals "50 800 1400"]')
    input_path = args[0]
    manual = None
    if '--bvals' in args:
        idx = args.index('--bvals')
        if idx + 1 >= len(args):
            raise RuntimeError('Missing value after --bvals')
        manual = parse_bvals_manual(args[idx + 1])

    nifti_path, source, img, dataobj, bvals, bvecs, bval_path, bvec_path, affine, header, shape = load_input(input_path, manual_bvals=manual)
    print('
Input:', nifti_path)
    print('Load method:', source)
    print('bval source:', 'manual command line' if manual is not None else bval_path)
    print('Detected b-values:', round_bvals(bvals).tolist())

    mask, ref = compute_mask(dataobj, bvals)
    stem = Path(strip_nii_ext(nifti_path.name)).name
    out_prefix = str(nifti_path.parent / stem)
    save_nii(Path(f'{out_prefix}_mask.nii.gz'), mask.astype(np.float32), affine, header)
    save_nii(Path(f'{out_prefix}_reference.nii.gz'), ref.astype(np.float32), affine, header)

    models, rejected, _ = propose_models(bvals)
    print('
Applicable models:')
    for k in sorted(models):
        print(f'{k} - {models[k][0]}')
        if models[k][2]:
            print('   outputs:', models[k][2])
    if rejected:
        print('
Rejected models:')
        for name, reason in rejected.items():
            print(f'- {name}: {reason}')

    cache = {}
    uniq_b, groups = group_by_bvals(bvals)

    while True:
        try:
            choice = int(input('
Enter model number: ').strip())
        except Exception:
            print('Invalid input.')
            continue
        if choice not in models:
            print('Unknown choice.')
            continue
        code = models[choice][1]
        if code == 'exit':
            break
        try:
            if code == 'adc':
                cache['adc'] = fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_adc', 'ADC', fit_adc_voxel, ['s0', 'adc', 'residual'])
            elif code == 'dki2d':
                cache['dki2d'] = fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_dki2d', '2D DKI', fit_dki2d_voxel, ['s0', 'd', 'k', 'residual'])
            elif code == 'ddc':
                cache['ddc'] = fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_ddc', 'DDC', fit_ddc_voxel, ['s0', 'ddc', 'alpha', 'residual'])
            elif code == 'ivim_seg':
                cache['ivim_seg'] = fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_ivim_seg', 'IVIM segmented', fit_ivim_seg_voxel, ['s0', 'f', 'dstar', 'd', 'residual'])
            elif code == 'ivim_full':
                cache['ivim_full'] = fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix + '_ivim_full', 'IVIM full', fit_ivim_full_voxel, ['s0', 'f', 'dstar', 'd', 'residual'])
            elif code == 'extrap':
                print('
Choose source fitted model:')
                options = [('adc', 'ADC'), ('dki2d', '2D DKI'), ('ddc', 'DDC'), ('ivim_seg', 'IVIM segmented'), ('ivim_full', 'IVIM full')]
                for i, (_, lab) in enumerate(options, 1):
                    print(f'{i} - {lab}')
                sub = int(input('Model number: ').strip())
                if sub < 1 or sub > len(options):
                    print('Invalid choice.')
                    continue
                key, lab = options[sub - 1]
                if key not in cache:
                    print(f'{lab} must be fitted first.')
                    continue
                b_target = float(input('Enter target higher b-value: ').strip())
                c = cache[key]
                if key == 'adc':
                    syn = mono_signal(b_target, c['s0'], c['adc'])
                elif key == 'dki2d':
                    syn = dki2d_signal(b_target, c['s0'], c['d'], c['k'])
                elif key == 'ddc':
                    syn = ddc_signal(b_target, c['s0'], c['ddc'], c['alpha'])
                else:
                    syn = ivim_signal(b_target, c['s0'], c['f'], c['dstar'], c['d'])
                syn = np.asarray(syn, np.float32) * mask.astype(np.float32)
                save_nii(Path(f'{out_prefix}_synthetic_b{int(round(b_target))}_{key}.nii.gz'), syn, affine, header)
        except Exception as e:
            print('Error:', e)


if __name__ == '__main__':
    main()
