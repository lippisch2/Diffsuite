import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def round_bvals(bvals):
    return np.round(bvals).astype(int)


def save_nii(path, arr, affine, header):
    import nibabel as nib
    arr = np.asarray(arr, dtype=np.float32)
    hdr = header.copy()
    hdr.set_data_shape(arr.shape)
    nib.save(nib.Nifti1Image(arr, affine, hdr), str(path))


def _fit_chunk(args):
    chunk, b, func = args
    results = []
    for i, sig in chunk:
        try:
            vals = func(sig, b)
            results.append((i, vals))
        except Exception:
            results.append((i, None))
    return results


def fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, name, func, out_names, workers=1, chunk_size=128, progress_callback=None, roi_mask=None):
    x, y, z, t = dataobj.shape
    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, bool)
        if roi_mask.shape == (x, y, z):
            mask = mask & roi_mask
    maps = {k: np.zeros((x, y, z), np.float32) for k in out_names}
    b = np.asarray(uniq_b, float)
    for k in range(z):
        sl = np.asarray(dataobj[:, :, k, :], np.float32)
        pav = np.zeros((x, y, len(groups)), np.float32)
        for i, idx in enumerate(groups):
            pav[..., i] = np.mean(sl[..., idx], axis=-1)
        vox = pav.reshape(-1, pav.shape[-1])
        m = mask[:, :, k].reshape(-1)
        idxs = np.where(m)[0]
        if workers and workers > 1 and len(idxs) > 0:
            chunks = [idxs[j:j + chunk_size] for j in range(0, len(idxs), chunk_size)]
            tasks = [([(i, vox[i]) for i in ch], b, func) for ch in chunks]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_fit_chunk, task) for task in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f'{name} slice {k+1}/{z}', unit='chunk', dynamic_ncols=True):
                    for i, vals in fut.result():
                        if vals is None:
                            continue
                        for key, val in zip(out_names, vals):
                            maps[key][:, :, k].reshape(-1)[i] = val
        else:
            pbar = tqdm(idxs, desc=f'{name} slice {k+1}/{z}', unit='vox', dynamic_ncols=True)
            for i in pbar:
                try:
                    vals = func(vox[i], b)
                    for key, val in zip(out_names, vals):
                        maps[key][:, :, k].reshape(-1)[i] = val
                except Exception:
                    pass
        if progress_callback is not None:
            progress_callback(k + 1, z)
    for key, arr in maps.items():
        save_nii(Path(f'{out_prefix}_{key}.nii.gz'), arr, affine, header)
    return maps