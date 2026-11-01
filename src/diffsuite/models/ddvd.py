"""
Diffusion-Derived Vessel Density (DDVD).

Ni et al. (2025) "Magnetic resonance diffusion-derived vessel density (DDVD)
as a valuable tissue perfusion biomarker for isocitrate dehydrogenase
genotyping in diffuse gliomas", BMC Med Imaging 25:79.

DDVD measures microvasculature density via the IVIM signal drop between
b=0 and a low b-value (typically b=10 s/mm²). A 3×3×3 uniform box smooth
is applied to each b-volume before differencing to replicate the paper's
ROI-mean paradigm as a sliding window.

Outputs (always):
  s0       — signal at b=0 (smoothed)
  ddvd     — S(b0) − S(b_low1)  [au/pixel]
  nddvd    — [S(b0) − S(b_low1)] / S(b0)  [×100, %]
  residual — 0 (analytical, 2 eq / 2 unknowns)

Outputs (when 2+ low-b values available, all ≤ 50 s/mm²):
  ddvd_diff — diffusion-corrected: [S(b0)−S(b_low1)] − [S(b_low1)−S(b_low2)]
            = S(b0) − 2·S(b_low1) + S(b_low2)

Optional motion correction:
  register=True aligns each low-b volume to the b0 volume (SimpleITK rigid
  registration; robust to rotation + translation) before differencing, so
  patient motion between the b0 and low-b acquisitions cannot create spurious
  vessel-density signal at tissue/fluid boundaries. resample=False keeps each
  registered volume on its native grid (default: resampled onto the b0 grid).
  Falls back to unregistered volumes (with a stderr warning) if SimpleITK is
  not installed.
"""

import sys

import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter

from .common import save_nii

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:  # pragma: no cover - depends on install
    sitk = None
    HAS_SITK = False


def _group_means(dataobj, groups):
    """Mean signal per b-group (same pattern as dki_ais._group_means)."""
    x, y, z, t = dataobj.shape
    pav = np.zeros((x, y, z, len(groups)), np.float32)
    for k in range(z):
        sl = np.asarray(dataobj[:, :, k, :], np.float32)
        for i, idx in enumerate(groups):
            pav[:, :, k, i] = np.mean(sl[..., idx], axis=-1)
    return pav


def _numpy_to_sitk(vol, affine):
    """Wrap a 3D numpy volume as a SimpleITK image with real voxel geometry.

    Correctly decomposes the NIfTI affine: voxel spacing is the length of each
    affine column, and the axis direction is the orthonormalized rotation.
    SimpleITK forbids negative spacing, so axis flips are carried in the
    direction matrix instead.
    """
    x, y, z = vol.shape
    img = sitk.GetImageFromArray(vol.transpose(2, 1, 0))
    m = np.asarray(affine[:3, :3], float)
    spacing = np.linalg.norm(m, axis=0)
    spacing[spacing == 0] = 1.0
    # orthonormal rotation (direction cosines), preserving axis signs
    direction = np.zeros((3, 3), float)
    for c in range(3):
        col = m[:, c]
        direction[:, c] = col / (spacing[c] or 1.0)
    img.SetSpacing(tuple(float(v) for v in spacing))
    img.SetOrigin((float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3])))
    dir_flat = tuple(float(v) for v in direction.flatten())
    try:
        img.SetDirection(dir_flat)
    except Exception:
        pass  # fall back to identity direction if the matrix is not a valid rotation
    return img


def _sitk_to_numpy(img):
    """Convert a SimpleITK image back to a (x, y, z) numpy float32 volume."""
    arr = sitk.GetArrayFromImage(img)
    return np.asarray(arr.transpose(2, 1, 0), np.float32)


def _register_low_b_to_b0(s0_vol, low_b_vols, affine, resample=True):
    """Register each low-b volume to the b0 volume (SimpleITK rigid).

    Uses sitk.ImageRegistrationMethod with a rigid (Euler3D) transform — 3
    rotations + 3 translations — the standard robust choice for within-exam
    motion between b0 and low-b, and the most reliable on smooth DWI volumes
    (affine over-parameterization tends to settle in local minima). Returns the
    registered low-b volumes. If resample=True they are resampled to the b0
    grid; if resample=False they are resampled onto their own (native) grid. If
    SimpleITK is unavailable, prints a stderr warning and returns the input
    volumes unchanged.
    """
    if not HAS_SITK or sitk is None:
        print("WARNING: SimpleITK not installed; skipping b0<->low-b "
              "registration (pip install SimpleITK to enable).", file=sys.stderr)
        return low_b_vols

    fixed = _numpy_to_sitk(np.asarray(s0_vol, np.float32), affine)

    def _register_one(vol):
        moving = _numpy_to_sitk(np.asarray(vol, np.float32), affine)
        transform = sitk.Euler3DTransform()
        transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(
            np.array(fixed.GetSize()) / 2.0))
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        r.SetMetricSamplingStrategy(r.REGULAR)
        r.SetMetricSamplingPercentage(0.2)
        r.SetInterpolator(sitk.sitkLinear)
        r.SetInitialTransform(transform, inPlace=True)
        r.SetOptimizerAsRegularStepGradientDescent(
            learningRate=0.01,
            minStep=1e-6,
            numberOfIterations=400,
            gradientMagnitudeTolerance=1e-6,
        )
        r.SetNumberOfThreads(1)
        try:
            r.Execute(fixed, moving)
        except Exception as e:  # registration can fail on degenerate input
            print(f"WARNING: b0<->low-b registration failed ({e}); "
                  f"using unregistered volume.", file=sys.stderr)
            return vol
        resampler = sitk.ResampleImageFilter()
        # resample onto the b0 grid (default) or the moving volume's native grid
        resampler.SetReferenceImage(fixed if resample else moving)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(transform)
        return _sitk_to_numpy(resampler.Execute(moving))

    return [_register_one(v) for v in low_b_vols]


def fit_ddvd_volume(dataobj, mask, uniq_b, groups, affine, header,
                    out_prefix, workers=1, progress_callback=None,
                    roi_mask=None, register=False, resample=True, **kwargs):
    """Volume-level DDVD computation (vectorized, no fit_volume)."""
    x, y, z, t = dataobj.shape
    m = np.asarray(mask, bool)
    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, bool)
        if roi_mask.shape == (x, y, z):
            m = m & roi_mask

    # Group-mean per b-value → shape (x, y, z, n_unique_b)
    pav = _group_means(dataobj, groups)

    # Identify b-value indices
    b_arr = np.asarray(uniq_b, float)
    b0_idx = int(np.argmin(b_arr))
    low_b_mask = (b_arr >= 10) & (b_arr <= 50)
    low_b_indices = np.where(low_b_mask)[0]
    # Exclude b0 itself from the low-b list
    low_b_indices = low_b_indices[low_b_indices != b0_idx]

    if len(low_b_indices) == 0:
        empty = {name: np.zeros((x, y, z), np.float32)
                 for name in ("s0", "ddvd", "nddvd", "residual")}
        for name in empty:
            save_nii(Path(f"{out_prefix}_{name}.nii.gz"), empty[name], affine, header)
        return empty

    # Sort low-b ascending
    low_b_indices = low_b_indices[np.argsort(b_arr[low_b_indices])]

    # 3×3×3 uniform box smooth each b-volume
    KERNEL = 3
    s0_smooth = uniform_filter(pav[:, :, :, b0_idx], size=KERNEL, mode="nearest")
    low_b_vols = [uniform_filter(pav[:, :, :, i], size=KERNEL, mode="nearest")
                  for i in low_b_indices]

    # Optional motion correction: align each low-b volume to b0 before differencing
    if register:
        low_b_vols = _register_low_b_to_b0(s0_smooth, low_b_vols, affine, resample=resample)

    # Basic DDVD
    ddvd = s0_smooth - low_b_vols[0]
    denom = np.where(s0_smooth > 0, s0_smooth, 1.0)
    nddvd = np.where(s0_smooth > 0, ddvd / denom, 0.0) * 100.0

    # Diffusion-corrected DDVD (if 2+ low-b values)
    have_diff = len(low_b_vols) >= 2
    if have_diff:
        ddvd_diff = s0_smooth - 2.0 * low_b_vols[0] + low_b_vols[1]

    # Apply mask and save
    maps = {}
    for name, vol in [("s0", s0_smooth), ("ddvd", ddvd), ("nddvd", nddvd)]:
        masked = np.where(m, vol, 0.0).astype(np.float32)
        maps[name] = masked
        save_nii(Path(f"{out_prefix}_{name}.nii.gz"), masked, affine, header)

    if have_diff:
        masked_diff = np.where(m, ddvd_diff, 0.0).astype(np.float32)
        maps["ddvd_diff"] = masked_diff
        save_nii(Path(f"{out_prefix}_ddvd_diff.nii.gz"), masked_diff, affine, header)

    residual = np.zeros((x, y, z), np.float32)
    maps["residual"] = residual
    save_nii(Path(f"{out_prefix}_residual.nii.gz"), residual, affine, header)

    # CLI warning for b_low > 30
    b_low_actual = float(b_arr[low_b_indices[0]])
    if b_low_actual > 30:
        print(f"WARNING: DDVD b_low={b_low_actual:.0f} s/mm² > 30. "
              f"Tissue diffusion contamination ~{(b_low_actual * 0.0015 * 100):.0f}%. "
              f"Protocol b=0,10,20 is recommended (Federau/Ni).", file=sys.stderr)

    return maps
