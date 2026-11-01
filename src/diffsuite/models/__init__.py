from .adc import fit_adc_volume
from .eadc import fit_eadc_volume
from .dki2d import fit_dki2d_volume
from .dki2d_fast import fit_dki2d_fast_volume
from .dki2d_fast_plus import model_dki2d_fast_plus
from .ddc import fit_ddc_volume
from .ivim_segmented import fit_ivim_segmented_volume
from .ivim_full import fit_ivim_full_volume
from .dki_ivim import fit_dki_ivim_volume
from .dki_ais import fit_dki_ais_u_volume, fit_dki_ais_c_volume, fit_dki_ais_sc_volume
from .akc_adci import fit_akc_volume
from .ddvd import fit_ddvd_volume
from .ctrw import fit_ctrw_volume
from .froc import fit_froc_volume
from .extrapolation import extrapolate_signal

# Registry: name -> fitting function
# Each function must accept (dataobj, mask, uniq_b, groups, affine, header, out_prefix, **kwargs)
# and return a dict of {param_name: 3D np.ndarray}.
MODEL_REGISTRY = {
    "adc": fit_adc_volume,
    "eadc": fit_eadc_volume,
    "dki2d": fit_dki2d_volume,
    "dki2d_fast": fit_dki2d_fast_volume,
    "dki2d_fast+": model_dki2d_fast_plus,
    "ddc": fit_ddc_volume,
    "ivim_seg": fit_ivim_segmented_volume,
    "ivim_full": fit_ivim_full_volume,
    "dki_ivim": fit_dki_ivim_volume,
    "dki_ais_u": fit_dki_ais_u_volume,
    "dki_ais_c": fit_dki_ais_c_volume,
    "dki_ais_sc": fit_dki_ais_sc_volume,
    "akc_adci": fit_akc_volume,
    "ddvd": fit_ddvd_volume,
    "ctrw": fit_ctrw_volume,
    "froc": fit_froc_volume,
    "extrap": extrapolate_signal,
}

# OOP stubs in base.py / dki_fit.py / exp_fit.py are unused legacy.

AVAILABLE = list(MODEL_REGISTRY.keys())

# Map a registry key to a filesystem-safe filename prefix. Keys with characters
# that are awkward in filenames/URLs (e.g. '+') are mapped to a safe alias so that
# the CLI and API produce identical output filenames (e.g. dki2d_fast_plus).
MODEL_FILE_PREFIX = {
    "dki2d_fast+": "dki2d_fast_plus",
}


def available_models():
    return AVAILABLE


def model_file_prefix(name):
    """Return the filesystem-safe filename prefix for a model registry key."""
    key = name.lower()
    return MODEL_FILE_PREFIX.get(key, key)


def get_model(name):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return MODEL_REGISTRY[key]