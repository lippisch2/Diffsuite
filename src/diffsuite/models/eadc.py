import numpy as np
from .common import fit_volume
from .adc import fit_adc_voxel


def fit_eadc_voxel(sig, b):
    """ADC monoexponential fit plus eADC at the maximum b-value.

    Reuses the ADC voxel fitter (same model, same b-values) so the 'adc' output
    map is identical to the standalone ADC model, and derives eADC from that ADC:
    EADC = exp(-ADC . b_ref) with b_ref = max b-value. eADC is dimensionless (0-1)
    and opposite in contrast to ADC (higher in restricted diffusion).
    """
    s0, adc, rmse = fit_adc_voxel(sig, b)
    b = np.asarray(b, float)
    b_ref = float(np.max(b)) if b.size else 0.0
    eadc = float(np.exp(-adc * b_ref))
    return s0, adc, eadc, rmse


def fit_eadc_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix, workers=1, **kwargs):
    return fit_volume(dataobj, mask, uniq_b, groups, affine, header, out_prefix,
                      'EADC', fit_eadc_voxel, ['s0', 'adc', 'eadc', 'residual'],
                      workers=workers, **kwargs)
