import numpy as np
from .adc import mono_signal
from .dki2d import dki2d_signal
from .ddc import ddc_signal
from .ivim_segmented import ivim_signal
from .dki_ivim import dki_ivim_signal
from .akc_adci import akc_extrapolate_signal
from .froc import froc_signal
from .ctrw import ctrw_signal


def extrapolate_signal(model_key, cache, b_target, **kwargs):
    if model_key in ('adc', 'eadc'):
        return mono_signal(b_target, cache['s0'], cache['adc'])
    if model_key == 'akc_adci':
        b_max = float(np.max(kwargs.get('uniq_b', [0])))
        return akc_extrapolate_signal(b_target, cache['s0'], cache['d'], cache['k'], b_max)
    if model_key in ('dki2d', 'dki2d_fast', 'dki2d_fast+',
                     'dki_ais_u', 'dki_ais_c', 'dki_ais_sc'):
        return dki2d_signal(b_target, cache['s0'], cache['d'], cache['k'])
    if model_key == 'ddc':
        return ddc_signal(b_target, cache['s0'], cache['ddc'], cache['alpha'])
    if model_key in ('ivim_seg', 'ivim_full'):
        return ivim_signal(b_target, cache['s0'], cache['f'], cache['dstar'], cache['d'])
    if model_key == 'dki_ivim':
        return dki_ivim_signal(b_target, cache['s0'], cache['f'], cache['dstar'], cache['d'], cache['k'])
    if model_key == 'ctrw':
        return ctrw_signal(b_target, cache['s0'], cache['d'], cache['alpha'], cache['beta'])
    if model_key == 'froc':
        return froc_signal(b_target, cache['s0'], cache['d'], cache['beta'], cache['mu'])
    raise KeyError(model_key)
