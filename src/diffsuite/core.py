from .models import get_model


def run_fit(model_name, dataobj=None, mask=None, uniq_b=None, groups=None, affine=None, header=None, out_prefix=None, **kwargs):
    func = get_model(model_name)
    return func(dataobj, mask, uniq_b, groups, affine, header, out_prefix, **kwargs)
