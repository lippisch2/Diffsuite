import numpy as np
from scipy.special import gamma

_MAX_TERMS = 500
_TOL = 1e-10


def mittag_leffler(a, z):
    """One-parameter Mittag-Leffler function E_a(z) for real z <= 0.

    E_a(z) = sum_{k=0}^{inf} z^k / Gamma(a*k + 1)

    For the real, non-positive argument used by the CTRW model the series is
    convergent. Terms are accumulated until convergence (|term| < tol) or the
    term budget is exhausted; a finite budget prevents pathological slow
    convergence on large |z| (high b / small beta) from hanging the fit.
    """
    a = float(a)
    z = float(z)
    acc = 0.0
    term = 1.0
    for k in range(_MAX_TERMS):
        acc += term
        if abs(term) < _TOL:
            break
        # term_{k+1} = term_k * z / Gamma(a*(k+1)+1) * Gamma(a*k+1)
        term *= z / (gamma(a * (k + 1) + 1.0) * (1.0 / gamma(a * k + 1.0)))
    return acc
