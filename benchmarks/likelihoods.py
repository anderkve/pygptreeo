"""Auxiliary log-likelihoods for the adaptive-sampling schedules.

These functions are completely separate from the emulator *targets* in
``problems.py``. The adaptive-sampling schedules (``schedule="de"`` and
``schedule="mcmc"``) use one of these likelihoods to decide where to
visit the input space; the emulator is then evaluated on the true
target ``fn(X)`` at exactly those visited points. This mimics the
"global-fit emulator" deployment setting: the fitter is sampling from
some posterior / objective landscape, and the emulator has to learn an
expensive auxiliary function from whatever draws it happens to get.

All likelihoods are defined on ``x in [0, 1]^d`` and return log-density
up to a constant. ``NINF`` is a moderately-negative value used when the
argument drops out of the unit cube, so that DE's mutation step does
not produce infinite objectives that break the optimiser.
"""

from __future__ import annotations

import numpy as np


# Moderately-negative "out of support" log-density, big enough that the
# sampler strongly avoids going there but not -inf so numerics behave.
_OUT_OF_BOX_PENALTY = -1e4


def _in_unit_cube(X: np.ndarray) -> np.ndarray:
    """Return a boolean mask of which rows of X lie in [0, 1]^d."""
    return np.all((X >= 0.0) & (X <= 1.0), axis=-1)


def bimodal_gauss(X: np.ndarray) -> np.ndarray:
    """Two isotropic Gaussian blobs centred at 0.3·1 and 0.7·1.

    Returns a 1-D array of log-densities (up to a constant). Both modes
    have `sigma = 0.1` so ~99 % of the mass sits within the cube. The
    function is smooth, bounded, and has two separated modes so DE and
    MCMC behave differently on it (DE finds both modes quickly; MCMC
    may get trapped in one with a modest proposal size).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu1 = 0.3
    mu2 = 0.7
    sigma = 0.1
    d1 = np.sum((X - mu1) ** 2, axis=-1) / (2.0 * sigma ** 2)
    d2 = np.sum((X - mu2) ** 2, axis=-1) / (2.0 * sigma ** 2)
    lp = np.logaddexp(-d1, -d2)
    # Penalise out-of-unit-cube evaluations so DE stays inside.
    in_box = _in_unit_cube(X)
    lp = np.where(in_box, lp, _OUT_OF_BOX_PENALTY)
    return lp


LIKELIHOODS = {
    "bimodal_gauss": bimodal_gauss,
}
