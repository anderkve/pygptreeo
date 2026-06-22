"""AutoSelectGPR: automatic per-region kernel selection (baseline vs additive).

A drop-in :class:`~pygptreeo.gp_interface.GPRegressorInterface` that wraps two
candidate GPs -- a cheap full-D *baseline* (e.g. a plain Matern) and a cheap
*additive* kernel (e.g. ``make_order_additive_kernel(..., rescue=False)``) -- and,
each time it is fitted, keeps whichever has the higher log marginal likelihood
(model evidence) on that region's data. It is the cheap, hard-selection
alternative to the rescue term: rather than always carrying both kernels in one
(expensive, 15-parameter) blend, it commits to a single cheap kernel per region.

Why this design
---------------
Because it implements the GP interface, it needs no changes to GPTree/GPNode: a
node treats it as an ordinary GP, and the verdict propagates through the
``clone()`` the tree already performs when it spawns children. The selection
exploits two structural facts about additivity (see the project README / design
notes):

* **An additive verdict is inheritable.** If ``f`` is a sum of low-order terms on
  a box it is on every sub-box too, so once a node selects the additive kernel
  its descendants inherit that choice and skip the comparison -- only the
  not-yet-additive frontier keeps testing. (A baseline verdict is *not*
  committed: children re-test, because smaller boxes tend to become additive.)
* **Additivity increases with depth**, since an order-``k`` interaction's
  variance scales like (box width)^{2k}; so the baseline frontier shrinks with
  depth and the test self-terminates.

A positive ``lml_margin`` requires the additive evidence to beat the baseline by
a margin before committing, so genuine ties default to the safe baseline.

Warm start
----------
Like the rest of the tree, a freshly cloned child predicts with its parent's
selected (fitted) GP until it retrains on its own data; ``clone()`` carries that
fitted GP across, plus the inherited additive verdict for the child's next fit.

Notes
-----
* Single-output only (the comparison is on a scalar log evidence).
* The two candidate GPs must be fit with the same data and observation noise for
  the LML comparison to be a valid model selection -- this class handles that.
"""

import numpy as np

from .gp_interface import GPRegressorInterface


class AutoSelectGPR(GPRegressorInterface):
    """Wrap a baseline and an additive GP; keep the higher-evidence one per fit.

    Parameters
    ----------
    baseline_gpr : GPRegressorInterface
        Untrained full-dimensional template (e.g. a plain Matern GP).
    additive_gpr : GPRegressorInterface
        Untrained additive template (e.g. ``make_order_additive_kernel(rescue=False)``).
    lml_margin : float, default=0.0
        Additive is selected only if ``LML_additive > LML_baseline + lml_margin``
        (nats). Larger values bias ambiguous regions toward the safe baseline.

    Attributes
    ----------
    verdict : {"additive", "baseline", None}
        The current selection (None before the first fit).
    """

    def __init__(self, baseline_gpr, additive_gpr, lml_margin=0.0):
        self._baseline_template = baseline_gpr
        self._additive_template = additive_gpr
        self.lml_margin = float(lml_margin)

        self._selected = None          # the fitted GP used for prediction
        self._committed_additive = False  # inheritable "this region is additive"
        self.verdict = None
        self.last_lml = None           # (lml_baseline, lml_additive) of last comparison
        # The tree sets per-point observation noise by assigning to `.alpha`
        # before fit(); we apply it to both candidates at fit time.
        self.alpha = None

    # ------------------------------------------------------------------ #
    # fitting / selection
    # ------------------------------------------------------------------ #
    def _prep(self, gpr):
        if self.alpha is not None:
            gpr.set_observation_noise(np.asarray(self.alpha))
        return gpr

    def fit(self, X, y):
        if self._committed_additive:
            # Region already judged additive (here or by an ancestor): skip the
            # comparison and just fit the additive kernel.
            g = self._prep(self._additive_template.clone())
            g.fit(X, y)
            self._selected = g
            self.verdict = "additive"
            return self

        gb = self._prep(self._baseline_template.clone()); gb.fit(X, y)
        ga = self._prep(self._additive_template.clone()); ga.fit(X, y)
        lml_b = gb.log_marginal_likelihood()
        lml_a = ga.log_marginal_likelihood()
        self.last_lml = (lml_b, lml_a)
        if lml_a > lml_b + self.lml_margin:
            self._selected = ga
            self.verdict = "additive"
            self._committed_additive = True   # commit -> children inherit, skip test
        else:
            self._selected = gb
            self.verdict = "baseline"          # provisional: children re-test
        return self

    # ------------------------------------------------------------------ #
    # prediction / state, delegated to the selected (or, pre-fit, baseline) GP
    # ------------------------------------------------------------------ #
    def _active(self):
        return self._selected if self._selected is not None else self._baseline_template

    def predict(self, X, return_std=False):
        return self._active().predict(X, return_std=return_std)

    def is_trained(self):
        return self._selected is not None and self._selected.is_trained()

    def set_observation_noise(self, alpha):
        self.alpha = alpha

    def get_kernel_covariance(self, X):
        return self._active().get_kernel_covariance(X)

    def get_kernel(self):
        return self._active().get_kernel()

    def set_kernel(self, kernel):
        self._active().set_kernel(kernel)

    def get_length_scales(self, n_features):
        return self._active().get_length_scales(n_features)

    def log_marginal_likelihood(self):
        if self._selected is None:
            return float("-inf")
        return self._selected.log_marginal_likelihood()

    def reset_training(self):
        self._selected = None
        self.verdict = None

    def clone(self):
        new = AutoSelectGPR(self._baseline_template.clone(),
                            self._additive_template.clone(),
                            lml_margin=self.lml_margin)
        # Inherit an additive commitment (additivity of a region implies it for
        # sub-regions); a baseline verdict is intentionally not committed.
        new._committed_additive = self._committed_additive
        # Warm start: let the child predict with the parent's fitted GP until it
        # retrains on its own data.
        if self._selected is not None:
            new._selected = self._selected.clone()
            new.verdict = self.verdict
        return new

    def __repr__(self):
        return (f"AutoSelectGPR(verdict={self.verdict}, "
                f"committed_additive={self._committed_additive}, "
                f"lml_margin={self.lml_margin})")


def make_auto_gpr(n_features, alpha=1e-6, n_restarts_optimizer=0, lml_margin=2.0,
                  max_order=2, base_kernel="matern", length_scale_bounds=(1e-5, 1e5)):
    """Convenience builder for an :class:`AutoSelectGPR` over ``n_features`` inputs.

    Pairs a plain full-D Matern baseline with a (rescue-free) per-order additive
    kernel, both wrapped in scikit-learn adapters, ready to pass to
    ``GPTree(GPR=...)``.

    Parameters mirror the usual GP/kernel settings; ``lml_margin`` (nats) controls
    how decisively the additive evidence must beat the baseline before the
    additive kernel is adopted for a region.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    from .adapters import SklearnGPAdapter
    from .kernels import make_order_additive_kernel

    baseline = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e8)) * Matern(
            nu=1.5, length_scale=[1.0] * n_features,
            length_scale_bounds=[length_scale_bounds] * n_features),
        alpha=alpha, n_restarts_optimizer=n_restarts_optimizer))
    additive = SklearnGPAdapter(GaussianProcessRegressor(
        kernel=make_order_additive_kernel(n_features, max_order=max_order,
                                          base_kernel=base_kernel, rescue=False),
        alpha=alpha, n_restarts_optimizer=n_restarts_optimizer))
    return AutoSelectGPR(baseline, additive, lml_margin=lml_margin)
