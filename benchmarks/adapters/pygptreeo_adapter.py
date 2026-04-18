"""Adapter exposing pygptreeo's GPTree through the common OnlineRegressor API."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from pygptreeo import GPTree
from pygptreeo.adapters import SklearnGPAdapter as _SklearnGPAdapter
from pygptreeo.kernels import AnisotropicRationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor

from .base import OnlineRegressor


def _make_configured_gpr_class(n_dims: int, kernel_spec: str = "matern+rq"):
    """Factory returning a fresh GPR subclass whose kernel matches ``n_dims``.

    We can't pass ``n_dims`` to ``__init__`` because sklearn's parameter
    validation requires every constructor kwarg to be a stored attribute that
    matches its declared type spec.

    ``kernel_spec`` selects the leaf-GP kernel:

    - ``"matern+rq"`` (default): ``Constant * (AnisotropicRQ + Matern-1.5)`` —
      the richer kernel used by pygptreeo's example scripts.
    - ``"matern"``: ``Constant * Matern-1.5`` — matches the single-kernel
      used by `sklearn_gp` and the GPyTorch SVGP adapter, so the variant
      is an apples-to-apples kernel comparison.
    """

    if kernel_spec == "matern+rq":
        def _build():
            return ConstantKernel(
                constant_value=1.0, constant_value_bounds=(1e-3, 1e8)
            ) * (AnisotropicRationalQuadratic(
                length_scale=[1.0] * n_dims,
                length_scale_bounds=(1e-5, 1e5),
                alpha=1.0,
                alpha_bounds=(1e-4, 1e4),
            ) + Matern(
                nu=1.5,
                length_scale=[1.0] * n_dims,
                length_scale_bounds=[(1e-5, 1e5)] * n_dims,
            ))
    elif kernel_spec == "matern":
        def _build():
            return ConstantKernel(
                constant_value=1.0, constant_value_bounds=(1e-3, 1e8)
            ) * Matern(
                nu=1.5,
                length_scale=[1.0] * n_dims,
                length_scale_bounds=[(1e-5, 1e5)] * n_dims,
            )
    else:
        raise ValueError(f"Unknown kernel_spec {kernel_spec!r}")

    class _ConfiguredGPR(GaussianProcessRegressor):
        def __init__(self, kernel=None, *, alpha=1e-6,
                     optimizer="fmin_l_bfgs_b", n_restarts_optimizer=1,
                     normalize_y=False, copy_X_train=True, n_targets=None,
                     random_state=None):
            super().__init__()
            self.kernel = _build()
            self.min_length_scale = 0.001
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts_optimizer = 1  # 3 was too slow for long benchmarks
            self.normalize_y = normalize_y
            self.copy_X_train = copy_X_train
            self.n_targets = n_targets
            self.random_state = random_state

    return _ConfiguredGPR


class PyGPTreeOAdapter(OnlineRegressor):
    """Wraps a GPTree with Nbar=200, median splits and calibrated sigma."""

    name = "pygptreeo"
    supports_uncertainty = True

    def __init__(self, n_dims: int, Nbar: int = 200, theta: float = 1e-4,
                 retrain_step: int = 200, sigma_rel: float = 1e-3,
                 kernel_spec: str = "matern+rq"):
        self.n_dims = n_dims
        self.sigma_rel = sigma_rel
        self.kernel_spec = kernel_spec
        gpr_cls = _make_configured_gpr_class(n_dims, kernel_spec=kernel_spec)
        self.tree = GPTree(
            GPR=_SklearnGPAdapter(gpr_cls()),
            Nbar=Nbar,
            theta=theta,
            split_position_method="median",
            split_dimension_criteria="max_uncertainty",
            retrain_every_n_points=retrain_step,
            use_calibrated_sigma=True,
            splitting_strategy="gradual",
            max_n_pred_leaves=3,
            aggregation="moe",
            use_hyperparameter_inheritance=False,
            use_standard_scaling=True,
            enable_point_rejection=False,
            enable_point_merging=False,
            enable_split_evaluation=True,
            n_split_candidates=4,
            split_eval_train_fraction=0.4,
            split_eval_min_points=20,
        )
        self._seen = 0

    def predict(self, X):
        if self._seen == 0:
            # Nothing learned yet; return zeros with large uncertainty.
            return (np.zeros((X.shape[0], 1)),
                    np.full((X.shape[0], 1), np.nan))
        mean, std = self.tree.predict(X, show_progress=False)
        return mean, std

    def update(self, x, y):
        sigma = self.sigma_rel * np.abs(y) + 1e-10
        self.tree.update_tree(x, y, sigma)
        self._seen += 1
