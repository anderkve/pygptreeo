"""pygptreeo: Online regression with dynamically growing GP trees.

This package implements a continual learning approach for regression tasks using
a dynamically growing binary tree of local Gaussian Process regressors. It is
designed for online learning scenarios where data arrives sequentially and fast
predictions with reliable uncertainty estimates are required.

Typical usage:
    from pygptreeo import GPTree

    gpt = GPTree(Nbar=100, theta=1e-4)

    for x, y, sigma in data_stream:
        y_pred, y_std = gpt.predict(x)
        gpt.update_tree(x, y, sigma)  # sigma: observation-noise std for this point

New nodes, sleeping and removed leaves and merged points are reported at DEBUG
level on the ``pygptreeo`` logger.
"""

from pygptreeo.default_gpr import Default_GPR
from pygptreeo.gpnode import GPNode
from pygptreeo.gptree import GPTree, solve_fit_task
from pygptreeo.trainer import Trainer
from pygptreeo.kernels import (
    AnisotropicRationalQuadratic,
    AdditiveKernel,
    NewtonGirardAdditiveKernel,
    AdditiveMaternKernel,
    AdditivePeriodicKernel,
    AdditivePeriodicMaternKernel,
    ObliqueMatern,
)
from pygptreeo.gp_interface import GPRegressorInterface
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.global_mean import GlobalMeanLearner, AdditiveGPGlobalMean
from pygptreeo.feature_net import FeatureNetLearner    # its use needs PyTorch

try:
    from pygptreeo.adapters import GPyTorchAdapter     # if GPyTorch is installed
except ImportError:
    pass
