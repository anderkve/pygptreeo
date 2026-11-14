"""pygptreeo: Online regression with dynamically growing GP trees.

This package implements a continual learning approach for regression tasks using
a dynamically growing binary tree of local Gaussian Process regressors. It is
designed for online learning scenarios where data arrives sequentially and fast
predictions with reliable uncertainty estimates are required.

Main components:
    - GPTree: The core tree structure with local GP models at leaf nodes
    - GPNode: Individual nodes managing local data and GP models
    - GPForest: Ensemble of multiple GPTree models for improved predictions
    - Default_GPR: Pre-configured Gaussian Process Regressor
    - GPRegressorInterface: Abstract interface for GP implementations
    - SklearnGPAdapter: Adapter for scikit-learn GP regressors
    - GPyTorchAdapter: Adapter for GPyTorch models (if GPyTorch is installed)
    - AnisotropicRationalQuadratic: Custom kernel for multi-scale learning
    - NewtonGirardAdditiveKernel: Sample-efficient low-order additive kernel
    - AdditiveMaternKernel: Shorthand for the additive + Matern catch-all leaf kernel
    - AdditivePeriodicKernel: Per-dimension additive periodic kernel
    - AdditivePeriodicMaternKernel: Shorthand for additive-periodic + catch-all leaf kernel
    - OutputBasis / OutputBasisLearner: Global PCA output basis for multi-output trees
    - AdditiveGPGlobalMean / GlobalMeanLearner: Tree-wide global model whose residual the leaves model
    - FeatureNetLearner / NetGlobalMean: Feature network as the tree-wide global model (needs PyTorch)

Typical usage:
    from pygptreeo import GPTree

    gpt = GPTree(Nbar=100, theta=1e-4)

    for x, y, sigma in data_stream:
        y_pred, y_std = gpt.predict(x)
        gpt.update_tree(x, y, sigma)  # sigma: observation-noise std for this point

New nodes, merged and rejected points and split evaluations are reported at
DEBUG level on the ``pygptreeo`` logger.
"""

from pygptreeo.default_gpr import Default_GPR
from pygptreeo.gpnode import GPNode
from pygptreeo.gptree import GPTree
from pygptreeo.gpforest import GPForest
from pygptreeo.kernels import (
    AnisotropicRationalQuadratic,
    AdditiveKernel,
    NewtonGirardAdditiveKernel,
    AdditiveMaternKernel,
    AdditivePeriodicKernel,
    AdditivePeriodicMaternKernel,
)
from pygptreeo.gp_interface import GPRegressorInterface
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.output_basis import OutputBasis, OutputBasisLearner
from pygptreeo.global_mean import (
    GlobalMeanLearner,
    GlobalMeanSnapshot,
    AdditiveGPGlobalMean,
    CoverageReservoir,
)

# Conditionally import GPyTorch adapter if available
try:
    from pygptreeo.adapters import GPyTorchAdapter
    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False

# The feature network global model ('net'); the module imports without torch, the classes need it.
from pygptreeo.feature_net import FeatureNetLearner, NetGlobalMean, TORCH_AVAILABLE as _TORCH_AVAILABLE
