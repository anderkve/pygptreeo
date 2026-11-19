"""pygptreeo: Online regression with dynamically growing GP trees.

This package implements a continual learning approach for regression tasks using
a dynamically growing binary tree of local Gaussian Process regressors. It is
designed for online learning scenarios where data arrives sequentially and fast
predictions with reliable uncertainty estimates are required.

Main components:
    - GPTree: The core tree structure with local GP models at leaf nodes
    - GPNode: Individual nodes managing local data and GP models
    - solve_fit_task: A leaf fit handed out by GPTree.update_tree(defer_fits=True), for another process
    - Trainer: Chooses the points at which to evaluate a target, to train a GPTree
    - Default_GPR: Pre-configured Gaussian Process Regressor
    - GPRegressorInterface: Abstract interface for GP implementations
    - SklearnGPAdapter: Adapter for scikit-learn GP regressors
    - GPyTorchAdapter: Adapter for GPyTorch models (if GPyTorch is installed)
    - AnisotropicRationalQuadratic: Custom kernel for multi-scale learning
    - NewtonGirardAdditiveKernel: Sample-efficient low-order additive kernel
    - AdditiveMaternKernel: Shorthand for the additive + Matern catch-all leaf kernel
    - AdditivePeriodicKernel: Per-dimension additive periodic kernel
    - AdditivePeriodicMaternKernel: Shorthand for additive-periodic + catch-all leaf kernel
    - ObliqueMatern: Matern kernel whose length scales need not lie along the inputs (Default_GPR(oblique=True))
    - OutputBasis / OutputBasisLearner: Global PCA output basis for multi-output trees
    - OutputWarp / AlignedCurves: Alignment of curve-valued outputs within a leaf (output_model='aligned')
    - GridWarp / AlignedSurfaces: The same for outputs that are a function on a grid of two dimensions (output_shape=(p1, p2))
    - AdditiveGPGlobalMean / GlobalMeanLearner: Tree-wide global model whose residual the leaves model
    - FeatureNetLearner / NetGlobalMean: Feature network as the tree-wide global model (needs PyTorch)

Typical usage:
    from pygptreeo import GPTree

    gpt = GPTree(Nbar=100, theta=1e-4)

    for x, y, sigma in data_stream:
        y_pred, y_std = gpt.predict(x)
        gpt.update_tree(x, y, sigma)  # sigma: observation-noise std for this point

New nodes, sleeping and removed leaves, merged and rejected points and split
evaluations are reported at DEBUG level on the ``pygptreeo`` logger.
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
from pygptreeo.output_alignment import AlignedCurves, AlignedSurfaces, GridWarp, OutputWarp
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
