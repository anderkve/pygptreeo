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

Typical usage:
    from pygptreeo import GPTree

    gpt = GPTree(Nbar=100, theta=1e-4)

    for x, y in data_stream:
        y_pred, y_std = gpt.predict(x)
        gpt.update_tree(x, y)
"""

# Standard library imports
from copy import deepcopy
from typing import Callable, Optional, Type, Union

# Third-party imports
import joblib
import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import resample
from tqdm import tqdm

# Local imports
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.gpnode import GPNode
from pygptreeo.gptree import GPTree
from pygptreeo.gpforest import GPForest
from pygptreeo.kernels import AnisotropicRationalQuadratic, AdditiveKernel
from pygptreeo.gp_interface import GPRegressorInterface
from pygptreeo.adapters import SklearnGPAdapter
from pygptreeo.kernel_performance_tracker import KernelPerformanceTracker

# Conditionally import GPyTorch adapter if available
try:
    from pygptreeo.adapters import GPyTorchAdapter
    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False
