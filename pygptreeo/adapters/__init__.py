"""
Adapters for different Gaussian Process implementations.

This module provides adapter classes that wrap different GP libraries
(scikit-learn, GPyTorch, etc.) to conform to pygptreeo's GPRegressorInterface.
"""

from .sklearn_adapter import SklearnGPAdapter

__all__ = ['SklearnGPAdapter']

# Import GPyTorch adapter only if GPyTorch is available
try:
    from .gpytorch_adapter import GPyTorchAdapter
    __all__.append('GPyTorchAdapter')
except ImportError:
    pass
