"""GPForest: Ensemble of GPTree models.

This module implements the GPForest class, which manages an ensemble of
multiple GPTree instances. By combining predictions from multiple trees,
GPForest can provide improved stability and accuracy compared to a single tree.

The ensemble approach allows for:
    - Better handling of uncertainty through multiple independent models
    - Improved robustness to hyperparameter choices
    - Potential for parallel training of individual trees
"""

# Standard library imports
from typing import Callable, Optional, Type, Union

# Third-party imports
import joblib
import numpy as np
from tqdm import tqdm

# Local imports
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.gptree import GPTree
from pygptreeo.gp_interface import GPRegressorInterface


class GPForest:
    """Manages an ensemble of GPTree models.

    A GPForest consists of multiple GPTree instances, each potentially
    trained with different hyperparameters (like Nbar or theta) or
    different Gaussian Process Regressors. This ensemble approach can
    lead to improved prediction stability and accuracy compared to a
    single GPTree.

    Attributes:
        GPR: The GPRegressorInterface or list of GPRs used by the trees.
        Nbar: Maximum number of training points per node in each GPTree.
        theta: Overlap parameter for node splitting in each GPTree.
        GPTrees: A list containing the individual GPTree instances.
        num_GPTrees: The total number of GPTrees in the forest.
    """
    def __init__(self,
                 GPR: Optional[Union[GPRegressorInterface, list]] = None,
                 Nbar: Optional[Union[int, list]] = 100,
                 theta: Optional[Union[float, list]] = 0.0001):
        """Initializes the GPForest.

        Args:
            GPR: The GPRegressorInterface instance or a list of such
                instances to be used for the GPTrees. If a single instance
                is provided, it will be replicated for all GPTrees in the
                forest. Defaults to `Default_GPR()` (scikit-learn adapter).
            Nbar: The maximum number of training points allowed in a leaf
                node for each GPTree. Can be an integer (applying to all
                trees) or a list of integers (one for each tree).
                Defaults to 100.
            theta: The overlap parameter used for splitting nodes in each
                GPTree. This parameter influences how much the data ranges
                of sibling nodes overlap. Can be a float (applying to all
                trees) or a list of floats (one for each tree).
                Defaults to 0.0001.
        Raises:
            AssertionError: If `Nbar` and `theta` are lists of different lengths.
        """

        # Use Default_GPR if no GPR is provided
        if GPR is None:
            GPR = Default_GPR()

        self.GPR = GPR
        self.Nbar = Nbar
        self.theta = theta

        self.GPTrees = []


        if type(self.Nbar) == int:
            self.Nbar = [self.Nbar]

        if type(self.theta) == int:
            self.theta = [self.theta]

        assert len(self.Nbar) == len(self.theta), "Nbar and theta must have the same number of elements"

        self.num_GPTrees = len(self.Nbar)

        # If GPR is a single instance (not a list), replicate it for all trees
        if not isinstance(self.GPR, list):
            self.GPR = self.num_GPTrees*[self.GPR]


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sigma_train: np.ndarray,
            show_progress: Optional[bool]=False):
        """Builds and trains all GPTrees in the forest.

        This method first initializes each GPTree in the `GPTrees` list
        according to the forest's configuration (`GPR`, `Nbar`, `theta`).
        Then, it trains each GPTree using the provided training data.

        Args:
            X_train: The training data features (numpy array).
            y_train: The training data targets (numpy array).
            sigma_train: Per-point uncertainties (standard deviations). Required.
            show_progress: If True, displays a progress bar for tree
                building and training. Defaults to False.
        """

        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Building forest'):
            self.GPTrees.append(GPTree(self.GPR[i], self.Nbar[i], self.theta[i]))

        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Training GPTrees'):
            self.GPTrees[i].fit(X_train, y_train, sigma_train, shuffle=True)

    def predict(self, X_test: np.ndarray, show_progress: Optional[bool]=False):
        """Predicts target values and their uncertainties using the GPForest.

        The trees are combined as a generalised product of experts with equal
        weights: the mean is the average of the trees' means weighted by their
        precisions ``1 / sigma_i^2``, and the variance is the harmonic mean of
        the trees' variances. The trees are fitted to the same data, so the
        combined sigma does not shrink with the number of trees.

        Args:
            X_test: The test data features (numpy array) for which
                predictions are to be made.
            show_progress: If True, displays a progress bar for the
                prediction process across trees. Defaults to False.

        Returns:
            A tuple containing:
                - mean (np.ndarray): The combined mean prediction.
                - std (np.ndarray): The combined standard deviation.
        """
        means, variances = [], []
        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Predicting'):
            mean_i, sigma_i = self.GPTrees[i].predict(X_test)
            means.append(mean_i)
            variances.append(sigma_i * sigma_i)
        means, variances = np.stack(means), np.stack(variances)

        # Precisions relative to the most confident tree: finite also where a sigma is zero
        var_min = variances.min(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_precision = np.where(variances > 0, var_min / variances, 1.0)
        norm = rel_precision.sum(axis=0)

        mean = (rel_precision * means).sum(axis=0) / norm
        std = np.sqrt(self.num_GPTrees * var_min / norm)

        return mean, std

    def save(self, path: str):
        """Saves the trained GPForest object to a file.

        This method serializes the entire GPForest instance, including all
        its trained GPTrees and their configurations, using `joblib.dump`.

        Args:
            path: The file path (string) where the GPForest object will be saved.
        """
        joblib.dump(self, path)


        
        

