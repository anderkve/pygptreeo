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

        This method aggregates predictions from all GPTrees in the forest.
        The final mean prediction is a weighted average of the means from
        individual trees. The weights are derived from the uncertainty
        (sigma_i) of each tree's prediction relative to a prior uncertainty
        (sigma_prior), as captured by the `alpha` and `T` variables in
        the implementation. The combined standard deviation provides an
        estimate of the overall prediction uncertainty.

        Args:
            X_test: The test data features (numpy array) for which
                predictions are to be made.
            show_progress: If True, displays a progress bar for the
                prediction process across trees. Defaults to False.

        Returns:
            A tuple containing:
                - mean (np.ndarray): The weighted average of mean predictions
                  from all GPTrees.
                - std (np.ndarray): The combined standard deviation, representing
                  the uncertainty of the predictions.
        """
        mean = np.zeros((X_test.shape[0], 1))
        std = np.zeros((X_test.shape[0], 1))

        mean_list = []
        std_list = []

        alpha = []
        T = []

        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Predicting'):
            mean_i, sigma_i = self.GPTrees[i].predict(X_test)

            mean_list.append(mean_i)
            std_list.append(sigma_i)

            T.append(1./(sigma_i*sigma_i))

            # Compute prior covariance using the GP interface
            cov_prior = self.GPTrees[i].GPR.get_kernel_covariance(X_test)
            sigma_prior = np.diag(cov_prior).reshape(-1, 1)

            alpha.append(0.5*(np.log(sigma_prior) - np.log(sigma_i)))

        sum_alpha = np.sum(alpha)
        
        for i in range(self.num_GPTrees):

            mean += mean_list[i]*alpha[i]*T[i]/sum_alpha

            std += alpha[i]*T[i]/sum_alpha
            
            
        std = 1./std
        
        mean *= std

        return mean, std

    def save(self, path: str):
        """Saves the trained GPForest object to a file.

        This method serializes the entire GPForest instance, including all
        its trained GPTrees and their configurations, using `joblib.dump`.

        Args:
            path: The file path (string) where the GPForest object will be saved.
        """
        joblib.dump(self, path)


        
        

