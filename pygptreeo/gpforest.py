import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import Callable, Optional, Type, Union
from tqdm import tqdm

from pygptreeo.gptree import GPTree # Added import
import joblib # Added import
from pygptreeo.default_gpr import Default_GPR


class GPForest:
    """Manages an ensemble of GPTree models.

    A GPForest consists of multiple GPTree instances, each potentially
    trained with different hyperparameters (like Nbar or theta) or
    different Gaussian Process Regressors. This ensemble approach can
    lead to improved prediction stability and accuracy compared to a
    single GPTree.

    Attributes:
        GPR: The GaussianProcessRegressor or list of GPRs used by the trees.
        Nbar: Maximum number of training points per node in each GPTree.
        theta: Overlap parameter for node splitting in each GPTree.
        GPTrees: A list containing the individual GPTree instances.
        num_GPTrees: The total number of GPTrees in the forest.
    """
    def __init__(self,
                 GPR: Optional[Union[GaussianProcessRegressor, list]] = Default_GPR(),
                 Nbar: Optional[Union[int, list]] = 100,
                 theta: Optional[Union[float, list]] = 0.0001,
                 use_standard_scaling: Optional[Union[bool, list]] = False): # jules standard scaling: Add use_standard_scaling parameter
        """Initializes the GPForest.

        Args:
            GPR: The GaussianProcessRegressor instance or a list of such
                instances to be used for the GPTrees. If a single instance
                is provided, it will be replicated for all GPTrees in the
                forest. Defaults to `Default_GPR()`.
            Nbar: The maximum number of training points allowed in a leaf
                node for each GPTree. Can be an integer (applying to all
                trees) or a list of integers (one for each tree).
                Defaults to 100.
            theta: The overlap parameter used for splitting nodes in each
                GPTree. This parameter influences how much the data ranges
                of sibling nodes overlap. Can be a float (applying to all
                trees) or a list of floats (one for each tree).
                Defaults to 0.0001.
            use_standard_scaling (Optional[Union[bool, list]]): Whether to use standard
                scaling in GPNodes. Can be a boolean (applying to all trees) or a
                list of booleans (one for each tree). Defaults to False. # jules standard scaling: Add docstring for use_standard_scaling
        Raises:
            AssertionError: If `Nbar` and `theta` are lists of different lengths.
        """
        
        self.GPR = GPR
        self.Nbar = Nbar
        self.theta = theta
        self.use_standard_scaling = use_standard_scaling # jules standard scaling: Store use_standard_scaling

        self.GPTrees = []

        
        if type(self.Nbar) == int:
            self.Nbar = [self.Nbar]

        if type(self.theta) == int:
            self.theta = [self.theta]
        
        assert len(self.Nbar) == len(self.theta), "Nbar and theta must have the same number of elements"

        self.num_GPTrees = len(self.Nbar)

        if type(self.GPR) == GaussianProcessRegressor or type(self.GPR) == Default_GPR:
            self.GPR = self.num_GPTrees*[self.GPR]

        # jules standard scaling: Ensure use_standard_scaling is a list of correct length
        if isinstance(self.use_standard_scaling, bool):
            self.use_standard_scaling = [self.use_standard_scaling] * self.num_GPTrees

        assert len(self.use_standard_scaling) == self.num_GPTrees, "use_standard_scaling list must have the same number of elements as Nbar/theta"


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, show_progress: Optional[bool]=False):
        """Builds and trains all GPTrees in the forest.

        This method first initializes each GPTree in the `GPTrees` list
        according to the forest's configuration (`GPR`, `Nbar`, `theta`).
        Then, it trains each GPTree using the provided training data.

        Args:
            X_train: The training data features (numpy array).
            y_train: The training data targets (numpy array).
            show_progress: If True, displays a progress bar for tree
                building and training. Defaults to False.
        """

        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Building forest'):
            # jules standard scaling: Pass use_standard_scaling to GPTree constructor
            self.GPTrees.append(GPTree(GPR=self.GPR[i],
                                       Nbar=self.Nbar[i],
                                       theta=self.theta[i],
                                       use_standard_scaling=self.use_standard_scaling[i]))
        
        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Training GPTrees'):
            self.GPTrees[i].fit(X_train, y_train, shuffle=True)

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

            sigma_prior = np.diag(self.GPTrees[i].GPR.kernel(X_test)).reshape(-1, 1)

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


        
        

