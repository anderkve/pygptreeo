import numpy as np
from binarytree import Node
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import Callable, Optional, Type, Union
from tqdm import tqdm

from pygptreeo.default_gpr import Default_GPR


class GPForest:
    def __init__(self,
                 GPR: Optional[Union[GaussianProcessRegressor, list]] = Default_GPR(),
                 Nbar: Optional[Union[int, list]] = 100,
                 theta: Optional[Union[float, list]] = 0.0001):
        
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

        if type(self.GPR) == GaussianProcessRegressor or type(self.GPR) == Default_GPR:
            self.GPR = self.num_GPTrees*[self.GPR]


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, show_progress: Optional[bool]=False):

        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Building forest'):
            self.GPTrees.append(GPTree(self.GPR[i], self.Nbar[i], self.theta[i]))
        
        for i in tqdm(range(self.num_GPTrees), disable=not show_progress, desc='Training GPTrees'):
            self.GPTrees[i].fit(X_train, y_train, shuffle=True)

    def predict(self, X_test: np.ndarray, show_progress: Optional[bool]=False):
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
        joblib.dump(self, path)


        
        

