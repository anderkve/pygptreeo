from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from typing import Callable, Optional, Type, Union

class Default_GPR(GaussianProcessRegressor):
        """ The default GaussianProcessRegressor of GPTree """
        def __init__(self, kernel=None, *, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, n_targets=None, random_state=None):
            self.kernel_alternatives = [
                ConstantKernel() * Matern(nu=1.5), 
                ConstantKernel() * Matern(nu=2.5),
                ConstantKernel() * RBF(),
            ]
            # self.kernel = ConstantKernel() * Matern(nu=1.5)
            self.kernel = self.kernel_alternatives[0]
            self.min_length_scale = 0.001
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts_optimizer = n_restarts_optimizer
            self.normalize_y = normalize_y
            self.copy_X_train = copy_X_train
            self.n_targets = n_targets
            self.random_state = random_state
    