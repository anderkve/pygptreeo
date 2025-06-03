import gpytorch
import torch
import numpy as np

# Define the ExactGPModel class
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_dims):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=n_dims))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPyTorchGPR:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.model = None
        self.likelihood = None
        self.kernel_alternatives = [
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.n_dims))
        ]
        self.min_length_scale = 0.001

    def fit(self, X, y, training_iter=50):
        train_x = torch.from_numpy(X).float()
        train_y = torch.from_numpy(y).float()

        # Initialize likelihood and model if they don't exist or if data dimensions change
        if self.likelihood is None or self.model is None or self.model.covar_module.base_kernel.ard_num_dims != self.n_dims:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # Use the first kernel from alternatives for now
            kernel = self.kernel_alternatives[0]
            self.model = ExactGPModel(train_x, train_y, self.likelihood, self.n_dims)
            # Re-assign the chosen kernel to the model if it's different
            if self.model.covar_module != kernel:
                 self.model.covar_module = kernel


        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    def predict(self, X, return_std=True):
        if self.model is None or self.likelihood is None:
            raise Exception("Model not trained yet. Call fit() first.")

        self.model.eval()
        self.likelihood.eval()

        test_x = torch.from_numpy(X).float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))

        mean = observed_pred.mean.numpy()
        if return_std:
            std = observed_pred.stddev.numpy()
            return mean, std
        return mean
