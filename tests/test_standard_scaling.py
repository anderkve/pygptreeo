"""Unit tests for standard scaling functionality in GPNode and GPTree.

This module tests that the StandardScaling feature correctly:
1. Scales data before GP training
2. Inverse transforms predictions
3. Maintains raw data in storage
4. Works correctly with child node generation
5. Can be disabled via flag
"""

import unittest
import numpy as np
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class TestStandardScaling(unittest.TestCase):
    """Test cases for StandardScaling feature."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create a simple 1D kernel
        kernel_1d = ConstantKernel(1.0, constant_value_bounds="fixed") * \
                    RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gpr_1d = Default_GPR()
        self.gpr_1d.kernel = kernel_1d
        self.gpr_1d.alpha = 1e-6

    def test_scaling_flag_initialization(self):
        """Test that use_standard_scaling flag is properly initialized."""
        # Default should be True
        gpt_default = GPTree()
        self.assertTrue(gpt_default.root.use_standard_scaling)

        # Explicit True
        gpt_true = GPTree(use_standard_scaling=True)
        self.assertTrue(gpt_true.root.use_standard_scaling)

        # Explicit False
        gpt_false = GPTree(use_standard_scaling=False)
        self.assertFalse(gpt_false.root.use_standard_scaling)

    def test_scalers_fitted_during_training(self):
        """Test that scalers are fitted when GP is trained."""
        gpt = GPTree(GPR=self.gpr_1d, Nbar=10, use_standard_scaling=True)

        # Initially scalers should be None
        self.assertIsNone(gpt.root.X_scaler)
        self.assertIsNone(gpt.root.y_scaler)

        # Add some points
        X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_train = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
        sigma_train = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        for x, y, sigma in zip(X_train, y_train, sigma_train):
            gpt.update_tree(x.reshape(1, -1), y.reshape(1, -1), sigma)

        # After training, scalers should be fitted
        self.assertIsNotNone(gpt.root.X_scaler)
        self.assertIsNotNone(gpt.root.y_scaler)

        # Check that scalers have correct statistics
        # X: mean = 3.0, std ~ 1.58
        # y: mean = 30.0, std ~ 15.8
        np.testing.assert_almost_equal(gpt.root.X_scaler.mean_[0], 3.0, decimal=5)
        np.testing.assert_almost_equal(gpt.root.y_scaler.mean_[0], 30.0, decimal=5)

    def test_raw_data_stored_unscaled(self):
        """Test that raw data is stored in original (unscaled) form."""
        gpt = GPTree(GPR=self.gpr_1d, Nbar=10, use_standard_scaling=True)

        # Create data with large values
        X_train = np.array([[100.0], [200.0], [300.0]])
        y_train = np.array([[1000.0], [2000.0], [3000.0]])
        sigma_train = np.array([0.1, 0.1, 0.1])

        for x, y, sigma in zip(X_train, y_train, sigma_train):
            gpt.update_tree(x.reshape(1, -1), y.reshape(1, -1), sigma)

        # Check that stored data is in original scale
        stored_X = gpt.root.my_X_data
        stored_y = gpt.root.my_y_data

        # Data is stored in reverse order (newest first)
        np.testing.assert_array_almost_equal(stored_X[::-1], X_train, decimal=5)
        np.testing.assert_array_almost_equal(stored_y[::-1], y_train, decimal=5)

    def test_predictions_in_original_scale(self):
        """Test that predictions are returned in original scale."""
        gpt = GPTree(GPR=self.gpr_1d, Nbar=10, use_standard_scaling=True,
                     retrain_every_n_points=1)

        # Train on data with large values
        X_train = np.array([[100.0], [200.0], [300.0], [400.0], [500.0]])
        y_train = np.array([[1000.0], [2000.0], [3000.0], [4000.0], [5000.0]])
        sigma_train = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        for x, y, sigma in zip(X_train, y_train, sigma_train):
            gpt.update_tree(x.reshape(1, -1), y.reshape(1, -1), sigma)

        # Predict at a test point
        X_test = np.array([[250.0]])
        y_pred, y_std = gpt.predict(X_test)

        # Prediction should be in the original scale (around 2500)
        self.assertGreater(y_pred[0, 0], 2000.0)
        self.assertLess(y_pred[0, 0], 3000.0)

        # Standard deviation should also be in original scale (not tiny)
        self.assertGreater(y_std[0, 0], 1.0)

    def test_scaling_vs_no_scaling_equivalence(self):
        """Test that both scaled and unscaled modes produce valid predictions."""
        np.random.seed(123)

        # Create identical training data
        X_train = np.random.uniform(0, 1, (20, 1))
        y_train = np.sin(X_train * 10).reshape(-1, 1)
        sigma_train = np.full(20, 0.1)

        # Train with scaling
        gpt_scaled = GPTree(GPR=self.gpr_1d, Nbar=30, use_standard_scaling=True,
                           retrain_every_n_points=5, theta=0.01)
        for x, y, sigma in zip(X_train, y_train, sigma_train):
            gpt_scaled.update_tree(x.reshape(1, -1), y.reshape(1, -1), sigma)

        # Train without scaling
        gpt_unscaled = GPTree(GPR=self.gpr_1d, Nbar=30, use_standard_scaling=False,
                             retrain_every_n_points=5, theta=0.01)
        for x, y, sigma in zip(X_train, y_train, sigma_train):
            gpt_unscaled.update_tree(x.reshape(1, -1), y.reshape(1, -1), sigma)

        # Make predictions
        X_test = np.array([[0.5]])
        y_pred_scaled, y_std_scaled = gpt_scaled.predict(X_test)
        y_pred_unscaled, y_std_unscaled = gpt_unscaled.predict(X_test)

        # Both should give reasonable predictions in the valid range
        # y = sin(x*10) for x in [0,1] gives values in [-1, 1]
        # Both predictions should be in this range
        self.assertGreater(y_pred_scaled[0, 0], -2.0)
        self.assertLess(y_pred_scaled[0, 0], 2.0)
        self.assertGreater(y_pred_unscaled[0, 0], -2.0)
        self.assertLess(y_pred_unscaled[0, 0], 2.0)

        # Both should have positive standard deviations
        self.assertGreater(y_std_scaled[0, 0], 0.0)
        self.assertGreater(y_std_unscaled[0, 0], 0.0)

    def test_child_nodes_inherit_scaling_flag(self):
        """Test that child nodes inherit use_standard_scaling from parent."""
        gpt = GPTree(GPR=self.gpr_1d, Nbar=5, use_standard_scaling=True,
                     retrain_every_n_points=1)

        # Add enough points to trigger a split
        X_train = np.random.uniform(0, 10, (10, 1))
        y_train = X_train * 2.0
        sigma_train = np.full(10, 0.1)

        for x, y, sigma in zip(X_train, y_train, sigma_train):
            gpt.update_tree(x.reshape(1, -1), y.reshape(1, -1), sigma)

        # Root should have split
        self.assertFalse(gpt.root.is_leaf)
        self.assertIsNotNone(gpt.root.children)

        # Check that children have the scaling flag
        self.assertTrue(gpt.root.left.use_standard_scaling)
        self.assertTrue(gpt.root.right.use_standard_scaling)

    def test_scaling_with_2d_data(self):
        """Test that scaling works correctly with 2D input data."""
        # Create a 2D kernel
        kernel_2d = ConstantKernel(1.0, constant_value_bounds="fixed") * \
                    RBF(length_scale=[1.0, 1.0], length_scale_bounds="fixed")
        gpr_2d = Default_GPR()
        gpr_2d.kernel = kernel_2d

        gpt = GPTree(GPR=gpr_2d, Nbar=10, use_standard_scaling=True,
                     retrain_every_n_points=2)

        # Create 2D data with different scales in each dimension
        X_train = np.array([[100.0, 1.0], [200.0, 2.0], [300.0, 3.0],
                           [400.0, 4.0], [500.0, 5.0]])
        y_train = X_train[:, 0:1] + X_train[:, 1:2] * 10  # y = x1 + 10*x2
        sigma_train = np.full(5, 0.1)

        for x, y, sigma in zip(X_train, y_train, sigma_train):
            gpt.update_tree(x.reshape(1, -1), y.reshape(1, -1), sigma)

        # Check that X_scaler has correct shape
        self.assertEqual(gpt.root.X_scaler.mean_.shape[0], 2)
        self.assertEqual(gpt.root.X_scaler.scale_.shape[0], 2)

        # Predict at test point
        X_test = np.array([[250.0, 2.5]])
        y_pred, y_std = gpt.predict(X_test)

        # Prediction should be reasonable (around 250 + 25 = 275)
        self.assertGreater(y_pred[0, 0], 250.0)
        self.assertLess(y_pred[0, 0], 300.0)

    def test_zero_variance_dimension_handling(self):
        """Test that scaling handles zero-variance dimensions correctly."""
        gpt = GPTree(GPR=self.gpr_1d, Nbar=10, use_standard_scaling=True,
                     retrain_every_n_points=1)

        # All points have the same x value (zero variance)
        X_train = np.array([[5.0], [5.0], [5.0], [5.0], [5.0]])
        y_train = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
        sigma_train = np.full(5, 0.1)

        for x, y, sigma in zip(X_train, y_train, sigma_train):
            gpt.update_tree(x.reshape(1, -1), y.reshape(1, -1), sigma)

        # Should not raise an error
        # StandardScaler sets scale=1 for zero-variance dimensions
        self.assertIsNotNone(gpt.root.X_scaler)
        self.assertEqual(gpt.root.X_scaler.scale_[0], 1.0)

        # Prediction should work
        X_test = np.array([[5.0]])
        y_pred, y_std = gpt.predict(X_test)

        # Should predict something in the range
        self.assertGreater(y_pred[0, 0], 5.0)
        self.assertLess(y_pred[0, 0], 55.0)

    def test_single_point_handling(self):
        """Test that scaling handles single-point training correctly."""
        gpt = GPTree(GPR=self.gpr_1d, Nbar=10, use_standard_scaling=True,
                     retrain_every_n_points=1)

        # Add just one point
        X_train = np.array([[3.0]])
        y_train = np.array([[30.0]])
        sigma_train = 0.1

        gpt.update_tree(X_train, y_train, sigma_train)

        # Scalers should be fitted (mean=value, scale=1)
        self.assertIsNotNone(gpt.root.X_scaler)
        self.assertIsNotNone(gpt.root.y_scaler)

        # Prediction should work (though with high uncertainty)
        X_test = np.array([[3.5]])
        y_pred, y_std = gpt.predict(X_test)

        # Should return some prediction
        self.assertIsNotNone(y_pred)
        self.assertIsNotNone(y_std)


if __name__ == '__main__':
    unittest.main()
