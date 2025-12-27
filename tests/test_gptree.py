import unittest
import numpy as np

# Add project root to Python path to allow direct import of pygptreeo
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from pygptreeo.adapters import SklearnGPAdapter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import warnings
from sklearn.exceptions import ConvergenceWarning

class TestGPTree(unittest.TestCase):

    def setUp(self):
        """Common setup for GPTree tests."""
        # Define a simple GPR for 1D data for most tests
        kernel_1d = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gpr_1d_template = Default_GPR(kernel=kernel_1d)

        # Define a simple GPR for 2D data for relevant tests
        kernel_2d = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=[1.0, 1.0], length_scale_bounds="fixed")
        self.gpr_2d_template = Default_GPR(kernel=kernel_2d)

        # Suppress console output from GPNode/GPTree - for now, accept prints
        pass

    def test_gptree_default_initialization(self):
        """Test GPTree creation with default parameters."""
        gpt = GPTree() # Uses Default_GPR() by default
        self.assertIsNotNone(gpt.root, "GPTree root should be initialized.")
        self.assertIsInstance(gpt.root, GPNode, "Root should be a GPNode instance.")
        self.assertIsInstance(gpt.GPR, SklearnGPAdapter, "Default GPR should be SklearnGPAdapter instance.")
        self.assertEqual(gpt.root.Nbar, 100, "Default Nbar should be 100.")
        self.assertEqual(gpt.theta, 0.0001, "Default theta should be 0.0001.")
        self.assertTrue(gpt.first_point, "first_point flag should be True initially.")
        self.assertEqual(gpt.n_features, 0, "n_features should be 0 initially.")

    def test_gptree_custom_initialization(self):
        """Test GPTree creation with custom parameters."""
        custom_nbar = 50
        custom_theta = 0.05
        # Use a fresh GPR instance for customization
        custom_kernel = ConstantKernel(0.5) * RBF(length_scale=0.5)
        custom_gpr = Default_GPR(kernel=custom_kernel, normalize_y=False)

        gpt = GPTree(GPR=custom_gpr, Nbar=custom_nbar, theta=custom_theta, use_calibrated_sigma=False)

        self.assertIsNotNone(gpt.root, "GPTree root should be initialized.")
        self.assertIsInstance(gpt.root, GPNode, "Root should be a GPNode instance.")
        self.assertEqual(gpt.root.Nbar, custom_nbar)
        self.assertEqual(gpt.theta, custom_theta)
        self.assertFalse(gpt.use_calibrated_sigma, "use_calibrated_sigma should be False.")

        # Check if the GPR in the tree is the one we passed (or a copy with same params)
        # GPTree root node gets a copy of the GPR instance
        self.assertIsInstance(gpt.GPR, SklearnGPAdapter)
        self.assertFalse(gpt.GPR.sklearn_gpr.normalize_y, "Custom GPR normalize_y should be False.")
        # Check root node's GPR
        self.assertIsInstance(gpt.root.my_GPR, SklearnGPAdapter)
        self.assertFalse(gpt.root.my_GPR.sklearn_gpr.normalize_y, "Root node's GPR normalize_y should be False.")
        # Compare kernel string representation as a proxy for checking if it's the custom one
        self.assertEqual(str(gpt.root.my_GPR.sklearn_gpr.kernel), str(custom_kernel))


    def test_gptree_initialization_sklearn_gpr(self):
        """Test GPTree creation with a direct scikit-learn GPR instance."""
        sklearn_gpr = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(1.0),
            alpha=1e-9,
            normalize_y=True
        )
        # To make it compatible with GPNode's expectation of kernel_alternatives and min_length_scale:
        sklearn_gpr.kernel_alternatives = [sklearn_gpr.kernel]
        sklearn_gpr.min_length_scale = 0.001

        gpt = GPTree(GPR=sklearn_gpr, Nbar=75)
        self.assertEqual(gpt.root.Nbar, 75)
        self.assertIsInstance(gpt.GPR, GaussianProcessRegressor)
        self.assertTrue(gpt.GPR.normalize_y)
        self.assertIsInstance(gpt.root.my_GPR, GaussianProcessRegressor)
        self.assertTrue(gpt.root.my_GPR.normalize_y)
        self.assertEqual(str(gpt.root.my_GPR.kernel), str(sklearn_gpr.kernel))


    # @ignore_warnings(category=ConvergenceWarning)
    def test_predict_basic_1d(self):
        """Test basic prediction with a 1D GPTree that has undergone at least one split."""
        custom_nbar = 2 # Ensure a split
        gpt = GPTree(GPR=self.gpr_1d_template, Nbar=custom_nbar, theta=0.001) # theta non-zero for prob_func

        X_train = np.array([[0.1], [0.2], [0.8], [0.9]])
        y_train = np.array([[1.0], [2.0], [8.0], [9.0]])
        sigma_train = np.array([0.1, 0.1, 0.1, 0.1])

        # Populate the tree enough to cause splits and have some data in leaves
        for i in range(X_train.shape[0]):
            x_sample = X_train[i:i+1, :]
            y_sample = y_train[i:i+1, :]
            sigma_sample = sigma_train[i]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gpt.update_tree(x_sample, y_sample, sigma_sample, allow_training=True) # Ensure leaf GPRs are trained

        # Ensure tree has split for a more meaningful predict test
        self.assertFalse(gpt.root.is_leaf, "Root should have split for this test.")
        if gpt.root.children: # Check if children exist
            for child in gpt.root.children:
                if child.is_leaf: # Only fit GPR if it's a leaf and has data
                    if child.n_points > 0:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            child.fit_my_GPR(force_training=True)

        X_test = np.array([[0.15], [0.5], [0.85]])

        # Test recursive prediction (default)
        y_pred_rec, y_std_rec = gpt.predict(X_test, mode='recursive')

        self.assertEqual(y_pred_rec.shape, (X_test.shape[0], 1), "Recursive prediction mean shape incorrect.")
        self.assertEqual(y_std_rec.shape, (X_test.shape[0], 1), "Recursive prediction std shape incorrect.")
        self.assertFalse(np.isnan(y_pred_rec).any(), "Recursive predictions should not be NaN.")
        self.assertFalse(np.isnan(y_std_rec).any(), "Recursive std deviations should not be NaN.")
        self.assertTrue(np.all(y_std_rec >= 0), "Recursive std deviations should be non-negative.")

        # Test loop prediction
        y_pred_loop, y_std_loop = gpt.predict(X_test, mode='loop')

        self.assertEqual(y_pred_loop.shape, (X_test.shape[0], 1), "Loop prediction mean shape incorrect.")
        self.assertEqual(y_std_loop.shape, (X_test.shape[0], 1), "Loop prediction std shape incorrect.")
        self.assertFalse(np.isnan(y_pred_loop).any(), "Loop predictions should not be NaN.")
        self.assertFalse(np.isnan(y_std_loop).any(), "Loop std deviations should not be NaN.")
        self.assertTrue(np.all(y_std_loop >= 0), "Loop std deviations should be non-negative.")

        # Optionally, check if recursive and loop predictions are close for this simple case
        # This might not always hold true perfectly due to floating point arithmetic or minor logic differences
        # For now, just checking shapes and NaNs is a good start.
        # np.testing.assert_allclose(y_pred_rec, y_pred_loop, rtol=1e-5, atol=1e-5, err_msg="Recursive and loop predictions are not close.")

    def test_predict_minimal_tree(self):
        """Test prediction with a minimal tree (single data point added)."""
        gpt = GPTree(GPR=self.gpr_1d_template, Nbar=10)
        X_test = np.array([[0.5]])

        # Prediction on an empty tree is problematic because n_features is not set.
        # GPTree.update_tree sets n_features from the first data point.
        # Let's add one point to define n_features, then predict.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gpt.update_tree(np.array([[0.1]]), np.array([[1.0]]), 0.1)
            gpt.root.fit_my_GPR(force_training=True) # Ensure the single node GPR is trained

        y_pred, y_std = gpt.predict(X_test)
        self.assertEqual(y_pred.shape, (X_test.shape[0], 1))
        self.assertEqual(y_std.shape, (X_test.shape[0], 1))
        self.assertFalse(np.isnan(y_pred).any())
        self.assertFalse(np.isnan(y_std).any())

    # @ignore_warnings(category=ConvergenceWarning)
    def test_update_tree_single_point_1d(self):
        """Test adding a single 1D data point to an empty tree."""
        gpt = GPTree(GPR=self.gpr_1d_template, Nbar=10)

        x_sample = np.array([[0.5]])
        y_sample = np.array([[1.0]]) # GPTree expects y to be 2D array for a single sample
        sigma_sample = 0.1

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gpt.update_tree(x_sample, y_sample, sigma_sample)

        self.assertFalse(gpt.first_point, "first_point flag should be False after first point.")
        self.assertEqual(gpt.n_features, 1, "n_features should be set to 1 for 1D data.")
        self.assertIsNotNone(gpt.root.my_X_data, "Root node's my_X_data should be initialized.")
        self.assertEqual(gpt.root.n_points, 1, "Root node should have 1 training point.")
        self.assertTrue(np.array_equal(gpt.root.my_X_data, x_sample), "Root node's X data incorrect.")
        self.assertTrue(np.array_equal(gpt.root.my_y_data, y_sample), "Root node's y data incorrect.")
        self.assertEqual(gpt.root.my_sigma_data[0, 0], sigma_sample, "Root node's sigma data incorrect.")
        self.assertTrue(gpt.root.is_leaf, "Root node should still be a leaf.")

    # @ignore_warnings(category=ConvergenceWarning)
    def test_update_tree_multiple_points_no_split_2d(self):
        """Test adding multiple 2D data points, not enough to cause a split."""
        custom_nbar = 5
        gpt = GPTree(GPR=self.gpr_2d_template, Nbar=custom_nbar)

        X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_train = np.array([[1.0], [2.0], [3.0]])
        sigma_train = np.array([0.1, 0.12, 0.11])

        for i in range(X_train.shape[0]):
            x_sample = X_train[i:i+1, :]
            y_sample = y_train[i:i+1, :]
            sigma_sample = sigma_train[i]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gpt.update_tree(x_sample, y_sample, sigma_sample)

        self.assertEqual(gpt.n_features, 2, "n_features should be set to 2 for 2D data.")
        self.assertEqual(gpt.root.n_points, 3, f"Root node should have 3 training points, got {gpt.root.n_points}.")
        self.assertTrue(gpt.root.is_leaf, "Root node should still be a leaf as Nbar not reached.")
        # Data is stored in reverse order (newest first)
        self.assertTrue(np.array_equal(gpt.root.my_X_data, X_train[::-1]))
        self.assertTrue(np.array_equal(gpt.root.my_y_data, y_train[::-1]))

    # @ignore_warnings(category=ConvergenceWarning)
    def test_update_tree_causes_split_1d(self):
        """Test adding 1D data points until Nbar is reached and a split occurs."""
        custom_nbar = 3 # Small Nbar to trigger split easily
        # Ensure retrain_every_n_points is <= Nbar for GPR model to be fit before split logic uses it.
        # The GPNode in GPTree is initialized with default retrain_every_n_points=1
        gpt = GPTree(GPR=self.gpr_1d_template, Nbar=custom_nbar)

        X_train = np.array([[0.1], [0.2], [0.3], [0.4]])
        y_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        sigma_train = np.array([0.1, 0.1, 0.1, 0.1])

        # Add points up to Nbar - 1
        for i in range(custom_nbar - 1):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gpt.update_tree(X_train[i:i+1, :], y_train[i:i+1, :], sigma_train[i])
            self.assertTrue(gpt.root.is_leaf, f"Root should be leaf before Nbar reached (point {i+1})")
            self.assertEqual(gpt.root.n_points, i + 1)

        # Add the Nbar-th point - this should trigger the split
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gpt.update_tree(X_train[custom_nbar-1:custom_nbar-1+1, :], y_train[custom_nbar-1:custom_nbar-1+1, :], sigma_train[custom_nbar-1])

        self.assertFalse(gpt.root.is_leaf, "Root node should have split and no longer be a leaf.")
        self.assertIsNotNone(gpt.root.children, "Root node should have children after split.")
        self.assertEqual(len(gpt.root.children), 2, "Root node should have two children.")

        left_child, right_child = gpt.root.children
        self.assertIsInstance(left_child, GPNode, "Left child should be a GPNode.")
        self.assertIsInstance(right_child, GPNode, "Right child should be a GPNode.")

        # Check if original root node's GPR and data are deleted (as per current GPTree.update_tree logic)
        self.assertFalse(hasattr(gpt.root, 'my_GPR'), "Split parent node should not have my_GPR.") # This will fail if GPNode.delete_my_GPR() is not called or effective
        self.assertFalse(hasattr(gpt.root, 'my_X_data'), "Split parent node should not have my_X_data.") # This will fail if GPNode.delete_data() is not called or effective

        # Check that data is distributed (total points in children should be custom_nbar)
        total_points_in_children = left_child.n_points + right_child.n_points
        self.assertEqual(total_points_in_children, custom_nbar,
                         f"Total points in children ({total_points_in_children}) should equal Nbar ({custom_nbar}).")

        # Check if children are leaves and have GPRs
        self.assertTrue(left_child.is_leaf, "Left child should be a leaf node.")
        self.assertTrue(hasattr(left_child, 'my_GPR') and left_child.my_GPR is not None, "Left child should have a GPR instance.")
        self.assertTrue(right_child.is_leaf, "Right child should be a leaf node.")
        self.assertTrue(hasattr(right_child, 'my_GPR') and right_child.my_GPR is not None, "Right child should have a GPR instance.")

        # Add one more point, it should go to one of the children
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gpt.update_tree(X_train[custom_nbar:custom_nbar+1, :], y_train[custom_nbar:custom_nbar+1, :], sigma_train[custom_nbar])
        total_points_in_children_after_one_more = left_child.n_points + right_child.n_points
        self.assertEqual(total_points_in_children_after_one_more, custom_nbar + 1,
                         "Total points in children should be Nbar + 1 after one more point.")


if __name__ == '__main__':
    unittest.main()
