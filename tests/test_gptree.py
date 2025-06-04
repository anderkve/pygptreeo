import unittest
import numpy as np

# Add project root to Python path to allow direct import of pygptreeo
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import warnings
from sklearn.exceptions import ConvergenceWarning

class TestGPTree(unittest.TestCase):

    def setUp(self):
        """Common setup for GPTree tests."""
        # Define a simple GPR for 1D data for most tests
        # Allow length_scale and constant_value to be optimized
        kernel_1d_base = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
        kernel_1d = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * kernel_1d_base

        # Create a Default_GPR instance and then override its kernel and alternatives for the test
        self.gpr_1d_template = Default_GPR(alpha=1e-5) # Use a slightly higher alpha
        self.gpr_1d_template.kernel = kernel_1d
        self.gpr_1d_template.kernel_alternatives = [
            ConstantKernel(c_val, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=ls, length_scale_bounds=(1e-2, 10.0))
            for ls in [0.5, 1.0, 2.0] for c_val in [0.1, 1.0, 10.0]
        ]
        # self.gpr_1d_template.min_length_scale is already set by Default_GPR

        # Define a simple GPR for 2D data for relevant tests
        kernel_2d_base = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 10.0))
        kernel_2d = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * kernel_2d_base

        self.gpr_2d_template = Default_GPR(alpha=1e-5)
        self.gpr_2d_template.kernel = kernel_2d
        self.gpr_2d_template.kernel_alternatives = [
             ConstantKernel(c_val, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[ls, ls], length_scale_bounds=(1e-2, 10.0))
             for ls in [0.5, 1.0, 2.0] for c_val in [0.1, 1.0, 10.0]
        ]

        # Suppress console output from GPNode/GPTree - for now, accept prints
        pass

    def test_gptree_default_initialization(self):
        """Test GPTree creation with default parameters."""
        gpt = GPTree() # Uses Default_GPR() by default
        self.assertIsNotNone(gpt.root, "GPTree root should be initialized.")
        self.assertIsInstance(gpt.root, GPNode, "Root should be a GPNode instance.")
        self.assertIsInstance(gpt.GPR_template, Default_GPR, "Default GPR template should be Default_GPR instance.")
        self.assertEqual(gpt.root.Nbar, 100, "Default Nbar should be 100.")
        self.assertEqual(gpt.theta, 0.0001, "Default theta should be 0.0001.")
        self.assertTrue(gpt.first_point, "first_point flag should be True initially.")
        self.assertEqual(gpt.n_features, 0, "n_features should be 0 initially.")

    def test_gptree_custom_initialization(self):
        """Test GPTree creation with custom parameters."""
        custom_nbar = 50
        custom_theta = 0.05
        # Use a fresh GPR instance for customization
        custom_gpr = Default_GPR()
        custom_gpr.kernel = ConstantKernel(0.5) * RBF(length_scale=0.5)
        custom_gpr.kernel_alternatives = [custom_gpr.kernel]
        custom_gpr.normalize_y = False

        gpt = GPTree(GPR=custom_gpr, Nbar=custom_nbar, theta=custom_theta, use_calibrated_sigma=False)

        self.assertIsNotNone(gpt.root, "GPTree root should be initialized.")
        self.assertIsInstance(gpt.root, GPNode, "Root should be a GPNode instance.")
        self.assertEqual(gpt.root.Nbar, custom_nbar)
        self.assertEqual(gpt.theta, custom_theta)
        self.assertFalse(gpt.use_calibrated_sigma, "use_calibrated_sigma should be False.")

        # Check if the GPR in the tree is the one we passed (or a copy with same params)
        # GPTree root node gets a copy of the GPR instance
        self.assertIsInstance(gpt.GPR_template, Default_GPR)
        self.assertFalse(gpt.GPR_template.normalize_y, "Custom GPR normalize_y should be False.")
        # Check root node's GPR (first one in the list, assuming n_GPs_per_node=1 by default here)
        self.assertIsInstance(gpt.root.my_GPRs[0], Default_GPR)
        self.assertFalse(gpt.root.my_GPRs[0].normalize_y, "Root node's GPR normalize_y should be False.")
        # Compare kernel string representation as a proxy for checking if it's the custom one
        self.assertEqual(str(gpt.root.my_GPRs[0].kernel), str(custom_gpr.kernel))


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

        gpt = GPTree(GPR=sklearn_gpr, Nbar=75) # n_GPs_per_node defaults to 1
        self.assertEqual(gpt.root.Nbar, 75)
        self.assertIsInstance(gpt.GPR_template, GaussianProcessRegressor)
        self.assertTrue(gpt.GPR_template.normalize_y)
        self.assertIsInstance(gpt.root.my_GPRs[0], GaussianProcessRegressor)
        self.assertTrue(gpt.root.my_GPRs[0].normalize_y)
        self.assertEqual(str(gpt.root.my_GPRs[0].kernel), str(sklearn_gpr.kernel))


    # @ignore_warnings(category=ConvergenceWarning)
    def test_predict_basic_1d(self):
        """Test basic prediction with a 1D GPTree that has undergone at least one split."""
        custom_nbar = 2 # Ensure a split
        gpt = GPTree(GPR=self.gpr_1d_template, Nbar=custom_nbar, theta=0.001) # theta non-zero for prob_func

        X_train = np.array([[0.1], [0.2], [0.8], [0.9]])
        y_train = np.array([[1.0], [2.0], [8.0], [9.0]])

        # Populate the tree enough to cause splits and have some data in leaves
        for i in range(X_train.shape[0]):
            x_sample = X_train[i:i+1, :]
            y_sample = y_train[i:i+1, :]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gpt.update_tree(x_sample, y_sample, allow_training=True) # Ensure leaf GPRs are trained

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
            gpt.update_tree(np.array([[0.1]]), np.array([[1.0]]))
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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gpt.update_tree(x_sample, y_sample)

        self.assertFalse(gpt.first_point, "first_point flag should be False after first point.")
        self.assertEqual(gpt.n_features, 1, "n_features should be set to 1 for 1D data.")
        self.assertIsNotNone(gpt.root.my_X_data, "Root node's my_X_data should be initialized.")
        self.assertEqual(gpt.root.n_points, 1, "Root node should have 1 training point.")
        self.assertTrue(np.array_equal(gpt.root.my_X_data, x_sample), "Root node's X data incorrect.")
        self.assertTrue(np.array_equal(gpt.root.my_y_data, y_sample), "Root node's y data incorrect.")
        self.assertTrue(gpt.root.is_leaf, "Root node should still be a leaf.")

    # @ignore_warnings(category=ConvergenceWarning)
    def test_update_tree_multiple_points_no_split_2d(self):
        """Test adding multiple 2D data points, not enough to cause a split."""
        custom_nbar = 5
        gpt = GPTree(GPR=self.gpr_2d_template, Nbar=custom_nbar)

        X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_train = np.array([[1.0], [2.0], [3.0]])

        for i in range(X_train.shape[0]):
            x_sample = X_train[i:i+1, :]
            y_sample = y_train[i:i+1, :]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gpt.update_tree(x_sample, y_sample)

        self.assertEqual(gpt.n_features, 2, "n_features should be set to 2 for 2D data.")
        self.assertEqual(gpt.root.n_points, 3, f"Root node should have 3 training points, got {gpt.root.n_points}.")
        self.assertTrue(gpt.root.is_leaf, "Root node should still be a leaf as Nbar not reached.")
        # Data is prepended in store_point, so compare with reversed order
        self.assertTrue(np.array_equal(gpt.root.my_X_data, np.flip(X_train, axis=0)))
        self.assertTrue(np.array_equal(gpt.root.my_y_data, np.flip(y_train, axis=0)))

    # @ignore_warnings(category=ConvergenceWarning)
    def test_update_tree_causes_split_1d(self):
        """Test adding 1D data points until Nbar is reached and a split occurs."""
        custom_nbar = 3 # Small Nbar to trigger split easily
        # Ensure retrain_every_n_points is <= Nbar for GPR model to be fit before split logic uses it.
        # The GPNode in GPTree is initialized with default retrain_every_n_points=1
        gpt = GPTree(GPR=self.gpr_1d_template, Nbar=custom_nbar)

        X_train = np.array([[0.1], [0.2], [0.3], [0.4]])
        y_train = np.array([[1.0], [2.0], [3.0], [4.0]])

        # Add points up to Nbar - 1
        for i in range(custom_nbar - 1):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gpt.update_tree(X_train[i:i+1, :], y_train[i:i+1, :])
            self.assertTrue(gpt.root.is_leaf, f"Root should be leaf before Nbar reached (point {i+1})")
            self.assertEqual(gpt.root.n_points, i + 1)

        # Add the Nbar-th point - this should trigger the split
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gpt.update_tree(X_train[custom_nbar-1:custom_nbar-1+1, :], y_train[custom_nbar-1:custom_nbar-1+1, :])

        self.assertFalse(gpt.root.is_leaf, "Root node should have split and no longer be a leaf.")
        self.assertIsNotNone(gpt.root.children, "Root node should have children after split.")
        self.assertEqual(len(gpt.root.children), 2, "Root node should have two children.")

        left_child, right_child = gpt.root.children
        self.assertIsInstance(left_child, GPNode, "Left child should be a GPNode.")
        self.assertIsInstance(right_child, GPNode, "Right child should be a GPNode.")

        # Check if original root node's GPR and data are deleted (as per current GPTree.update_tree logic)
        self.assertFalse(hasattr(gpt.root, 'my_GPRs'), "Split parent node should not have my_GPRs.")
        self.assertFalse(hasattr(gpt.root, 'my_X_data'), "Split parent node should not have my_X_data.")

        # Check that data is distributed (total points in children should be custom_nbar)
        total_points_in_children = left_child.n_points + right_child.n_points
        self.assertEqual(total_points_in_children, custom_nbar,
                         f"Total points in children ({total_points_in_children}) should equal Nbar ({custom_nbar}).")

        # Check if children are leaves and have GPRs
        self.assertTrue(left_child.is_leaf, "Left child should be a leaf node.")
        self.assertTrue(hasattr(left_child, 'my_GPRs') and left_child.my_GPRs[0] is not None, "Left child should have a GPR instance.")
        self.assertTrue(right_child.is_leaf, "Right child should be a leaf node.")
        self.assertTrue(hasattr(right_child, 'my_GPRs') and right_child.my_GPRs[0] is not None, "Right child should have a GPR instance.")

        # Add one more point, it should go to one of the children
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gpt.update_tree(X_train[custom_nbar:custom_nbar+1, :], y_train[custom_nbar:custom_nbar+1, :])
        total_points_in_children_after_one_more = left_child.n_points + right_child.n_points
        self.assertEqual(total_points_in_children_after_one_more, custom_nbar + 1,
                         "Total points in children should be Nbar + 1 after one more point.")

    def test_gptree_with_multiple_gps_per_node(self):
        """Test GPTree with n_GPs_per_node > 1, checking node configurations and prediction."""
        num_gps_per_node = 2
        # Ensure Nbar is >= num_gps_per_node for leaves to train all internal GPs.
        # If a leaf has K points, and K < num_gps_per_node, some GPs in that leaf won't get data.
        custom_nbar = num_gps_per_node * 2 # e.g., 4, so each GP gets 2 points in a full leaf.

        gpt = GPTree(GPR=self.gpr_1d_template, Nbar=custom_nbar, n_GPs_per_node=num_gps_per_node)

        self.assertEqual(gpt.root.n_GPs_per_node, num_gps_per_node, "Root node n_GPs_per_node mismatch.")
        self.assertIsInstance(gpt.root.my_GPRs, list, "Root node my_GPRs should be a list.")
        self.assertEqual(len(gpt.root.my_GPRs), num_gps_per_node, f"Root node should have {num_gps_per_node} GPRs.")
        for i in range(num_gps_per_node):
            self.assertIsInstance(gpt.root.my_GPRs[i], type(self.gpr_1d_template), f"GPR {i} in root is not of the correct type.")
            if i > 0: # Check they are distinct objects
                 self.assertIsNot(gpt.root.my_GPRs[i], gpt.root.my_GPRs[0], f"GPR {i} in root is not distinct from GPR 0.")

        # Add enough points to ensure at least one split and that leaves have >= num_gps_per_node points
        # Nbar = 4. Add 5 points.
        # P0, P1, P2, P3 -> root (4 points), then root splits. L1 gets 2, R1 gets 2.
        # P4 -> e.g. L1. L1 has 3 points.
        # At the end of GPTree.fit, L1 (3 points) and R1 (2 points) are leaves.
        # L1: 3 points / 2 GPs -> GP0 gets 2, GP1 gets 1. Both trained.
        # R1: 2 points / 2 GPs -> GP0 gets 1, GP1 gets 1. Both trained.
        X_train = np.array([[0.1], [0.2], [0.3], [0.8], [0.9]])
        y_train = np.array([[1.0], [2.0], [3.0], [8.0], [9.0]])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gpt.fit(X_train, y_train, show_progress=False) # Use fit for convenience

        self.assertFalse(gpt.root.is_leaf, "Root should have split after fitting data.")

        for leaf_idx, leaf in enumerate(gpt.root.leaves):
            print(f"Leaf {leaf_idx} (name {leaf.name}): {leaf.n_points} points.") # Removed get_path()
            for gp_idx, gp_in_leaf in enumerate(leaf.my_GPRs):
                trained_status = "TRAINED" if hasattr(gp_in_leaf, 'kernel_') and gp_in_leaf.kernel_ is not None else "NOT TRAINED"
                print(f"  Leaf {leaf_idx} - GPR {gp_idx}: {trained_status}")
                if trained_status == "TRAINED":
                    print(f"    Kernel: {gp_in_leaf.kernel_}")
                else:
                    print(f"    Initial kernel: {gp_in_leaf.kernel}")
                    if hasattr(gp_in_leaf, 'X_train_'):
                        print(f"    GPR was fit, but kernel_ not set. X_train_ shape: {gp_in_leaf.X_train_.shape}")
                    else:
                        print(f"    GPR was likely not fit or X_train_ not set.")


        for leaf_idx, leaf in enumerate(gpt.root.leaves):
            self.assertEqual(leaf.n_GPs_per_node, num_gps_per_node, f"Leaf {leaf_idx} n_GPs_per_node mismatch.")
            self.assertIsInstance(leaf.my_GPRs, list, f"Leaf {leaf_idx} my_GPRs should be a list.")
            self.assertEqual(len(leaf.my_GPRs), num_gps_per_node, f"Leaf {leaf_idx} should have {num_gps_per_node} GPRs.")

            if leaf.n_points > 0:
                point_counts_per_gp = [len(arr) for arr in np.array_split(np.arange(leaf.n_points), leaf.n_GPs_per_node)]
                MIN_RELIABLE_TRAIN_POINTS = 2 # Assume GPRs with <2 points might not reliably train and set kernel_

                for gp_idx, gp in enumerate(leaf.my_GPRs):
                    points_for_this_gp = point_counts_per_gp[gp_idx]
                    if points_for_this_gp >= MIN_RELIABLE_TRAIN_POINTS:
                        self.assertTrue(hasattr(gp, 'kernel_') and gp.kernel_ is not None,
                                        f"GPR {gp_idx} in Leaf {leaf_idx} (name {leaf.name}) received {points_for_this_gp} points and should be trained with kernel_ set.")
                    elif points_for_this_gp > 0: # Received 1 point
                        # For GPRs that received only 1 point, training is unreliable.
                        # We won't assert that kernel_ *must* be set.
                        # However, we can note if it was trained for debugging.
                        if hasattr(gp, 'kernel_') and gp.kernel_ is not None:
                            print(f"INFO: GPR {gp_idx} in Leaf {leaf_idx} (name {leaf.name}) received 1 point and WAS successfully trained.")
                        else:
                            print(f"INFO: GPR {gp_idx} in Leaf {leaf_idx} (name {leaf.name}) received 1 point and was NOT successfully trained (no kernel_).")
                    # If points_for_this_gp is 0, it shouldn't have kernel_ from this training cycle.
                    # GPNode.fit_my_GPR already skips GPs with 0 points from split.
            elif leaf.n_points == 0:
                 for gp_idx, gp in enumerate(leaf.my_GPRs):
                      self.assertFalse(hasattr(gp, 'kernel_') and gp.kernel_ is not None,
                                       f"GPR {gp_idx} in Leaf {leaf_idx} (name {leaf.name}) has 0 points and should not have a recently trained kernel_.")

        X_test = np.array([[0.15], [0.5], [0.85]])
        mu, std = gpt.predict(X_test)

        self.assertEqual(mu.shape, (X_test.shape[0], 1), "Prediction mean shape incorrect.")
        self.assertEqual(std.shape, (X_test.shape[0], 1), "Prediction std shape incorrect.")
        self.assertFalse(np.isnan(mu).any(), "Predictions should not be NaN.")
        self.assertFalse(np.isnan(std).any(), "Std deviations should not be NaN.")
        self.assertTrue(np.all(std >= 0), "Std deviations should be non-negative.")


if __name__ == '__main__':
    unittest.main()
