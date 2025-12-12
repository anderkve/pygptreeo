import unittest
import numpy as np

# Add project root to Python path to allow direct import of pygptreeo
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from sklearn.gaussian_process.kernels import RBF

import warnings
from sklearn.exceptions import ConvergenceWarning

class TestGPNode(unittest.TestCase):

    def setUp(self):
        """Set up common resources for tests."""
        self.default_gpr_instance = Default_GPR()

        # A simple kernel for 1D data
        kernel_1d = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gpr_for_1d = Default_GPR()
        self.gpr_for_1d.kernel = kernel_1d
        self.gpr_for_1d.kernel_alternatives = [kernel_1d]


    def test_gpnode_creation_default_gpr(self):
        """Test basic GPNode creation with Default_GPR."""
        try:
            node = GPNode(0, my_GPR=self.default_gpr_instance, Nbar=50, name="init_node_1")
            self.assertIsNotNone(node, "Node should not be None")
            self.assertEqual(node.value, 0, "Node value should be 0 (n_points)")
            self.assertEqual(node.Nbar, 50, "Nbar should be set to 50")
            self.assertTrue(node.is_leaf, "A new node should be a leaf")
            self.assertIsNone(node.children, "A new node should not have children")
            self.assertIsInstance(node.my_GPR, Default_GPR, "my_GPR should be an instance of Default_GPR")
        except Exception as e:
            self.fail(f"GPNode creation raised an exception: {e}")

    def test_gpnode_creation_custom_params(self):
        """Test GPNode creation with some custom parameters."""
        try:
            # Note: GPNode value is n_points, which is managed internally by store_point.
            # Direct initialization of value in constructor is for binarytree.Node, but GPNode overrides it.
            node = GPNode(0, my_GPR=Default_GPR(normalize_y=False), Nbar=100, split_position_method='mean', name="custom_node_1")
            self.assertEqual(node.n_points, 0)
            self.assertEqual(node.Nbar, 100)
            self.assertEqual(node.split_position_method, 'mean')
            self.assertEqual(node.name, "custom_node_1")
            self.assertFalse(node.my_GPR.normalize_y, "GPR normalize_y should be False")
        except Exception as e:
            self.fail(f"GPNode creation with custom params raised an exception: {e}")

    def test_gpnode_init_training_set(self):
        """Test the init_data_set method."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, name="init_set_node")
        n_features = 1
        node.init_data_set(n_features)

        self.assertEqual(node.my_X_data.shape, (0, n_features), "my_X_data shape incorrect")
        self.assertEqual(node.my_y_data.shape, (0, 1), "my_y_data shape incorrect")
        self.assertEqual(node.my_sigma_data.shape, (0, 1), "my_sigma_data shape incorrect")
        self.assertEqual(node.n_features, n_features, "n_features not set correctly")
        self.assertEqual(node.n_points, 0, "n_points should be 0")
        self.assertEqual(node.n_points_since_retrain, 0, "n_points_since_retrain should be 0")

    def test_gpnode_add_single_training_data(self):
        """Test adding a single data point to GPNode."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, name="add_single_data_node")
        n_features = 1
        node.init_data_set(n_features) # Initialize training set structure

        x_sample = np.array([[0.5]])
        y_sample = np.array([[1.0]]) # Ensure y is 2D as per typical usage in GPTree
        sigma_sample = 0.1

        node.store_point(x_sample, y_sample, sigma_sample) # y should be float or (1,1) array for GPNode

        self.assertEqual(node.n_points, 1, "n_points should be 1")
        self.assertEqual(node.n_points_since_retrain, 1, "n_points_since_retrain should be 1")
        self.assertTrue(np.array_equal(node.my_X_data, x_sample), "my_X_data not stored correctly")
        self.assertTrue(np.array_equal(node.my_y_data, y_sample), "my_y_data not stored correctly")
        self.assertEqual(node.my_sigma_data[0, 0], sigma_sample, "my_sigma_data not stored correctly")

    def test_gpnode_add_multiple_training_data(self):
        """Test adding multiple data points."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, name="add_multi_data_node")
        n_features = 1
        node.init_data_set(n_features)

        x_data = [np.array([[0.1]]), np.array([[0.2]]), np.array([[0.3]])]
        y_data = [np.array([[1.0]]), np.array([[2.0]]), np.array([[3.0]])] # Ensure y is 2D
        sigma_data = [0.1, 0.15, 0.12]

        for x, y, sigma in zip(x_data, y_data, sigma_data):
            node.store_point(x, y, sigma)

        self.assertEqual(node.n_points, 3)
        self.assertEqual(node.n_points_since_retrain, 3)
        # Data is stored in reverse order (newest first)
        self.assertTrue(np.array_equal(node.my_X_data, np.vstack(x_data[::-1])))
        self.assertTrue(np.array_equal(node.my_y_data, np.vstack(y_data[::-1])))

    def test_gpnode_fit_my_gpr_simple(self):
        """Test fitting the GPR with a few data points."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, retrain_every_n_points=1, name="fit_gpr_node")
        n_features = 1
        node.init_data_set(n_features)

        # Add enough data to trigger a fit (retrain_every_n_points = 1)
        x_sample1 = np.array([[0.1]])
        y_sample1 = np.array([[1.0]])
        sigma_sample1 = 0.1
        node.store_point(x_sample1, y_sample1, sigma_sample1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            did_train = node.fit_my_GPR() # Should train as n_points_since_retrain (1) == retrain_every_n_points (1)
        self.assertTrue(did_train, "GPR should have been trained")
        self.assertIsNotNone(node.my_GPR.kernel_.theta, "GPR kernel theta should be set after fitting")
        self.assertEqual(node.n_points_since_retrain, 0, "Buffer should be reset after training")

        # Add another point
        x_sample2 = np.array([[0.2]])
        y_sample2 = np.array([[2.0]])
        sigma_sample2 = 0.12
        node.store_point(x_sample2, y_sample2, sigma_sample2)

        self.assertEqual(node.n_points, 2)
        self.assertEqual(node.n_points_since_retrain, 1) # Incremented after add

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            did_train_again = node.fit_my_GPR()
        self.assertTrue(did_train_again, "GPR should have been trained again")
        self.assertEqual(node.n_points_since_retrain, 0, "Buffer should be reset after training again")

    def test_gpnode_fit_my_gpr_force_training(self):
        """Test GPR fitting with force_training=True."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, retrain_every_n_points=5, name="force_fit_node") # retrain_every_n_points is high
        n_features = 1
        node.init_data_set(n_features)

        x_sample = np.array([[0.5]])
        y_sample = np.array([[1.5]])
        sigma_sample = 0.1
        node.store_point(x_sample, y_sample, sigma_sample) # n_points_since_retrain = 1

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            did_train_normal = node.fit_my_GPR() # Should not train as buffer (1) < retrain_every_n_points (5)
        self.assertFalse(did_train_normal, "GPR should not train with insufficient buffer points")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            did_train_forced = node.fit_my_GPR(force_training=True)
        self.assertTrue(did_train_forced, "GPR should train when force_training is True")
        self.assertEqual(node.n_points_since_retrain, 0, "Buffer should be reset after forced training")


if __name__ == '__main__':
    unittest.main()
