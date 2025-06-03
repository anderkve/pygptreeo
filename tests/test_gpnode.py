import unittest
import numpy as np

# Attempt to import GPNode and Default_GPR from the correct locations
# Assuming the tests directory is at the same level as the pygptreeo directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from sklearn.gaussian_process.kernels import RBF

class TestGPNode(unittest.TestCase): # Renamed class for broader scope

    def setUp(self):
        """Set up common resources for tests."""
        self.default_gpr_instance = Default_GPR()
        # Re-initialize kernel for Default_GPR for n_features=1 if needed by tests
        # Many tests will use 1D data for simplicity.
        # Default_GPR's kernel_alternatives might be initialized with n_dims from example.py
        # For isolated node tests, we often want to define n_features explicitly or implicitly.

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
            self.assertEqual(node.value, 0, "Node value should be 0 (num_training_points)")
            self.assertEqual(node.Nbar, 50, "Nbar should be set to 50")
            self.assertTrue(node.is_leaf, "A new node should be a leaf")
            self.assertIsNone(node.children, "A new node should not have children")
            self.assertIsInstance(node.my_GPR, Default_GPR, "my_GPR should be an instance of Default_GPR")
        except Exception as e:
            self.fail(f"GPNode creation raised an exception: {e}")

    def test_gpnode_creation_custom_params(self):
        """Test GPNode creation with some custom parameters."""
        try:
            # Note: GPNode value is num_training_points, which is managed internally by add_training_data.
            # Direct initialization of value in constructor is for binarytree.Node, but GPNode overrides it.
            node = GPNode(0, my_GPR=Default_GPR(normalize_y=False), Nbar=100, split_position_method='mean', name="custom_node_1")
            self.assertEqual(node.num_training_points, 0)
            self.assertEqual(node.Nbar, 100)
            self.assertEqual(node.split_position_method, 'mean')
            self.assertEqual(node.name, "custom_node_1")
            self.assertFalse(node.my_GPR.normalize_y, "GPR normalize_y should be False")
        except Exception as e:
            self.fail(f"GPNode creation with custom params raised an exception: {e}")

    def test_gpnode_init_training_set(self):
        """Test the init_training_set method."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, name="init_set_node")
        n_features = 1
        node.init_training_set(n_features)

        self.assertEqual(node.my_X_data.shape, (0, n_features), "my_X_data shape incorrect")
        self.assertEqual(node.my_y_data.shape, (0, 1), "my_y_data shape incorrect")
        self.assertEqual(node.n_features, n_features, "n_features not set correctly")
        self.assertEqual(node.num_training_points, 0, "num_training_points should be 0")
        self.assertEqual(node.num_buffer_points, 0, "num_buffer_points should be 0")

    def test_gpnode_add_single_training_data(self):
        """Test adding a single data point to GPNode."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, name="add_single_data_node")
        n_features = 1
        node.init_training_set(n_features) # Initialize training set structure

        x_sample = np.array([[0.5]])
        y_sample = np.array([[1.0]]) # Ensure y is 2D as per typical usage in GPTree

        node.add_training_data(x_sample, y_sample) # y should be float or (1,1) array for GPNode

        self.assertEqual(node.num_training_points, 1, "num_training_points should be 1")
        self.assertEqual(node.num_buffer_points, 1, "num_buffer_points should be 1")
        self.assertTrue(np.array_equal(node.my_X_data, x_sample), "my_X_data not stored correctly")
        self.assertTrue(np.array_equal(node.my_y_data, y_sample), "my_y_data not stored correctly")

    def test_gpnode_add_multiple_training_data(self):
        """Test adding multiple data points."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, name="add_multi_data_node")
        n_features = 1
        node.init_training_set(n_features)

        x_data = [np.array([[0.1]]), np.array([[0.2]]), np.array([[0.3]])]
        y_data = [np.array([[1.0]]), np.array([[2.0]]), np.array([[3.0]])] # Ensure y is 2D

        for x, y in zip(x_data, y_data):
            node.add_training_data(x, y)

        self.assertEqual(node.num_training_points, 3)
        self.assertEqual(node.num_buffer_points, 3)
        self.assertTrue(np.array_equal(node.my_X_data, np.vstack(x_data)))
        self.assertTrue(np.array_equal(node.my_y_data, np.vstack(y_data)))

    def test_gpnode_fit_my_gpr_simple(self):
        """Test fitting the GPR with a few data points."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, retrain_every_n_points=1, name="fit_gpr_node")
        n_features = 1
        node.init_training_set(n_features)

        # Add enough data to trigger a fit (retrain_every_n_points = 1)
        x_sample1 = np.array([[0.1]])
        y_sample1 = np.array([[1.0]])
        node.add_training_data(x_sample1, y_sample1)

        did_train = node.fit_my_GPR() # Should train as num_buffer_points (1) == retrain_every_n_points (1)
        self.assertTrue(did_train, "GPR should have been trained")
        self.assertIsNotNone(node.my_GPR.kernel_.theta, "GPR kernel theta should be set after fitting")
        self.assertEqual(node.num_buffer_points, 0, "Buffer should be reset after training")

        # Add another point
        x_sample2 = np.array([[0.2]])
        y_sample2 = np.array([[2.0]])
        node.add_training_data(x_sample2, y_sample2)

        self.assertEqual(node.num_training_points, 2)
        self.assertEqual(node.num_buffer_points, 1) # Incremented after add

        did_train_again = node.fit_my_GPR()
        self.assertTrue(did_train_again, "GPR should have been trained again")
        self.assertEqual(node.num_buffer_points, 0, "Buffer should be reset after training again")

    def test_gpnode_fit_my_gpr_force_training(self):
        """Test GPR fitting with force_training=True."""
        node = GPNode(0, my_GPR=self.gpr_for_1d, Nbar=10, retrain_every_n_points=5, name="force_fit_node") # retrain_every_n_points is high
        n_features = 1
        node.init_training_set(n_features)

        x_sample = np.array([[0.5]])
        y_sample = np.array([[1.5]])
        node.add_training_data(x_sample, y_sample) # num_buffer_points = 1

        did_train_normal = node.fit_my_GPR() # Should not train as buffer (1) < retrain_every_n_points (5)
        self.assertFalse(did_train_normal, "GPR should not train with insufficient buffer points")

        did_train_forced = node.fit_my_GPR(force_training=True)
        self.assertTrue(did_train_forced, "GPR should train when force_training is True")
        self.assertEqual(node.num_buffer_points, 0, "Buffer should be reset after forced training")

if __name__ == '__main__':
    unittest.main()
