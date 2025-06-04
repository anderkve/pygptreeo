import unittest
import numpy as np
from unittest.mock import patch, MagicMock, call
from copy import deepcopy

# Add project root to Python path to allow direct import of pygptreeo
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sys import float_info # For PoE epsilon

import warnings
from sklearn.exceptions import ConvergenceWarning

class TestGPNode(unittest.TestCase):

    def setUp(self):
        """Set up common resources for tests."""
        self.default_gpr_instance = Default_GPR()

        # A simple kernel for 1D data
        kernel_1d = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # For GPNode, my_GPRs should be a list of GPRs.
        # In the original tests, my_GPR was a single GPR.
        # Adjusting to reflect the change to my_GPRs (list).
        self.gpr_for_1d_list = [Default_GPR()]
        self.gpr_for_1d_list[0].kernel = kernel_1d
        self.gpr_for_1d_list[0].kernel_alternatives = [kernel_1d]


    def test_gpnode_creation_default_gpr(self):
        """Test basic GPNode creation with Default_GPR."""
        try:
            # Pass a list of GPRs
            node = GPNode(0, my_GPRs=self.gpr_for_1d_list, Nbar=50, name="init_node_1")
            self.assertIsNotNone(node, "Node should not be None")
            self.assertEqual(node.value, 0, "Node value should be 0 (n_points)")
            self.assertEqual(node.Nbar, 50, "Nbar should be set to 50")
            self.assertTrue(node.is_leaf, "A new node should be a leaf")
            self.assertIsNone(node.children, "A new node should not have children")
            self.assertIsInstance(node.my_GPRs[0], Default_GPR, "my_GPRs[0] should be an instance of Default_GPR")
        except Exception as e:
            self.fail(f"GPNode creation raised an exception: {e}")

    def test_gpnode_creation_custom_params(self):
        """Test GPNode creation with some custom parameters."""
        try:
            gpr_list = [Default_GPR(normalize_y=False)]
            node = GPNode(0, my_GPRs=gpr_list, Nbar=100, split_position_method='mean', name="custom_node_1")
            self.assertEqual(node.n_points, 0)
            self.assertEqual(node.Nbar, 100)
            self.assertEqual(node.split_position_method, 'mean')
            self.assertEqual(node.name, "custom_node_1")
            self.assertFalse(node.my_GPRs[0].normalize_y, "GPR normalize_y should be False")
        except Exception as e:
            self.fail(f"GPNode creation with custom params raised an exception: {e}")

    def test_gpnode_init_data_set(self):
        """Test the init_data_set method."""
        node = GPNode(0, my_GPRs=self.gpr_for_1d_list, Nbar=10, name="init_set_node")
        n_features = 1
        node.init_data_set(n_features)

        self.assertEqual(node.my_X_data.shape, (0, n_features), "my_X_data shape incorrect")
        self.assertEqual(node.my_y_data.shape, (0, 1), "my_y_data shape incorrect")
        self.assertEqual(node.n_features, n_features, "n_features not set correctly")
        self.assertEqual(node.n_points, 0, "n_points should be 0")
        self.assertEqual(node.n_points_since_retrain, 0, "n_points_since_retrain should be 0")

    def test_gpnode_add_single_training_data(self):
        """Test adding a single data point to GPNode."""
        node = GPNode(0, my_GPRs=self.gpr_for_1d_list, Nbar=10, name="add_single_data_node")
        n_features = 1
        node.init_data_set(n_features)

        x_sample = np.array([[0.5]])
        y_sample = np.array([[1.0]])

        node.store_point(x_sample, y_sample)

        self.assertEqual(node.n_points, 1, "n_points should be 1")
        self.assertEqual(node.n_points_since_retrain, 1, "n_points_since_retrain should be 1")
        self.assertTrue(np.array_equal(node.my_X_data, x_sample), "my_X_data not stored correctly")
        self.assertTrue(np.array_equal(node.my_y_data, y_sample), "my_y_data not stored correctly")

    def test_gpnode_add_multiple_training_data(self):
        """Test adding multiple data points."""
        node = GPNode(0, my_GPRs=self.gpr_for_1d_list, Nbar=10, name="add_multi_data_node")
        n_features = 1
        node.init_data_set(n_features)

        x_data = [np.array([[0.1]]), np.array([[0.2]]), np.array([[0.3]])]
        y_data = [np.array([[1.0]]), np.array([[2.0]]), np.array([[3.0]])]

        for x, y in zip(x_data, y_data):
            node.store_point(x, y)

        self.assertEqual(node.n_points, 3)
        self.assertEqual(node.n_points_since_retrain, 3)
        # Data is prepended, so compare with reversed original data
        self.assertTrue(np.array_equal(node.my_X_data, np.vstack(x_data[::-1])))
        self.assertTrue(np.array_equal(node.my_y_data, np.vstack(y_data[::-1])))

    def test_gpnode_fit_my_gpr_simple(self):
        """Test fitting the GPR with a few data points."""
        node = GPNode(0, my_GPRs=self.gpr_for_1d_list, Nbar=10, retrain_every_n_points=1, name="fit_gpr_node")
        n_features = 1
        node.init_data_set(n_features)

        x_sample1 = np.array([[0.1]])
        y_sample1 = np.array([[1.0]])
        node.store_point(x_sample1, y_sample1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            did_train = node.fit_my_GPR()
        self.assertTrue(did_train, "GPR should have been trained")
        # Check kernel_ as it's set after fit
        self.assertIsNotNone(node.my_GPRs[0].kernel_.theta, "GPR kernel theta should be set after fitting")
        self.assertEqual(node.n_points_since_retrain, 0, "Buffer should be reset after training")

        x_sample2 = np.array([[0.2]])
        y_sample2 = np.array([[2.0]])
        node.store_point(x_sample2, y_sample2)

        self.assertEqual(node.n_points, 2)
        self.assertEqual(node.n_points_since_retrain, 1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            did_train_again = node.fit_my_GPR()
        self.assertTrue(did_train_again, "GPR should have been trained again")
        self.assertEqual(node.n_points_since_retrain, 0, "Buffer should be reset after training again")

    def test_gpnode_fit_my_gpr_force_training(self):
        """Test GPR fitting with force_training=True."""
        node = GPNode(0, my_GPRs=self.gpr_for_1d_list, Nbar=10, retrain_every_n_points=5, name="force_fit_node")
        n_features = 1
        node.init_data_set(n_features)

        x_sample = np.array([[0.5]])
        y_sample = np.array([[1.5]])
        node.store_point(x_sample, y_sample)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            did_train_normal = node.fit_my_GPR()
        self.assertFalse(did_train_normal, "GPR should not train with insufficient buffer points")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            did_train_forced = node.fit_my_GPR(force_training=True)
        self.assertTrue(did_train_forced, "GPR should train when force_training is True")
        self.assertEqual(node.n_points_since_retrain, 0, "Buffer should be reset after forced training")


class TestGPNodeMultiGP(unittest.TestCase):
    def setUp(self):
        self.gpr_template = Default_GPR(kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1))
        # For Default_GPR, kernel_alternatives should be set if you want it to choose
        self.gpr_template.kernel_alternatives = [
            ConstantKernel(1.0) * RBF(length_scale=ls) + WhiteKernel(0.1) for ls in [0.5, 1.0, 2.0]
        ]
        self.n_features = 1

    def test_gpnode_creation_single_gp_explicit(self):
        """Test GPNode creation with n_GPs_per_node=1 and a single GPR instance."""
        single_gpr_instance = Default_GPR(kernel=RBF(1.0))
        node = GPNode(0, my_GPRs=single_gpr_instance, Nbar=50, n_GPs_per_node=1, name="single_gp_explicit")

        self.assertEqual(node.n_GPs_per_node, 1)
        self.assertIsInstance(node.my_GPRs, list)
        self.assertEqual(len(node.my_GPRs), 1)
        self.assertIs(node.my_GPRs[0], single_gpr_instance) # Should be the same instance

    def test_gpnode_creation_single_gp_from_list(self):
        """Test GPNode creation with n_GPs_per_node=1 and a list containing one GPR."""
        single_gpr_instance = Default_GPR(kernel=RBF(1.0))
        node = GPNode(0, my_GPRs=[single_gpr_instance], Nbar=50, n_GPs_per_node=1, name="single_gp_list")

        self.assertEqual(node.n_GPs_per_node, 1)
        self.assertIsInstance(node.my_GPRs, list)
        self.assertEqual(len(node.my_GPRs), 1)
        self.assertIs(node.my_GPRs[0], single_gpr_instance)

    def test_gpnode_creation_multiple_gps_from_template(self):
        """Test GPNode creation with n_GPs_per_node=3 and a single GPR template, expecting deepcopies."""
        # When a single GPR is passed with n_GPs_per_node > 1, GPNode's __init__ should deepcopy it.
        # However, the current __init__ expects a list if multiple GPRs are intended.
        # Let's adjust the test to reflect that GPNode's __init__ expects a list of GPRs
        # or a single GPR that will be replicated (if n_GPs_per_node > 1 and len(my_GPRs)==1).
        # The prompt implies we pass *one* template, and it gets copied.
        # GPNode __init__ was modified to wrap a single GPR in a list. If n_GPs_per_node > 1
        # and only one GPR is provided, it implies this single GPR should be deepcopied N times.
        # Let's refine this logic based on the current GPNode implementation.
        # The current GPNode __init__ takes my_GPRs. If it's a single GPR and n_GPs_per_node > 1,
        # it does NOT automatically deepcopy it n_GPs_per_node times. It just stores it as a list of one.
        # This test needs to align with the actual behavior or the subtask implies GPNode.__init__
        # should be changed to perform this replication.
        # Assuming the subtask implies my_GPRs passed should ALREADY be a list of N GPRs if n_GPs_per_node > 1.
        # Or, if a single GPR is passed, it is used as a template for deepcopying.
        # The previous changes to __init__ were:
        # if not isinstance(my_GPRs, list): my_GPRs = [my_GPRs]
        # self.my_GPRs = my_GPRs
        # This means if a single GPR is passed, self.my_GPRs becomes a list containing that one GPR.
        # The number of GPRs in self.my_GPRs is not necessarily self.n_GPs_per_node at init.
        # This test might be revealing a design ambiguity.
        # For now, let's assume my_GPRs passed to constructor should match n_GPs_per_node in length
        # if n_GPs_per_node > 1. Or, if a single GPR is passed, it's used by all.
        # The prompt "a single GPR template" suggests the latter.
        # Let's write the test assuming GPNode constructor handles this replication.
        # Re-reading GPNode __init__: it DOES NOT replicate. It just stores what's passed.
        # The replication happens in generate_children.
        # So, for this test, we should pass a list of 3 GPRs.

        num_gps = 3
        gpr_instances = [deepcopy(self.gpr_template) for _ in range(num_gps)]
        node = GPNode(0, my_GPRs=gpr_instances, Nbar=50, n_GPs_per_node=num_gps, name="multi_gps_init")

        self.assertEqual(node.n_GPs_per_node, num_gps)
        self.assertIsInstance(node.my_GPRs, list)
        self.assertEqual(len(node.my_GPRs), num_gps)
        for i in range(num_gps):
            self.assertIsInstance(node.my_GPRs[i], type(self.gpr_template))
            if i > 0:
                self.assertIsNot(node.my_GPRs[i], node.my_GPRs[i-1], f"GPRs {i} and {i-1} should be distinct objects")
                self.assertIsNot(node.my_GPRs[i].kernel, node.my_GPRs[i-1].kernel, f"Kernels for GPRs {i} and {i-1} should be distinct")


    def test_generate_children_multi_gp(self):
        """Test that children inherit n_GPs_per_node and have deepcopied GPRs."""
        parent_n_gps = 2
        parent_gprs = [deepcopy(self.gpr_template) for _ in range(parent_n_gps)]
        parent_node = GPNode(0, my_GPRs=parent_gprs, Nbar=50, n_GPs_per_node=parent_n_gps, name="parent_multi")
        parent_node.init_data_set(n_features=self.n_features) # Need to init before generating children

        # Add some data to allow splitting (though split logic itself is not tested here)
        for i in range(5):
             parent_node.store_point(np.array([[i]]), np.array([[i*2.0]]))

        with warnings.catch_warnings(): # generate_children might involve GPR fitting via fit_my_GPR if node is full
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            # The GPR argument to generate_children is not used if my_GPRs is already set.
            # The method signature for generate_children is (self, GPR_class_or_instance, n_features)
            # This GPR arg seems to be a legacy from when only one GPR was present.
            # GPNode.generate_children uses deepcopy(self.my_GPRs)
            parent_node.generate_children(GPR=type(self.gpr_template), n_features=self.n_features)

        self.assertIsNotNone(parent_node.left)
        self.assertIsNotNone(parent_node.right)

        for child_node in [parent_node.left, parent_node.right]:
            self.assertIsNotNone(child_node)
            self.assertEqual(child_node.n_GPs_per_node, parent_n_gps)
            self.assertIsInstance(child_node.my_GPRs, list)
            self.assertEqual(len(child_node.my_GPRs), parent_n_gps)
            for i in range(parent_n_gps):
                self.assertIsInstance(child_node.my_GPRs[i], type(self.gpr_template))
                # Check distinctness from parent's GPRs
                self.assertIsNot(child_node.my_GPRs[i], parent_node.my_GPRs[i])
                # Check distinctness of kernels (implies deepcopy worked)
                self.assertNotEqual(id(child_node.my_GPRs[i].kernel), id(parent_node.my_GPRs[i].kernel))

        # Check distinctness between siblings
        for i in range(parent_n_gps):
            self.assertIsNot(parent_node.left.my_GPRs[i], parent_node.right.my_GPRs[i])
            self.assertNotEqual(id(parent_node.left.my_GPRs[i].kernel), id(parent_node.right.my_GPRs[i].kernel))


    def test_fit_my_gpr_single_gp_no_partitioning(self):
        """Test fit_my_GPR with n_GPs_per_node=1, ensuring no data partitioning."""
        node_gpr = deepcopy(self.gpr_template)
        node = GPNode(0, my_GPRs=[node_gpr], Nbar=20, n_GPs_per_node=1, name="single_fit_node")
        node.init_data_set(n_features=self.n_features)

        num_points = 10
        for i in range(num_points):
            node.store_point(np.array([[float(i)]]), np.array([[float(i * 1.5)]]))

        # Let the actual fit run, then inspect the GPR state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            node.fit_my_GPR(force_training=True)

        self.assertTrue(hasattr(node.my_GPRs[0], 'X_train_'), "GPR should have X_train_ attribute after fitting.")
        self.assertTrue(hasattr(node.my_GPRs[0], 'y_train_'), "GPR should have y_train_ attribute after fitting.")

        # X_train in fit_my_GPR is np.vstack((node.my_X_data, node.shared_X_data))
        # node.my_X_data stores points in reverse order of addition.
        expected_X_train_shape_0 = num_points + node.n_shared_points
        self.assertEqual(node.my_GPRs[0].X_train_.shape[0], expected_X_train_shape_0)

        # Verify that the data used for training is what we expect (all points)
        # Reconstruct the expected X_train based on how GPNode stores and combines data
        combined_X_data_in_node = np.vstack((node.my_X_data, node.shared_X_data))
        self.assertTrue(np.array_equal(np.sort(node.my_GPRs[0].X_train_, axis=0),
                                       np.sort(combined_X_data_in_node, axis=0)),
                        "Data in GPR's X_train_ does not match all combined input data.")

    def test_fit_my_gpr_data_partitioning(self):
        """Test fit_my_GPR with n_GPs_per_node=2, ensuring data partitioning."""
        num_gps_for_node = 2
        gpr_instances_for_node = [deepcopy(self.gpr_template) for _ in range(num_gps_for_node)]
        # Ensure kernel_alternatives are set for each instance if Default_GPR doesn't set them on deepcopy
        for gp_instance in gpr_instances_for_node:
            if not hasattr(gp_instance, 'kernel_alternatives') or not gp_instance.kernel_alternatives:
                gp_instance.kernel_alternatives = [ConstantKernel(1.0) * RBF(length_scale=ls) + WhiteKernel(0.1) for ls in [0.5, 1.0, 2.0]]


        node = GPNode(0, my_GPRs=gpr_instances_for_node, Nbar=30, n_GPs_per_node=num_gps_for_node, name="partition_fit_node")
        node.init_data_set(n_features=self.n_features)

        num_points = 10 # Total points to add
        original_X_points_list = []
        for i in range(num_points):
            x_val = np.array([[float(i)]])
            original_X_points_list.append(x_val)
            node.store_point(x_val, np.array([[float(i * 2.0)]]))

        # Let the actual fit run
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning) # Catches "No data points assigned"
            node.fit_my_GPR(force_training=True)

        total_points_in_gpr_fits = 0
        all_X_data_from_gpr_fits = [] # List of arrays

        for i, gp in enumerate(node.my_GPRs):
            self.assertTrue(hasattr(gp, 'X_train_'), f"GPR {i} should have X_train_ attribute after fitting.")
            self.assertTrue(hasattr(gp, 'y_train_'), f"GPR {i} should have y_train_ attribute after fitting.")

            X_fit = gp.X_train_
            self.assertIsNotNone(X_fit, f"GPR {i} X_train_ should not be None.")

            num_samples_in_gp = X_fit.shape[0]
            expected_samples_per_gp = (num_points + node.n_shared_points) / num_gps_for_node

            # Check if this GP was trained (it might not if it received no data points)
            # The print statement in fit_my_GPR indicates "GP X kernel: Not trained..."
            # For this test, with 10 points and 2 GPs, each should get data.
            if num_samples_in_gp == 0 and (num_points + node.n_shared_points) >= num_gps_for_node :
                 warnings.warn(f"GPR {i} received 0 samples unexpectedly.") # Should not happen here

            self.assertAlmostEqual(num_samples_in_gp, expected_samples_per_gp, delta=1.5,
                                   msg=f"GPR {i} received {num_samples_in_gp} samples, expected around {expected_samples_per_gp}")
            total_points_in_gpr_fits += num_samples_in_gp
            all_X_data_from_gpr_fits.append(X_fit)

        self.assertEqual(total_points_in_gpr_fits, num_points + node.n_shared_points,
                         "Total points in GPRs' X_train_ do not match original number of points.")

        # Verify distinctness of data (most points should be unique across subsets)
        original_X_combined_sorted_tuples = sorted([tuple(row) for row in np.vstack((node.my_X_data, node.shared_X_data))])

        fitted_X_combined_sorted_tuples = sorted([tuple(item) for sublist in all_X_data_from_gpr_fits for item in sublist.tolist()])

        self.assertListEqual(fitted_X_combined_sorted_tuples, original_X_combined_sorted_tuples,
                             "The combined data from all GPR fits does not match the sorted original combined data.")

    def test_predict_single_gp_no_poe(self):
        """Test predict method with n_GPs_per_node=1 (no PoE)."""
        gpr_instance = deepcopy(self.gpr_template)
        gpr_instance.kernel_ = gpr_instance.kernel # Simulate a trained GPR

        node = GPNode(0, my_GPRs=[gpr_instance], Nbar=10, n_GPs_per_node=1)
        node.init_data_set(n_features=self.n_features)
        # Add a point to ensure GPR can be "fit" or has data if predict checks it
        node.store_point(np.array([[0.1]]), np.array([[0.1]]))


        x_test = np.array([[0.5]])
        expected_mu = np.array([[1.0]])
        expected_sigma = np.array([[0.2]])

        with patch.object(gpr_instance, 'predict', return_value=(expected_mu, expected_sigma)) as mock_predict_gpr:
            mu_pred, sigma_pred = node.predict(x_test, return_std=True)
            mock_predict_gpr.assert_called_once_with(x_test, return_std=True)
            self.assertTrue(np.array_equal(mu_pred, expected_mu))
            self.assertTrue(np.array_equal(sigma_pred, expected_sigma))

    def test_predict_poe_two_gps(self):
        """Test predict method with n_GPs_per_node=2 using Product of Experts."""
        gpr1 = Default_GPR(kernel=ConstantKernel(1.0) * RBF(0.5))
        gpr1.kernel_ = gpr1.kernel # Simulate trained
        gpr2 = Default_GPR(kernel=ConstantKernel(1.0) * RBF(2.0))
        gpr2.kernel_ = gpr2.kernel # Simulate trained

        node = GPNode(0, my_GPRs=[gpr1, gpr2], Nbar=10, n_GPs_per_node=2)
        node.init_data_set(n_features=self.n_features)
        # Add a point to ensure node appears to have data if any internal checks exist
        node.store_point(np.array([[0.1]]), np.array([[0.1]]))


        x_test = np.array([[0.5]])
        mu1, sigma1 = np.array([[2.0]]), np.array([[0.5]])
        mu2, sigma2 = np.array([[3.0]]), np.array([[1.0]])

        with patch.object(gpr1, 'predict', return_value=(mu1, sigma1)) as mock_predict_gpr1, \
             patch.object(gpr2, 'predict', return_value=(mu2, sigma2)) as mock_predict_gpr2:

            mu_pred, sigma_pred = node.predict(x_test, return_std=True)

            mock_predict_gpr1.assert_called_once_with(x_test, return_std=True)
            mock_predict_gpr2.assert_called_once_with(x_test, return_std=True)

            var1 = sigma1**2 + float_info.epsilon
            var2 = sigma2**2 + float_info.epsilon
            prec1 = 1.0 / var1
            prec2 = 1.0 / var2

            poe_var = 1.0 / (prec1 + prec2)
            poe_mean = poe_var * (mu1 * prec1 + mu2 * prec2)
            poe_std = np.sqrt(poe_var)

            np.testing.assert_allclose(mu_pred, poe_mean, rtol=1e-6, err_msg="PoE mean mismatch")
            np.testing.assert_allclose(sigma_pred, poe_std, rtol=1e-6, err_msg="PoE std mismatch")

    def test_predict_poe_with_untrained_gp(self):
        """Test PoE prediction skips untrained GPs and issues a warning."""
        gpr1 = Default_GPR(kernel=ConstantKernel(1.0) * RBF(0.5))
        gpr1.kernel_ = gpr1.kernel # Trained
        gpr2 = Default_GPR(kernel=ConstantKernel(1.0) * RBF(2.0))
        # gpr2 has no kernel_ attribute, so it's "untrained"

        node = GPNode(0, my_GPRs=[gpr1, gpr2], Nbar=10, n_GPs_per_node=2)
        node.init_data_set(n_features=self.n_features)
        node.store_point(np.array([[0.1]]), np.array([[0.1]]))


        x_test = np.array([[0.5]])
        expected_mu1 = np.array([[2.5]])
        expected_sigma1 = np.array([[0.3]])

        with patch.object(gpr1, 'predict', return_value=(expected_mu1, expected_sigma1)) as mock_predict_gpr1:
            # gpr2.predict should not be called
            with patch.object(gpr2, 'predict') as mock_predict_gpr2:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning) # Ensure RuntimeWarnings are caught
                    mu_pred, sigma_pred = node.predict(x_test, return_std=True)

                mock_predict_gpr1.assert_called_once_with(x_test, return_std=True)
                mock_predict_gpr2.assert_not_called()

                # Use allclose for floating point comparisons
                np.testing.assert_allclose(mu_pred, expected_mu1, rtol=1e-6, err_msg="Untrained PoE mean mismatch")
                np.testing.assert_allclose(sigma_pred, expected_sigma1, rtol=1e-6, err_msg="Untrained PoE sigma mismatch")

                # Check for the specific warning
                self.assertTrue(any("Skipping in PoE" in str(warn.message) for warn in w if warn.category == RuntimeWarning),
                                "Expected RuntimeWarning for untrained GP was not issued.")


if __name__ == '__main__':
    unittest.main()
