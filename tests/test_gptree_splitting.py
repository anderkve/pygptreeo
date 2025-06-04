import unittest
import numpy as np

# Add project root to Python path to allow direct import of pygptreeo
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode
from pygptreeo.default_gpr import Default_GPR

import warnings
from sklearn.exceptions import ConvergenceWarning


class TestGradualSplitting(unittest.TestCase):
    # Test cases for the 'gradual' splitting strategy

    def test_gradual_splitting_behavior(self):
        # Test the core logic of gradual splitting

        # Setup
        nbar = 4
        simple_gpr = Default_GPR()

        # Initialize GPTree with gradual splitting
        # Explicitly set split_position_method to median for predictability in test
        tree = GPTree(Nbar=nbar, GPR=simple_gpr, splitting_strategy='gradual',
                      split_dimension_criteria='max_spread', split_position_method='median')

        initial_X_data = np.array([[1.0], [2.0], [3.0], [10.0]]) # Shape (N, n_features=1)
        initial_y_data = np.array([[1.1], [2.1], [3.1], [10.1]]) # Shape (N,1)

        self.assertEqual(initial_X_data.shape[0], nbar, "Initial dataset size should match Nbar")

        # Add Nbar points (root node gets full)
        for i in range(nbar):
            # Pass y as (1,1) as expected by GPNode.store_point via GPTree.update_tree
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                tree.update_tree(initial_X_data[i].reshape(1, -1), initial_y_data[i].reshape(1, 1))

        # After Nbar points, the root node should have Nbar points AND should have split.
        # Let's check is_leaf status first.
        self.assertFalse(tree.root.is_leaf, "Root node should have split after Nbar points were added")
        # Check root node's split_position_method. This attribute should still be on the root node object.
        self.assertEqual(tree.root.split_position_method, 'median', "Root node split_position_method mismatch")

        # Verification after root split (due to Nbar points added)
        self.assertIsNotNone(tree.root.children, "Root node should have children after split")
        self.assertEqual(len(tree.root.children), 2, "Root node should have two children")

        child1, child2 = tree.root.children

        # Each child should inherit all data when using the 'gradual' mode
        self.assertEqual(child1.n_points, nbar, "Child 1 should have Nbar points (all parent data)")
        self.assertEqual(child2.n_points, nbar, "Child 2 should have Nbar points (all parent data)")

        np.testing.assert_array_almost_equal(np.sort(child1.my_X_data, axis=0), np.sort(initial_X_data, axis=0),
                                             err_msg="Child 1 X_data mismatch (should have all parent data)")
        np.testing.assert_array_almost_equal(np.sort(child1.my_y_data, axis=0), np.sort(initial_y_data, axis=0),
                                             err_msg="Child 1 y_data mismatch (should have all parent data)")
        np.testing.assert_array_almost_equal(np.sort(child2.my_X_data, axis=0), np.sort(initial_X_data, axis=0),
                                             err_msg="Child 2 X_data mismatch (should have all parent data)")
        np.testing.assert_array_almost_equal(np.sort(child2.my_y_data, axis=0), np.sort(initial_y_data, axis=0),
                                             err_msg="Child 2 y_data mismatch (should have all parent data)")

        parent_split_index = tree.root.split_index
        parent_split_position = tree.root.split_position

        self.assertEqual(child1.parent_split_index, parent_split_index, "Child1 parent_split_index mismatch")
        self.assertEqual(child1.parent_split_position, parent_split_position, "Child1 parent_split_position mismatch")
        self.assertEqual(child2.parent_split_index, parent_split_index, "Child2 parent_split_index mismatch")
        self.assertEqual(child2.parent_split_position, parent_split_position, "Child2 parent_split_position mismatch")

        # Verify parent split calculations based on initial_X_data and 'median' method
        self.assertEqual(parent_split_index, 0, "Parent split index should be 0 for 1D data with max_spread")
        self.assertAlmostEqual(parent_split_position, 2.5, msg="Parent split position is not as expected (2.5 for [1,2,3,10] median)")

        # Trigger discard mechanism in one child directly. We will use child1 for this test.
        # Based on parent_split_position = 2.5 and initial_X_data in child1:
        # Distances: |1-2.5|=1.5, |2-2.5|=0.5, |3-2.5|=0.5, |10-2.5|=7.5
        # The point X=[10.0] (from initial_X_data[3]) is furthest and should be discarded.
        point_should_be_discarded_X_val = initial_X_data[3] # This is np.array([10.0])

        new_X_point_val = np.array([[0.1]])   # Shape (1,1) - new distinct point
        new_y_point_val = np.array([[0.11]])  # Shape (1,1)

        # Call store_point directly on child1 to test discard logic without further tree splits
        child1.store_point(new_X_point_val, new_y_point_val)

        # Verification of discard
        self.assertEqual(child1.n_points, nbar,
                         "Child 1 should still have Nbar points after adding new point (due to discard)")

        # Verify the correct point was discarded from child1
        is_discarded_point_present_X = any(np.array_equal(row, point_should_be_discarded_X_val) for row in child1.my_X_data)
        self.assertFalse(is_discarded_point_present_X,
                         f"The point X={point_should_be_discarded_X_val} should have been discarded from Child 1. Child data: {child1.my_X_data}")

        # Verify the new point was added to child1
        is_new_point_present_X = any(np.array_equal(row, new_X_point_val[0]) for row in child1.my_X_data)
        self.assertTrue(is_new_point_present_X,
                        f"The new point X={new_X_point_val[0]} should be present in Child 1. Child data: {child1.my_X_data}")

        is_new_point_present_y = any(np.array_equal(row, new_y_point_val[0]) for row in child1.my_y_data)
        self.assertTrue(is_new_point_present_y,
                        f"The new point y={new_y_point_val[0]} should be present in Child 1. Child data: {child1.my_y_data}")

        # Verify child2's data remains unchanged (still initial_X_data)
        self.assertEqual(child2.n_points, nbar, "Child 2 should still have Nbar points")
        np.testing.assert_array_almost_equal(np.sort(child2.my_X_data, axis=0), np.sort(initial_X_data, axis=0),
                                             err_msg="Child 2 X_data should remain unchanged (all parent data)")
        np.testing.assert_array_almost_equal(np.sort(child2.my_y_data, axis=0), np.sort(initial_y_data, axis=0),
                                             err_msg="Child 2 y_data should remain unchanged (all parent data)")


if __name__ == '__main__':
    unittest.main()
