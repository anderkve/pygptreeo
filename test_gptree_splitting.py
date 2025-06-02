# jules gradual splitting test: Unit tests for GPTree splitting strategies
import unittest
import numpy as np

# Attempt to import from pygptreeo, assuming it's installed or in PYTHONPATH
# If running tests directly from repo root, relative imports might be needed if pygptreeo is not installed
try:
    from pygptreeo.gptree import GPTree
    from pygptreeo.gpnode import GPNode
    from pygptreeo.default_gpr import Default_GPR
except ImportError:
    # This is a fallback for cases where the package structure is not directly available in the path.
    # For robust testing, the package should be installed, e.g., in editable mode.
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    from pygptreeo.gptree import GPTree
    from pygptreeo.gpnode import GPNode
    from pygptreeo.default_gpr import Default_GPR


class TestGradualSplitting(unittest.TestCase):
    # jules gradual splitting test: Test cases for the 'gradual' splitting strategy

    def test_gradual_splitting_behavior(self):
        # jules gradual splitting test: Test the core logic of gradual splitting

        # Setup
        nbar = 4
        simple_gpr = Default_GPR()

        # jules gradual splitting test: Initialize GPTree with gradual splitting
        # Explicitly set split_position_method to median for predictability in test
        # This is passed to the root GPNode via **kwargs in GPTree.__init__
        tree = GPTree(Nbar=nbar, GPR=simple_gpr, splitting_strategy='gradual',
                      split_dimension_criteria='max_spread', split_position_method='median')

        initial_X_data = np.array([[1.0], [2.0], [3.0], [10.0]]) # Shape (N, n_features=1)
        initial_y_data = np.array([[1.1], [2.1], [3.1], [10.1]]) # Shape (N,1)

        self.assertEqual(initial_X_data.shape[0], nbar, "Initial dataset size should match Nbar")

        # jules gradual splitting test: Add Nbar points (root node gets full)
        for i in range(nbar):
            # Pass y as (1,1) as expected by GPNode.add_training_data via GPTree.update_tree
            tree.update_tree(initial_X_data[i].reshape(1, -1), initial_y_data[i].reshape(1, 1))

        # jules gradual splitting test: After Nbar points, the root node should have Nbar points AND should have split.
        # self.assertEqual(tree.root.num_training_points, nbar, "Root node should have Nbar points before split")
        # The above assertion is tricky because after split, tree.root.num_training_points might be reset or reflect children.
        # Let's check is_leaf status first.
        self.assertFalse(tree.root.is_leaf, "Root node should have split after Nbar points were added")
        # jules gradual splitting test: Check root node's split_position_method (if it's still relevant for a non-leaf)
        # This attribute should still be on the root node object.
        self.assertEqual(tree.root.split_position_method, 'median', "Root node split_position_method mismatch")

        # jules gradual splitting test: Verification after root split (due to Nbar points added)
        self.assertIsNotNone(tree.root.children, "Root node should have children after split")
        self.assertEqual(len(tree.root.children), 2, "Root node should have two children")

        child1, child2 = tree.root.children

        # jules gradual splitting test: Children inherit all data in 'gradual' mode
        self.assertEqual(child1.num_training_points, nbar, "Child 1 should have Nbar points (all parent data)")
        self.assertEqual(child2.num_training_points, nbar, "Child 2 should have Nbar points (all parent data)")

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

        # jules gradual splitting test: Verify parent split calculations based on initial_X_data and 'median' method
        self.assertEqual(parent_split_index, 0, "Parent split index should be 0 for 1D data with max_spread")
        self.assertAlmostEqual(parent_split_position, 2.5, msg="Parent split position is not as expected (2.5 for [1,2,3,10] median)")

        # jules gradual splitting test: Action: Trigger discard mechanism in one child directly
        # We will use child1 for this test.
        # Based on parent_split_position = 2.5 and initial_X_data in child1:
        # Distances: |1-2.5|=1.5, |2-2.5|=0.5, |3-2.5|=0.5, |10-2.5|=7.5
        # The point X=[10.0] (from initial_X_data[3]) is furthest and should be discarded.
        point_should_be_discarded_X_val = initial_X_data[3] # This is np.array([10.0])

        new_X_point_val = np.array([[0.1]])   # Shape (1,1) - new distinct point
        new_y_point_val = np.array([[0.11]])  # Shape (1,1)

        # Call add_training_data directly on child1 to test discard logic without further tree splits
        child1.add_training_data(new_X_point_val, new_y_point_val)

        # jules gradual splitting test: Verification of discard
        self.assertEqual(child1.num_training_points, nbar,
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

        # jules gradual splitting test: Verify child2's data remains unchanged (still initial_X_data)
        self.assertEqual(child2.num_training_points, nbar, "Child 2 should still have Nbar points")
        np.testing.assert_array_almost_equal(np.sort(child2.my_X_data, axis=0), np.sort(initial_X_data, axis=0),
                                             err_msg="Child 2 X_data should remain unchanged (all parent data)")
        np.testing.assert_array_almost_equal(np.sort(child2.my_y_data, axis=0), np.sort(initial_y_data, axis=0),
                                             err_msg="Child 2 y_data should remain unchanged (all parent data)")

# Ensure the main block for running tests is present
if __name__ == '__main__':
    unittest.main()
