# jules standard scaling: Test file for standard scaling feature
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygptreeo.gptree import GPTree
from pygptreeo.gpnode import GPNode # Import GPNode to check leaf.my_X_data
from pygptreeo.default_gpr import Default_GPR

def test_gp_standard_scaling():
    # jules standard scaling: Test function for standard scaling in GPTree

    # jules standard scaling: Setup data
    X_train = np.array([
        [1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4],
        [5.0, 6.0], [5.1, 6.1], [5.2, 6.2], [5.3, 6.3], [5.4, 6.4],
        [10.0, 11.0], [10.1, 11.1], [10.2, 11.2], [10.3, 11.3], [10.4, 11.4]
    ])
    y_train = np.array([
        [3.0], [3.2], [3.4], [3.6], [3.8],
        [11.0], [11.2], [11.4], [11.6], [11.8],
        [21.0], [21.2], [21.4], [21.6], [21.8]
    ])
    X_test = np.array([[1.05, 2.05], [5.15, 6.15], [10.15, 11.15]])

    N_train_samples = X_train.shape[0]
    Nbar_test = 5 # Ensure splits

    # jules standard scaling: Test with scaling enabled
    print("\n--- Testing with use_standard_scaling = True ---")
    tree_scaled = GPTree(
        GPR=Default_GPR(),
        Nbar=Nbar_test,
        use_standard_scaling=True,
        split_position_method='median' # ensure more deterministic splits for testing
    )
    tree_scaled.fit(X_train, y_train)

    assert tree_scaled.root is not None
    assert len(tree_scaled.root.leaves) > 0, "Tree did not split, no leaves to test."

    found_leaf_with_data_and_scaler = False
    for leaf in tree_scaled.root.leaves:
        print(f"Checking scaled leaf: {leaf.name}, num_training_points: {leaf.num_training_points}, my_X_data shape: {leaf.my_X_data.shape if leaf.my_X_data is not None else 'None'}") # jules standard scaling: Added more debug info
        assert leaf.use_standard_scaling is True, f"Leaf {leaf.name} should have use_standard_scaling=True"

        # After splitting, parent nodes delete their GPR and data.
        # Leaf nodes should have their GPR and scaler if they received data and were trained.
        if leaf.my_GPR is not None and leaf.num_training_points > 0:
            # jules standard scaling: Check if my_X_data is also present, as fit_my_GPR needs it
            if leaf.my_X_data is None or leaf.my_X_data.shape[0] == 0:
                print(f"Warning: Leaf {leaf.name} has {leaf.num_training_points} points but my_X_data is empty/None before scaler check. This might indicate an issue if it was expected to train.")

            assert leaf.scaler is not None, f"Leaf {leaf.name} (GPR exists, {leaf.num_training_points} pts, X_data_shape: {leaf.my_X_data.shape if leaf.my_X_data is not None else 'None'}) should have a scaler."
            assert isinstance(leaf.scaler, StandardScaler), f"Leaf {leaf.name}'s scaler is not StandardScaler."
            assert hasattr(leaf.scaler, 'mean_') and leaf.scaler.mean_ is not None, f"Leaf {leaf.name}'s scaler is not fitted (no mean_)."
            found_leaf_with_data_and_scaler = True

            # jules standard scaling: Prediction test for this leaf
            if leaf.my_X_data is not None and leaf.my_X_data.shape[0] > 0:
                # Take a sample from this leaf's original training data
                x_sample_original = leaf.my_X_data[0:1, :] # Take the first point

                # 1. Manually scale this sample using the leaf's scaler
                x_sample_scaled_manual = leaf.scaler.transform(x_sample_original)

                # 2. Predict directly using the leaf's GPR with the manually scaled sample
                # Ensure GPR was actually trained (has X_train_ attribute)
                assert hasattr(leaf.my_GPR, 'X_train_'), f"Leaf {leaf.name}'s GPR doesn't seem to be trained (no X_train_)."
                # jules standard scaling: Added return_std=True to GPR predict call
                mu_gpr_direct, _ = leaf.my_GPR.predict(x_sample_scaled_manual, return_std=True)

                # 3. Predict using the leaf's comprehensive predict() method with the original unscaled sample
                mu_leaf_predict, _ = leaf.predict(x_sample_original)

                print(f"Leaf {leaf.name} original sample: {x_sample_original}")
                print(f"Leaf {leaf.name} manually scaled sample: {x_sample_scaled_manual}")
                print(f"Leaf {leaf.name} GPR direct prediction on scaled: {mu_gpr_direct}")
                print(f"Leaf {leaf.name} leaf.predict() on original: {mu_leaf_predict}")

                assert np.allclose(mu_gpr_direct, mu_leaf_predict), \
                    f"Leaf {leaf.name}: GPR direct prediction and leaf.predict() method should yield close results. Got {mu_gpr_direct} vs {mu_leaf_predict}"
            else:
                print(f"Skipping prediction consistency check for leaf {leaf.name} as it has no my_X_data post-fit (normal for leaves if data is cleared after GPR training).")

    assert found_leaf_with_data_and_scaler, "No leaf node was found with data and a fitted scaler."
    mu_scaled_tree, _ = tree_scaled.predict(X_test)
    print(f"Scaled tree predictions: {mu_scaled_tree.flatten()}")

    # jules standard scaling: Test with scaling disabled
    print("\n--- Testing with use_standard_scaling = False ---")
    tree_unscaled = GPTree(
        GPR=Default_GPR(),
        Nbar=Nbar_test,
        use_standard_scaling=False, # Explicitly false
        split_position_method='median'
    )
    tree_unscaled.fit(X_train, y_train)

    assert tree_unscaled.root is not None
    assert len(tree_unscaled.root.leaves) > 0, "Unscaled tree did not split."

    for leaf in tree_unscaled.root.leaves:
        print(f"Checking unscaled leaf: {leaf.name}")
        assert leaf.use_standard_scaling is False, f"Leaf {leaf.name} should have use_standard_scaling=False"
        assert leaf.scaler is None, f"Leaf {leaf.name} should not have a scaler when scaling is disabled."

    mu_unscaled_tree, _ = tree_unscaled.predict(X_test)
    print(f"Unscaled tree predictions: {mu_unscaled_tree.flatten()}")

    # jules standard scaling: Assert that predictions are different (scaling should have an effect)
    # This depends on the data and problem. For this data, they should differ.
    assert not np.allclose(mu_scaled_tree, mu_unscaled_tree), \
        "Predictions from scaled and unscaled trees should ideally differ if scaling has an effect."

    print("\n--- Test scaling completed ---")

if __name__ == "__main__":
    # jules standard scaling: Allow running the test directly
    test_gp_standard_scaling()
    print("Test passed.")
