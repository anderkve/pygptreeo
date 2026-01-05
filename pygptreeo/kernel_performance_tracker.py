"""Kernel Performance Tracker for adaptive kernel selection.

This module implements the KernelPerformanceTracker class, which tracks
the performance of different kernel types across the tree to enable
intelligent kernel selection during node splitting.
"""

import numpy as np


class KernelPerformanceTracker:
    """Tracks kernel performance across the tree for intelligent selection.

    This class maintains statistics about how well different kernel types
    perform when tested during node splits. It uses a simple win-rate
    strategy to identify which kernels work best, allowing the tree to
    preferentially test high-performing kernels while still exploring
    new options.

    Attributes:
        n_kernels (int): Total number of kernel types available
        n_tests (np.ndarray): Number of times each kernel has been tested
        n_wins (np.ndarray): Number of times each kernel was selected (won)
    """

    def __init__(self, n_kernels):
        """Initialize the kernel performance tracker.

        Args:
            n_kernels (int): Total number of kernel types to track
        """
        self.n_kernels = n_kernels
        self.n_tests = np.zeros(n_kernels)  # How many times tested
        self.n_wins = np.zeros(n_kernels)   # How many times selected

    def get_win_rate(self, kernel_idx):
        """Get win rate for a kernel (wins / tests).

        Args:
            kernel_idx (int): Index of the kernel

        Returns:
            float: Win rate between 0 and 1, or 0 if never tested
        """
        if self.n_tests[kernel_idx] == 0:
            return 0.0  # Untested kernels
        return self.n_wins[kernel_idx] / self.n_tests[kernel_idx]

    def update(self, tested_kernels, selected_kernel):
        """Update statistics after a kernel selection event.

        Args:
            tested_kernels (list): List of kernel indices that were tested
            selected_kernel (int): Index of the kernel that was selected (won)
        """
        for k_idx in tested_kernels:
            self.n_tests[k_idx] += 1
            if k_idx == selected_kernel:
                self.n_wins[k_idx] += 1

    def select_kernels_to_test(self, parent_kernel_idx, k_best=2, n_random=1):
        """Select which kernels to test: k best (by win rate) + n random.

        This method implements an exploration-exploitation strategy:
        1. Always test the parent's current kernel (for comparison)
        2. Test k_best kernels with highest win rates (exploitation)
        3. Test n_random additional kernels randomly (exploration)
        4. Prioritize untested kernels to ensure all get evaluated

        Args:
            parent_kernel_idx (int or None): Current kernel index (always tested if not None)
            k_best (int): Number of best-performing kernels to test
            n_random (int): Number of random exploratory kernels to test

        Returns:
            list: Indices of kernels to test
        """
        # Always include parent kernel for comparison
        kernels_to_test = [parent_kernel_idx] if parent_kernel_idx is not None else []

        available_indices = [i for i in range(self.n_kernels) if i != parent_kernel_idx]

        if len(available_indices) == 0:
            return kernels_to_test

        # Separate untested from tested kernels
        untested = [i for i in available_indices if self.n_tests[i] == 0]
        tested = [i for i in available_indices if self.n_tests[i] > 0]

        # Prioritize untested kernels (exploration)
        if untested:
            n_from_untested = min(k_best, len(untested))
            kernels_to_test.extend(np.random.choice(untested, size=n_from_untested, replace=False))
            k_best -= n_from_untested  # Reduce k_best by what we already added

        # Add k_best kernels by win rate (exploitation)
        if k_best > 0 and tested:
            win_rates = [(i, self.get_win_rate(i)) for i in tested]
            win_rates.sort(key=lambda x: x[1], reverse=True)  # Sort by win rate descending
            best_kernels = [i for i, _ in win_rates[:k_best]]
            kernels_to_test.extend(best_kernels)

            # Remove selected kernels from available pool
            for k in best_kernels:
                available_indices.remove(k)

        # Add n_random exploratory kernels from remaining pool
        remaining = [i for i in available_indices if i not in kernels_to_test]
        if remaining and n_random > 0:
            n_random_actual = min(n_random, len(remaining))
            random_kernels = np.random.choice(remaining, size=n_random_actual, replace=False)
            kernels_to_test.extend(random_kernels)

        return kernels_to_test

    def print_statistics(self, kernel_names=None):
        """Print kernel performance statistics.

        Args:
            kernel_names (dict, optional): Mapping from kernel index to name string
        """
        print("\n" + "=" * 80)
        print("Kernel Selection Performance Statistics")
        print("=" * 80)
        print(f"{'Kernel':<8} {'Tests':<10} {'Wins':<10} {'Win Rate':<12}")
        print("-" * 80)

        for i in range(self.n_kernels):
            win_rate = self.get_win_rate(i)
            name = kernel_names.get(i, f"Kernel {i}") if kernel_names else f"Kernel {i}"

            if self.n_tests[i] > 0:
                print(f"{i:<8} {int(self.n_tests[i]):<10} {int(self.n_wins[i]):<10} {win_rate:<12.2%}")
            else:
                print(f"{i:<8} {int(self.n_tests[i]):<10} {int(self.n_wins[i]):<10} {'N/A':<12}")

        print("=" * 80)
