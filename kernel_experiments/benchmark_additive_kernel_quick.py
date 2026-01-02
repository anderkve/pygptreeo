"""Quick benchmark test - reduced scope for faster testing.

Tests just 2 kernels, 2 dimensions, 2 target functions with fewer points.
"""

import sys
import numpy as np

# Modify config for quick test
class QuickBenchmarkConfig:
    """Configuration for quick benchmark tests."""
    DIMENSIONS = [3, 8]  # Just 2 dimensions
    TARGET_FUNCTIONS = {
        'rastrigin': None,  # Will be imported
        'levy': None,
    }
    N_POINTS = 2000  # Reduced from 10000
    NBAR = 100  # Reduced from 200
    THETA = 1e-4
    RETRAIN_STEP = 100  # Reduced from 200
    X_MIN = 0.0
    X_MAX = 1.0
    EVAL_BATCH_SIZE = 500  # Reduced from 1000
    RANDOM_SEED = 42

# Monkeypatch the config in the main script
import benchmark_additive_kernel as main_script
main_script.BenchmarkConfig = QuickBenchmarkConfig

# Import target functions
from target_functions import Rastrigin, Levy
QuickBenchmarkConfig.TARGET_FUNCTIONS = {
    'rastrigin': Rastrigin,
    'levy': Levy,
}

# Run with reduced kernel types
if __name__ == '__main__':
    print("\n" + "="*80)
    print("QUICK BENCHMARK TEST")
    print("="*80)
    print("Testing: 3 kernels × 2 dimensions × 2 targets = 12 tests")
    print("This should take ~5-10 minutes")
    print("="*80 + "\n")

    # Modify run to test fewer kernels
    original_run = main_script.run_all_benchmarks

    def quick_run():
        """Run with reduced kernel set."""
        # Test only 3 kernel types: matern, additive_d1, additive_d2
        config = main_script.BenchmarkConfig()
        kernel_types = ['matern', 'additive_d1', 'additive_d2']
        results = []

        total_tests = len(config.DIMENSIONS) * len(config.TARGET_FUNCTIONS) * len(kernel_types)
        test_num = 0

        for target_name, target_func in config.TARGET_FUNCTIONS.items():
            for n_dims in config.DIMENSIONS:
                print(f"\n{target_name.upper()} - {n_dims}D:")
                print("-" * 60)

                for kernel_type in kernel_types:
                    test_num += 1
                    print(f"[{test_num}/{total_tests}] ", end='')

                    try:
                        result = main_script.run_single_benchmark(
                            target_func, n_dims, kernel_type, config
                        )
                        result['target_function'] = target_name
                        result['n_dims'] = n_dims
                        result['kernel_type'] = kernel_type
                        results.append(result)
                    except Exception as e:
                        print(f"FAILED: {e}")
                        results.append({
                            'target_function': target_name,
                            'n_dims': n_dims,
                            'kernel_type': kernel_type,
                            'error': str(e)
                        })

        import pandas as pd
        from pathlib import Path
        df = pd.DataFrame(results)

        output_dir = Path('benchmark_results')
        output_dir.mkdir(exist_ok=True)
        csv_path = output_dir / 'quick_benchmark.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n\nQuick test results saved to: {csv_path}")

        return df

    df = quick_run()
    main_script.print_summary(df)

    print("\n✓ Quick benchmark complete!")
    print("\nTo run full benchmark, execute: python benchmark_additive_kernel.py")
