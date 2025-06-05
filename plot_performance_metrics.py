import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys # Import sys for error handling

def plot_metrics(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_file_path} is empty.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file at {csv_file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print(f"CSV file at {csv_file_path} is empty or contains only headers.", file=sys.stderr)
        sys.exit(1)

    # Expected columns for conversion and checks
    expected_numeric_cols = ["true_y", "predicted_y", "prediction_uncertainty", "predict_time_s", "update_tree_time_s"]

    # Optional: Check if all expected columns are present
    missing_cols = [col for col in expected_numeric_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing expected columns in CSV: {', '.join(missing_cols)}", file=sys.stderr)
        sys.exit(1)

    # Attempt to convert columns to numeric, coercing errors to NaN
    for col in expected_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check if conversion introduced NaNs where it shouldn't have (e.g. if a column was entirely non-numeric)
    if df[expected_numeric_cols].isnull().all().any(): # if any whole column is all NaN after conversion
        all_nan_cols = df[expected_numeric_cols].isnull().all()
        print(f"Error: One or more numeric columns could not be parsed correctly (became all NaNs): {', '.join(all_nan_cols[all_nan_cols].index)}", file=sys.stderr)
        sys.exit(1)


    processed_points = np.arange(1, len(df) + 1)

    print(f"Successfully read {len(df)} rows from {csv_file_path}.")
    print("Calculating actual performance metrics...")

    window_size = 2000
    min_p = 1 # min_periods for rolling calculations

    # Metric 1: Average prediction time
    avg_predict_time = df['predict_time_s'].rolling(window=window_size, min_periods=min_p).mean()

    # Metric 2: Average tree update time
    avg_update_time = df['update_tree_time_s'].rolling(window=window_size, min_periods=min_p).mean()

    # Metric 3: RMSE
    squared_error = (df['predicted_y'] - df['true_y'])**2
    mean_squared_error = squared_error.rolling(window=window_size, min_periods=min_p).mean()
    rmse = np.sqrt(mean_squared_error)

    # Metric 4: Fraction of predictions within 1% of true value
    # Handle true_y == 0 to avoid division by zero; replace with NaN for this calculation
    # Create a temporary Series, not a column in the DataFrame
    true_y_for_ratio = df['true_y'].replace(0, np.nan)
    abs_percentage_error = abs(df['predicted_y'] - df['true_y']) / abs(true_y_for_ratio)
    within_1_percent = abs_percentage_error < 0.01
    within_10_percent = abs_percentage_error < 0.10
    # Calculate mean on the boolean series (True=1, False=0)
    fraction_within_1_percent = within_1_percent.rolling(window=window_size, min_periods=min_p).mean()
    fraction_within_10_percent = within_10_percent.rolling(window=window_size, min_periods=min_p).mean()

    # Metric 5: Empirical coverage of prediction uncertainty
    abs_error = abs(df['predicted_y'] - df['true_y'])
    within_uncertainty = abs_error <= df['prediction_uncertainty']
    # Calculate mean on the boolean series (True=1, False=0)
    empirical_coverage = within_uncertainty.rolling(window=window_size, min_periods=min_p).mean()

    print("Metrics calculated.")

    # Create figure and subplots
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(15, 15))
    fig.suptitle("PyGPTreeo performance metrics", fontsize=16)

    # Plot 1: Average prediction time
    axes[0].plot(processed_points, avg_predict_time, label="Avg. predict time (last 2000 pts)", linewidth=2.0)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Average prediction time per point")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Average tree update time
    axes[1].plot(processed_points, avg_update_time, label="Avg. update time (last 2000 pts)", color='orange', linewidth=2.0)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Average tree update time per point")
    axes[1].grid(True)
    axes[1].legend()

    # Plot 3: RMSE
    axes[2].plot(processed_points, rmse, label="RMSE (last 2000 pts)", color='green', linewidth=2.0)
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("RMSE for predictions")
    axes[2].grid(True)
    axes[2].legend()

    # Plot 4: Fraction of predictions within 1% of true value
    axes[3].plot(processed_points, fraction_within_1_percent, label="Fraction < 1% Error (last 2000 pts)", color='red', linewidth=2.0)
    axes[3].plot(processed_points, fraction_within_10_percent, label="Fraction < 10% Error (last 2000 pts)", color='orange', linewidth=2.0)
    axes[3].set_ylabel("Fraction")
    axes[3].set_title("Fraction of predictions within x% of true value")
    axes[3].grid(True)
    axes[3].legend()

    # Plot 5: Empirical coverage of prediction uncertainty
    axes[4].plot(processed_points, empirical_coverage, label="Empirical coverage (last 2000 pts)", color='purple', linewidth=2.0)
    axes[4].plot([processed_points[0], processed_points[-1]], [0.68, 0.68], "--", color='black', linewidth=2.0)
    axes[4].set_ylabel("Fraction")
    axes[4].set_title("Empirical coverage of prediction uncertainty")
    axes[4].grid(True)
    axes[4].legend()

    # Common X-axis label
    axes[-1].set_xlabel("Total points processed")

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

    output_plot_file = "performance_metrics.png"
    plt.savefig(output_plot_file, dpi=300)
    print(f"Plot saved to {output_plot_file}")
    # plt.show() # Optionally display the plot

def main():
    parser = argparse.ArgumentParser(description="Plot performance metrics from pygptreeo run_output.csv.")
    parser.add_argument("csv_file", help="Path to the input CSV file (e.g., run_output.csv)")

    args = parser.parse_args()

    plot_metrics(args.csv_file)

if __name__ == "__main__":
    main()
