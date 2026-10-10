import numpy as np
import tinygp as tgp
from pygptreeo import GPTree
from pygptreeo.adapters import TinyGPAdapter

def synthetic_target(X: np.ndarray) -> np.ndarray:
    """Simple 2D benchmark function: f(x1, x2) = sin(3*x1) * cos(3*x2)."""
    return np.sin(3.0 * X[:, 0]) * np.cos(3.0 * X[:, 1])

np.random.seed(42)

n_dims = 2
n_pts = 200
Nbar = 30
theta = 1e-3
retrain_step = 10

X_input = np.random.uniform(0.0, 1.0, size=(n_pts, n_dims))
y_input = synthetic_target(X_input)

base_kernel = 1.0 * tgp.kernels.Matern32(scale=0.3)
adapter = TinyGPAdapter(kernel=base_kernel, alpha=1e-6)

gpt = GPTree(
    GPR=adapter,
    Nbar=Nbar,
    theta=theta,
    split_position_method="median",
    split_dimension_criteria="max_variance",
    retrain_every_n_points=retrain_step,
    use_calibrated_sigma=True,
    splitting_strategy="gradual",
)

print(f"Running sequential update with {n_pts} points...")

for i, (x_row, y_val) in enumerate(zip(X_input, y_input), 1):
    x = x_row.reshape(1, n_dims)
    y = np.array([[y_val]])

    y_pred, y_pred_std = gpt.predict(x, show_progress=False)

    noise_std = 0.01 * np.abs(y) + 1e-4
    gpt.update_tree(x, y, noise_std)

    if i % 20 == 0 or i == n_pts:
        err = np.abs(y[0, 0] - y_pred[0, 0])
        print(f"Point {i:3d}/{n_pts} | True: {y[0, 0]:.4f} | Pred: {y_pred[0, 0]:.4f} | Std: {y_pred_std[0, 0]:.4f} | Err: {err:.4f}")

print("\nFinal Tree Structure:")
print(gpt.root)
print("\nTest completed successfully.")