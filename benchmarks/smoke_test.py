"""Minimal smoke test for each adapter."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/pygptreeo")

from benchmarks.adapters import (
    PyGPTreeOAdapter, SklearnGPAdapter, GPyTorchSVGPAdapter,
    RandomForestAdapter, RiverKNNAdapter,
)
from benchmarks.problems import PROBLEMS


def main():
    rng = np.random.default_rng(0)
    prob = PROBLEMS["smooth_sines_2d"]
    X, y = prob.sample(400, rng)
    X_test, y_test = prob.sample(200, rng)

    adapters = [
        ("pygptreeo", lambda d: PyGPTreeOAdapter(d, Nbar=100, retrain_step=50)),
        ("sklearn_gp", lambda d: SklearnGPAdapter(d, retrain_every=100, max_train_points=400)),
        ("gpytorch_svgp", lambda d: GPyTorchSVGPAdapter(d, retrain_every=200, n_epochs=10, n_inducing=32)),
        ("random_forest", lambda d: RandomForestAdapter(d, retrain_every=100, n_estimators=30)),
        ("river_knn", lambda d: RiverKNNAdapter(d, n_neighbors=5)),
    ]

    for name, factory in adapters:
        print(f"--- {name} ---")
        method = factory(prob.dim)
        for i in range(len(X)):
            method.update(X[i:i+1], np.array([[y[i]]]))
        mean, std = method.predict(X_test)
        err = np.sqrt(np.mean((mean.ravel() - y_test) ** 2))
        print(f"  rmse={err:.4f}  std_ok={np.all(np.isfinite(std))}  "
              f"mean_std={np.nanmean(std):.4f}")


if __name__ == "__main__":
    main()
