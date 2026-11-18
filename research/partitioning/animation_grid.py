"""Predictions over the unit square at the end of the README animation's stream.

    python animation_grid.py seed noise [leaf|neighbours|guarded|withleaf] [option=value ...]

The stream and the tree are those of `examples/make_readme_gif.py --alt`
(Himmelblau, 1500 points, additive kernel), with the declared noise as a share
of the target value; the options go to the tree. Prints the largest prediction
on a 70 x 70 grid and the errors of the cells near the data and away from it.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is loaded
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'examples')]
warnings.simplefilter('ignore')
import numpy as np                                               # noqa: E402
from scipy.spatial import cKDTree                                # noqa: E402
from pygptreeo import GPTree, Default_GPR, AdditiveMaternKernel  # noqa: E402
from target_functions import Himmelblau                          # noqa: E402

N_POINTS, AWAY = 1500, 0.05
START, END, BOW, HOLD = np.array([0.20, 0.20]), np.array([0.80, 0.70]), 0.12, 0.45
WIDTH_FIRST, WIDTH_LAST = 0.20, 0.04


def target(x):
    return 10.0 * (Himmelblau(x) + 1.0)


def centre(t):
    """The centre of the cloud the stream draws from, along a bowed path."""
    p = min(t / (1.0 - HOLD), 1.0)
    along = END - START
    normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
    return np.clip(START + along * p * p * (3.0 - 2.0 * p) + BOW * np.sin(np.pi * p) * normal, 0.03, 0.97)


def stream(seed):
    np.random.seed(seed)
    X = np.empty((N_POINTS, 2))
    for i, t in enumerate((np.arange(N_POINTS) + 0.5) / N_POINTS):
        width = WIDTH_LAST + (WIDTH_FIRST - WIDTH_LAST) * (1.0 - t) ** 1.5
        while True:
            X[i] = centre(t) + np.random.normal(0.0, 1.0, 2) * width
            if X[i].min() >= 0.0 and X[i].max() <= 1.0:
                break
    return X, np.array([target(x) for x in X]).reshape(-1, 1)


def main():
    seed, noise = int(sys.argv[1]), float(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else 'neighbours'
    options = {name: float(value) for name, value in (item.split('=') for item in sys.argv[4:])}
    X, y = stream(seed)
    tree = GPTree(GPR=Default_GPR(kernel=AdditiveMaternKernel(d=2, order=2, nu=1.5), alpha=1e-6), Nbar=100, theta=0.05,
                  splitting_strategy='gradual', split_position_method='median', retrain_every_n_points=2,
                  use_calibrated_sigma=True, predict_from='leaf' if mode == 'leaf' else 'neighbours', **options)
    if mode == 'guarded':
        from guarded import guard
        guard(tree)
    elif mode == 'withleaf':
        from with_leaf import with_leaf
        with_leaf(tree)
    mode = ' '.join([mode] + sys.argv[4:])
    for i in range(N_POINTS):
        tree.update_tree(X[i:i + 1], y[i:i + 1], noise * np.abs(y[i:i + 1]))
    axis = np.linspace(0.0, 1.0, 70)
    grid = np.column_stack([g.ravel() for g in np.meshgrid(axis, axis)])
    truth = np.array([target(x) for x in grid])
    mean = tree.predict(grid)[0][:, 0]
    error, away = np.abs(mean - truth), cKDTree(X).query(grid)[0] > AWAY
    print('seed %d noise %g %-36s largest prediction %6.1f (target at most %.1f)  rmse away from the data %6.2f, near it %6.3f  '
          'largest error %6.1f  cells off by more than 30: %d' % (seed, noise, mode, mean.max(), truth.max(), np.sqrt((error[away] ** 2).mean()),
                                                                 np.sqrt((error[~away] ** 2).mean()), error.max(), (error > 30).sum()), flush=True)


if __name__ == '__main__':
    main()
