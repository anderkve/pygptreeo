"""Predictions over the unit square at the end of the README animation's stream.

    python animation_grid.py seed noise [leaf|neighbours|guarded|withleaf|smooth] [option=value ...]

The stream and the tree are those of `examples/make_readme_gif.py --alt`
(Himmelblau, 1500 points, additive kernel), with the declared noise as a share
of the target value; the options go to the tree, or to its neighbour search
(`variants.neighbour_tree`). Prints the largest prediction
on a 70 x 70 grid, the errors of the cells near the data and away from it,
and the steps of the mean along straight lines across the square.
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
from pygptreeo import Default_GPR, AdditiveMaternKernel          # noqa: E402
from variants import neighbour_tree                              # noqa: E402
from target_functions import Himmelblau                          # noqa: E402

N_POINTS, AWAY, LINES = 1500, 0.05, 40
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
    options = {name: None if value == 'None' else float(value) for name, value in (item.split('=') for item in sys.argv[4:])}
    prototype = {name: options.pop(name) for name in ('inside', 'outside', 'far', 'width') if name in options}
    X, y = stream(seed)
    tree = neighbour_tree(GPR=Default_GPR(kernel=AdditiveMaternKernel(d=2, order=2, nu=1.5), alpha=1e-6), Nbar=100, theta=0.05,
                          splitting_strategy='gradual', split_position_method='median', retrain_every_n_points=2,
                          use_calibrated_sigma=True, predict_from='leaf' if mode == 'leaf' else 'neighbours', **options)
    if mode == 'guarded':
        from guarded import guard
        guard(tree)
    elif mode == 'withleaf':
        from with_leaf import with_leaf
        with_leaf(tree)
    elif mode == 'smooth':
        from smooth_guard import smooth
        smooth(tree, **prototype)
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
    # steps of the mean along straight lines across the square, as in jitter.py
    rng, steps, largest = np.random.RandomState(0), [], []
    for _ in range(LINES):
        a, b = rng.rand(2), rng.rand(2)
        mu, sd = tree.predict(a + np.linspace(0.0, 1.0, 1001)[:, None] * (b - a))
        step = np.diff(mu[:, 0])
        jump = np.abs(step[1:-1] - 0.5 * (step[:-2] + step[2:])) / sd[1:-2, 0]
        steps.append([(jump > level).sum() for level in (0.1, 1.0)])
        largest.append(jump.max())
    print('seed %d noise %g %-36s steps per line across the square: %.1f above 0.1 sigma, %.1f above 1 sigma; largest %.1f sigma at the median, %.1f on the worst line'
          % (seed, noise, mode, *np.mean(steps, axis=0), np.median(largest), np.max(largest)), flush=True)

if __name__ == '__main__':
    main()
