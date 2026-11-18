"""
Images f(u, v; theta) as multi-output targets: 'shared' and 'aligned' leaves

Learns a family of images on a grid of 32 x 32 points from a set of images, with the
flattened image as the multi-output target, and prints the error on test images, the
share of the leaves that hold an alignment and the times. With
``output_shape=(32, 32)`` the leaves of ``output_model='aligned'`` warp both axes of
the grid. Without it they take the outputs for a curve of 1024 points.

The families:
    'blob'       a blob whose centre moves along both axes, and its width
    'two_blobs'  two blobs side by side that move in opposite directions
    'ring'       a ring about the centre of the image, with a radius that changes
    'smooth'     three fixed shapes with varying amplitudes: nothing moves

Usage:
    python examples/multioutput_image_alignment.py [family] [n_train]

    family   one of the families above (default 'blob')
    n_train  number of images (default 800)
"""

import sys
import time
import warnings

import numpy as np

from pygptreeo import GPTree, Default_GPR

warnings.filterwarnings("ignore")

P = 32
_points = (np.arange(P) + 0.5) / P
U_GRID, V_GRID = np.meshgrid(_points, _points, indexing="ij")     # first and second axis of the image


def _column(t, lo, hi):
    return (lo + (hi - lo) * t)[:, None, None]


def blob(theta):
    cu, cv, width = _column(theta[:, 0], 0.25, 0.75), _column(theta[:, 1], 0.25, 0.75), _column(theta[:, 2], 0.06, 0.12)
    return np.exp(-0.5 * ((U_GRID - cu) ** 2 + (V_GRID - cv) ** 2) / width ** 2)


def two_blobs(theta):
    shift, cu, width = _column(theta[:, 0], -0.2, 0.2), _column(theta[:, 1], 0.45, 0.55), _column(theta[:, 2], 0.06, 0.1)
    one = lambda u0, v0: np.exp(-0.5 * ((U_GRID - u0) ** 2 + (V_GRID - v0) ** 2) / width ** 2)
    return one(cu - 0.22, 0.5 + shift) + one(cu + 0.22, 0.5 - shift)


def ring(theta):
    radius, width, asymmetry = _column(theta[:, 0], 0.15, 0.4), _column(theta[:, 1], 0.03, 0.06), _column(theta[:, 2], -0.5, 0.5)
    r = np.sqrt((U_GRID - 0.5) ** 2 + (V_GRID - 0.5) ** 2)
    return np.exp(-0.5 * (r - radius) ** 2 / width ** 2) * (1.0 + asymmetry * (U_GRID - 0.5) / np.maximum(r, 1e-6))


def smooth(theta):
    a, b, c = _column(theta[:, 0], 0.5, 1.5), _column(theta[:, 1], -1.0, 1.0), _column(theta[:, 2], -1.0, 1.0)
    return (a * np.sin(np.pi * U_GRID) * np.sin(np.pi * V_GRID) + b * np.sin(2 * np.pi * U_GRID) * np.sin(np.pi * V_GRID)
            + c * np.sin(np.pi * U_GRID) * np.sin(2 * np.pi * V_GRID))


FAMILIES = dict(blob=blob, two_blobs=two_blobs, ring=ring, smooth=smooth)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "blob"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    family = FAMILIES[name]

    rng = np.random.RandomState(0)
    theta, theta_test = rng.rand(n_train, 3), rng.rand(300, 3)
    Y, truth = family(theta).reshape(n_train, -1), family(theta_test).reshape(300, -1)   # an image is flattened row by row
    value_range = truth.max() - truth.min()
    sigma = np.full(Y.shape, 1e-4 * value_range)

    print(f"{name}: 3 parameters, images of {P} x {P} points, {n_train} images\n")
    print(f"{'leaves':<34s} {'test NRMSE':>11s} {'coverage':>9s} {'leaves aligned':>15s} {'s to train':>11s} {'ms/prediction':>14s}")
    for label, kwargs in (("'shared'", dict(output_model='shared')),
                          ("'aligned', as a curve", dict(output_model='aligned')),
                          (f"'aligned', output_shape=({P}, {P})", dict(output_model='aligned', output_shape=(P, P)))):
        np.random.seed(0)
        gpt = GPTree(GPR=Default_GPR(), Nbar=100, retrain_every_n_points=25, n_outputs=P * P, **kwargs)
        t0 = time.time()
        gpt.batch_update_tree(theta, Y, sigma)
        t1 = time.time()
        mean, sd = gpt.predict(theta_test, mode='loop')
        t2 = time.time()
        aligned = np.mean([leaf.alignment is not None for leaf in gpt.root.leaves])
        print(f"{label:<34s} {np.sqrt(np.mean((mean - truth) ** 2)) / value_range:>11.2e} {np.mean(np.abs(mean - truth) < sd):>9.2f} "
              f"{100 * aligned:>14.0f}% {t1 - t0:>11.0f} {1e3 * (t2 - t1) / 300:>14.1f}")


if __name__ == "__main__":
    main()
