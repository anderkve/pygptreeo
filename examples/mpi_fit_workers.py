"""A tree whose leaves are fitted by other MPI processes.

Rank 0 is the trainer: it holds the tree and takes the points of a stream. With
``update_tree(defer_fits=True)`` it does not fit a leaf itself but gets the fit
as a task, which it sends to a fit worker (the other ranks). The worker
optimises the hyperparameters with ``solve_fit_task`` and sends them back, and
the trainer gives them to the leaf with ``apply_fit_result``. Every point is
predicted before the tree is updated with it.

    mpiexec -n 4 python examples/mpi_fit_workers.py     # a trainer and 3 fit workers
    mpiexec -n 1 python examples/mpi_fit_workers.py     # for comparison: the trainer fits the leaves itself

Needs mpi4py. Run with OMP_NUM_THREADS=1.
"""
import argparse
import time
from collections import deque

import numpy as np
from mpi4py import MPI

from pygptreeo import GPTree, solve_fit_task
import target_functions as tf

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def fit_worker(comm):
    """Solve the tasks that the trainer sends, until it sends None."""
    while True:
        task = comm.recv(source=0)
        if task is None:
            return
        try:
            result = solve_fit_task(task)
        except Exception as error:
            # The trainer has to hear of a failed fit too: the leaf waits for its result
            print(f"fit task {task['id']} failed: {error}", flush=True)
            result = {'id': task['id'], 'hyperparameters': None}
        comm.send(result, dest=0)


def trainer(comm, a):
    rng = np.random.RandomState(a.seed)
    np.random.seed(a.seed)
    X = tf.make_stream(a.stream, tf.GaussianPeaks, a.d, a.N, rng)
    y = tf.GaussianPeaks(X.T)
    sigma = np.maximum(1e-3 * np.abs(y), 1e-6)

    gpt = GPTree(Nbar=a.Nbar, retrain_every_n_points=a.retrain)
    n_workers = comm.Get_size() - 1
    free = list(range(1, n_workers + 1))    # the fit workers without a task
    queue = deque()                         # the tasks that wait for a free worker
    sent_at = {}                            # task id -> the point at which the task was made
    lags = []
    status = MPI.Status()

    def take_results(i, wait=False):
        """Apply the results that have arrived; with `wait`, those of every worker with a task."""
        while comm.iprobe(status=status) or (wait and len(free) < n_workers):
            result = comm.recv(source=MPI.ANY_SOURCE, status=status)
            gpt.apply_fit_result(result)
            free.append(status.Get_source())
            lags.append(i - sent_at.pop(result['id']))

    pred, std = np.empty(a.N), np.empty(a.N)
    t_train = 0.0
    for i in range(a.N):
        xi = X[i:i + 1]
        mu, sd = gpt.predict(xi)
        pred[i], std[i] = mu[0, 0], sd[0, 0]

        t0 = time.perf_counter()
        take_results(i)
        task = gpt.update_tree(xi, y[i:i + 1, None], sigma[i:i + 1, None], defer_fits=n_workers > 0)
        if task is not None:
            sent_at[task['id']] = i
            queue.append(task)
        while free and queue:
            comm.send(queue.popleft(), dest=free.pop())
        t_train += time.perf_counter() - t0

    take_results(a.N, wait=True)
    for worker in range(1, n_workers + 1):
        comm.send(None, dest=worker)

    last = slice(a.N - a.N // 4, a.N)
    err = pred[last] - y[last]
    print(f"{n_workers} fit workers, {a.N} points of a {a.stream} stream in {a.d} dimensions, {len(gpt.root.leaves)} leaves")
    print(f"  trainer: {1e3 * t_train / a.N:.2f} ms per point ({a.N / t_train:.0f} points per second)")
    print(f"  last quarter of the predictions: NRMSE {np.sqrt(np.mean(err ** 2)) / np.ptp(y[last]):.4f}, "
          f"coverage {np.mean(np.abs(err) <= std[last]):.2f}")
    if lags:
        print(f"  {len(lags)} fits by the workers; their results came {np.mean(lags):.0f} points later on average "
              f"({np.max(lags)} at most)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=12000)
    ap.add_argument('--d', type=int, default=6)
    ap.add_argument('--stream', default='focusing', choices=('uniform', 'focusing', 'sweeping', 'walker'))
    ap.add_argument('--Nbar', type=int, default=100)
    ap.add_argument('--retrain', type=int, default=25, help='retrain_every_n_points')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        trainer(comm, a)
    else:
        fit_worker(comm)


if __name__ == '__main__':
    main()
