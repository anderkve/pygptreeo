# pyGPTreeO examples

Install the package first, from the repository root:

```bash
pip install -e .            # ".[neural]" adds PyTorch, ".[gpytorch]" GPyTorch
pip install matplotlib      # for the examples that plot
```

Run times are for one core. `OMP_NUM_THREADS=1` avoids threading overhead in
the small linear-algebra calls of the leaves and gives consistent timings.

| Script | Shows | Needs | Run time |
|---|---|---|---|
| `example.py` | The basic loop: predict each point, then update the tree with it | | 6 s |
| `example_hybrid.py` | The hybrid (`global_mean='net'`) against the plain tree on a 6D stream | PyTorch | 1 min |
| `multioutput_function_learning_pca.py` | The three multi-output models (`independent`, `shared`, `pca`) on a function f(t; x) observed on a t-grid | matplotlib for the figure | 15 s |
| `gpytorch_backend.py` | GPyTorch models as the leaf GPs, against the scikit-learn default | GPyTorch | 10 s |
| `streaming_performance.py` | Accuracy, coverage and time per point along a stream, for the plain tree, the tree that predicts from each query's nearest points, the hybrid and the additive global GP, as a figure. `--max_memory_mb` limits the memory of the trees | matplotlib (PyTorch for the hybrid) | 2 min |
| `mpi_fit_workers.py` | An MPI trainer that holds the tree and hands the leaf fits to fit workers (`update_tree(defer_fits=True)`), against a trainer that fits the leaves itself | mpi4py | 15 s on 4 processes |
| `make_readme_gif.py` | The animation in the top-level README: a 2D target learned locally from a moving cluster of points | matplotlib, Pillow | 3 min |

```bash
python examples/example.py
OMP_NUM_THREADS=1 python examples/example_hybrid.py
python examples/multioutput_function_learning_pca.py [n_train] [easy|hard] [noise_std]
python examples/gpytorch_backend.py
OMP_NUM_THREADS=1 python examples/streaming_performance.py --target eggholder --d 3 --N 20000
OMP_NUM_THREADS=1 python examples/streaming_performance.py --target relic_density --stream walker --configs tree,hybrid,global_gp
OMP_NUM_THREADS=1 python examples/streaming_performance.py --target gaussian_peaks --d 6 --stream walker --configs tree,neighbours
OMP_NUM_THREADS=1 python examples/streaming_performance.py --target gaussian_peaks --d 6 --stream walker --configs tree --max_memory_mb 10
OMP_NUM_THREADS=1 mpiexec -n 4 python examples/mpi_fit_workers.py     # -n 1: no fit workers
python examples/make_readme_gif.py --alt      # --quick for a fast preview
```

The scripts write their figures to the current directory, except
`make_readme_gif.py`, which writes to `examples/example_plots/animation/`.

## Target functions

- **`target_functions.py`**: closed-form N-dimensional test functions on the
  unit cube. They take one column per point: `y = Eggholder(X.T)`.
  - `Eggholder`, `Himmelblau`, `Rosenbrock`, `Rastrigin`, `Levy` and `Custom`
    are sums of terms in one or two adjacent coordinates: an exact low-order
    additive structure that an additive kernel or the additive global GP can
    exploit.
  - `RotatedRosenbrock` and `GaussianPeaks` couple every input through a fixed
    random rotation and have no such structure.
  - `ActiveSubspacePeaks`, `Michalewicz`, `Ackley`, `Griewank`, `StepRidge` and
    `Chirp` are harder in one respect each: a hidden 2-plane of ten inputs,
    steep narrow valleys, ripples on a bowl, an interaction of every order, a
    jump across a hyperplane, a length scale that varies across the box.
  - `make_stream(kind, target, d, N, rng)` generates the input streams
    `'uniform'`, `'focusing'`, `'sweeping'` and `'walker'`.

- **`target_functions_physics.py`**: eleven cheap targets with the structure of
  quantities emulated in particle physics (`resonance_xsec`,
  `dilepton_spectrum`, `jet_spectrum`, `oscillation_probability`,
  `branching_ratio`, `loop_decay_width`, `relic_density`, `neutralino_mass`,
  `eft_signal_strength`, `signal_yield_mc` with Monte Carlo noise, and
  `global_fit_loglike`, a likelihood with nuisance parameters), in 4 to 10
  inputs. They take one row per point:

  ```python
  from target_functions_physics import TARGETS
  target = TARGETS['relic_density']
  X = rng.rand(1000, target.d)
  truth = target(X)                        # noise-free values
  y, sigma, truth = target.sample(X, rng)  # observations and their declared sigma
  ```
