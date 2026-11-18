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
| `multioutput_function_learning_pca.py` | The multi-output models `independent`, `shared` and `pca` on a function f(t; x) observed on a t-grid, against a B-spline representation of the curves | matplotlib for the figure | 15 s |
| `multioutput_physics_curves.py` | The multi-output models on a stream of physics-like curves f(x; theta), e.g. a resonance, a spectrum with a peak that moves, a kinematic edge. `aligned` against `shared` as a figure | matplotlib for the figure | 20 s |
| `multioutput_surfaces.py` | `shared` and `aligned` leaves on families of functions f(u, v; theta) on a grid of two dimensions, e.g. a peak that moves along both axes, with and without `output_shape` | | 7 min |
| `gpytorch_backend.py` | GPyTorch models as the leaf GPs, against the scikit-learn default | GPyTorch | 10 s |
| `streaming_performance.py` | Accuracy, coverage and time per point along a stream, for the plain tree, the hybrid and the additive global GP, as a figure. `--max_memory_mb` limits the memory of the trees | matplotlib (PyTorch for the hybrid) | 2 min |
| `mpi_fit_workers.py` | An MPI trainer that holds the tree and hands the leaf fits to fit workers (`update_tree(defer_fits=True)`), against a trainer that fits the leaves itself | mpi4py | 15 s on 4 processes |
| `make_readme_gif.py` | The animation in the top-level README: a 2D target learned locally from a moving cluster of points | matplotlib, Pillow | 3 min |

```bash
python examples/example.py
OMP_NUM_THREADS=1 python examples/example_hybrid.py
python examples/multioutput_function_learning_pca.py [n_train] [easy|hard] [noise_std]
OMP_NUM_THREADS=1 python examples/multioutput_physics_curves.py zprime_spectrum 400 shared,aligned
OMP_NUM_THREADS=1 python examples/multioutput_surfaces.py peak 800
python examples/gpytorch_backend.py
OMP_NUM_THREADS=1 python examples/streaming_performance.py --target eggholder --d 3 --N 20000
OMP_NUM_THREADS=1 python examples/streaming_performance.py --target relic_density --stream walker --configs tree,hybrid,global_gp
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

- **`target_functions_curves.py`**: twelve families of curves f(x; theta) with
  the structure of functions emulated in physics, as multi-output targets: the
  curve on a fixed grid in x for parameters theta in the unit cube.
  - `matter_power`, `eft_distribution` and `jet_spectrum` change shape slowly
    and by a few modes.
  - `dust_sed`, `z_lineshape`, `zprime_spectrum` and `kinematic_edge` have a
    cutoff, a peak or an edge whose position depends on the parameters.
  - `oscillation_probability`, `cmb_peaks`, `form_factor` and `ringdown`
    oscillate with a phase or a frequency that depends on the parameters.
  - `peak_counts_mc` has Monte Carlo noise that differs from bin to bin.

  ```python
  from target_functions_curves import FAMILIES
  family = FAMILIES['zprime_spectrum']
  U = rng.rand(500, family.d)
  truth = family(U)                        # noise-free curves, shape (500, len(family.x))
  Y, sigma, truth = family.sample(U, rng)  # observations and their declared sigma
  ```
