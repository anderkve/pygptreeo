# Benchmarks

## Tree-global hyperparameter pooling (`GPTree(pool_hyperparameters=True)`)

### What it does
Each leaf of a `GPTree` fits its own local GP, including the kernel
hyperparameters. With pooling enabled, every leaf shares a tree-global pool of
learned hyperparameters: before fitting, a leaf is **warm-started** from the
robust (elementwise-median) consensus of the other leaves' current
hyperparameters; after fitting, it contributes its result back. The pool stores
hyperparameters in the kernel's native log-space (`theta`) and is computed in
the leaves' standardized coordinate system, which is what makes the values
comparable across leaves (and compatible with `use_standard_scaling`, unlike the
parent→child `use_hyperparameter_inheritance`).

The mechanism is **opt-in and off by default**; it is a no-op for GP backends
that do not implement `get_hyperparameters`/`set_hyperparameters`.

### How to reproduce
```bash
# Default isotropic kernel (the common case)
OMP_NUM_THREADS=1 python benchmarks/benchmark_hp_pooling.py --kernel iso

# Anisotropic + noise kernel (4 hyperparameters, hard to fit per leaf)
OMP_NUM_THREADS=1 python benchmarks/benchmark_hp_pooling.py --kernel ard

# Most favorable regime for pooling: hard-to-estimate but spatially-homogeneous
OMP_NUM_THREADS=1 python benchmarks/benchmark_hp_pooling.py --kernel ard --target aniso
```
Node routing and tree growth are independent of the GP hyperparameters, so with
a fixed seed the pooling-on and pooling-off trees have **identical structure and
data ordering** — the only difference is how each leaf is hyperparameter-fitted.
This isolates the effect of pooling. Metrics (RMSE, NLPD, cumulative train time)
are reported at checkpoints as a function of the number of points ingested,
averaged over several seeds.

### Findings (honest summary)
Across the configurations tested, pooling did **not** improve prediction
accuracy or sample efficiency:

| Kernel | Restarts | Final RMSE (off → on) | Early-N RMSE | Train time |
|--------|----------|-----------------------|--------------|------------|
| `iso`  | 0 | identical | unchanged | ~2% faster |
| `ard` (mixed) | 0 | identical | **up to ~3× worse** transiently | ~4–5% faster |
| `ard` (aniso) | 0 | identical | ~23% worse (mean, first half) | ~10% faster |
| `ard` | 2 | identical | unchanged | ~2% faster |

Interpretation:

1. **Default isotropic kernel: no effect.** With only two easily-identified
   hyperparameters, a leaf fits them reliably from very few points, so warm-
   starting from the consensus lands at the same optimum. Pooling changes
   nothing except a small reduction in optimizer iterations.

2. **Harder (ARD) kernels with no optimizer restarts: pooling can hurt.** A
   single optimization started from the median can get *stuck* near the global
   consensus, which is the wrong local geometry for that particular leaf — this
   suppresses exactly the per-region hyperparameter adaptation that a GP tree
   exists to capture. The damage is transient (everything agrees once leaves
   mature) but real during the sample-efficient regime we care about.

3. **The only consistent benefit is a small wall-clock speedup** (~2–10%, larger
   when optimization is more expensive), because a good initial point reduces
   the number of optimizer iterations. With `n_restarts_optimizer >= 1` the
   restarts rescue any bad pooled init, so accuracy is unchanged and the speedup
   remains with no downside.

### Recommendation
Leave pooling **off by default**. It is not a sample-efficiency lever in these
tests. Consider enabling it only as a *warm-start speedup* when hyperparameter
optimization is expensive (e.g. `n_restarts_optimizer >= 1`) and you have reason
to believe the optimal local hyperparameters are similar across regions. Do not
enable it with single-shot optimization on functions whose local length scales
vary by region.

Plots: `hp_pooling_iso.png`, `hp_pooling_ard.png`.
