# Split-dimension criteria: benchmark summary

Streaming, 20 000 points, `Nbar=200`, `retrain_every_n_points=50`, anisotropic
Matérn(3/2) ARD kernel, gradual splitting, MoE aggregation. Reproduce with
`python examples/benchmark_split_direction.py <target> 20000`.

Final-batch NRMSE per criterion:

| criterion       | aniso_chirp | rosenbrock | rastrigin | eggholder |
|-----------------|------------:|-----------:|----------:|----------:|
| max_spread      | 0.0010      | 0.0001     | **0.0543** | 0.0431   |
| max_variance    | 0.0009      | 0.0001     | 0.0552    | 0.0444    |
| max_uncertainty | **0.0001**  | **0.0000** | 0.0604    | **0.0428** |
| min_lengthscale | **0.0001**  | **0.0000** | 0.0599    | 0.0446    |
| random          | 0.0021      | 0.0002     | 0.0620    | 0.0467    |

- On `aniso_chirp` (one dimension genuinely rougher, but with smaller input
  spread) the GP-aware criteria are ~10× better than the spread/variance defaults,
  which split the wrong (smooth, wide) dimension.
- `min_lengthscale` matches the best (`max_uncertainty`) everywhere and is the
  best or within noise of the best on every target, while being cheaper — it
  reuses the GP's fitted length scales instead of probing the GP on a grid.
- On isotropic/separable functions all criteria are within ~0.01; the split axis
  barely matters there. Coverage stays ~0.68 throughout.

**Takeaway:** `min_lengthscale` is the recommended split-dimension criterion for
ARD kernels — best-or-tied accuracy at low cost, and a no-op when there is no
anisotropy to exploit.

An oblique (non-axis-aligned) criterion was also prototyped but removed: it only
helped on a synthetic target built for it and was a net negative on the standard
functions at higher cost (see git history, commit `3b3dd88`).
