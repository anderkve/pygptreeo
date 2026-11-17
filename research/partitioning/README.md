# Partitioning study

Scripts behind section 8 of `docs/benchmarks.md`: other ways to partition the
data than cuts along the inputs, and predictions from a query's nearest
points. None of this is part of the package; the results that were kept are
`GPTree(predict_from='neighbours')` and the index of the leaves.

Run from this folder, one core per process (`run_matrix.py` starts four).
Records go to `results/` (not in the repository).

| File | What it does |
|---|---|
| `bench.py` | Targets, streams and one prequential run; each run stores its per-point predictions |
| `variants.py` | The models compared, by name: `base`, `nbf` (the shipped neighbours), `nb:...`, `oblique:...`, `mr`, `knn:...`, `fisheye:...`, `anchored:...`, `tapered:...`, `blended:...`, `mixed:...`, `hybrid:...` |
| `run_matrix.py` | Runs variants x targets x streams x seeds in parallel; finished runs are skipped |
| `report.py`, `tables.py`, `plots.py` | Tables and figures from the records |
| `oblique.py` | Hyperplane cuts with a free normal (among them two clusters of the leaf's points), cut positions from the GP's gradient, decorrelated leaf inputs |
| `multires.py` | A model in every node: descendants fit the residual of their ancestors' frozen GPs |
| `anchored.py` | Neighbours of one of a few cells per leaf, shared by the queries of the cell |
| `tapered.py` | The prototype of `neighbour_fade`: neighbours whose noise grows towards the edge of their set |
| `blended.py` | The prototypes of `neighbour_blend`: predictions blended, or the leaves' kernels mixed, in a zone around each cut |
| `jitter.py` | Steps of the predicted mean along short paths |
| `sleepy.py` | Leaves asleep by default with predictions from neighbours: memory, time and wake-ups |
| `memory_limit.py` | `max_memory_mb` with predictions from the leaves and from neighbours |
| `knn.py` | The first version of the neighbours (brute-force search, optional far points and per-level samples) |
| `diag_knn.py`, `diag_batch.py`, `diag_knn_report.py` | Replay: the leaf's prediction against that of its kernel on the nearest points, for every size of the neighbour set, and rules for the test against the kernel |
| `diag_kernel.py`, `diag_kernel_report.py` | Replay: which leaf's kernel the neighbours should use |
| `diag_select.py` | Replay: a leaf chooses between its own GP and the neighbours by their recent errors |
| `drift_bench.py` | Cost along a steadily drifting stream |
| `time_predict.py` | Update and prediction times with one process |
| `time_scaling.py` | The same at checkpoints along one long stream |
| `index_scaling.py` | The search for the nearest leaves, and the leaves of a prediction with zones at the cuts, up to a million points, without GPs |
| `profile_search.py` | The parts of a neighbour search on a growing tree |

```bash
python run_matrix.py --variants "base;nbf" --set dev --N 5000 --seeds 1
python report.py --variants "nbf" --base base --N 5000 --per_run
python tables.py long 20000 1,2 nbf base
```
