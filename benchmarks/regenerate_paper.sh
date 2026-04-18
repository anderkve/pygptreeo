#!/usr/bin/env bash
# Regenerate the full paper-snapshot data set and paper-ready artefacts.
#
# This script codifies the exact command sequence used to produce the
# canonical iteration-07 results. It is idempotent — every run check
# inside `run_all.py` skips `.npz` files that already exist, so a
# second invocation is a no-op.
#
# Usage (from the repo root):
#     bash benchmarks/regenerate_paper.sh
#
# Environment: set OMP_NUM_THREADS=1 to keep the per-run timing
# comparable across methods (the driver spawns a subprocess per run
# and lets the parent control parallelism).
set -euo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1

ITER_DIR=benchmarks/iterations/iteration_08

# ---- 1. iid main sweep ------------------------------------------------
# Main methods with _A baseline configurations on all 4 default problems.
# Seeds 0..4 (i.e. n=5). Uses the 300 s subprocess cap; any method that
# hits the cap flushes its partial .npz and the driver writes an aborted
# stub if no checkpoint was reached.

python benchmarks/run_all.py \
    --methods pygptreeo_A sklearn_gp_A gpytorch_svgp_A random_forest_A river_knn_A \
    --problems smooth_sines_2d rosenbrock_2d friedman1_5d borehole_8d \
    --seeds 0 1 2 3 4 \
    --n-stream 2000 --checkpoint-every 200 --n-test 1000 \
    --max-wall-time 300

# ---- 2. variant sweeps ------------------------------------------------
# Stress / ablation variants: pygptreeo (B Nbar=100, C Matern-only),
# sklearn_gp_B (bigger reservoir), gpytorch_svgp_B (2x inducing / 3x
# training steps), river_knn_B (smaller k), pygptreeo_poe.

python benchmarks/run_all.py \
    --methods pygptreeo_B pygptreeo_C pygptreeo_poe \
    --problems rosenbrock_2d friedman1_5d \
    --seeds 0 1 2 \
    --n-stream 2000 --checkpoint-every 200 --n-test 1000 \
    --max-wall-time 300

python benchmarks/run_all.py \
    --methods sklearn_gp_B \
    --problems rosenbrock_2d friedman1_5d \
    --seeds 0 1 2 \
    --n-stream 2000 --checkpoint-every 200 --n-test 1000 \
    --max-wall-time 600

python benchmarks/run_all.py \
    --methods gpytorch_svgp_B river_knn_B \
    --problems rosenbrock_2d friedman1_5d \
    --seeds 0 1 2 \
    --n-stream 2000 --checkpoint-every 200 --n-test 1000 \
    --max-wall-time 300

# ---- 3. covariate-shift stress test -----------------------------------

python benchmarks/run_all.py \
    --methods pygptreeo_A gpytorch_svgp_A random_forest_A river_knn_A \
    --schedules shift \
    --problems rosenbrock_2d friedman1_5d \
    --seeds 0 1 \
    --n-stream 2000 --checkpoint-every 200 --n-test 1000 \
    --max-wall-time 300

# ---- 4. long-stream asymptote ----------------------------------------
# 5000-point runs, one seed each, separate out-dir to avoid colliding
# with the iid 2000-point seed-0 files.

mkdir -p benchmarks/data_long
python benchmarks/run_all.py \
    --methods pygptreeo_A \
    --problems rosenbrock_2d friedman1_5d borehole_8d \
    --seeds 0 \
    --n-stream 5000 --checkpoint-every 500 --n-test 1000 \
    --max-wall-time 1200 \
    --out-dir benchmarks/data_long

# ---- 5. plots + tables ------------------------------------------------
# Writes comparison, headline, wilcoxon_*, pareto, calibration, shift_vs_iid,
# scaling, paper_table.md, paper_table.tex into `$ITER_DIR/` and into
# benchmarks/plots/ (canonical HEAD).

python benchmarks/make_plots.py \
    --problems smooth_sines_2d rosenbrock_2d friedman1_5d borehole_8d \
    --iter-dir "$ITER_DIR"

echo ""
echo "Done. Canonical paper tables at:"
echo "  benchmarks/iterations/iteration_07/paper_table.{md,tex}   # pinned"
echo "  $ITER_DIR/paper_table.{md,tex}   # regenerated this run"
echo "Main headline figure: $ITER_DIR/headline.png"
