# Iterative benchmark development log

Each numbered subdirectory corresponds to one round of
**reviewer → implementer → run → plot** refinement of the benchmark. Later
iterations strictly supersede earlier ones, but we keep everything for audit
and provenance.

Layout of each iteration directory:

```
iteration_NN/
├── review.md                # the reviewer's instructions for this iteration
├── summary.md               # the implementer's summary + observations
├── comparison.png           # main multi-panel figure at iteration end
├── summary.png              # final-step bar charts
├── pareto.png               # accuracy vs compute scatter
├── calibration.png          # reliability diagrams
└── data/*.npz               # snapshot of the raw results used for the plots
```

## Roles

**Reviewer (Plan agent).** Reads the previous iteration's code + figures +
summary, writes `review.md` with a short prioritised punch-list and explicit
out-of-scope items, and states acceptance criteria.

**Implementer (this agent).** Applies the reviewer's punch-list, runs the
benchmark, writes `summary.md` with honest observations, and snapshots the
resulting plots + data to the iteration directory.

## Iteration log

| # | Focus                                                              |
| - | ------------------------------------------------------------------ |
| 00 | Baseline + NLPD bug investigation                                  |
| 01 | Robust metrics (median/trimmed NLPD, CRPS, multi-level coverage), multi-seed, fair per-method budgets, emulation-community problems (borehole, Friedman-1), distribution-shift schedule option |
| 02 | TBD — populated after iteration 01's reviewer step                 |
