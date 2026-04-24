# coverage_1sigma drift across all pygptreeo* runs

Definition: a run is **in-band** iff `coverage_1sigma[-1] ∈ [0.6, 0.76]`. The nominal value is 0.6827.

The band only has prescriptive meaning when training and test distributions are aligned. Under `shift`, `mcmc`, `de`, and emulator-assisted MCMC, the test set is uniform-iid but the training stream is not, so the emulator's coverage on the test set is *expected* to drift outside this band; that is the substantive content of the iter-11/13/14 chapters, not a methods bug.

## Headline

- Aligned (iid, lhs): **214 / 234 in-band (91.5 %)**
- Non-aligned (shift, de, mcmc, assisted, delayed): 25 / 103 in-band (24.3 %)
- Combined: 239 / 337 in-band (70.9 %)

## Aligned-stratum out-of-band cells (20)
These would be a regression — investigate.

| iteration | method | problem | schedule | seed | coverage_1sigma[-1] |
|---|---|---|---|---|---|
| iteration_01 | pygptreeo | friedman1_5d | iid | 0 | 0.558 |
| iteration_10 | pygptreeo_A | borehole_8d | lhs | 1 | 1.000 |
| iteration_13 | pygptreeo_A | borehole_8d | iid | 0 | 0.789 |
| iteration_13 | pygptreeo_A | borehole_8d | iid | 0 | 0.922 |
| iteration_13 | pygptreeo_A | borehole_8d | iid | 0 | 0.922 |
| iteration_13 | pygptreeo_A | borehole_8d | iid | 0 | 0.961 |
| iteration_13 | pygptreeo_A | rosenbrock_2d | iid | 0 | 0.915 |
| iteration_13 | pygptreeo_A | rosenbrock_2d | iid | 0 | 0.903 |
| iteration_13 | pygptreeo_A | rosenbrock_2d | iid | 0 | 1.000 |
| iteration_13 | pygptreeo_A | rosenbrock_2d | iid | 0 | 1.000 |
| iteration_13 | pygptreeo_A | rosenbrock_2d | iid | 0 | 0.419 |
| iteration_13 | pygptreeo_D | borehole_8d | iid | 0 | 0.536 |
| iteration_13 | pygptreeo_D | borehole_8d | iid | 0 | 0.842 |
| iteration_13 | pygptreeo_D | borehole_8d | iid | 0 | 0.931 |
| iteration_13 | pygptreeo_D | borehole_8d | iid | 0 | 0.933 |
| iteration_13 | pygptreeo_D | rosenbrock_2d | iid | 0 | 0.899 |
| iteration_13 | pygptreeo_D | rosenbrock_2d | iid | 0 | 0.926 |
| iteration_13 | pygptreeo_D | rosenbrock_2d | iid | 0 | 0.957 |
| iteration_13 | pygptreeo_D | rosenbrock_2d | iid | 0 | 0.994 |
| iteration_13 | pygptreeo_D | rosenbrock_2d | iid | 0 | 0.419 |

## Non-aligned out-of-band cells: 78 (expected; not flagged)
