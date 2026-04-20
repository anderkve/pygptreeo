# CLAUDE.md

## Scope

This file applies **only to the benchmarking work in `benchmarks/`**. It
does *not* govern changes to the core `pygptreeo` package, the example
scripts, or the test suite — those follow ordinary contribution norms.

## Agent-based workflow for benchmark iterations

The benchmarking work is run as a recurring **reviewer ↔ implementer**
loop, with a periodic **referee** sweep when a chapter draft is ready.
The Claude Code parent session is the conductor: it spawns the agents,
gates progress on their output, commits, and pushes. It never invents
review.md or summary.md content itself when an agent should be doing it.

### Roles

- **Reviewer agent.** After every implementer iteration completes, the
  reviewer reads the just-produced `summary.md` plus the relevant prior
  `review.md` / `summary.md` files and the user's most recent guidance.
  It writes the next iteration's
  `benchmarks/iterations/iteration_<N>/review.md` — succinct, with a
  *Goal*, a numbered *Plan* naming files and parameters, an
  *Out-of-scope* list, and *Acceptance criteria*. The reviewer never
  writes summaries or runs sweeps.

- **Implementer agent (or the parent session acting as one).** Reads
  the latest `review.md`, makes the listed code changes, runs the
  sweep, generates plots into `iteration_<N>/plots/`, writes
  `iteration_<N>/summary.md`, and commits. The implementer never
  drafts the next iteration's review.

- **Referee agent.** Spawned only after a chapter draft exists (after
  ~3 iterations of reviewer+implementer work). Acts as a critical
  external referee: pushes back on choice of comparison methods,
  whether the tests measure what they claim to measure, whether some
  tests should be dropped, plot quality, etc. Writes a referee report
  (`benchmarks/referee/report_<N>.md`). Does not propose specific code
  changes.

### Loop

1. Reviewer agent → `iteration_<N>/review.md`
2. Implementer → code changes, sweep, plots, `iteration_<N>/summary.md`,
   commit, push
3. Repeat 1–2 for ≈3 iterations
4. Reviewer agent → `chapter/draft_<R>.md` (a benchmarking chapter for
   the paper, with figures, tables, and references)
5. Referee agent → `referee/report_<R>.md` (critical review of the
   draft)
6. Reviewer + implementer → 2–3 more iterations addressing the referee
7. Reviewer agent → `chapter/draft_<R+1>.md` (revised chapter)
8. Referee agent → `referee/report_<R+1>.md`

### Writing constraints (apply to **all** agents in this loop)

- **No bloat.** Succinct, dense prose, in the style of a strong
  academic paper. Lead with the result, justify in one sentence,
  avoid filler.
- No emoji, no padding adverbs, no "in conclusion" paragraphs.
- Tables before paragraphs whenever the data is tabular.
- Cite filenames with `path:line` so the reader can navigate.

## Per-iteration directory layout

```
benchmarks/iterations/iteration_<N>/
├── review.md       # written by reviewer agent
├── summary.md      # written by implementer
├── data/           # all .npz files for this iteration
└── plots/          # all .png/.pdf figures for this iteration
```

The global `benchmarks/plots/` always mirrors the *latest* iteration's
plots, so the head-of-branch view shows the current state. The
per-iteration `plots/` subfolders preserve history.

## Reliability invariant

Every iteration's `summary.md` opens with a single quotable line:

> `Reliability: M / N pygptreeo* runs have frac_pathological_std[-1] == 0 (P %)`

Target is 100 %. A drop is a regression and must be triaged before the
next iteration starts.

## Commit / push norms for benchmark commits

- One commit per iteration (code + data + plots + summary together).
  Partial-data commits during a long sweep are allowed if the parent
  session needs to flush the workspace; the final commit lands the
  rest.
- Commit subject: `Iteration <N>: <one-line headline>`.
- Body: bulleted "what landed" and the headline finding(s); end with
  the reliability line.
- Push to `claude/benchmark-pygptreeo-comparison-FjW3K` (the working
  branch).
