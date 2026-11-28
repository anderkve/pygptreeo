# Search-based symbolic regression and its benchmarks: state of the art as of November 2026

Reading status (28 November 2026). "Full" = whole text read as plain text extracted from the arXiv HTML or PDF; "targeted" = method, protocol, results, limitations and relevant appendices read, the rest skimmed.

- Full or near-full: PySR [2305.01582](https://arxiv.org/abs/2305.01582) (software-implementation sections 2.3 to 2.5 skimmed); SRBench [2107.14351](https://arxiv.org/abs/2107.14351); SRBench 2.0 "Call for action" [2505.03977](https://arxiv.org/abs/2505.03977); SRSD [2206.10540](https://arxiv.org/abs/2206.10540) (the per-equation tables 8 to 18 not read line by line); UDFS "Scaling up unbiased search-based SR" [2506.19626](https://arxiv.org/abs/2506.19626); MDBench [2509.20529](https://arxiv.org/abs/2509.20529) (PDE dataset appendix skipped); uDSR main paper ([NeurIPS 2022 PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf)); ERBench [2606.09276](https://arxiv.org/abs/2606.09276); FastSRB [2508.14481](https://arxiv.org/abs/2508.14481).
- Targeted: DSR [1912.04871](https://arxiv.org/abs/1912.04871); PhySO [2303.03192](https://arxiv.org/abs/2303.03192); SISSO [1710.03319](https://arxiv.org/abs/1710.03319); Aldeia's thesis "Current challenges of SR" [2512.01682](https://arxiv.org/abs/2512.01682); SymbolicArena [2609.35113](https://arxiv.org/abs/2609.35113); MOSAIC-SR [2609.20997](https://arxiv.org/abs/2609.20997); LEE [2605.27245](https://arxiv.org/abs/2605.27245); ParFam [2310.05537](https://arxiv.org/abs/2310.05537); SRBench 2022 competition analysis [2304.01117](https://arxiv.org/abs/2304.01117); LLM-SRBench [2504.10415](https://arxiv.org/abs/2504.10415); "Fundamental limits" [2204.02704](https://arxiv.org/abs/2204.02704); SMC-SR [2512.10849](https://arxiv.org/abs/2512.10849); Wilstrup and Kasak [2103.15147](https://arxiv.org/abs/2103.15147); cp3-bench [2406.15531](https://arxiv.org/abs/2406.15531) (abstract and data description only).
- Could not be opened: the Operon paper (Burlacu, Kronberger, Kommenda, GECCO Companion 2020, [doi 10.1145/3377929.3398099](https://doi.org/10.1145/3377929.3398099)). The ACM site returned 403, Semantic Scholar lists it as closed access, and no free copy was found. Everything said about Operon below comes from other sources and is marked as such. The uDSR supplementary material (appendices with hyperparameters) returned 404 and was not read.
- Numbers that exist only in figures in the papers (SRBench Fig. 3, uDSR Fig. 2, UDFS Fig. 5, ParFam Fig. 4, DSR Fig. 4) could not be read from the text. For SRBench 2021 and SRBench 2025 the gap was closed by computing tables directly from the result files in the benchmark repository; these tables are labelled "computed for these notes".
- Several sources are from mid to late 2026 (ERBench June 2026, MOSAIC-SR and SymbolicArena September 2026); they were read from arXiv, not recalled. MOSAIC-SR and SymbolicArena are v1 preprints with no sign of peer review; ERBench carries a "Reviewed on OpenReview" line.

## 1. For each benchmark: how many points, what noise, how many inputs, what budget, and what counts as success?

### Takeaway
The benchmarks with a ground truth use large clean samples (10,000 training points in SRBench and 8,000 in SRSD), long budgets (hours per run) and a symbolic-equivalence test done with SymPy; noise is an add-on at 0.1 %, 1 % and 10 % of the target's RMS. Only EmpiricalBench and the first-principles track of SRBench 2025 sit in the 6 to 236 point regime, and neither scores automatic symbolic recovery (one is judged by eye, the other by position on an accuracy-size plot).

### Cited Findings

#### Summary table

| Benchmark | Problems | Training points | Inputs | Noise | Budget per run | Runs | Success criterion |
|---|---|---|---|---|---|---|---|
| SRBench 2021, ground-truth track | 130 (116 Feynman + 14 Strogatz) | 75 % of the data, capped at 10,000 rows | Feynman up to 9; Strogatz 2 | Gaussian on target, γ ∈ {0, 0.001, 0.01, 0.1} × RMS(y) | 1M evaluations or 8 h, 1 core | 10 | Symbolic solution (Def. 4.1, SymPy); also R² > 0.999 on test |
| SRBench 2021, black-box track | 122 PMLB sets | 75 %, capped at 10,000 | 2 to 124 | none added | 500k evaluations per training or 48 h, incl. tuning | 10 | test R², model size |
| SRBench 2025 ("2.0"), black-box | 12 PMLB sets | 75 % of 47 to 31,104 rows | 3 to 51 columns incl. target | none added | 6 h tuning + 1 h final fit, 1 core, 10 GB | 30 | AUC of a performance profile over test R²; model size |
| SRBench 2025, phenomenological and first-principles | 12 | 75 % of 6 to 236 rows | 1 to 3 | real data, or generated with 1 %, 10 %, 20 % noise | same | 30 | Position on test-R² vs size Pareto plot relative to the accepted equation; no automatic symbolic metric |
| EmpiricalBench (PySR paper) | 9 | whole data set (8 to 100 rows, sizes from SRBench 2025 Table 1) | 1 to 3 | real or "realistic" noise | 1 h, 8 cores | 5 | By eye: true expression is somewhere in the returned Pareto front |
| SRSD-Feynman | 120 (30 easy, 40 medium, 50 hard) + 120 with dummy variables | 8,000 (plus 1,000 validation, 1,000 test) | as Feynman, plus 1 to 3 dummies | none in main tables; γ ∈ {0, 1e-3, 1e-2, 1e-1} on Easy only (App. E) | 24 h job, up to 100 hyperparameter sets or seeds, best chosen on validation | 1 pick | R² > 0.999; solution rate (SRBench def.); normalised edit distance |
| Nguyen (as run by DSR) | 12 | 20 uniform points | 1 or 2 | 0 to 0.1 × RMS(y) in a separate experiment | 2M expression evaluations | 100 | Exact symbolic equivalence (SymPy) |
| SRBench 2022 competition | 5 synthetic tasks × 3 or 4 levels + 1 real-world | not stated in the sections read | 1 to 3 in the formulas shown | 0.05 to 0.2 ratio | 1 h (≤ 1,000 samples), 10 h (≤ 10,000) | 10 in the qualification stage | harmonic mean of ranks on R², simplicity, task score |
| MDBench | 63 ODE + 14 PDE systems | first 60 % of the trajectory (+20 % validation) | 1 to 4 states (ODE) | multiplicative, SNR 40, 30, 20, 10 dB on the states | 12 h incl. tuning | not stated | NMSE of derivatives; complexity; coefficients within ±5 % (PDE table) |
| FastSRB (SRSD equations) | 120 | 200 | as SRSD | none | 30 min, single process, early stop on success | 5 | Match to a curated list of acceptable forms after SymPy simplification and rounding |
| ERBench | public development set + secret evaluation set | varies by design | varies | varies | user-run, defaults | 5 | Numeric check of scale or shift equivalence on 1,000 points in [-100,100]^d; Jaccard index; tree edit distance |
| SymbolicArena Core50 | 50 distilled from 664 | per source benchmark | per source | clean; 1 % and 5 % of std(y) as supplement | 3 h | 3 seeds | six axes; symbolic equivalence judged by an LLM |
| LLM-SRBench | 239 | per problem | per problem | none stated | 1,000 LLM calls | n/a | symbolic accuracy judged by GPT-4o; Acc_0.1; NMSE in and out of domain |

#### SRBench 2021
- Settings table: 130 ground-truth problems, 14 algorithms, 10 trials per dataset, 75/25 split, "Tuned set from Black-box problems", termination "1M evaluations or 8 hours", target noise "0, 0.001, 0.01, 0.1", 54,600 runs, 436.8K core hours. Black-box: 122 problems, 21 algorithms, "500k evaluations/train or 48 hours", 5-fold halving grid search — [SRBench Table 2](https://arxiv.org/abs/2107.14351).
- Each job "received one CPU core and up to 16GB of RAM"; "training data exceeding 10,000 samples was randomly subset to 10,000 rows"; the final models got "an additional hour of computing time with 8GB of RAM to be simplified with sympy and assessed by the solution criteria" — [SRBench App. A.5](https://arxiv.org/abs/2107.14351).
- Each SR method was tuned from six hyperparameter combinations on the black-box problems; "The most common parameter setting chosen during the black-box regression experiments was then used as the 'tuned' version of each algorithm for the ground-truth problems, with updates to 1) include any mathematical operators needed for those problems and 2) double the evaluation budget." — [SRBench App. A.5](https://arxiv.org/abs/2107.14351).
- Feynman data: "each dataset was down-sampled from 1 million samples to 100,000"; the version "not simplified by dimensional analysis" was used; II.11.17 was missing and "we excluded three datasets from our analysis that contained arcsin and arccos functions". Strogatz: "Each dataset is one state of a 2-state system of first-order, ordinary differential equations" — [SRBench App. A.4](https://arxiv.org/abs/2107.14351). The repository result file contains 116 Feynman and 14 Strogatz problems — [SRBench results, v2.0](https://github.com/cavalab/srbench/tree/master/results) (counted for these notes).
- The Feynman problems have up to 9 input variables (ParFam notes that handling "9 dimensions and arbitrary constants" is "necessary for the Feynman dataset") — [ParFam App. G](https://arxiv.org/abs/2310.05537).
- Noise: "White gaussian noise was added to the target as a fraction of the signal root mean square value", y_noise = y + ε, ε ~ N(0, γ·sqrt((1/N) Σ y_i²)) — [SRBench App. A.4](https://arxiv.org/abs/2107.14351).
- Success: "Definition 4.1 (Symbolic Solution). A model φ̂(x, θ̂) is a Symbolic Solution to a problem with ground-truth model y = φ*(x, θ*) + ε, if φ̂ does not reduce to a constant, and if either of the following conditions are true: 1) φ* − φ̂ = a; or 2) φ*/φ̂ = b, b ≠ 0, for some constants a and b." "Prior to assessing symbolic solutions, each model underwent sympy simplification, as did the conditions above." The authors add: "because models can be represented in myriad ways, and sympy's simplification procedure is non-optimal, we cannot guarantee that all symbolic solutions are captured with perfect fidelity by this metric." — [SRBench Sec. 4.3](https://arxiv.org/abs/2107.14351).
- The weaker criterion, "Accuracy Solutions", is "models with R² > 0.999 on the Feynman and Strogatz problems" — [SRBench Fig. 8](https://arxiv.org/abs/2107.14351). SRSD notes that the SRBench code "ignores coefficient terms whose absolute values are less than 10^-4" — [SRSD footnote 13](https://arxiv.org/abs/2206.10540).
- Complexity is "the number of mathematical operators, features and constants in the model", also computed after SymPy simplification — [SRBench Sec. 4.3](https://arxiv.org/abs/2107.14351).

#### SRBench 2025 ("SRBench 2.0", "Call for action")
- Changes: 25 methods; "increasing the number of independent runs from 10 to 30"; "selecting 24 datasets, divided into two tracks: (i) black-box (drawn from SRBench 1.0) and (ii) phenomenological & first-principles (sourced from (Cranmer, 2023))" — [SRBench 2.0 Sec. 1](https://arxiv.org/abs/2505.03977). The Feynman and Strogatz ground-truth track, with its symbolic solution rate, is not part of this edition (no such track or metric appears in Sec. 3 or 4).
- First-principles data sets and row counts: absorption 14, bode 8, hubble 32, ideal_gas 30 ("Generated, 10% noise"), kepler 6, leavitt 26, newton 30 (10 % noise), planck 100 (10 % noise), rydberg 50 (1 % noise), schechter 27 (20 % noise), supernovae_zr 236, tully_fisher 18; 2 to 4 columns including the target — [SRBench 2.0 Table 1](https://arxiv.org/abs/2505.03977). "All these datasets are particularly challenging because of the small number of points and the diverse and unknown natural distribution of the phenomena." — [SRBench 2.0 Sec. 3.1](https://arxiv.org/abs/2505.03977).
- "All datasets contain no missing values, have only numeric features, and are standardized before training." — [SRBench 2.0 Sec. 3.1](https://arxiv.org/abs/2505.03977).
- Tuning: "Hyper-parameter optimization is performed before each run using a 3-fold cross-validation on the training data (75%) and selecting the best configuration based on the average R² score"; "we included the default (off-the-shelf) configuration as one of the candidate settings, since preliminary results suggest that, for some problems, is outperforms tuned configurations." — [SRBench 2.0 Sec. 3.2](https://arxiv.org/abs/2505.03977).
- Budget: "Each job was allocated 10 GB of RAM ... Jobs were not allowed to spawn subprocesses or utilize multiple cores. The time budget was set to 6 hours for hyperparameter search and 1 hour for training the algorithm with the optimal configuration" — [SRBench 2.0 Sec. 3.4](https://arxiv.org/abs/2505.03977).
- Scoring: a performance profile gives, for each R² threshold, the share of runs reaching it; the area under it is the summary ("a value of 1.0 indicates that all 30 runs achieved the maximum R²=1, whereas a value of 0.0 indicates that all runs resulted in R² ≤ 0"); the aggregation over the 30 runs in the paper is the maximum. Model size is the SymPy node count without simplification "as simplification using SymPy is unreliable and can increase the model size" — [SRBench 2.0 Sec. 3.3](https://arxiv.org/abs/2505.03977).
- The repository README (read 28 Nov 2026) calls SRBench 2025 "the current edition" and says: "If you are choosing baselines for a new symbolic regression paper, please use this roster and these results rather than the 2021 tables below." — [SRBench README](https://github.com/cavalab/srbench).

#### EmpiricalBench (PySR paper)
- Nine historical laws (Hubble, Kepler, Newton, Planck, Leavitt, Schechter, Bode, ideal gas, Rydberg); original data where available, otherwise "generated from the equation with realistic ranges of variables, with noise applied"; no physical constants are given to the algorithm — [PySR Sec. 3.2, Table 2](https://arxiv.org/abs/2305.01582).
- Budget: "We allow every method to use 8 cores on an AMD Rome CPU running Rocky Linux, and allow them to search for up to 1 hour." — [PySR Sec. 3.2.1](https://arxiv.org/abs/2305.01582).
- "Each algorithm is fed the entire dataset for training. This is because our test is the output expression, rather than a separate test dataset." — [PySR Sec. 3.2.1](https://arxiv.org/abs/2305.01582).
- Counting success: "each of the Pareto fronts was analyzed, by eye, to see whether the true expression was contained within it. This was performed manually, as oftentimes automated equality checking with sympy produced incorrect results." Categories: "(1) correct if the Pareto front contained the true expression, (2) almost if one of the expressions in the Pareto front was off of the true expression by a constant factor somewhere, (3) failed, if the search produced no results ... and (4) incorrect". Crashed runs count against the method — [PySR Sec. 3.2.1](https://arxiv.org/abs/2305.01582).
- The study is of "the entire Pareto front of each algorithm, rather than only a single expression, as was done in SRBench" — [PySR Sec. 3.2.1](https://arxiv.org/abs/2305.01582).

#### SRSD (Matsubara et al.)
- "Each of the 120 SRSD datasets consists of 10,000 samples and has train, val, and test splits with ratio of 8:1:1." — [SRSD App. A](https://arxiv.org/abs/2206.10540).
- Sampling: physical constants are constants with their SI values; ranges are set "in SI units to correspond to each typical physics experiment"; "Generally, the ranges are set to be sampled on log scales within their orders as 10^2"; angles uniformly; signed variables in their sign range; integers where physically integer — [SRSD Sec. 3.2.1](https://arxiv.org/abs/2206.10540).
- Difficulty classes from the number of operations in the true tree and the range of the sampling domains: 30 easy, 40 medium, 50 hard — [SRSD Sec. 3.2.2](https://arxiv.org/abs/2206.10540).
- Dummy variables: 1 to 3 per data set, each log-uniform in (10^(s-1), 10^(s+1)) with s drawn from {-32, ..., 32}, at random column positions — [SRSD Sec. 3.3](https://arxiv.org/abs/2206.10540).
- Metrics: accuracy as R² > 0.999 on test; solution rate as in SRBench; and normalised edit distance d̄ = min(1, d(f_pred, f_true)/|f_true|) on equation trees (Zhang-Shasha) after SymPy simplification, with constants replaced by a symbol — [SRSD Sec. 4.1, Eq. 3](https://arxiv.org/abs/2206.10540).
- Model choice: for each method the best trained model among all its runs is chosen by mean squared relative error on the validation set (Eq. 4), then scored — [SRSD Sec. 4.2](https://arxiv.org/abs/2206.10540).
- Budget: "each HPC job is designed to run for up to 24 hours"; "each of our HPC jobs runs up to 100 separate training sessions with different hyperparameter values"; nodes with 5 to 20 cores — [SRSD Sec. 5.2](https://arxiv.org/abs/2206.10540).
- Noise appears only in App. E (Easy set, five older baselines, same γ levels as SRBench) and is listed as a limitation: "these aspects are not thoroughly discussed in this study" — [SRSD App. E and G.2](https://arxiv.org/abs/2206.10540).

#### Nguyen as used by DSR
- 12 expressions; training data U(a, b, 20), that is 20 random points per input variable range; libraries without constants; "Each experiment consists of 2 million expression evaluations"; "All experiments were replicated with 100 different random seeds" — [DSR Sec. 4 and App. D Table 2](https://arxiv.org/abs/1912.04871).
- "We use the strictest definition of recovery: exact symbolic equivalence, as determined using a computer algebra system, e.g. SymPy". With noise: "we defined recovery as exact symbolic equivalence on any expression along the reward-complexity Pareto front at the end of training" — [DSR Sec. 4](https://arxiv.org/abs/1912.04871).

#### SRBench 2022 competition (de França et al., "SRBench++")
- "a candidate SR algorithm had a pre-specified time budget of 1 hour for datasets up to 1000 samples and 10 hours for datasets up to 10000 samples", tuning included; participants did not know the data sets in advance — [Competition Sec. III](https://arxiv.org/abs/2304.01117).
- Five synthetic tasks: exact rediscovery, feature selection, escaping local optima, extrapolation, sensitivity to noise; the final rank is the harmonic mean of ranks on R², simplicity = round(−log5(s), 1) with s the SymPy node count, and a task score — [Competition Sec. III-B](https://arxiv.org/abs/2304.01117).
- Extrapolation task: f(x) = erf(0.22x) + 0.17 sin(5.5x), training on −15 ≤ x ≤ 15, validation on 15 < x ≤ 40, noise ratios 0.05, 0.1, 0.2; the score is R² on the validation range — [Competition Sec. III-B4](https://arxiv.org/abs/2304.01117).
- Noise task: f(x) = (0.11x⁴ − 1.4x³)/(0.68x² + 1), noise ratios 0.05, 0.1, 0.15, evaluated on noiseless data — [Competition Sec. III-B5](https://arxiv.org/abs/2304.01117).

#### MDBench
- "MDBench assesses 12 algorithms on 14 partial differential equations (PDEs) and 63 ordinary differential equations (ODEs) under varying levels of noise"; ODE systems have "one to four state variables"; inputs are states, targets their time derivatives — [MDBench abstract, Sec. 3.1](https://arxiv.org/abs/2509.20529).
- Noise: "Gaussian noise at signal-to-noise ratios (SNRs) of 40 dB, 30 dB, 20 dB, and 10 dB", multiplicative, u → (1 + ξ)u with σ = 10^(−SNR/20); derivatives are computed from the noisy states by finite differences and the error is measured against clean derivatives — [MDBench Sec. 3.2](https://arxiv.org/abs/2509.20529).
- Split along time 60/20/20; "we impose a time limit of 12 hours per experiment, covering both training and hyperparameter tuning"; the model is chosen by s = 1/(1 + NMSE) + λ·exp(−l(f)/L) with λ = 1, L = 200 — [MDBench Sec. 3.2, Eq. 2](https://arxiv.org/abs/2509.20529).
- There is no symbolic-equivalence metric for ODEs; for PDEs a check mark means "all coefficients being in ±5% of ground truth" — [MDBench Table 3](https://arxiv.org/abs/2509.20529).

#### FastSRB, ERBench, SymbolicArena, LLM-SRBench
- FastSRB: SRSD equations, "we sample 200 points for the training and the test dataset, respectively"; "Each expression is searched five times allowing a maximum search time of 30 min for each run"; no multiprocessing; success = the simplified, rounded candidate is in a curated list of acceptable forms whose size is at most 20 % above the reference; a callback every 15 s stops the search on success — [FastSRB Sec. 3 and 4.1](https://arxiv.org/abs/2508.14481).
- ERBench: recovery is equivalence up to shift or scale; in practice a numeric test on "k = 1 000 randomly sampled validation points ... chosen independently of the sample point domains to lie in the interval [-100,100]^d", because it gave the same values as the SymPy test and is much faster; partial credit by Jaccard index over subexpressions and tree edit distance; there is a secret evaluation set — [ERBench Sec. 4.1, 4.3](https://arxiv.org/abs/2606.09276).
- SymbolicArena: 664 tasks from LLM-SRBench (240), SRSD (232), SRBench 1.0 (133), Keijzer (15), the SRBench 2025 first-principles subset (12), Nguyen (12), Korns (12), Vladislavleva (8), distilled to 50; "Runs have a three hour budget", seeds 520 to 522; axes ID, OOD (both a clipped log10 NMSE mapped to [0,1]), SYM, MIN, EFF, STAB; "Opus 5 postprocessing simplifies final expressions and provides symbolic judgments for SYM" — [SymbolicArena Sec. 3.3, 4.1, Table 4](https://arxiv.org/abs/2609.35113).
- LLM-SRBench: 239 problems; symbolic accuracy uses "GPT-4o as an evaluator", with 94.6 % agreement with two human raters on 130 problems; numeric metrics on in-domain and out-of-domain test sets; 1,000 LLM calls per problem — [LLM-SRBench Sec. 2.3, 3.1](https://arxiv.org/abs/2504.10415).

### Inferences
- A protocol at 100 noisy points has no established large benchmark with an automatic symbolic score. The nearest published settings are: FastSRB (SRSD equations at 200 clean points, 30 min), EmpiricalBench and the SRBench 2025 first-principles track (6 to 236 real or noisy points, judged by eye or by a plot), the DSR Nguyen runs (20 clean points), and SRBench or SRSD with the standard γ noise added (but 8,000 to 10,000 points).
- "Success" has at least five operational meanings in use (SymPy difference or ratio is a constant; any member of the Pareto front matches by eye; match to a curated list; numeric constancy of difference or ratio far outside the training box; an LLM's judgment). Results across papers are comparable only when this is held fixed.
- Whether the Pareto front or a single returned expression is scored matters: EmpiricalBench and the DSR noise experiment credit a hit anywhere on the front, SRBench and SRSD score one expression.

### Gaps
- The number of input variables per SRSD problem and the per-task sample sizes of the 2022 competition were not extracted (they are in tables or supplementary material not read).
- The exact test for "nearly correct" in EmpiricalBench is a human judgment and is not specified further.

## 2. Which methods are at the top of each benchmark now, with what solution rates, and how do the rates fall with noise?

### Takeaway
No single method leads everywhere. On the SRBench Feynman and Strogatz problems the best published symbolic solution rates are roughly 50 to 60 % without noise (AI Feynman 53 %, PhySO 58.5 % with units; by their own figures uDSR is above PhySO and UDFS+Aug above AI Feynman), and they collapse with noise for most methods (AI Feynman 52.6 % → 0.9 % at 10 % noise) except those with strong structural priors (PhySO > 53 %, DSR ≈ 17 %). On SRSD, PySR and uDSR led the original study (60 % / 30 % / 4 % for PySR), and later method papers report higher numbers under their own protocols.

### Cited Findings

#### SRBench 2021, 14 methods, symbolic solution rate (%) by target noise
Computed for these notes from `results/ground-truth_results.feather` of the SRBench repository (69,088 runs): mean over trials within a problem, then mean over the 130 problems. The paper shows these only as Fig. 3; its text gives AI Feynman 53 % and GP-GOMEA 27 % at zero noise, which this computation reproduces — [SRBench results](https://github.com/cavalab/srbench/tree/master/results), [SRBench Sec. 5](https://arxiv.org/abs/2107.14351).

| Method | γ = 0 | 0.001 | 0.01 | 0.1 |
|---|---|---|---|---|
| AIFeynman | 52.6 | 31.9 | 12.6 | 0.9 |
| GP-GOMEA | 27.1 | 10.6 | 4.7 | 1.5 |
| AFP_FE | 26.2 | 21.2 | 20.0 | 12.3 |
| ITEA | 20.8 | 13.8 | 7.7 | 1.5 |
| AFP | 20.5 | 19.0 | 16.3 | 12.2 |
| DSR | 19.7 | 19.2 | 18.9 | 16.6 |
| Operon | 16.0 | 12.3 | 1.9 | 0.1 |
| gplearn | 15.5 | 16.2 | 16.0 | 15.8 |
| SBP-GP | 12.6 | 0.7 | 0.0 | 0.0 |
| EPLEX | 12.0 | 9.5 | 8.2 | 9.3 |
| BSR | 2.3 | 0.6 | 0.1 | 0.0 |
| FEAT | 0.1 | 0.0 | 0.0 | 0.0 |
| FFX | 0.0 | 0.0 | 0.0 | 0.1 |
| MRGP | 0.0 | 0.0 | 0.0 | 0.0 |

Feynman only (116 problems): AIFeynman 55.8 / 33.1 / 13.0 / 0.7; AFP_FE 27.0 / 21.9 / 20.8 / 13.5; GP-GOMEA 26.8 / 11.0 / 5.1 / 1.6; DSR 19.7 / 19.1 / 19.0 / 16.8; Operon 16.6 / 13.2 / 2.1 / 0.1; gplearn 16.3 / 17.0 / 16.9 / 16.6. Strogatz only (14): GP-GOMEA 29.5 / 7.1 / 1.4 / 0.0; AIFeynman 27.1 / 22.1 / 9.3 / 2.1; DSR 19.6 / 19.3 / 18.6 / 15.0 (same computation).

Same data, accuracy solution rate (share of problems with test R² > 0.999, %):

| Method | γ = 0 | 0.001 | 0.01 | 0.1 |
|---|---|---|---|---|
| MRGP | 88.2 | 87.1 | 83.4 | 3.1 |
| Operon | 85.3 | 85.8 | 84.9 | 67.9 |
| SBP-GP | 74.9 | 73.7 | 73.8 | 50.4 |
| AIFeynman | 73.1 | 72.6 | 66.1 | 10.6 |
| GP-GOMEA | 71.1 | 72.3 | 72.2 | 66.9 |
| AFP_FE | 55.0 | 52.0 | 50.8 | 50.2 |
| DSR | 21.6 | 25.1 | 25.2 | 25.0 |

- The paper's own reading: "when the target is free of noise, we observe that AIFeynman identifies exact solutions 53% of the time, nearly twice as often as the next closest method (GP-GOMEA, 27%). However, at noise levels above 0.01, four other methods recover exact solutions more often: DSR, gplearn, AFP_FE, and AFP." "the best method on real-world data, Operon, struggles to recover solutions to these problems, despite finding many candidate solutions with near prefect test set scores." — [SRBench Sec. 5](https://arxiv.org/abs/2107.14351).
- Slightly different values for the same data are quoted by SRSD (FSRD: gplearn 15.7 %, AFP 20.41 %, AFP-FE 26.08 %, AIF 53.0 %, DSR 19.1 %) — [SRSD Table 23](https://arxiv.org/abs/2206.10540).
- Median training time at zero noise (same file, computed for these notes): DSR 635 s, gplearn 1,456 s, GP-GOMEA 1,886 s, Operon 2,015 s, AFP 2,956 s; AFP_FE, SBP-GP, BSR and AIFeynman sit at the 8 h limit (28,800 to 29,200 s) — [SRBench results](https://github.com/cavalab/srbench/tree/master/results).
- In SRBench 2021, DSR was run without a constant token; PhySO states "DSR [81] was not permitted to use any free parameters when generating expressions, greatly hindering its capabilities" — [PhySO Sec. IV.1](https://arxiv.org/abs/2303.03192).

#### Methods published after SRBench, on the same Feynman and Strogatz problems (self-reported)
- uDSR: "uDSR outperforms all other 14 benchmarked methods both in symbolic solution (by a large margin) and accuracy solution rates", "for both noiseless settings and for multiple noise levels", with a budget of "up to 2,000,000 expression evaluations per sub-problem, or until a 24 hr walltime limit" (twice the evaluations and three times the wall time of SRBench, as ParFam also points out). The rates are in Fig. 2 only — [uDSR Sec. 5](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf), [ParFam App. G](https://arxiv.org/abs/2310.05537).
- PhySO (Feynman only, 5 seeds, 1M expressions, units given): "With DSR's score at roughly 42%, our method's 58.5% score highlights the significant benefits of incorporating dimensional analysis." "at a 10% noise level, where most methods' recovery rates dip below 20%, and even high performers like uDSR and AI Feynman 2.0 score only 10.7% and 0.7% respectively, PhySO continues to accurately recover expressions over 53% of the time." "In noiseless scenarios PhySO is only surpassed by uDSR". "with a minor noise level of 0.1%, many high-performing methods see their recovery rates almost halved." The 42 % for DSR is the value with constants allowed from the uDSR ablation at 0.5M evaluations — [PhySO Sec. IV.1, IV.2](https://arxiv.org/abs/2303.03192).
- UDFS+Aug (unbiased DAG search with variable augmentation): "at every noise level except the largest, UDFS + Aug recovers the most expressions. At the highest noise level, UDFS + Aug is still the third best regressor in terms of recovery." The comparison set is described as the 14 SRBench 2021 methods ("Results from the SRBench paper for the other symbolic regressors were used as provided by the authors"), which do not include PySR or uDSR. Rates are in Fig. 5 and Fig. 12 only. Without augmentation, average recovery on Feynman rises with search size from 0.11 (1 intermediary node) to 0.43 (4 nodes, 80,000 skeletons) — [UDFS Sec. 6.1, App. C.2, C.3, Table 1](https://arxiv.org/abs/2506.19626).
- ParFam (8 CPU h, 1M evaluations, own reruns of PySR, uDSR, EndToEnd): "PySR, ParFam, AI Feynman, DL-ParFam, and uDSR outperform all other competitors by a substantial margin (over 20%) when it comes to symbolic solution rate. Among those 5 algorithms, PySR performs the best, followed by ParFam. These two algorithms are also the most robust to noise". For accuracy solution: "ParFam outperforms all competitors with and without noise, followed by PySR, MRGP, Operon, uDSR, and DL-ParFam." Rates are in Fig. 4 only — [ParFam Sec. 3](https://arxiv.org/abs/2310.05537).
- LEE paper's table of mean test R² and SymPy complexity (baseline numbers taken from the GenSR paper and SRBench data), Feynman at γ = 0 / 0.01 / 0.1: Operon .989 (70) / .988 (88) / .985 (89); GP-GOMEA .996 (35) / .997 (45) / .996 (46); DSR .844 (15) / .878 (16) / .878 (16); AIFeynman2 .931 (124) / .873 (155) / .225 (177); TPSR .992 (57) / .991 (64) / .984 (67); GenSR .987 (23) / .987 (23) / .989 (24). This is fit quality, not recovery — [LEE Table 1](https://arxiv.org/abs/2605.27245).

#### SRSD-Feynman (original study, one model per method chosen on validation data)

| Metric | Group | gplearn | AFP | AFP-FE | AIF | DSR | E2E | uDSR | PySR |
|---|---|---|---|---|---|---|---|---|---|
| R² > 0.999 | Easy | 6.67 | 20.0 | 26.7 | 33.3 | 63.3 | 26.7 | 100.0 | 66.7 |
| | Medium | 7.50 | 2.50 | 2.50 | 5.00 | 45.0 | 17.5 | 75.0 | 45.0 |
| | Hard | 2.00 | 4.00 | 4.00 | 6.00 | 28.0 | 14.0 | 20.0 | 38.0 |
| Solution rate | Easy | 6.67 | 20.0 | 23.3 | 30.0 | 46.7 | 0.00 | 50.0 | 60.0 |
| | Medium | 0.00 | 2.50 | 2.50 | 2.50 | 10.0 | 0.00 | 17.5 | 30.0 |
| | Hard | 0.00 | 0.00 | 0.00 | 2.00 | 2.00 | 0.00 | 4.00 | 4.00 |
| NED | Easy | 0.866 | 0.727 | 0.693 | 0.646 | 0.524 | 1.00 | 0.478 | 0.269 |
| | Medium | 0.917 | 0.873 | 0.897 | 0.936 | 0.793 | 1.00 | 0.781 | 0.537 |
| | Hard | 0.968 | 0.946 | 0.954 | 0.927 | 0.839 | 0.987 | 0.949 | 0.785 |

Values in %, NED as a fraction — [SRSD Table 4](https://arxiv.org/abs/2206.10540).

- With dummy variables, solution rates (Easy / Medium / Hard): PySR 20.0 / 5.00 / 0.00; uDSR 10.0 / 7.50 / 0.00; DSR 10.0 / 0.00 / 2.00; AFP and AFP-FE 16.7 / 0 / 0; AIF 0 / 0 / 0. "None of the baseline methods is robust against dummy variables". About 45.1 % of DSR's equations with R² > 0.999 use at least one dummy variable and all of E2E's do; none of PySR's predictions that use a dummy variable reaches R² > 0.999 — [SRSD Table 5, Table 6, Sec. 5.3](https://arxiv.org/abs/2206.10540).
- Over all 120 problems the solution rates are gplearn 1.67 %, AFP 5.83 %, AFP-FE 6.67 %, AIF 9.17 %, DSR 15.8 % — [SRSD Table 23](https://arxiv.org/abs/2206.10540). PySR's 60 / 30 / 4 corresponds to 26.7 % overall, the figure FastSRB quotes — [FastSRB abstract](https://arxiv.org/abs/2508.14481).
- AI Feynman falls from 53.0 % on the original Feynman data to 9.17 % on SRSD, DSR from 19.1 % to 15.8 % (not significant, p = 0.479) — [SRSD Table 23, App. F](https://arxiv.org/abs/2206.10540).
- Noise on the Easy set, NED at γ = 0 / 1e-3 / 1e-2 / 1e-1: gplearn 0.876 / 0.928 / 0.940 / 0.948; AFP 0.703 / 0.799 / 0.824 / 0.823; AFP-FE 0.712 / 0.814 / 0.880 / 0.960; AIF 0.646 / 0.797 / 0.870 / 0.882; DSR 0.551 / 0.820 / 0.793 / 0.841. PySR and uDSR were not run with noise — [SRSD Table 22](https://arxiv.org/abs/2206.10540).
- FastSRB (200 points, 30 min, curated equivalence lists), rediscovery Easy / Medium / Hard: PySR as reported by SRSD 60.0 / 30.0 / 4.0; SymbolicRegression.jl under FastSRB 86.0 / 46.0 / 17.4 (44.7 % overall); TiSR 100.0 / 78.0 / 42.6 (69.4 % overall) — [FastSRB Table 1](https://arxiv.org/abs/2508.14481).
- MOSAIC-SR (September 2026 preprint; 3,600 s per problem, five seeds, authors' own runs of the newer baselines), symbolic solution rate Easy / Medium / Hard: MOSAIC-SR 84.7 / 55.0 / 34.8; SR4MDL 63.3 / 43.0 / 21.2; TPSR 47.3 / 44.0 / 20.0; GP-GOMEA 24.7 / 19.5 / 10.0; Operon 2.0 / 7.0 / 6.0; BSR 0 / 0 / 0. With dummy variables: MOSAIC-SR 74.0 / 54.5 / 31.2; SR4MDL 40.0 / 14.0 / 4.8; TPSR 32.0 / 36.0 / 17.2. PySR and uDSR rows are copied from the SRSD paper, not rerun — [MOSAIC-SR Tables 1, 2](https://arxiv.org/abs/2609.20997).

#### EmpiricalBench (9 laws, 5 trials, correct = true law on the Pareto front)

| Law | PySR | Operon | DSR | EQL | QLattice | SR-Transformer |
|---|---|---|---|---|---|---|
| Hubble | 5/5 | 0/5 (5 almost) | 1/5 (4 failed) | 0/5 | 0/5 (5 almost) | 0/5 |
| Kepler | 5/5 | 0/5 (5 almost) | 4/5 | 0/5 | 0/5 | 0/5 |
| Newton | 5/5 | 1/5 | 1/5 (4 failed) | 0/5 | 0/5 | 0/5 |
| Planck | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| Leavitt | 5/5 | 0/5 | 5/5 | 0/5 | 0/5 | 0/5 |
| Schechter | 5/5 | 5/5 | 5/5 | 0/5 | 5/5 | 0/5 |
| Bode | 5/5 | 3/5 | 1/5 | 0/5 | 0/5 | 0/5 |
| Ideal gas | 5/5 | 0/5 | 5/5 | 0/5 | 0/5 | 0/5 |
| Rydberg | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

— [PySR Table 3](https://arxiv.org/abs/2305.01582). Totals (my arithmetic): PySR 35/45, DSR 22/45, Operon 9/45, QLattice 5/45, EQL and SR-Transformer 0/45. The author notes: "because we are the authors of PySR, we are therefore more likely to be running in a stable environment".

#### SRBench 2025
- Black-box: "Among the highest AUC scores are GP-based algorithms such as Brush, GP-GOMEA, and GPZGD"; "Most of the top-performing algorithms incorporate constant optimization as a local search step"; "there is no algorithm that performs exceptionally well on all datasets" — [SRBench 2.0 Sec. 5.2](https://arxiv.org/abs/2505.03977).
- Critical-difference analysis in the thesis version: "EQL achieves the best average R² test rank, followed by a large cluster of well-performing algorithms ranging from PS-Tree to GP-GOMEA"; EQL (rank 7.5) and PS-Tree (8.5) are 20th and 24th in model size; "The only algorithm that ranks among the top 10 in both R² and model size is Rils-Rols" (7th and 9th) — [Aldeia thesis Sec. 35.3](https://arxiv.org/abs/2512.01682).
- First-principles track: "for almost every problem, no SR method could discover an expression that strictly dominates the original equation"; "only ITEA successfully retrieved one of the first-principles equations (Leavitt). In other cases, SR methods either produced a more complex yet more accurate expression or a simpler but less precise model." — [SRBench 2.0 Sec. 5.4, Table 3](https://arxiv.org/abs/2505.03977).
- Computed for these notes from `results/first-principles-tuning/results.feather` (30 runs per method and data set, 13 data sets in the file; median over data sets of the per-data-set median): test R² and model size are QLattice 0.987 (19), PySR 0.913 (25), Operon 0.913 (14), Rils-Rols 0.901 (12), Bingo 0.894 (26), GPZGD 0.887 (39), AFP_FE 0.840 (20), ITEA 0.824 (13.5), Brush 0.789 (19.5), TPSR 0.788 (25), E2E 0.783 (24.5), gplearn 0.769 (19.5). On Newton (30 rows, 10 % noise) the median test R² is 0.08 for PySR, 0.19 for Operon, 0.20 for QLattice, 0.67 for GP-GOMEA, with median sizes of 28, 18, 24 and 55 nodes against a true law of roughly 7 to 9 nodes (my count). On Kepler (6 rows) the median sizes run from 1 (GP-GOMEA, a constant) and 3 (Operon, gplearn) to 22 (E2E, ITEA). The test folds here hold a quarter of 6 to 236 rows, so these R² values are very noisy — [SRBench 2025 results](https://github.com/cavalab/srbench/tree/srbench_2025/results).
- uDSR could not be run in part of the 2025 study ("uDSR was not compatible with the library during the experiments") and is absent from the first-principles result file; in the black-box file its median-of-median R² is −0.03 (computed for these notes), which indicates a broken configuration, not the method's ability — [SRBench 2.0 Sec. 5.1](https://arxiv.org/abs/2505.03977), [SRBench 2025 results](https://github.com/cavalab/srbench/tree/srbench_2025/results).

#### Other 2024 to 2026 rankings
- 2022 competition, synthetic tracks: "QLattice ranked first, followed by pysr and uDSR"; the first five are not statistically separable (Nemenyi, α = 0.05). On R² alone Operon was best; on simplicity PySR, QLattice, Bingo. Exact rediscovery "was only achieved for problems in the easier and easy difficulty levels. Further, only pysr, uDSR, operon, and Bingo were capable of finding exact solutions in these levels." — [Competition Sec. IV-B](https://arxiv.org/abs/2304.01117).
- ERBench (secret evaluation set, defaults, five runs), recovery: PySR 0.29 ± 0.04, gplearn 0.05 ± 0.01, DSR, E2E, Operon and linear regression 0.0; Jaccard index PySR 0.49, DSR 0.28, linear baseline 0.25, Operon 0.13. PySR's recovery is "near-perfect" for 1 to 3 operators and "near zero for problems with more than 13 operators" — [ERBench Table 3, Sec. 5.2](https://arxiv.org/abs/2606.09276).
- SymbolicArena Core50, clean, 3 h, default or recommended settings, 15 methods (0 to 100 scale; SYM judged by an LLM); ten of the fifteen rows shown:

| Method | ID | OOD | SYM | MIN |
|---|---|---|---|---|
| iMCTS | 77.91 | 73.35 | 42.15 | 80.37 |
| PySR | 70.34 | 66.28 | 44.60 | 82.62 |
| uDSR | 64.84 | 55.94 | 41.25 | 51.41 |
| FePySR | 59.84 | 55.45 | 48.13 | 89.55 |
| SymbolFit | 59.04 | 48.46 | 29.96 | 38.92 |
| E2ESR | 57.09 | 42.40 | 29.96 | 31.50 |
| DSO | 38.56 | 34.14 | 43.19 | 79.90 |
| PyOperon | 38.89 | 30.64 | 25.06 | 44.88 |
| LLM-SR | 42.59 | 32.78 | 35.89 | 59.06 |
| gplearn | 31.37 | 24.96 | 28.25 | 53.55 |

"iMCTS leads ID and OOD, FePySR leads SYM and MIN ... PySR ranks second on ID, OOD, SYM, and MIN." — [SymbolicArena Table 2, Sec. 4.3](https://arxiv.org/abs/2609.35113).
- Same benchmark under label noise of 1 % and 5 % of std(y), OOD score clean / 1 % / 5 %: iMCTS 73.35 / 49.40 / 40.73; PySR 66.28 / 41.50 / 35.53; uDSR 55.94 / 32.86 / 27.44; FePySR 55.45 / 42.77 / 36.92; DSO 34.14 / 32.95 / 32.84; PyOperon 30.64 / 27.85 / 24.42. SYM for PySR: 44.60 clean, 30.53 at 1 %, 30.66 at 5 %; DSO 43.19 / 43.32 / 42.83 — [SymbolicArena Tables 13 to 15](https://arxiv.org/abs/2609.35113).
- MDBench: "linear methods and genetic programming methods achieve the lowest prediction error for PDEs and ODEs, respectively. Moreover, linear models are in general more robust against noise." "GP algorithms showed a tendency to overfit in noisy settings, producing highly complex and inaccurate expressions." PySR identified the full equation on six of 14 PDE systems, Operon on four, uDSR on none — [MDBench abstract, Sec. 4.1, Table 3](https://arxiv.org/abs/2509.20529).
- LLM-SRBench: "the best-performing system so far achieves only 31.5% symbolic accuracy" — [LLM-SRBench abstract](https://arxiv.org/abs/2504.10415).
- Nguyen (DSR paper, 2M evaluations, 100 seeds), average recovery: DSR 83.6 %, PQT 75.2 %, Eureqa 73.9 %, GP 60.1 %, VPG 46.7 % — [DSR Table 1](https://arxiv.org/abs/1912.04871).

### Inferences
- Three patterns under noise in the SRBench 2021 data. Methods that rely on problem decomposition or on fitting many constants lose almost everything by 1 % noise (AI Feynman 52.6 → 12.6, GP-GOMEA 27.1 → 4.7, Operon 16.0 → 1.9, SBP-GP 12.6 → 0.0). Methods without fitted constants or with a small expression budget are nearly flat (gplearn 15.5 → 15.8, DSR 19.7 → 16.6). Methods in between lose about half (AFP_FE 26.2 → 12.3).
- The accuracy criterion (R² > 0.999 on clean-range test data) and the symbolic criterion rank methods almost in reverse: Operon is second on accuracy (85 %) and seventh on symbolic recovery (16 %); MRGP is first on accuracy and last on recovery. A new method should report both.
- PySR was never part of the SRBench 2021 ground-truth run. Its numbers on those problems come from third parties (ParFam's figure) or from other benchmarks (SRSD, ERBench, SymbolicArena), where it is consistently first or second among search-based methods for symbolic recovery. That is the strongest support for it as the main baseline.
- Operon looks weak for symbolic recovery in every study that scores recovery (SRBench 2021, EmpiricalBench, ERBench, SymbolicArena, MOSAIC-SR's rerun) and strong on fit quality. A comparison on recovery alone would understate it; one on R² alone would overstate it.
- The claims of the method papers (uDSR, PhySO, UDFS, ParFam, MOSAIC-SR) are each made under a protocol chosen by the authors (budget, seeds, whether Pareto fronts were inspected, whether baselines were rerun). They are not mutually comparable, and none is confirmed by an independent benchmark run except through SRSD (uDSR, PySR) and the 2022 competition.

### Gaps
- Exact solution rates of uDSR, UDFS+Aug, ParFam and PySR on the SRBench Feynman and Strogatz problems exist only as figures; only the numbers quoted in running text are given above.
- No independent 2025 or 2026 study reran the Feynman and Strogatz symbolic-solution track with the current roster of methods; SRBench 2025 dropped the track.
- No table was found that gives solution rate as a function of noise for PySR on SRBench or SRSD specifically.

## 3. What is known about few data points (tens to a few hundred), and about noise in that regime?

### Takeaway
Very little is measured. One 2026 table gives recovery against sample size (PySR 96.5 % at 5 points per dimension rising to 100 % at 80, enumeration 100 % throughout, apparently on clean data), DSR has a figure on noise and a tenfold larger sample, and a theory paper shows a sharp noise threshold beyond which the true formula is no longer the most plausible one for any method, shifting with N. On small real data sets all current methods return expressions several times larger than the accepted law.

### Cited Findings
- Recovery against sample size, 40 Feynman equations, five training sets per size, averages ± s.d.:

| Samples per dimension | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|
| UDFS | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 |
| PySR | 0.965 ± 0.05 | 0.975 ± 0.03 | 0.98 ± 0.02 | 0.99 ± 0.02 | 1.0 ± 0.0 |

"In contrast to enumeration-based search algorithms, the performance of genetic programming algorithms seems to depend more on the size of the training sample" — [ERBench Table 1, Sec. 3.3](https://arxiv.org/abs/2606.09276). The 40 equations are evidently ones both methods can solve; noise is not mentioned for this table.
- ERBench diagnostic plots for PySR on its Feynman subset: "as noise increases, the recovery rate steadily decreases. In contrast, increasing the sample size leads to a steady improvement in recovery"; the authors conclude "problem complexity, rather than data scarcity or moderate noise, is the principal bottleneck" (values in Fig. 6 only) — [ERBench Sec. 5.2](https://arxiv.org/abs/2606.09276).
- Sampling-based searches depend on where the data lie: recovery of f(x) = exp(x) − x³ from 100 equally spaced points varies with a shift ε of the interval [−5+ε, 5+ε] for two distribution-based algorithms, averaged over 100 runs — [ERBench Fig. 3 (right)](https://arxiv.org/abs/2606.09276).
- DSR: noise with standard deviation proportional to the RMS of y from 0 to 0.1 on the Nguyen problems (20 points); "Increasing the dataset size may help prevent overfitting by smoothing the reward function. Thus, we repeated the noise experiments with 10-fold larger training datasets ... For each dataset size, DSR consistently outperforms all baselines." Values in Fig. 4 only — [DSR Sec. 4](https://arxiv.org/abs/1912.04871).
- Learnability transition: with N ∈ {25, 50, 100, 200, 400} points, inputs uniform in [−2, 2], and noise s_ε, "this model-learning problem displays a transition from a low-noise phase in which the true model can be learned, to a phase in which the observation noise is too high for the true model to be learned by any method." In the example of Fig. 2, at s_ε ≳ 0.6 some sampled models become more plausible (shorter description length) than the true one, and for s_ε > 2 "the true model is unlearnable for virtually all training sets"; "the transition shifts towards higher values of s_ε with increasing size of D". Prediction error of the minimum-description-length model is optimal on both sides and worst at the transition — [Fundamental limits, abstract and "Phenomenology of the learnability transition"](https://arxiv.org/abs/2204.02704).
- Same paper: "the minimum description length (MDL) model is the most compressive and the most plausible one, and any approach selecting, from D, a model that is not the MDL model violates those basic postulates." — [Fundamental limits, Results](https://arxiv.org/abs/2204.02704).
- Small real data sets in SRBench 2025 (6 to 236 rows, 75 % for training): "We notice that some first-principle equations fails to achieve perfect R² due to noise in the original data — notably in the Hubble and Tully-Fisher datasets. In such cases, obtaining a higher R² with a more complex model often indicates that the model is fitting the noise rather than the underlying relationship." A suggested direction: "the incorporation of the uncertainties information of the data" — [SRBench 2.0 Sec. 5.4](https://arxiv.org/abs/2505.03977).
- The thesis version: "SR methods were shown to improve model accuracy but often at the cost of overfitting noise, producing small accurate models that fail to capture the underlying ground-truth equation" — [Aldeia thesis Sec. 38.4](https://arxiv.org/abs/2512.01682).
- EmpiricalBench (8 to 100 points, real noise, 1 h on 8 cores): PySR recovers 7 of 9 laws in 5 of 5 trials; Planck and Rydberg are found by no method; Hubble's linear law is missed by four of six methods — [PySR Table 3, Sec. 3.3](https://arxiv.org/abs/2305.01582).
- Bayesian SR by sequential Monte Carlo, 12 hard Feynman problems (3 to 8 inputs), 500 training points, noise 10 % of median |y|, 20 repeats, population 2,000: "Overfitting is seen in almost all datasets for the GPSR approaches. SMC-SR, however, only overfits on 5 of the 12 datasets". Choosing the mode of the final population beats choosing on a validation set of 20 % extra data, which beats best training loss. A 25-point demonstration is included — [SMC-SR Sec. 1.2, 2, Fig. 4 to 6](https://arxiv.org/abs/2512.10849). The baselines are the authors' own GP variants, not PySR or Operon.
- Noise costs in other studies: 2022 competition, "the noise data still had an impact on the accuracy reducing the R² by up to 20%" — [Competition Sec. V-D](https://arxiv.org/abs/2304.01117); SymbolicArena, PySR's OOD score falls from 66.28 to 41.50 at 1 % label noise — [SymbolicArena Table 13](https://arxiv.org/abs/2609.35113).
- Prediction (not recovery) on small tabular data: with 250 training rows from each of 48 PMLB sets (240 cases), "In 132 out of 240 cases, symbolic regression achieves a higher R² than any of the other models on the out-of-sample data"; random forest is best in 37 — [Wilstrup and Kasak, abstract](https://arxiv.org/abs/2103.15147).
- SISSO claims stability for small samples on one materials data set: 82 materials, 23 primary features, candidate space of about 10^11 features; "The LOOCV descriptor agrees with the one found over all data 79, 73, 58 times out of 82 iterations"; it also quotes a compressed-sensing condition for identifying an n-dimensional descriptor, #P ≥ k·n·log(#Φ) with k around 1 to 10 — [SISSO "Solution algorithm", "Benchmark"](https://arxiv.org/abs/1710.03319).
- Pre-trained methods fix the sample size in their design: E2E, TPSR and MOSAIC-SR take 200 input points; ParFam caps data at 1,000 — [SymbolicArena Table 17](https://arxiv.org/abs/2609.35113), [MOSAIC-SR Sec. 4.2](https://arxiv.org/abs/2609.20997), [ParFam Table 7](https://arxiv.org/abs/2310.05537).
- PySR offers, but does not enable by default, a Gaussian-process denoising step (kernel RBF + white noise + constant; the GP's predictions replace the targets) and per-point weights 1/σ² — [PySR Sec. 2.1 "Additional Features"](https://arxiv.org/abs/2305.01582); `denoise` default False — [PySR source](https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py).

### Inferences
- A solution rate curve against N with noise, for several methods, at N between 20 and 500, appears to be unpublished. A prototype aimed at about 100 noisy points could supply that curve itself, and it would be new measured content, not only a comparison.
- The learnability result gives a principled reason to report, per problem and noise level, whether the truth is recoverable at all: when the true expression is not the best-scoring model under the method's own criterion, failure to return it is not a search failure. For a method with an exact evidence this check is cheap (score the true formula and compare).
- On small noisy data the common failure is over-large expressions (median 18 to 55 nodes on a 30-row Newton set), so model selection, more than search, separates methods there. Published selection rules that helped under noise are description length or evidence (fundamental-limits paper, Operon's current default, QLattice's BIC) and the posterior mode (SMC-SR).
- The ERBench table suggests that at about 5 points per dimension and no noise, a competent search loses only a few percent of otherwise solvable problems. The difficulty of the 100-point regime therefore probably comes from noise and from formulas that cannot be told apart in the sampled range, not from the number of points alone.

### Gaps
- Values in DSR Fig. 4 and ERBench Fig. 6 (recovery against noise and N) were not readable from text.
- A search-result snippet (not opened) describes a 2026 bioRxiv study of SR on population-dynamics time series, reporting failure below 10 points per cycle; and DGSR (arXiv 2401.00282) and SEGVAE (arXiv 2301.06064) reportedly contain sub-sampling or dataset-size plots. None was read.
- Exhaustive SR with a description-length criterion (Bartlett et al.) is cited by UDFS and ERBench as the comparable Bayesian or MDL approach for one input; it was not read here.

## 4. What budgets and settings would a referee expect for PySR, Operon, AI Feynman and DSR/uDSR?

### Takeaway
Published comparisons give each baseline between 30 minutes and 24 hours per run (most often 1 hour, or 1M evaluations / 8 hours on one core), 5 to 30 seeds, and an operator set matched to the problems; PySR is almost never run with its factory defaults, whose operator set is only + − × ÷. Several recent papers do run "defaults", and the SRBench maintainers now ask that the default configuration at least be included alongside a small tuned grid.

### Cited Findings

#### What recent studies gave the baselines

| Study | Budget per run | Seeds | Tuning | Cores |
|---|---|---|---|---|
| SRBench 2021 ground truth | 1M evaluations or 8 h | 10 | best of 6 settings from the black-box track | 1 |
| uDSR paper | 2M evaluations per sub-problem or 24 h | 10 | reused published settings | n/s |
| PhySO | 1M expressions | 5 | fixed | n/s |
| PySR EmpiricalBench | 1 h | 5 | none stated | 8 |
| SRSD | 24 h job, up to 100 sessions | best of sessions | Optuna for gplearn/AFP; fixed sets for others | 5 to 20 |
| 2022 competition | 1 h (≤ 1,000 samples), 10 h (≤ 10,000) | 10 | inside the budget | n/s |
| ParFam | 8 CPU h and 1M evaluations | n/s | grid on first 20 Feynman problems | n/s |
| SRBench 2025 | 6 h tuning + 1 h | 30 | 3-fold CV grid incl. default | 1 |
| FastSRB | 30 min | 5 | none | 1 |
| MDBench | 12 h incl. tuning | 1 | grid; authors' defaults for GP methods | 24-core host |
| ERBench | not fixed | 5 | "default hyperparameter configurations" | n/s |
| MOSAIC-SR | 3,600 s (SRSD); "default budgets" (others) | 5 | "settings recommended by their original implementations" | n/s |
| SymbolicArena | 3 h (1 h for the fidelity study) | 3 | "official default or recommended configurations" | 1 to 4 |

Sources: [SRBench Table 2](https://arxiv.org/abs/2107.14351); [uDSR Sec. 5](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf); [PhySO Sec. IV.1](https://arxiv.org/abs/2303.03192); [PySR Sec. 3.2.1](https://arxiv.org/abs/2305.01582); [SRSD Sec. 5.2, App. C](https://arxiv.org/abs/2206.10540); [Competition Sec. III](https://arxiv.org/abs/2304.01117); [ParFam Sec. 3, App. G](https://arxiv.org/abs/2310.05537); [SRBench 2.0 Sec. 3.2 to 3.4](https://arxiv.org/abs/2505.03977); [FastSRB Sec. 4.1](https://arxiv.org/abs/2508.14481); [MDBench Sec. 3.2](https://arxiv.org/abs/2509.20529); [ERBench Sec. 5.1](https://arxiv.org/abs/2606.09276); [MOSAIC-SR Sec. 4.2, 4.3](https://arxiv.org/abs/2609.20997); [SymbolicArena Sec. 4.1, 5, Table 17](https://arxiv.org/abs/2609.35113).

#### PySR
- Current factory defaults (PySR 2.7.3, source read 28 Nov 2026): `binary_operators` "Default is ["+", "-", "*", "/"]", `unary_operators` "Default is None"; `niterations=100`, `populations=31`, `population_size=27`, `ncycles_per_iteration=380`, `maxsize=30`, `maxdepth=None`, `parsimony=0.0`, `adaptive_parsimony_scaling=1040.0`, `timeout_in_seconds=None`, `tournament_selection_n=15`, `tournament_selection_p=0.982`, `crossover_probability=0.2`, `optimizer_algorithm="BFGS"`, `optimizer_nrestarts=2`, `optimizer_iterations=8`, `optimize_probability=0.14`, `turbo=False`, `denoise=False`, `select_k_features=None`, `model_selection="best"` — [PySR source, `PySRRegressor.__init__` and docstring](https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py).
- The returned expression under `model_selection="best"`: "'score' selects the candidate model with the highest score. Score is defined as the negated derivative of the log-loss with respect to complexity ... 'best' selects the candidate model with the highest score among expressions with a loss better than at least 1.5x the most accurate model." — [PySR source docstring](https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py).
- The paper's pseudocode lists different values (40 populations of 1,000 expressions, 300,000 mutations per evolve call, crossover probability 0.01, tournament size 12 with p = 0.9, annealing scale α = 0.1) — [PySR Algorithms 1 to 4](https://arxiv.org/abs/2305.01582).
- Settings others used: SRSD — procs 5, populations 10, population_size 40, ncyclesperiteration 500, niterations 50,000, timeout 82,800 s, maxsize 50, binary {*, +, −, /}, unary {sin, cos, exp, log}, nested constraints forbidding sin/cos inside sin/cos and exp in exp, log in log, turbo on — [SRSD Table 20](https://arxiv.org/abs/2206.10540). ParFam — "6 populations of size 50 and 500 cycles per iteration ... maximum depth of 10 and a maximum size of 50. These are the default parameters shown in the tutorial, which also performed the best on a grid hyperparameter search on the first 20 problems of the Feynman dataset" (PySR 0.18.1) — [ParFam App. G](https://arxiv.org/abs/2310.05537). MDBench — 100 iterations of 1,000 cycles, 20 populations of 100, max size 40, max depth 20, unary {sin, exp, log, sqrt} — [MDBench App. C](https://arxiv.org/abs/2509.20529). SymbolicArena — 8 populations of 64, 500 cycles per iteration, unary {square, cube, exp, log, sin, cos}, serial, max size 30, max depth 10, parsimony 0.001, "best" selection, 3 h — [SymbolicArena Table 17](https://arxiv.org/abs/2609.35113). FastSRB — SymbolicRegression.jl with {+, −, *, /, pow, neg, exp, log, sqrt, pow2, pow3, sin, cos, tanh}, nesting of exp/log and of sin/cos/tanh forbidden, maximum complexity 50 % above the reference, 30 min — [FastSRB Sec. 4.1](https://arxiv.org/abs/2508.14481).
- Runtime PySR took under "default budgets" in the MOSAIC-SR cross-benchmark table: 99.6 s per equation on Nguyen, 150.6 s on Strogatz, 619.6 s on Korns — [MOSAIC-SR Table 3](https://arxiv.org/abs/2609.20997).

#### Operon
- SRBench 2021 grid: population and pool 500 or 100, max_length 50 or 25, symbols {add, mul, aq, constant, variable} with or without {exp, log, sin, tanh}, local_iterations 5, tournament 5 or 3, reinserter keep-best, max_evaluations 500,000 (doubled for the ground-truth track) — [SRBench Table 6](https://arxiv.org/abs/2107.14351).
- pyoperon `SymbolicRegressor` defaults (source read 28 Nov 2026): population_size 1000, generations 1000, max_evaluations 1,000,000, max_length 50, max_depth 10, tournament_size 5, optimizer 'lm', local_search_probability 1.0, `model_selection_criterion='minimum_description_length'` (alternatives: mean squared error, BIC, AIC), linear scale and intercept terms on, n_threads 1 — [pyoperon source](https://github.com/heal-research/pyoperon/blob/main/pyoperon/sklearn.py).
- In the 2022 competition Operon "is paired with Optuna for tuning search hyper-parameters" — [Competition Table II](https://arxiv.org/abs/2304.01117); SRBench 2025 notes Operon "includes an inner hyperparameter tuning step" — [SRBench 2.0 Sec. 5.1](https://arxiv.org/abs/2505.03977).
- Others: MDBench — {add, sub, mul, aq, sin, constant, variable}, brood size 10, max depth 10, max length 50, pool and population 1000 — [MDBench App. C](https://arxiv.org/abs/2509.20529). SymbolicArena — population 500, pool 500, {add, sub, mul, div, aq, exp, log, sin, cos, tanh, sqrt, square}, 4 threads, max length 50, max depth 10, LM optimiser — [SymbolicArena Table 17](https://arxiv.org/abs/2609.35113).

#### AI Feynman
- SRBench 2021 grid: BF_try_time 60 with NN_epochs 4000, or BF_try_time 600 with NN_epochs 400, operator files 10ops, 14ops or 19ops — [SRBench Table 5](https://arxiv.org/abs/2107.14351). In that run its median training time is at the 8 h limit (computed for these notes) — [SRBench results](https://github.com/cavalab/srbench/tree/master/results).
- SRSD: ten settings, bftt 60 or 120, epoch 300 or 500, op 7/10/14/19ops, poly_deg 3 or 4 — [SRSD Table 19](https://arxiv.org/abs/2206.10540).
- It is absent from SRBench 2025's roster, from ERBench's and from SymbolicArena's evaluated methods — [SRBench 2.0 Table 2](https://arxiv.org/abs/2505.03977), [ERBench Sec. 5.1](https://arxiv.org/abs/2606.09276), [SymbolicArena Table 2](https://arxiv.org/abs/2609.35113).

#### DSR and uDSR
- SRBench 2021 tuned only DSR's batch size over {10, 100, 1000, 10000, 100000} and ran it without constants — [SRBench Table 5](https://arxiv.org/abs/2107.14351), [PhySO Sec. IV.1](https://arxiv.org/abs/2303.03192).
- DSR paper: 2M evaluations; tuned on Nguyen-7 and -10 over batch size {250, 500, 1000}, learning rate {0.0003, 0.0005, 0.001}, entropy weight {0.01, 0.05, 0.1}, risk factor {0.05, 0.10, 0.15}; average single-core runtime for 2M expressions 1,921 ± 343 s — [DSR App. D, Table 11](https://arxiv.org/abs/1912.04871).
- uDSR: tokens +, −, ×, ÷, sin, cos, exp, log, sqrt, 1.0, const and the LINEAR token with polynomial terms up to degree 3 — [uDSR Sec. 5](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf). SRSD ran five seeds each of two settings (batch 1000, lr 0.0005, entropy 0.03; batch 500, lr 0.0025, entropy 0.3) with the poly token — [SRSD Table 20](https://arxiv.org/abs/2206.10540). ParFam reran it at 8 CPU h and 1M evaluations with the repository's `config_regression_gp.json`: "the performance we report is slightly worse" — [ParFam App. G](https://arxiv.org/abs/2310.05537).
- Running time: in MDBench "EQL and uDSR are significantly slower, differing by several orders of magnitude" from SINDy and Operon on ODEs — [MDBench Sec. 4.1](https://arxiv.org/abs/2509.20529).

#### Guidance from the benchmark maintainers
- "We propose standardizing the hyperparameter tuning by requiring each method to define a small set (e.g., four) of hyperparameter configurations. A simple grid search is then applied using these configurations, each limited to a maximum runtime (e.g., one hour). The off-the-shelf configuration should automatically be included by the experiment scripts, as we observed some methods perform best with their default settings." — [SRBench 2.0 Sec. 5.3](https://arxiv.org/abs/2505.03977).
- "During the preparation for running the benchmark, it was observed that, when attempts were made to provide improved configurations, the default parameters often produced better results." "many methods did not utilize the maximum allowed time" — [Aldeia thesis Sec. 35.4](https://arxiv.org/abs/2512.01682).
- Evaluation counts were dropped as a budget unit because they are "hard to measure uniformly across the variety of different approaches with different computational runtime"; the 2022 competition "introduced a runtime limit instead of using the number of evaluations, stimulating the efficient implementation of the algorithms" — [SRBench 2.0 Sec. 2](https://arxiv.org/abs/2505.03977).
- Open issue named by the maintainers: "the fair comparison between GPU-based and CPU-based approaches, as well as single-core versus multi-core implementations. While every algorithm with a max_time hyperparameter respects its limit, some terminate far earlier" — [SRBench 2.0 Sec. 5.5](https://arxiv.org/abs/2505.03977).
- Statistical reporting used: Wilcoxon signed-rank tests with Bonferroni correction (SRBench 2021); Friedman test and critical-difference diagrams (SRBench 2025 thesis version); Nemenyi test (competition) — [SRBench App. A.7](https://arxiv.org/abs/2107.14351), [Aldeia thesis Sec. 35.3](https://arxiv.org/abs/2512.01682), [Competition Sec. IV-B](https://arxiv.org/abs/2304.01117).
- SymbolicArena on its own use of defaults: "Algorithm specific hyperparameter optimization is not included. Absolute scores and relative rankings may change with dedicated tuning." — [SymbolicArena Sec. 5](https://arxiv.org/abs/2609.35113).

### Inferences
- "PySR with default settings and 60 seconds" is weaker than any baseline configuration found in the literature on two counts. The default operator set has no unary functions, so formulas with exp, log, sin or sqrt cannot be represented unless operators were set by hand. And 60 s is 30 to 500 times shorter than the budgets used elsewhere (30 min to 8 h), although MOSAIC-SR's table shows default-budget PySR finishing in 100 to 620 s per problem.
- A comparison likely to be accepted, pieced together from the practices above: (i) the same operator set for every method, stated; (ii) PySR and Operon at equal wall time and cores as the new method and, separately, at a long budget of the order of 1 h (the SRBench 2025 and competition figure), so that both "equal cost" and "baseline at its best" are shown; (iii) the default plus a small grid (about four to six settings), chosen without looking at the test criterion; (iv) at least 5 and preferably 10 to 30 seeds per problem with a paired test; (v) for PySR, state `model_selection` and consider also scoring the whole Pareto front, since EmpiricalBench counts a hit anywhere on the front; (vi) report failures and timeouts as failures.
- A 60 s search on 4 cores is about 4 CPU-minutes, roughly 8 times less than FastSRB's 30 single-process minutes, the shortest budget found in a published comparison. Showing how each method's solution rate grows with budget would address the objection directly.
- AI Feynman is no longer in current rosters and is slow (8 h limit); a referee is more likely to ask for PySR, Operon and one of uDSR/DSO, and, for physics formulas with units, PhySO. TiSR, iMCTS, SR4MDL and TPSR appear as strong newer baselines in single studies only.
- Because Operon's Python wrapper now selects by description length and PySR by its "best" score, both return a single model by a parsimony-aware rule; older comparisons that took the most accurate model are not comparable with new ones.

### Gaps
- No source states what referees require; the above is inferred from what published comparisons did and from the maintainers' proposals.
- Operon's own paper could not be opened, so its recommended settings and its stated evaluation budget are not reported from the primary source.
- uDSR's hyperparameter appendix (Table 3 of its supplement) could not be fetched.

## 5. Which techniques make the strongest search-based methods strong?

### Takeaway
The benchmark studies single out local optimisation of constants as the common factor of the top methods; the rest of the advantage comes from keeping a complexity-indexed front with a parsimony-aware pick, from structural priors (units, nesting limits, decomposition), and from breadth (many populations, restarts, or outright enumeration of small expressions). Simplification matters mainly as a guard against over-parameterised expressions and as a precondition for equivalence checks.

### Cited Findings

#### Constant optimisation
- "Most of the top-performing algorithms incorporate constant optimization as a local search step —using linear, gradient-based, or non-linear methods–, highlighting the importance of parameter optimization for achieving high performance." — [SRBench 2.0 Sec. 5.2](https://arxiv.org/abs/2505.03977). In the black-box ranking, "Methods with performance superior than Genetic Engine performs parameter optimization at some stage" — [Aldeia thesis Sec. 35.2](https://arxiv.org/abs/2512.01682).
- PySR: an "evolve-simplify-optimize loop"; "The 'Optimize' part of this algorithm is a few iterations of a classical optimization algorithm (BFGS [63] is used by default, though any optimizer in the Optim.jl package can be used [64]) to optimize constants in every equation explicitly"; several mutations are done before each simplify-optimise stage — [PySR Sec. 2.1](https://arxiv.org/abs/2305.01582). Current defaults: BFGS, 8 iterations, 2 restarts "with different initial conditions", applied with probability 0.14 per iteration; Nelder-Mead is the alternative — [PySR source](https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py).
- Operon (secondary sources): "a GP method that incorporates non-linear least squares constant optimization using the Levenberg-Marquadt algorithm" — [SRBench Sec. 4.1](https://arxiv.org/abs/2107.14351); 5 local iterations per individual in the SRBench grid — [SRBench Table 6](https://arxiv.org/abs/2107.14351); it "systematically associates a parameter with every terminal and optimizes them using a non-linear optimization method, although this often leads to over-parameterized expressions" — [Aldeia thesis Sec. 9.1](https://arxiv.org/abs/2512.01682); "Similar ideas of a linear basis of terms is used in the initialization of the 'Operon' genetic algorithm" — [PySR Sec. 1](https://arxiv.org/abs/2305.01582). At 10 % noise Operon's expressions contain on average 17 free constants after simplification, MRGP's 88 — [PhySO Sec. IV.3](https://arxiv.org/abs/2303.03192).
- DSR: constants are placeholders optimised "using a nonlinear optimization algorithm, e.g. BFGS" inside the reward for every sampled expression — [DSR Sec. 3.1](https://arxiv.org/abs/1912.04871). PhySO: "the L-BFGS [116] optimization routine in pytorch (optimizing over 20 steps and using an mean squared error metric)" with automatic differentiation — [PhySO Sec. III.3](https://arxiv.org/abs/2303.03192).
- UDFS: one parameter node per expression (p = 1), fitted by a hierarchical grid search, "grids of size 50 in a two-level hierarchy starting from the interval [-1,1]", chosen because a whole grid is evaluated in one pass; other constants arise from operations on that one — [UDFS Sec. 4, App. C.1](https://arxiv.org/abs/2506.19626).
- ParFam: a global optimiser over a fixed parametric family, "the number of iterations for basin-hopping to 10, and the maximal number of BFGS steps for the local search to 100 times the dimension of the problem", with regularisation λ = 0.001 — [ParFam Sec. 3 "Hyperparameters"](https://arxiv.org/abs/2310.05537).
- Linear and non-linear parameters separately: "OLS should always be used whenever linear decomposition is available, as it provides a closed-form solution. For mixed cases, local non-linear optimization (LM) should be combined with global linear optimization (OLS)"; "LM is sensitive to starting points, making initialization crucial"; "Non-linear optimization alone suffers from overparameterization and high computational cost" — [Aldeia thesis Sec. 38.1](https://arxiv.org/abs/2512.01682).
- uDSR's LINEAR token "identifies sparse coefficients of a linear combination of basis functions"; in the 48-way ablation "For a fixed N, GP and LM tend to yield the largest marginal performance gains" and "Both the LINEAR token and the inclusion of our GP component almost always provide large improvements in both symbolic solution and accuracy solution" — [uDSR Sec. 1, 6](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf).
- A cost of numerical fitting for recovery: in the competition an Operon model matched the truth except for coefficients 0.19 and 0.21 where 0.2 was needed, so that "the internal optimization of the numerical parameters may suffer from imprecision that makes it harder to simplify the expression" — [Competition Sec. IV-B1](https://arxiv.org/abs/2304.01117).
- SRBench 2021 on the state of practice: "the field lacks a comprehensive study that provides guidelines for the appropriate hyperparameters for constant optimization (learning rate schedule, iterations, batch size, etc.)" — [SRBench App. A.3](https://arxiv.org/abs/2107.14351).

#### Simplification and equality saturation
- PySR simplifies "to an equivalent form during the loop using a set of algebraic equivalencies", only occasionally, because "some equations are only accessible via a redundant intermediate state" (the example is x·x − x·x → x·x − x·y) — [PySR Sec. 2.1](https://arxiv.org/abs/2305.01582).
- Equality saturation is cited for reducing over-parameterisation before constant fitting: models "with too many parameters are difficult to interpret and suffer from ill-conditioning, which slows down the parameter optimization" (de França and Kronberger, GECCO 2023, as summarised in the thesis); and it is proposed for assessing model equivalence in future SRBench versions — [Aldeia thesis Sec. 9.3, 38.4](https://arxiv.org/abs/2512.01682).
- "Simplification should always be applied, at least to the final model"; but neutrality (redundant code) "may promote broad exploration of the search space", so early simplification may hinder search — [Aldeia thesis Sec. 9.3, 38.3](https://arxiv.org/abs/2512.01682).
- SymPy as simplifier is "unreliable and can increase the model size" — [SRBench 2.0 Sec. 3.3](https://arxiv.org/abs/2505.03977).

#### Selection from the Pareto front
- PySR keeps "the best expressions at each complexity" (hall of fame) and by default returns the highest-score member whose loss is within 1.5× of the most accurate — [PySR Algorithm 1](https://arxiv.org/abs/2305.01582), [PySR source](https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py).
- pyoperon selects from its Pareto front by minimum description length by default; QLattice was run with "BIC model selection" — [pyoperon source](https://github.com/heal-research/pyoperon/blob/main/pyoperon/sklearn.py), [SymbolicArena Table 17](https://arxiv.org/abs/2609.35113).
- UDFS: "we return the best fitting model with a complexity below a given threshold. If there is no such model, we simply return the smallest model." — [UDFS Sec. 5.2](https://arxiv.org/abs/2506.19626).
- AI Feynman 2.0 "integrates Pareto optimization with an information-theoretic complexity metric to improve robustness to noise" — [SRBench Sec. 4.1](https://arxiv.org/abs/2107.14351).
- SRSD attributes PySR's lead in structural closeness to its complexity handling: "it may be important in SRSD problems to penalize overcomplex solutions in a similar way to PySR" — [SRSD Sec. 5.3](https://arxiv.org/abs/2206.10540).
- Under 10 % noise, choosing the posterior mode beat choosing by a validation set, which beat best training loss; "accurate model selection can significantly improve SR results" — [SMC-SR Sec. 2, Fig. 6](https://arxiv.org/abs/2512.10849).
- "Half of the SR methods we test use Pareto optimization in some form during training." — [SRBench Sec. 4.1](https://arxiv.org/abs/2107.14351).

#### Parsimony pressure
- PySR replaces a constant penalty by an adaptive one: "we adaptively tune a per-complexity penalty such that the number of expressions at each complexity is approximately the same", ℓ(E) = ℓ_pred(E)·exp(frecency[C(E)]); default `parsimony` is 0.0 and complexity is the node count unless configured — [PySR Sec. 2.1, Eq. 2](https://arxiv.org/abs/2305.01582), [PySR source](https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py).
- Hard constraints in PySR: maximum size, depth, per-operator argument size, and nesting limits such as {sin: {sin: 0, cos: 1}} — [PySR Sec. 2.1 "Constraints"](https://arxiv.org/abs/2305.01582). DSR enforces during sampling: length 4 to 30, no all-constant children, no inverse of its parent, no trigonometric function inside another — [DSR Sec. 3.1](https://arxiv.org/abs/1912.04871).
- Noise raises complexity where parsimony is weak: median simplified complexity from γ = 0 to 0.1 goes 35 → 78 for Operon, 20 → 48 for GP-GOMEA, 104 → 207 for SBP-GP, 12 → 13 for DSR (computed for these notes) — [SRBench results](https://github.com/cavalab/srbench/tree/master/results).

#### Dimensional constraints
- PhySO computes the units required of each token while the expression is being generated and removes tokens that would violate them; in a five-symbol example this "reduces the search space from 268 expressions to only 6" — [PhySO Fig. 1, Sec. III.2](https://arxiv.org/abs/2303.03192).
- Effect: 58.5 % against roughly 42 % for the same kind of search without units on Feynman, and > 53 % at 10 % noise; on six astrophysical cases the full method recovers 100 %, 47 % without the units prior but with a units-informed network, 63 % with the prior but an uninformed network, 37 % with neither — [PhySO Sec. IV.2, Table 5](https://arxiv.org/abs/2303.03192).
- PhySO used a batch of 10,000 expressions (DSR: 1,000) and reinforces the best 5 %, attributing the larger batch to "the very strong constraints" — [PhySO Sec. III.3](https://arxiv.org/abs/2303.03192).
- SISSO builds its feature space with "dimensional analysis ... to retain only 'meaningful' combinations (e.g., no unphysical items like 'size + energy')" — [SISSO "Features space construction"](https://arxiv.org/abs/1710.03319).
- uDSR's authors: "For real-world problems with physical units, we highly recommend that users first perform dimensional analysis on their SR problem." — [uDSR Sec. 6](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf).
- PySR has a soft version: "Additive penalty for if dimensional analysis of an expression fails. By default, this is 1000.0" — [PySR source](https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py).

#### Feature selection and many inputs
- PySR: "PySR uses a gradient-boosting tree algorithm to first fit the dataset, then select the most important features", for a user-given number of features; off by default (the current docstring says "using random forests") — [PySR Sec. 2.1](https://arxiv.org/abs/2305.01582), [PySR source](https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py).
- SISSO: sure independence screening ranks up to 10^10 or more candidate features by correlation with the target (then with the residual), keeps a subspace, and an ℓ0 search picks the best n-tuple; unlike orthogonal matching pursuit it "both reselects features and reoptimizes coefficients at each dimensional increment" — [SISSO "Solution algorithm"](https://arxiv.org/abs/1710.03319).
- No baseline in SRSD was robust to 1 to 3 irrelevant inputs (PySR's Easy solution rate 60 % → 20 %) — [SRSD Tables 4, 5](https://arxiv.org/abs/2206.10540). cp3-bench found dimensionality of the feature space and precision of the data "highly important" for success — [cp3-bench abstract](https://arxiv.org/abs/2406.15531).

#### Restarts, islands and diversity
- PySR is "a multi-population evolutionary algorithm with multiple evolutions performed asynchronously", with migration from other populations and from a global hall of fame, replacement of the oldest member (age-regularised evolution) "rather than the individual with the lowest fitness", and simulated annealing of mutation acceptance — [PySR Sec. 2.1, 2.2](https://arxiv.org/abs/2305.01582).
- Benchmarks treat restarts as part of the method: SRBench 2025 reports the maximum over 30 runs as "an optimistic perspective on model performance, assuming a user performs the same number of runs" — [SRBench 2.0 Sec. 4](https://arxiv.org/abs/2505.03977); SRSD picks the best of up to 100 sessions (for DSR and uDSR, ten seeds) on validation data — [SRSD Sec. 4.2, 5.2, Tables 19, 20](https://arxiv.org/abs/2206.10540).
- DSR's risk-seeking gradient trains on the top ε fraction of each batch, optimising "best-case performance, possibly at the expense of average performance" — [DSR Sec. 3.2, 4](https://arxiv.org/abs/1912.04871).

#### Decomposition and enumeration
- AI Feynman "trains a NN on the data and uses it to estimate functional modularities (e.g., symmetry and/or separability), which are used to partition the data into simpler problems and recurse" — [SRBench Sec. 4.1](https://arxiv.org/abs/2107.14351). These strategies "are extremely noise sensitive" — [PhySO Sec. IV.2](https://arxiv.org/abs/2303.03192). As a wrapper in uDSR it gives "strict improvement over the base SR algorithm" at "a roughly m-fold increase in computational cost" — [uDSR Sec. 6](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf).
- UDFS samples DAG skeletons uniformly (five intermediary nodes, 200,000 skeletons) and labels operators exhaustively; common subexpressions are shared in the DAG. Its "variable augmentation" searches small parameter-free expressions z(x) and keeps the one that most improves a polynomial regression on (x, z), then solves the augmented problem. On ten one-variable tasks: UDFS+Aug recovers 1.0 in 26 s, UDFS 0.8 in 389 s, exhaustive SR 0.5 in 1,294 s — [UDFS Sec. 4, 5, 6, Table 4](https://arxiv.org/abs/2506.19626).
- "If the true expression has a small representation, then it will be recovered by an enumeration-based search algorithm, independently of the sampling domain." — [ERBench Sec. 3.2](https://arxiv.org/abs/2606.09276).

### Inferences
- The techniques group by what they protect against: constant fitting protects accuracy; front-plus-parsimony selection and description-length criteria protect against over-large answers under noise; units, nesting limits and decomposition cut the search space; islands, restarts and enumeration protect against search failure.
- Two recurring weaknesses of the strongest evolutionary baselines concern noise and irrelevant inputs: constant-heavy expressions that fit noise (Operon, MRGP, SBP-GP) and no built-in protection against dummy variables. A method whose score is an evidence that penalises unused flexibility is addressing a documented gap; the data to show it are the γ = 0.01 and 0.1 columns and SRSD's dummy-variable sets.
- Exact fitting of linear weights with non-linear parameters handled separately (LM+OLS in the thesis, the LINEAR token in uDSR) is the published analogue of a model that is linear in some weights and marginalises them.
- A greedy beam over rewrite rules is closer to the enumeration and decomposition family (UDFS, AI Feynman) than to the evolutionary one; ERBench's remark that such methods are insensitive to the sampling domain, and PhySO's that decomposition is noise sensitive, are the two claims a referee would probably want tested for it.

### Gaps
- Operon's design (linear tree encoding, offspring generators, concurrency model) is known here only through a search-result summary of its abstract and through its documentation front page; no details or numbers are reported from the paper itself.
- No ablation was found that isolates the number of optimiser restarts or iterations for constants in PySR or Operon.
- Equality saturation was read only as summarised in the thesis; the original papers (de França and Kronberger) were not opened.

## 6. Known criticisms of the benchmarks

### Takeaway
The Feynman data are the main target: constants sampled as variables in arbitrary narrow ranges, duplicate and indistinguishable problems, no irrelevant inputs, uniform sampling and public, memorisable answers. The scoring is the second target: SymPy equivalence misses correct answers (changing only the equivalence check raised PySR's SRSD rate from 26.7 % to 44.7 %), test R² rewards over-fitted expressions, and aggregate ranks hide per-problem behaviour.

### Cited Findings

#### Data generation
- Constants as variables and unrealistic ranges: "the strategies do not distinguish between constants and variables e.g., speed of light is treated as a variable and randomly sampled in range of 1 to 5"; "most of the sampling domains are far from values we could observe in the real world" — [SRSD Sec. 3.1, issue 2](https://arxiv.org/abs/2206.10540).
- Formulas that cannot be told apart in the sampled range: "When sampled ranges of the distributions are narrow, we cannot distinguish Lorentz transformation from Galilean transformation e.g. I.15.10 and I.16.6 ..., or the black body radiation can be misestimated to Stephan-Boltzmann law or the Wien displacement law e.g. I.41.16" — [SRSD Sec. 3.1, issue 2](https://arxiv.org/abs/2206.10540).
- Duplicates: "approximately 25% of the symbolic regression problems in the original FSRD have 1 - 5 duplicates" (same functional form and same sampling ranges, e.g. F = μN and F = qE) — [SRSD Sec. 3.1, issue 3](https://arxiv.org/abs/2206.10540).
- Wrong variable types: integers and angles sampled as positive reals; arcsin and arccos used "just to obtain angle variables" — [SRSD Sec. 3.1, issue 4](https://arxiv.org/abs/2206.10540).
- No dummy variables: "The existing SR datasets consist of samples using only necessary input variables to symbolically express the true models." — [SRSD Sec. 3.1, issue 5](https://arxiv.org/abs/2206.10540).
- Constants given away: "in the Feynman benchmark dataset of [73], all expressions are listed with the associated physical constants—but these constants originally had to be discovered along with the equation"; Nguyen has "no non-integral constants" — [PySR Sec. 3.2](https://arxiv.org/abs/2305.01582).
- The same feature favours decomposition methods: treating constants of nature "as continuously varying input variables ... places a higher emphasis on the implementation of the problem simplification schemes" — [PhySO Sec. IV](https://arxiv.org/abs/2303.03192).
- Noise model: "the noise models are very simple: perfectly Gaussian and including no heteroscedasticity"; SRBench "only contained a single real-world task (with an unknown ground truth)" — [PySR Sec. 3.2](https://arxiv.org/abs/2305.01582).
- Uniform sampling inflates results: for PySR, with uniform sampling "the recovery rate is up to approximately 10% higher than when the data follows the diverse distribution" of Kamienny et al. — [ERBench Sec. 5.2](https://arxiv.org/abs/2606.09276).
- Black-box track bias: "in the original SRBench, half of the black-box problems were variations of the Friedman datasets (Friedman, 2001), which introduced bias in the results" — [SRBench 2.0 Sec. 3.1](https://arxiv.org/abs/2505.03977); SRBench itself noted they "may have an out-sized effect" — [SRBench App. A.6.1](https://arxiv.org/abs/2107.14351).
- Public answers: existing benchmarks have "only a small number of groundtruth formulas that are publicly known" — [ERBench abstract](https://arxiv.org/abs/2606.09276); for LLM-based methods, "The sharp drops in numeric error curves and considerably lower symbolic error for Feynman problems suggest memorization rather than gradual discovery" — [LLM-SRBench Fig. 1](https://arxiv.org/abs/2504.10415).
- Transfer to a domain: "we find no indication that performance of algorithms on standardized datasets are good indicators of performance on particular cosmological and astrophysical datasets" — [cp3-bench abstract](https://arxiv.org/abs/2406.15531).
- Difficulty of the standard problems is limited: PySR is near-perfect up to 3 operators and near zero beyond 13 — [ERBench Sec. 5.2](https://arxiv.org/abs/2606.09276); in the competition exact recovery succeeded only at the two easiest levels — [Competition Sec. IV-B1](https://arxiv.org/abs/2304.01117).

#### Scoring
- SymPy equivalence fails on correct answers, e.g. v1·v2/(v3·(v4² − v5²)) against v1·v2/(v3·(v4 − v5)·(v4 + v5)), or 0.39894·exp(−v1²/2) against 0.39894/√exp(v1²) — [FastSRB Sec. 1](https://arxiv.org/abs/2508.14481). With curated acceptable forms, PySR's SRSD rediscovery "increases from 26.7% to 44.7%" (other conditions also differ: 200 points, 30 min, per-problem size limit) — [FastSRB Sec. 4.2](https://arxiv.org/abs/2508.14481).
- "Requiring exact rediscovery of an expression does neither fully align with SR's optimization objectives, nor does it reflect a real-world use case." — [FastSRB Sec. 1](https://arxiv.org/abs/2508.14481).
- Manual checks are in use because of this: EmpiricalBench was scored by eye; PhySO inspected "the first few expressions (in accuracy) of the Pareto front ... making a marginal difference of approximately 1% in recovery rate" — [PySR Sec. 3.2.1](https://arxiv.org/abs/2305.01582), [PhySO Sec. IV.1](https://arxiv.org/abs/2303.03192).
- All-or-nothing scoring ignores partial recovery: an answer that has the correct dependence on three of four variables counts as a failure; a Jaccard index over subexpressions (0.47 in that example) changes rankings ("DSR now ranks higher than AIFeynman") — [UDFS Sec. 6.1, Fig. 7](https://arxiv.org/abs/2506.19626). SRSD's edit distance has the same motive: "solution rate may be not informative for challenging datasets" — [SRSD Sec. 5.3](https://arxiv.org/abs/2206.10540).
- R²-based accuracy is "vulnerable to dummy variables": about 45 % of DSR's and 100 % of E2E's "correct" equations by R² > 0.999 used a dummy variable — [SRSD Sec. 5.3, Table 6](https://arxiv.org/abs/2206.10540). In a user study with 23 researchers the correlation of human judgment with R² was 0.005 (p = 0.913) and with NED −0.416 (p ≈ 1.9e-24) — [SRSD Table 7](https://arxiv.org/abs/2206.10540).
- High fit without recovery: on SRSD Easy, SBP-GP has 87.3 % accuracy and 8.0 % symbolic rate — [MOSAIC-SR Table 1](https://arxiv.org/abs/2609.20997); on Core50, "Near zero numerical error can coexist with failure of strict symbolic equivalence" — [SymbolicArena Sec. 4.3](https://arxiv.org/abs/2609.35113); AI Feynman, when it fails, returns expressions with "an average of 147 symbols and 18 free constants" — [PhySO Sec. IV.3](https://arxiv.org/abs/2303.03192).
- In-domain test error as a proxy: random train/test splits "can be a misleading proxy for true model discovery and generalization. The obvious alternative is to measure out-of-domain accuracy. However, obtaining challenging out-of-domain test data is a non-trivial problem. Therefore, we focus on equation recovery" — [ERBench abstract](https://arxiv.org/abs/2606.09276).
- Extrapolation as a criterion: "current equation discovery benchmarks largely overlook this aspect ... we advocate for explicit OOD assessment" — [LLM-SRBench Sec. 3.3](https://arxiv.org/abs/2504.10415). In the competition's extrapolation task accuracy was "often with a negative R² for the test set, even for the easiest levels", with the caveat: "given that a certain dataset can have many different solutions inside the interpolation region that behaves differently outside this region, we cannot expect the algorithm to make the right choice without any prior information." — [Competition Sec. IV-B4, V-B](https://arxiv.org/abs/2304.01117).
- Noise makes the accepted law sub-optimal by the score: on Hubble and Tully-Fisher data the first-principles equation does not reach R² = 1, and more complex models score higher by fitting noise — [SRBench 2.0 Sec. 5.4](https://arxiv.org/abs/2505.03977). Above a noise threshold the true model is not the most plausible one for any method — [Fundamental limits, abstract](https://arxiv.org/abs/2204.02704).
- Aggregation: SRBench 1.0 "presents final results using only aggregated metrics, which can hide finer details"; criticisms of "aggregated views of the results" were raised at GECCO — [SRBench 2.0 Sec. 1, 2](https://arxiv.org/abs/2505.03977). "there is hardly a single winner or a single approach that could be named state-of-the-art" — [Competition Sec. V-A](https://arxiv.org/abs/2304.01117).
- Complexity measures differ by method and node counts miss interpretability (f(g(h(x))) is smaller than f(x) + h(x)·g(x)) — [Competition Sec. V-A](https://arxiv.org/abs/2304.01117), [UDFS App. C.2](https://arxiv.org/abs/2506.19626).
- For dynamical systems, "there is no established metric for quantifying equation fidelity"; low derivative NMSE occurred with equations that "do not capture the true dynamics" — [MDBench Sec. 4.3](https://arxiv.org/abs/2509.20529).
- LLM judges: symbolic scores in LLM-SRBench and SymbolicArena come from language models; SymbolicArena reports "inconsistent pairwise symbolic judgments for 3 clean, 2 one percent noise, and 3 five percent noise algorithm task groups" and states "A positive model judgment provides evaluation evidence and does not constitute formal mathematical proof." — [SymbolicArena App. E.3, E.5](https://arxiv.org/abs/2609.35113).

#### Protocol and upkeep
- Budgets in evaluations were "hard to measure uniformly"; implementations differ in cores and GPU use; some stop early — [SRBench 2.0 Sec. 2, 5.5](https://arxiv.org/abs/2505.03977). SRBench 2021 "have not benchmarked the methods in a setting that allows them to exploit parallelism" — [SRBench Sec. 6](https://arxiv.org/abs/2107.14351).
- "During the benchmarking process, a number of bugs had to be fixed for most evaluated algorithms. Few of the corresponding repositories are being maintained" — [SRBench 2.0 Sec. 5.5](https://arxiv.org/abs/2505.03977). MDBench reports failures from "runtime errors within the published codebases" — [MDBench Sec. 4.2](https://arxiv.org/abs/2509.20529).
- Hyperparameters: "The current methods showed to be difficult to tune, and the papers proposing these algorithms often fail to conduct sensitivity analyses" — [Aldeia thesis Sec. 35.4](https://arxiv.org/abs/2512.01682).
- Benchmark over-fitting: optimising methods for benchmark performance "risks making benchmarks less useful over time" (Radwan et al. 2024, as cited) — [Aldeia thesis Sec. 10](https://arxiv.org/abs/2512.01682).
- Author-run comparisons: the PySR author notes his familiarity advantage — [PySR Sec. 3.2.1](https://arxiv.org/abs/2305.01582); ERBench has users run their own methods, so "computation times of algorithms from different users are no longer directly comparable" — [ERBench Sec. 6](https://arxiv.org/abs/2606.09276).

### Inferences
- Judging "right" by error outside the training range has support in the literature (competition extrapolation track, LLM-SRBench and SymbolicArena OOD axes) but is not the primary accepted criterion, and the competition's caveat applies: without prior information several formulas agree in range and differ outside. A referee will probably expect symbolic recovery (SRBench Def. 4.1, ideally with a tolerant equivalence check and manual audit of near-misses) as the headline, with out-of-range error, normalised edit distance or Jaccard index, and expression size as supporting numbers.
- Several criticisms describe exactly the conditions of a small, noisy, physics-style setting: indistinguishable formulas in a narrow range, noise making the truth sub-optimal, dummy variables. A method that can leave part of the answer unspecified has a natural response to the first two, but the existing scores have no category for "correct as far as it goes"; the partial-credit metrics (NED, Jaccard) are the closest existing instruments, and how a remaining unknown function is scored would need to be defined and justified.
- Because SymPy-based checks under-count by a large margin (26.7 % against 44.7 % for the same method on SRSD), any comparison should apply one equivalence procedure to all methods and state it; mixing a lenient check for the new method with published strict numbers for baselines would not be accepted.
- SRSD's sampling (realistic SI ranges, true constants) and dummy-variable sets answer most data criticisms and are the natural source of formulas, subsampled to the small-N regime as FastSRB did with 200 points.
- SRBench 2025 standardises all data before training, and the "closest" equations it prints (Table 3, e.g. "0.090 + D" for Hubble, "−0.94·log P" for Leavitt) are in standardised variables. If the target is standardised too, a pure product or power law acquires offsets and is no longer the simplest exact form, which would by itself make symbolic recovery of the textbook law harder on that track. This is my reading of Sec. 3.1 and Table 3, not a statement in the paper.

### Gaps
- The GECCO critiques cited by SRBench 2.0 (Dick 2022; de França 2023) were not read; they are represented only through that paper's summary.
- No published criticism was found that addresses how to score partially specified answers (expressions containing an unknown function); this may not exist.
- cp3-bench's result tables and its noise analysis were not read beyond the abstract.
