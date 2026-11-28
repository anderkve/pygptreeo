# Closest prior art to symbolic regression with GP holes

Search date: 28 November 2026. "The method" below is the one in the brief: a
sum of terms `a(x)·H(u(x)) + w·b(x) + const`, optionally inside one outer
function, with every hole `H` a GP that is marginalised; a greedy beam search
from plain GP regression by rewrite rules; every candidate scored by its
evidence with a complexity penalty; a hole may remain in the answer.

How to read these notes:

- "Read" after a paper means I had its full text and read the sections cited.
  "Abstract only" or "search summary only" means I did not have the full text.
- Statements under "Cited Findings" are what the papers say. Everything that is
  my own judgement (the ranking, what is new, what a paper should show) is
  under "Inferences".
- No statement here comes from the memory store. Where I add something from my
  own background knowledge it is marked "(from memory, not checked)".

Tools used: web search (about 60 queries, including every term listed in the
brief), the arXiv listing API (abstract search), the OpenAlex API for "cited
by" lists of AI Feynman, AI Feynman 2.0, compositional kernel search, the
Automatic Statistician paper, Kennedy & O'Hagan, SHAREs, NEAR and KAN, each
filtered by topic words, and the full texts of the candidates.

## 1. Is there a symbolic regression or equation discovery method in which the unknown part of an expression carries a GP (or other nonparametric Bayesian) prior that is marginalised, so that partial expressions can be scored exactly?

### Takeaway

No paper was found that does this inside a search over expressions. The two
closest in mechanics are compositional kernel search (same greedy
search-and-evidence loop, but over kernels) and the sparse Bayesian equation
discovery of Zhu et al. with a marginalised GP discrepancy term (same exact
evidence for "formula terms + GP", but a fixed library and an additive GP that
is never refined). One part of the method is not new: reading scaling and
translation symmetries, power-law and exponential factors from linear relations
between a fitted function and its derivatives was published in August 2026 for
a neural surrogate (NestyNet-SR), in nearly the same form.

### Cited Findings

#### Ranked list of the closest prior work (ten works, all read)

**1. Duvenaud, Lloyd, Grosse, Tenenbaum, Ghahramani (2013), "Structure
discovery in nonparametric regression through compositional kernel search",
ICML 2013, arXiv:1302.4922; with Lloyd et al. (2014), arXiv:1402.4304.**

- Same: a greedy search over GP models by rewrite rules, "at each stage, we
  choose the highest scoring kernel and expand it by applying all possible
  operators" (§3). The score is the GP marginal likelihood, with the integral
  over kernel parameters approximated by BIC after optimising them (§3,
  "Scoring kernel families"). New parameters start from the values of the
  model that was refined (§3). — [arXiv:1302.4922](https://arxiv.org/abs/1302.4922)
- Same: the language already holds parametric parts and mixed models. `Lin` is
  Bayesian linear regression, `Lin × Lin × …` polynomial regression,
  `Lin + SE` a "linear trend with local deviations" and `Lin × SE` a "linearly
  growing amplitude" (§2, "Example expressions"). The paper names "building a
  composite model out of an interpretable, parametric part (such as linear
  regression) and a 'catch-all' nonparametric part" as a special case, "a sum
  of SE and Lin" (§3). In the terms of the method, `SE1 × Lin2` is
  `x2·H(x1)`. — [arXiv:1302.4922](https://arxiv.org/abs/1302.4922)
- Same: Lloyd et al. add changepoints "through addition and multiplication
  with sigmoidal functions" (§2 of that paper), i.e. a known function times a
  GP, and state that the language "can represent both simple parametric forms
  of f such as linear or polynomial and also complex nonparametric functions".
  Search and BIC score are as in the 2013 paper (§3). The paper compares its
  output with the equations that Eureqa gives for the same time series (§5.3)
  and has Eureqa among the baselines for prediction (§8). — [arXiv:1402.4304](https://arxiv.org/abs/1402.4304)
- Differs: the authors set their work apart from equation learning: "Because
  we learn expressions describing the covariance structure rather than the
  functions themselves, we are able to capture structure which does not have a
  simple parametric form" (§4, citing Schmidt & Lipson 2009 and Todorovski &
  Džeroski 1997). The result is a kernel, not a formula. — [arXiv:1302.4922](https://arxiv.org/abs/1302.4922)
- Differs: the search grows upward from single base kernels on single inputs
  (§3: "begins by proposing all base kernel families applied to all input
  dimensions"); operators are `S → S + B`, `S → S × B`, `B → B'` only. There is
  no merge of inputs, no outer function, no library of nonlinear factors, and
  no rule read from a fitted model. — [arXiv:1302.4922](https://arxiv.org/abs/1302.4922)
- Evaluation: decompositions of time series (§5), recovery of known kernels
  from 300 points sampled from a GP prior at several signal-to-noise ratios
  (§6), extrapolation on the airline data and prediction on multi-input data
  against linear regression, fixed-kernel GPs and additive models (§7). — [arXiv:1302.4922](https://arxiv.org/abs/1302.4922)

**2. Zhu, Gardner, Wagg, Barthorpe, Cross, Fuentes (2022), "Robust equation
discovery considering model discrepancy: a sparse Bayesian and Gaussian process
approach", Mech. Syst. Signal Process. 168, 108717.**

- Same: the model is `y = Φθ + δ(x, ψ) + ε` with candidate terms `Φ`, weights
  `θ` and a GP `δ` for what the terms leave out (§2, Eq. 7). Both `θ` (prior
  `N(0, A⁻¹)`, the relevance vector machine) and `δ` (`N(m, K)`) are
  integrated out, and the marginal likelihood `p(y | α, ψ, σ²)` is maximised
  (§3, Eqs. 8–11). Terms are added to and deleted from the model one at a time
  by their effect on this marginal likelihood (§5, §6 steps 4–5). This is an
  exact evidence for "symbolic terms + GP remainder", used to choose terms. — [doi:10.1016/j.ymssp.2021.108717](https://doi.org/10.1016/j.ymssp.2021.108717) (open copy: [White Rose](https://eprints.whiterose.ac.uk/id/eprint/181580/))
- Same regime in one example: 100 points, noise of 10% of the output (§7.1). — [doi:10.1016/j.ymssp.2021.108717](https://doi.org/10.1016/j.ymssp.2021.108717)
- Differs: the candidate terms are a fixed design matrix chosen by the user,
  linear in the parameters (§2); there is no search over expression structure.
  The GP enters only as an additive term, and it is never refined into
  symbols: its stated purpose is to stop missing physics from biasing `θ`
  (§1, §2). Examples are a toy regression and oscillators (§7). — [doi:10.1016/j.ymssp.2021.108717](https://doi.org/10.1016/j.ymssp.2021.108717)
- A 2026 follow-up line: Manna, Rogers & Bull constrain the discrepancy GP to
  be orthogonal to the dictionary "to favour the physics-inspired bases over
  the black-box" (§1.1). — [arXiv:2606.21199](https://arxiv.org/abs/2606.21199)

**3. Ibata, Tenachi, Diakogiannis, Ibata, Shankar (2026), "NestyNet. III.
Symbolic regression from analytic neural surrogates", arXiv:2608.21051 (v2,
25 August 2026; AASTeX draft, astro-ph.IM).**

- Same: an expression tree whose unknown leaves are fitted nonparametric
  functions that are replaced step by step. "That atom is a neural atom while
  it wraps a NestyNet surrogate (its state throughout Stage A and in the
  unresolved branches of Stage B) and an analytical leaf once it has been
  rewritten in closed form"; atoms take symbolic input expressions such as
  `g(x0²/x1, x2)` (§3). Stage A splits additively and multiplicatively and
  finds compound coordinates (§4); Stage B "is implemented as a rule-driven
  search over AST rewrites", each candidate refitted and accepted or rejected
  (§7.2, §7.4). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- Same, and close to identical to the method's "symmetry" rule: the
  "determining operator" (§6.1). For the affine vector field
  `V = (Ax + b)·∇x + (α + βy)∂y` the condition
  `∇f(x)·(Ax + b) − α − βf(x) = 0` "is linear in all n² + n + 2 unknowns
  jointly, so every sampled point contributes one row of a design matrix, and
  the full Lie algebra of affine symmetries of the target is the nullspace of
  the stacked system, obtained by a single SVD" (Eq. 8). An accepted generator
  gives `f = −α/β + e^{βs}·H(z)` or `f = αs + H(z)` with `z` the invariant
  coordinates (Eq. 9). "Solving Eq. (8) in the chart `ui = log xi` … turns
  scalings into translations, so monomial coordinates `∏ xi^ai` appear as
  linear invariant covectors whose components are snapped to small rationals";
  the chart `1/xi` is also used (§6.2). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- Same: power-law merges from first derivatives. If `f = g(∏ xi^ai)` the
  weighted gradients `xi ∂f/∂xi` are collinear with the exponent vector;
  tested by an SVD rank-1 test, exponents snapped to simple values (§4.3).
  Affine, radial and centred coordinates have the like signatures (§4.4).
  Homogeneity over a subset of inputs from the "restricted Euler relation"
  `Σ_{i∈S} xi ∂f/∂xi ≈ k_S f` (§8, Eq. 14). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- Same: an outer function. A registry of 15 output transforms (powers, log,
  exp, trigonometric and inverse) is tried when no structure is found in `y`
  (§9). Simple values for exponents are preferred (§4.3). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- Differs: the surrogate is a neural network with analytic derivatives, not a
  GP, and nothing is marginalised. Candidates are compared by fitted loss with
  bonuses for simpler forms (§7.4), or by confidence bounds on loss
  differences on a sealed audit sample (§7.5). A symmetry reduction is kept
  "only if it survives a complexity-penalized comparison against the unreduced
  expression at matched held-out accuracy" (§6.1). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- Differs: data. 2,000 training and 2,000 validation points per equation; the
  noise runs use a further sealed audit set of 20,000 rows; units of all
  variables are given to the method (§11.1). The paper states that symmetry
  discovery "inherits the surrogate's derivative quality, certifying only
  while gradients are accurate to ∼10⁻³" (§11.3). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- Differs: the reported result is a closed form or nothing; the statistical
  mode "declines any candidate it cannot defend on sealed data against a
  simpler rival" (§11.1). I found no statement that an answer with a neural
  atom left in it is reported as a result. — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- Evaluation: all 120 SRBench Feynman equations recovered without noise,
  median 6 minutes on one CPU core; with noise `σ/σy` of 10⁻³, 10⁻², 10⁻¹:
  105, 94, 56 of 120 with the heuristic acceptance and 103, 91, 54 with the
  statistical selection (§11.1, Table 4). Two astrophysical case studies
  (§11.3, §11.4). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)

**4. Udrescu & Tegmark (2020), "AI Feynman", Science Advances,
arXiv:1905.11481; Udrescu et al. (2020), "AI Feynman 2.0", NeurIPS 2020,
arXiv:2006.10782.**

- Same: fit a surrogate, test it for structure, recurse on smaller problems.
  AI Feynman tests a neural network for translational symmetry and its
  generalisations (a pair of inputs replaced by its sum, difference, product
  or ratio) and for additive and multiplicative separability (§II). — [arXiv:1905.11481](https://arxiv.org/abs/1905.11481)
- Same: structure from derivatives of the surrogate. AI Feynman 2.0: "all our
  other types of graph modularity (compositionality, symmetry and generalized
  additivity) can be revealed by ∇f" (§2.1). Generalised symmetry
  `f = g[h(x′), x″]` is detected because the normalised gradient with respect
  to `x′` does not depend on `x″`; compositionality `f = g(h(x))` from
  `∇f ∝ ∇h`; generalised additivity from the ratio of two partial derivatives
  being multiplicatively separable (§2.1, Table 1). Table 1 also has "Solve
  for g ≡ ln f" for positive `f` and snapping of parameters to integers and
  rationals. — [arXiv:2006.10782](https://arxiv.org/abs/2006.10782)
- Same: a greedy choice among the candidate decompositions, "at each step we
  compare the decomposition in a unified way and try only the most accurate
  one" (§2.2). — [arXiv:2006.10782](https://arxiv.org/abs/2006.10782)
- Differs: the surrogate is a neural network (4 hidden layers of 128, 128, 64,
  64 units, 80% of the data for training; §2.4), candidates are scored by
  error and description length on a Pareto frontier (§2.2), and after a
  decomposition each part is a separate regression problem; there is no one
  probabilistic model of the whole expression and no part is left open in the
  answer. — [arXiv:2006.10782](https://arxiv.org/abs/2006.10782)
- Differs: data. Each problem has 10⁵ rows (§III of the first paper).
  "equations that require the use of a neural network to be solved need
  significantly more data points (between 10² and 10⁶)" (§III.C). With noise,
  "most of the equations can still be recovered exactly with an ε-value of
  10⁻⁴ or less, while almost half of them are still solved for ε = 10⁻²"
  (§III.D). — [arXiv:1905.11481](https://arxiv.org/abs/1905.11481)

**5. Kacprzyk & van der Schaar (2024), "Shape Arithmetic Expressions: advancing
scientific discovery beyond closed-form equations", AISTATS 2024 (PMLR 238),
arXiv:2404.09788.**

- Same: expressions with placeholders for unknown functions, kept in the
  answer by design. A SHARE "consists of a finite number of shape functions,
  binary operations, variables, and numeric constants", e.g.
  `s1(x4·s2(x2)) + x1/s3(x3) − 2.3`; the shape functions "need to be plotted
  next to the equation" (§3, Eq. 5). Motivation: relationships "lacking
  inherent closed-form expressions" (abstract, §2.1). — [arXiv:2404.09788](https://arxiv.org/abs/2404.09788)
- Same: the result is a family of models from fully symbolic to fully
  nonparametric: the best expression for each number of shape functions is
  reported, e.g. `τ = r·F·s3(θ)` with `s3` matching the sine, from 100 rows of
  Feynman I.18.12 (§5.2, Fig. 5). Shape functions that look linear are
  replaced by fitted lines by hand (§5.2, §5.3). — [arXiv:2404.09788](https://arxiv.org/abs/2404.09788)
- Differs: shape functions are univariate neural networks ("We decided to use
  only univariate functions", §3). The search is "a modified genetic
  programming algorithm (based on gplearn)" with the shape functions fitted by
  gradient descent for every candidate (§5, App. B). The fitness is the fit;
  models are compared on a Pareto front of R² against the number of shape
  functions (App. D.11). No prior over functions, no evidence, no noise model. — [arXiv:2404.09788](https://arxiv.org/abs/2404.09788)
- Differs: structure is constrained by two "transparency" rules, so that each
  variable appears once and a shape function is never applied to another
  (§4, Definition 1); 86 of the 100 Feynman equations fit this form (§4). — [arXiv:2404.09788](https://arxiv.org/abs/2404.09788)
- The authors call the implementation "mainly for illustrative purposes, and
  thus … time-intensive"; "the main bottleneck comes from nested optimization
  and the necessity of fitting a separate neural network for every equation"
  (§6, App. D.2). — [arXiv:2404.09788](https://arxiv.org/abs/2404.09788)

**6. Shah, Zhan, Sun, Verma, Yue, Chaudhuri (2020), "Learning differentiable
programs with admissible neural heuristics" (NEAR), NeurIPS 2020,
arXiv:2007.12101.**

- Same: a top-down search over partial programs in which the unknown part is
  filled by a nonparametric function so that the partial program can be
  scored. "For each partial architecture u encountered during this search, the
  relaxation amounts to substituting the unknown part of u with a neural
  network with free parameters … the training loss for the relaxation can be
  viewed as an (approximately) admissible heuristic" (§1; §3.1, Eq. 4). Rules
  have structural costs that are added to the score (§2). — [arXiv:2007.12101](https://arxiv.org/abs/2007.12101)
- Differs: the score of a partial program is a training loss used as a lower
  bound for A* and branch-and-bound (§3.2), not a marginal likelihood; the
  goal is a complete program, and a hole is not part of an answer. The tasks
  are classifiers of behaviour sequences (mice, flies, basketball) with
  thousands of trajectories (§4.1), not formulas. — [arXiv:2007.12101](https://arxiv.org/abs/2007.12101)
- A later paper states that this heuristic is admissible only under
  assumptions that rarely hold and replaces it by bounds from abstract
  interpretation (search summary of Mell, Zdancewic & Bastani, POPL 2024). — [arXiv:2602.14717](https://arxiv.org/abs/2602.14717)

**7. Planas, Oune, Bostanabad (2021), "Evolutionary Gaussian processes",
J. Mech. Des. 143(11), 111703.**

- Same: a GP whose mean is a weighted sum of symbolic terms found from the
  data, `η(x) = f(x)β + ξ(x)` (§2.1, Eq. 1), fitted by maximum likelihood with
  `β` in closed form (Eqs. 4–7). The loop starts from "an ordinary GP, i.e.,
  one without any parametric bases" (Fig. 2), runs symbolic regression on
  values generated from the GP with the known terms taken out,
  `y* = y*_GP − f(x*)β` (§3), adds the terms that pass, and refits. "If this
  assumption is incorrect (or EGP fails to find bases that explain the data
  well), the output of our algorithm would be an ordinary GP" (§3.4). "While
  the goal of EGP is not symbolic regression, it can be used for that purpose"
  (§3.4, App. B). — [doi:10.1115/1.4050746](https://doi.org/10.1115/1.4050746) (author copy: [UCI](https://pmacslab.eng.uci.edu/files/2023/01/GP-Symbolic-Regression.pdf))
- Differs: the symbolic terms come from an outside symbolic regression program
  (Eureqa, §4). They are accepted by the stability of their coefficients over
  bootstrap samples (§3.3) and by the error of a GP fitted to the inner 80% of
  the domain on the outer part (§3.1), not by the evidence. The GP is only an
  additive remainder; there is no hole inside a product or behind a merged
  input. The stated aim is extrapolation of emulators (§1). — [doi:10.1115/1.4050746](https://doi.org/10.1115/1.4050746)
- Evaluation: eight analytical functions with small and large noise and one
  materials problem, against an ordinary GP and Eureqa (§4, Tables 1–2). — [doi:10.1115/1.4050746](https://doi.org/10.1115/1.4050746)

**8. Kahlmeyer, Fischer, Giesen (2025), "Dimension reduction for symbolic
regression", AAAI 2025, arXiv:2506.19537.**

- Same: a beam search over small symbolic substitutions, each scored without
  knowing the rest of the formula. An input substitution replaces some inputs
  by `g(x_I)` (a merge); an "out-input substitution" `h(x_I, y)` changes the
  target, e.g. `y/x1` (a factor or an outer function) (§2). "every candidate
  substitution becomes a node in the search tree, but now the nodes are
  weighted by a score", and the tree is explored by beam search (§4). — [arXiv:2506.19537](https://arxiv.org/abs/2506.19537)
- Differs: the score is a rank-based measure of functional dependence between
  the transformed inputs and the output (Chatterjee's coefficient, CODEC,
  KMAC), "a functional dependence score between 0 and 1" (§3, §4), not a
  likelihood. The search only reduces the number of inputs; the reduced
  problem is then "passed on to an established symbolic regression algorithm"
  (§4). No additive split, no model of the whole, no partial answer. — [arXiv:2506.19537](https://arxiv.org/abs/2506.19537)
- Evaluation: reduction rate and recovery rate on the Feynman and "eponymous
  equations" sets with 1,000 points and several noise levels, with and without
  the method in front of PySR, Operon, DSR, a transformer and UDFS (§5,
  Tables 3–4, supplement). — [arXiv:2506.19537](https://arxiv.org/abs/2506.19537)

**9. Deng et al. (2026), "Deep divide-and-reduce in symbolic regression"
(DDRSR), arXiv:2608.02628 (v3, 16 September 2026; venue not stated).**

- Same: the two merges of the method as criteria on first derivatives. If
  `(∂f/∂x1) ÷ (∂f/∂x2) = c` for all `X`, then `f = f(c1·x1 + c2·x2, …)`; if
  `(∂f/∂x1) ÷ (∂f/∂x2) × x1/x2 = c`, then `f = f(x1^c1 × x2^c2, …)`, with
  `c1/c2 = c` (§3.1). Also separability with shared inputs, a product plus a
  constant, and forms inside an outer function (§3.2). — [arXiv:2608.02628](https://arxiv.org/abs/2608.02628)
- Differs: it is a decomposition step in front of other symbolic regression
  programs. In the main experiments "function values and required derivatives
  are evaluated directly from the analytical ground-truth functions"; the
  noise test uses finite differences and thresholds (§4.1). The authors list
  as a limitation that such methods need "access to sufficiently accurate
  data" and name "more accurate surrogate models for reliable estimation of
  function values and derivative information" as future work (Limitations). — [arXiv:2608.02628](https://arxiv.org/abs/2608.02628)
- The paper calls NestyNet-SR "concurrent work" (§2). — [arXiv:2608.02628](https://arxiv.org/abs/2608.02628)

**10. Bowers, Lew, Mansinghka, Tenenbaum, Solar-Lezama (2024), "Toward
probabilistic coarse-to-fine program synthesis", LAFI 2024 (extended
abstract).**

- Same idea, in program synthesis: put a random element where the program is
  not yet known, so that the partial program has a likelihood. "Unlike
  programs with holes … probabilistic programs have well-defined semantics, so
  we can evaluate the likelihood of each yi given xi. This likelihood provides
  signal as to whether a partial program is on the right track. Promising
  programs can then be iteratively refined by replacing high-entropy
  expressions with low-entropy ones … synthesis can be stopped early to yield
  a probabilistic program" (§1). The paper lists "a way to evaluate the
  marginal likelihoods that probabilistic programs assign to outputs, to guide
  the search" as a requirement (§3). — [LAFI 2024 PDF](https://mlb2251.github.io/lafi2024_synthesis_final.pdf)
- Differs: discrete programs (list functions, classifiers that amount to
  decision trees), with coin flips and random digits as the placeholders; no
  regression, no function-valued prior. Two pages with preliminary results
  (§4). A LAFI 2026 talk by the same group, "Sequential Monte Carlo program
  synthesis with refinement proposals", continues it (title and abstract
  line from a search result only). — [LAFI 2024 PDF](https://mlb2251.github.io/lafi2024_synthesis_final.pdf); [POPL 2026 listing](https://popl26.sigplan.org/details/lafi-2026-papers/2/Sequential-Monte-Carlo-Program-Synthesis-with-Refinement-Proposals)

#### What the searches for a direct match returned

- "Semiparametric symbolic regression", "partially symbolic", "nonparametric
  component", "holes", "sketch", "grey-box … Gaussian process", "partially
  linear", "single-index … symbolic", "additive model to formula": no paper
  that puts a GP prior on a part of an expression and scores partial
  expressions by evidence. The hits were SHAREs, NEAR, Zhu et al., the hybrid
  works in section 2 below, and skeleton methods whose placeholders stand for
  constants. — searches of 28 Nov 2026; e.g. [arXiv:2404.09788](https://arxiv.org/abs/2404.09788), [arXiv:2007.12101](https://arxiv.org/abs/2007.12101)
- arXiv listing, abstracts with both "symbolic regression" and "Gaussian
  process(es)": 9 papers up to 23 Nov 2026. In all of them the GP is a
  denoiser, an emulator to be explained, an active-learning model, or a noise
  model beside the formula; none has a GP inside an expression. Examples:
  GPSINDy uses a GP "to denoise the data" before SINDy. — [arXiv:2309.11076](https://arxiv.org/abs/2309.11076); Ansari et al. use GP-based active learning to choose simulations and then run symbolic regression — [arXiv:2108.03293](https://arxiv.org/abs/2108.03293)
- arXiv listing, abstracts with "symbolic regression" and "marginal
  likelihood": 2 papers, both with the evidence of complete expressions only
  (Bomarito et al. 2022; Bomarito & Leser 2025). — [arXiv:2205.01626](https://arxiv.org/abs/2205.01626); [arXiv:2512.10849](https://arxiv.org/abs/2512.10849)
- Works that cite AI Feynman and mention Gaussian processes (29 in OpenAlex),
  nonparametric (6), semiparametric (0), marginal likelihood (3): none is a
  match. Works that cite compositional kernel search and mention symbolic
  regression (3) or equation discovery (3): none is a match; the nearest is
  Schaechtle et al. (2015), where "symbolic expressions" are kernel structures
  learned by Bayesian structure learning (abstract only). — [OpenAlex](https://api.openalex.org/works?filter=cites:W2947630964,title_and_abstract.search:gaussian%20process); [arXiv:1512.05665](https://arxiv.org/abs/1512.05665)
- The two recent reviews named in NOTES.md contain no mention of Gaussian
  processes, nonparametric or semiparametric models (full-text search of
  both): Guimerà & Sales-Pardo's review of Bayesian symbolic regression and
  the 2026 survey of uncertainty quantification in symbolic regression. — [arXiv:2507.19540](https://arxiv.org/abs/2507.19540); [arXiv:2606.06567](https://arxiv.org/abs/2606.06567)
- Bayesian symbolic regression scores complete expressions. Bomarito et al.
  use "model evidence (i.e., marginal likelihood)" of free-form equations with
  a fractional Bayes factor (§2); the 2025 paper uses a normalised marginal
  likelihood within sequential Monte Carlo (abstract). — [arXiv:2205.01626](https://arxiv.org/abs/2205.01626); [arXiv:2512.10849](https://arxiv.org/abs/2512.10849)

#### GP-side precedents for single rules (not symbolic regression)

- Linear combination of inputs behind a GP: the GP single-index model of
  Gramacy & Lian, "combining parametric (linear) projection with univariate
  nonparametric (non-linear) regression models" (abstract). — [arXiv:1009.4241](https://arxiv.org/abs/1009.4241)
- Split into GPs of fewer inputs, found from a first GP: Muehlenstaedt et al.
  estimate the interaction structure "using a first initial standard Kriging
  model", draw it as a graph, and build a block-additive kernel from its
  cliques (abstract and search summary only). Additive GPs with interactions
  of all orders: Duvenaud, Nickisch & Rasmussen (search summary only). — [doi:10.1007/s11222-011-9259-7](https://doi.org/10.1007/s11222-011-9259-7); [arXiv:1112.4394](https://arxiv.org/abs/1112.4394)
- Bayes factor between a symbolic and a GP form of one term: Wu, Jongerling &
  Mulder give each predictor "a standardized linear component with a g-prior
  and an orthogonalized Gaussian process component" and test linear against
  nonlinear against no effect (abstract; 22 Oct 2026). Mulder's 2021 paper is
  the one-predictor case (search summary only). — [arXiv:2610.31025](https://arxiv.org/abs/2610.31025); [arXiv:2109.07166](https://arxiv.org/abs/2109.07166)
- Power-law combinations from gradients in log coordinates: Constantine, del
  Rosario & Iaccarino show that "all empirical models are ridge functions" of
  the logs of the inputs and find the groups with active subspaces (§2, §3). — [arXiv:1708.04303](https://arxiv.org/abs/1708.04303)

### Inferences

The order of the ranked list is my judgement: 1 and 2 share the scoring
mechanics, 3 and 4 the purpose and the recursive use of a surrogate, 5 and 6
the idea of a placeholder inside an expression, 7 to 10 one element each.

Which elements are new, as far as these searches show:

| Element | Status | Closest published | My confidence that no closer work exists |
|---|---|---|---|
| A GP prior on the unknown part of a symbolic expression, including inside products and behind symbolic inputs, marginalised, with the evidence as the score of every partial expression in a search for a formula | Not found | Duvenaud 2013 (kernels, not formulas); Zhu 2022 (additive GP, fixed library) | Moderate to high, about 75–80% |
| Search from plain GP regression downward, each step replacing nonparametric by symbolic structure | Not found as one probabilistic model | AI Feynman and NestyNet-SR do this with a neural surrogate and separate sub-problems; kernel search goes the other way (from simple kernels up) | Moderate to high |
| A hole of one or more inputs left in the answer because the evidence does not favour any refinement | Partly published | SHAREs (univariate neural placeholders by design, no evidence); "formula + GP remainder" (Zhu 2022, Planas 2021, Otte 2013); NestyNet-SR declines to answer instead | Moderate; the combination "kept because the evidence says so, with error bars" was not found |
| The rule "symmetry": linear relations between a function and its derivatives give scalings, translations, power-law and exponential factors, merges | **Published** for neural surrogates | NestyNet-SR §6 (Aug 2026), nearly the same equations; DDRSR §3.1 (Jul 2026) for pairs; AI Feynman 2.0 §2.1 (2020) for the general case; Constantine 2017 for power laws | High that the idea is published; not found with GP posterior derivatives |
| Single rewrite rules (factor, merge, split, outer function, formula from the hole's values) | Each has a precedent | See the list above and section 3 | High |
| The regime of about 100 noisy points | Not shown by the divide-and-conquer methods | AI Feynman needs 10²–10⁶ points for its network steps; NestyNet-SR uses 2,000 + 2,000; Zhu 2022 has one example at 100 points and 10% noise | Moderate |

Further inferences:

- The fact that `a(x)·H(u(x))` is a GP with kernel
  `a(x)·k(u(x), u(x′))·a(x′)` is textbook kernel algebra, as is a GP with a
  weighted sum of known functions as its mean (from memory, not checked:
  Rasmussen & Williams 2006, ch. 4 and §2.7), and the model class
  with `a` a polynomial or a sigmoid and `u` a single input is already inside
  the kernel grammar of Duvenaud 2013 and Lloyd 2014. So the model class alone
  is not the contribution; what was not found is its use as the state of a
  search for a formula, with nonlinear symbolic factors, merged inputs, an
  outer function and removal of the GP as the goal.
- NOTES.md lists "the rules as linear constraints on a hole … a principle in
  place of a list" under "what looks new". After NestyNet-SR §6 and DDRSR
  §3.1 this can no longer be claimed as new. What remains specific to the
  method: the derivatives are those of a GP posterior mean (available in
  closed form at 100 noisy points, where the cited works assume gradients
  accurate to about 10⁻³ or use exact ones), the relations only propose
  candidates that the evidence then accepts or rejects, and the second-order
  relations (sine factor, "linear in `u^p`") — I did not find these two in
  NestyNet-SR, which finds trigonometric structure by a grid over frequencies
  (§8), but I did not check its companion papers.
- The objection to expect is no longer only "AI Feynman with a GP". It is
  "NestyNet-SR / AI Feynman with a GP surrogate, scored like the Automatic
  Statistician". The answer has to rest on the first two rows of the table and
  on the data regime.
- Reviewer 2 of NEAR asked whether a small neural network "that predicts the
  residual after running your interpretable model" had been considered (see
  section 7). The idea of keeping a nonparametric remainder is
  therefore natural to reviewers; the evidence-based decision between
  remainder and formula is the part that needs to be shown.

### Gaps

- OpenReview could not be opened from this environment (a verification page
  for both direct download and page fetch). So: the full text of Cohrs et al.
  (ICLR 2024 workshop, section 2) was not read, submissions under review
  (for example to ICLR 2027, if already public) could not be searched on the
  site itself, and no OpenReview reviews were read. Web search for such submissions returned none
  that match, but search engines index them poorly.
- Google Scholar was not queried directly; the Semantic Scholar API refused
  the requests (rate limit). "Cited by" lists came from OpenAlex, which
  searches titles and abstracts only and lacks abstracts for part of its
  records. A paper that mentions a GP prior on a subexpression only in its
  body would be missed.
- The companion papers NestyNet I, II and IV were not read. Paper IV
  (differential equations) is said to extend the symmetry layer (§12 of
  Paper III).
- Theses, patents, non-English work and software without a paper were not
  searched.
- Not read in full, cited from abstracts or search summaries: Muehlenstaedt
  2012, Mulder 2021, Wu et al. 2026, Schaechtle 2015, Mell et al. 2024, the
  LAFI 2026 talk.

## 2. Is there a method that returns an answer that is partly symbolic and partly nonparametric by design, and says so?

### Takeaway

Yes, in three forms: expressions with univariate shape functions (SHAREs),
"formula + GP remainder" models, and hybrid models with a neural network for
one named term. None chooses between the nonparametric part and a formula by
the evidence, and only SHAREs searches over where the nonparametric parts sit
inside the expression.

### Cited Findings

- SHAREs (see section 1, item 5): shape functions inside arithmetic
  expressions, reported as a Pareto front over the number of shape functions;
  "any GAM is an example of a SHARE", and closed-form expressions are SHAREs
  whose shape functions are from a fixed set (§3, Remark 1). — [arXiv:2404.09788](https://arxiv.org/abs/2404.09788)
- Otte (ESANN 2013): "The combination of a symbolic regression model with a
  residual Gaussian Process is proposed for providing an interpretable model
  with improved accuracy"; `f(x) = a(x) + g_γ(r(x))` with `a` from Eureqa, `r`
  a GP on the residuals and `g_γ` a clip to ±γ, so that "the maximal deviation
  from the symbolic model is always below a defined limit" (abstract, §2). The
  two parts are fitted one after the other; evaluation on the SARCOS robot
  data (§3). The journal version (Neurocomputing 143, 2014, "Interpretable
  semi-parametric regression models with defined error bounds") was not
  opened. — [ESANN 2013](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-3.pdf); [doi:10.1016/j.neucom.2013.11.042](https://doi.org/10.1016/j.neucom.2013.11.042)
- Planas et al. (2021), symbolic bases as the mean of a GP, and Zhu et al.
  (2022), sparse terms plus a GP discrepancy: section 1, items 7 and 2.
- Yang & Affenzeller (2022): kriging on the residual of a symbolic expression,
  whose standard deviation is used as the uncertainty of the expression
  (search summary only). — [GP bibliography entry](https://gpbib.cs.ucl.ac.uk/gp-html/YANG_2022_procs.html)
- Ziatdinov et al.: "structured Gaussian Processes containing probabilistic
  models of the possible system's behaviors (hypotheses)" as GP mean
  functions, with a policy that chooses among hypotheses during an experiment
  (abstract); a later paper builds the hypotheses from a small subset of data
  and says it "combines the elements from the symbolic regression methods such
  as SISSO and active learning" (abstract). — [arXiv:2112.06649](https://arxiv.org/abs/2112.06649); [arXiv:2301.02665](https://arxiv.org/abs/2301.02665)
- Cohrs, Varando, Guimerà, Sales-Pardo, Camps-Valls (ICLR 2024 workshop
  AI4DifferentialEquations in Science), "Semiparametric inference and equation
  discovery with the Bayesian machine scientist": a hybrid of Bayesian
  symbolic regression with a flexible part, on the Q10 model of ecosystem
  respiration; reported to recover the equations and to extrapolate better
  than non-hybrid and deep-learning baselines (abstract as shown on the
  conference page; a search summary says the flexible part is a neural
  network). Full text not opened. — [ICLR page](https://www.iclr.cc/virtual/2024/21401); [OpenReview](https://openreview.net/forum?id=7YqV4deFAc)
- Asadzadeh, Gänser & Mücke (2021): "hybrid SR models, where the regression
  tree is partly fixed according to a-priori knowledge", for a sheet bending
  process; here the known part is symbolic and the searched part is symbolic
  too (abstract only). — [doi:10.1016/j.apples.2021.100049](https://doi.org/10.1016/j.apples.2021.100049)
- Universal differential equations: neural networks stand for the unknown
  terms of a differential equation and are turned into symbols afterwards with
  sparse regression (§2.3). — [arXiv:2001.04385](https://arxiv.org/abs/2001.04385)
- HER-PDE (Aug 2026): an agent "discovers compositional PDE structure together
  with nonparametric, time-invariant coefficient fields"; structures are
  scored by transfer between two excitations (abstract only). — [arXiv:2608.27475](https://arxiv.org/abs/2608.27475)
- KAN: spline functions of one input are replaced by symbols one at a time
  ("Symbolification", §2.5.1), with `suggest_symbolic` proposing candidates
  and the affine parameters retrained (§2.5.2). A network in which only some
  activations have been fixed is a mixed model, but the paper treats it as a
  stage, not a result. — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)

### Inferences

- "Partly symbolic by design" is published (SHAREs), and so is "formula plus a
  GP for the rest". The method differs in that the nonparametric part can sit
  inside a product or behind a merged input, can have several inputs, has a
  posterior with error bars, and stays only when the evidence does not prefer
  a formula.
- SHAREs argues for the partly symbolic answer on scientific grounds (laws
  that have no closed form: stress–strain curves, phase changes). That
  argument can be cited rather than made again.

### Gaps

- Cohrs et al. and the Neurocomputing version of Otte were not read in full.
- No search of the statistics literature on semiparametric model selection
  beyond the two Bayes-factor papers in section 1.

## 3. Divide-and-conquer symbolic regression other than AI Feynman: which methods detect separability, symmetry or compositional structure from a surrogate or from the data and recurse? Does any use a GP as the surrogate or a Bayesian test for the structure?

### Takeaway

There is a line of such methods from 2017 to 2026. One uses a GP as the
surrogate, but only to generate points for a correlation test of additive
separability. None uses a Bayesian test for the structure; NestyNet-SR has a
frequentist one (confidence bounds on a sealed sample).

### Cited Findings

- Luo, Chen & Jiang (2017): the target is divided into sub-functions when a
  "bi-correlation test" finds it separable; each part is then found by genetic
  programming (abstract, §4). — [arXiv:1705.08061](https://arxiv.org/abs/1705.08061)
- Liu, Yang, Zhong & Wang (2021), "Genetic programming with separability
  detection for symbolic regression": "by using a gaussian process surrogate
  model, the proposed method does not require the system be able to generate
  new data for separability detection". The GP predicts the output at four
  constructed input matrices; correlation coefficients equal to 1 mean
  additive separability. Additive separability only; the GP hyperparameters
  are set by an evolutionary optimiser; the article names no marginal
  likelihood or model comparison; 15 synthetic problems against three genetic
  programming variants. (From a fetched summary of the article page; tables
  not seen.) — [doi:10.1007/s40747-020-00240-6](https://doi.org/10.1007/s40747-020-00240-6)
- Taylor genetic programming (He et al. 2022): features of a Taylor polynomial
  fitted to the data, among them variable separability, monotonicity and
  parity (§3.2), guide the search (section headings and abstract only). — [arXiv:2205.09751](https://arxiv.org/abs/2205.09751)
- ScaleSR (Chu et al. 2023): a neural network is learned as a "data
  generator", one input is varied at a time, and single-input formulas are
  combined (abstract only). — [arXiv:2306.04718](https://arxiv.org/abs/2306.04718)
- KAN 2.0 (Liu et al. 2024): separability from the Hessian of the network (of
  `f` for sums, of `log|f|` for products), "generalized separability" from
  ratios of first derivatives, and generalised symmetry, applied recursively
  by a "tree converter" (§4.2.2). — [arXiv:2408.10205](https://arxiv.org/abs/2408.10205)
- KAN-SR (2025): KANs with "simplification strategies such as translational
  symmetries and separabilities" after AI Feynman (abstract, §"Symbolic
  simplification via separability and symmetry detection"), an output
  transformation such as the logarithm when no fit is found, and BIC or AIC to
  choose among formulas; evaluated on the SRSD Feynman sets. — [arXiv:2509.10089](https://arxiv.org/abs/2509.10089)
- Kahlmeyer et al. (2025), DDRSR (2026), NestyNet-SR (2026): section 1, items
  8, 9 and 3.
- uDSR (Landajuela et al., NeurIPS 2022) uses AI Feynman's "recursive problem
  simplification" as one of five combined strategies (§1, §2). — [NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf)
- Decomposable neural symbolic regression (Morales & Sheppard, TMLR 2026): a
  transformer proposes one-input skeletons from an opaque model, which are
  merged step by step (abstract only). — [arXiv:2511.04124](https://arxiv.org/abs/2511.04124)
- On the tests: NestyNet-SR's statistical mode removes a candidate only when
  "the upper bound on E[d_ij] is negative", with one-sided simultaneous 95%
  bounds from a bootstrap of paired loss differences on a sealed audit sample
  (§7.5). AI Feynman 2.0 prunes candidates on a Pareto frontier of
  description length against error (§2.2), which its authors describe as
  replacing the thresholds of the first paper (author feedback, section 7). Kahlmeyer et al. use a dependence score in [0, 1]
  (§4). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051); [arXiv:2006.10782](https://arxiv.org/abs/2006.10782); [arXiv:2506.19537](https://arxiv.org/abs/2506.19537)

### Inferences

- A GP surrogate in this line exists (Liu et al. 2021), so "a GP in place of
  the network" is not new by itself. Not found: the structure test as a
  comparison of the evidence of two GP models, and the surrogate and the
  partial formula as one model.
- Three papers of 2025–2026 (Kahlmeyer, DDRSR, NestyNet-SR) extend AI
  Feynman's decompositions at the same time. The area is active; a paper on
  the method will be read against them.

### Gaps

- Liu et al. 2021 was read through a fetched summary, not the PDF.
- TaylorGP, ScaleSR, decomposable neural SR: abstracts only.
- "A comparative evaluation of additive separability tests for
  physics-informed machine learning" (2023, doi:10.1007/978-3-031-48316-5_13)
  was found but not opened; it may compare tests that involve GPs.

## 4. Search methods that score partial expressions: how do they score an expression that is not finished, and is any of those scores a marginal likelihood over the unfinished part?

### Takeaway

Four ways were found: complete the expression at random and score the result
(Monte Carlo tree search), learn a value network, fill the gap with a fitted
function or a fitted family and score the fit (NEAR, uDSR, NestyNet-SR), or
compute a bound. The one score that is a likelihood of an unfinished program
is in the program-synthesis abstract of Bowers et al., for discrete programs.

### Cited Findings

- Monte Carlo tree search: in the Symbolic Physics Learner, "if the current
  node is non-terminal, the agent performs one or multiple independent
  simulations starting from the current node until reaching the terminal
  state" (§2), and the reward is that of the finished expression (§3). — [arXiv:2205.13134](https://arxiv.org/abs/2205.13134)
- A 2025 paper on tree search for symbolic regression defines the reward "as
  zero if the resulting state s′ is still an incomplete expression" and uses
  rollouts (§2.2). — [arXiv:2509.15929](https://arxiv.org/abs/2509.15929)
- DGSR-MCTS (Kamienny et al. 2023) values expressions with a learned critic
  network, `v(f) = C_ψ(D, f)` (§3.1). — [arXiv:2302.11223](https://arxiv.org/abs/2302.11223)
- NEAR: the hole is replaced by a neural network and the training loss is the
  heuristic value (section 1, item 6). — [arXiv:2007.12101](https://arxiv.org/abs/2007.12101)
- uDSR: a `LINEAR` token "serves as a placeholder" for a sparse linear model
  of basis functions inside a nonlinear expression; it is solved by inverting
  the operators above it and a linear fit, and "it only works if there is
  exactly one LINEAR token" (§4, "Integrating LM via the novel LINEAR
  token"). — [NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf)
- NestyNet-SR's factorized search: "A carrier is never evaluated on its own.
  Its score is the loss L(s) above, obtained after fitting the best outer map
  and residual head", so that "a skeleton receives partial credit whenever
  some fitted outer map and residual head explain the target well" (§5); the
  outer maps are a polynomial, a power law, a rational function, a sinusoid
  and an exponential (§5.1). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- Kahlmeyer et al.: a substitution is scored by whether the output is still a
  function of the reduced inputs (section 1, item 8). — [arXiv:2506.19537](https://arxiv.org/abs/2506.19537)
- Bounds: mixed-integer formulations of symbolic regression prune with lower
  bounds from relaxations (search summary of Austel et al. only); Mell et al.
  bound the value of sets of programs by abstract interpretation (abstract). — [arXiv:1710.10720](https://arxiv.org/abs/1710.10720); [arXiv:2602.14717](https://arxiv.org/abs/2602.14717)
- Likelihood of a partial program: Bowers et al. (section 1, item 10). — [LAFI 2024 PDF](https://mlb2251.github.io/lafi2024_synthesis_final.pdf)

### Inferences

- No symbolic regression search was found whose score of an unfinished
  expression is a marginal likelihood over the unfinished part.
- NestyNet-SR's "carrier scored through the best of a family of outer maps" is
  the nearest in spirit to a hole of one input: both ask whether some function
  of a proposed inner expression explains the data. It uses five parametric
  families and least squares; a GP hole uses a prior over all smooth functions
  and the evidence.
- NEAR's loss is an optimistic bound by intent; the evidence of a GP hole is
  not a bound on the scores of its refinements. A reviewer may ask what
  guarantee the greedy search then has.

### Gaps

- Austel et al. was not read. LLM-based searches (e.g. LLM-SR, FunctionEvolve)
  were not examined for how they score skeletons beyond their abstracts.

## 5. Symmetry or invariance detection from a fitted GP's gradients: is the idea of reading power-law, exponential and merge structure from linear relations between a function and its derivatives published in the context of symbolic regression?

### Takeaway

Yes, with neural surrogates or exact derivatives, most completely in
NestyNet-SR (August 2026) and DDRSR (July 2026), and in a less general form in
AI Feynman 2.0 (2020). With the derivatives of a GP it was not found.

### Cited Findings

- NestyNet-SR §6.1–6.2, §4.3, §8: one linear determining equation in the
  function and its gradient, solved by SVD over the data points, in the charts
  `x`, `log x` and `1/x`; normal forms `e^{βs}H(z)` and `αs + H(z)`; exponents
  snapped to simple values (quoted in section 1, item 3). The paper presents
  this as "replacing a catalog of special cases by geometry" (§6). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- The same paper on noise: "On trained surrogates, whose gradients carry
  relative errors of order 10⁻³, the global nullspace solve can lose bracket
  closure", so pairwise tests are used as well (§6.2); bootstrap angles and
  held-out residuals guard the result (§6.1). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- DDRSR §3.1: constant ratio of two partial derivatives gives
  `c1·x1 + c2·x2`; constant ratio times `x1/x2` gives `x1^c1·x2^c2` (quoted in
  section 1, item 9). — [arXiv:2608.02628](https://arxiv.org/abs/2608.02628)
- AI Feynman 2.0 §2.1: generalised symmetry from the direction of the
  gradient; the inner function `h` is then searched for separately ("only
  performing the time-consuming task of searching for an h … after
  determining that a solution exists"). — [arXiv:2006.10782](https://arxiv.org/abs/2006.10782)
- KAN 2.0 §4.2.2: Hessian and gradient-ratio tests on a KAN. — [arXiv:2408.10205](https://arxiv.org/abs/2408.10205)
- Outside symbolic regression: Constantine et al. (2017) find products of
  powers of the inputs as active subspaces in log coordinates (§2–3); a 2026
  preprint finds dimensionless groups by SVD of log-transformed data
  (abstract only). — [arXiv:1708.04303](https://arxiv.org/abs/1708.04303); [arXiv:2606.16737](https://arxiv.org/abs/2606.16737)
- Reviewer concern about surrogate derivatives: reviewers of AI Feynman 2.0
  asked how overfitting of the network is handled and whether the system
  functions "on small training sets (say, tens of points)" (Reviews 1 and 3;
  exact wording in section 7). — [NeurIPS 2020 reviews](https://proceedings.neurips.cc/paper/2020/file/33a854e247155d590883b93bca53848a-Review.html)
- Searches for "Gaussian process" with "symmetry detection", "scaling
  symmetry", "Lie symmetry", "homogeneous function" and "symbolic regression":
  no paper that reads such relations from GP derivatives. — searches of 28 Nov 2026

### Inferences

- The idea of the rule is published; a paper cannot present it as its own
  principle. It can present: the same relations from a GP posterior mean and
  its closed-form derivatives at about 100 noisy points; the relations as
  proposals whose acceptance is left to the evidence, in place of thresholds
  and certificates; and the comparison with NestyNet-SR's requirement of
  gradients accurate to about 10⁻³.
- NestyNet-SR's affine field `Ax + b` is more general than relations of one to
  three inputs in one coordinate (it covers rotations and Lorentz boosts,
  §6.2). The method's second-order relations (sine factor, linear in `u^p`)
  were not found there.
- Both 2026 papers must be cited where the rule is described. NOTES.md should
  be corrected on this point.

### Gaps

- Not checked: whether the GP literature has posterior tests of such
  differential relations under other names (e.g. tests of invariance of a GP,
  GPs constrained by linear operators). One hit, "Constraining Gaussian
  processes to systems of linear ordinary differential equations" (NeurIPS
  2022), builds the constraint in rather than detecting it; not read.
- NestyNet papers I, II, IV not read.

## 6. Kernel or GP methods for equation discovery: how close are they?

### Takeaway

Close in one respect only: Zhu et al. (2022) and its orthogonal-kernel
follow-up have "selected terms + marginalised GP". The other GP and kernel
methods use the GP for the measured signal (smoothing, derivatives, the
solution of the equation) and keep the equation a sparse choice from a fixed
dictionary.

### Cited Findings

- Zhu et al. (2022): section 1, item 2. Manna et al. (2026): GP constrained to
  be orthogonal to the dictionary, "grey-box SINDy" (§1.1, §2). — [doi:10.1016/j.ymssp.2021.108717](https://doi.org/10.1016/j.ymssp.2021.108717); [arXiv:2606.21199](https://arxiv.org/abs/2606.21199)
- KBASS (Long et al., AISTATS 2024): "We use kernel regression to estimate the
  target function and its derivatives" and "a Bayesian spike-and-slab prior to
  select equation operators and to estimate the operator weights" from a
  dictionary of operators; inference by expectation propagation within EM
  (§1, §3). The kernel models the solution, not an unknown part of the
  equation. — [arXiv:2310.05387](https://arxiv.org/abs/2310.05387)
- GPSINDy (Hsin et al.): GP regression "to denoise the data", then SINDy
  (abstract). — [arXiv:2309.11076](https://arxiv.org/abs/2309.11076)
- Raissi & Karniadakis (2017): GPs to learn the parameters of linear
  differential operators of a given form (abstract). — [arXiv:1701.02440](https://arxiv.org/abs/1701.02440)
- Kennedy & O'Hagan-type discrepancy with symbolic regression in applications:
  a 2025 clinical-model paper notes that "a Gaussian Process (GP) is commonly
  used to capture the model discrepancy" and describes as an alternative a
  grey-box model whose incomplete parts are filled by a neural network that is
  then regressed into symbols (search summary only). — [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10206906)
- Works citing Kennedy & O'Hagan that mention symbolic regression (8) or
  equation discovery (10) in OpenAlex: applications and discrepancy models;
  none searches over expressions with a GP inside. — [OpenAlex](https://api.openalex.org/works?filter=cites:W1973333099,title_and_abstract.search:symbolic%20regression)

### Inferences

- Zhu et al. is the reference for "exact evidence of formula terms with a GP
  for the rest, used to select terms", and it predates the method. The step
  beyond it: the GP is not a nuisance to protect the parameters but the thing
  to be explained away, it may enter multiplicatively and through symbolic
  inputs, and the terms are searched rather than listed.
- KBASS, GPSINDy and Raissi-type methods are not close.

### Gaps

- The engineering literature on grey-box identification with GPs is large;
  only the works that surfaced through the searches and the "cited by" lists
  were looked at.

## 7. For the three or four closest works: where were they published, how were they evaluated, and what did reviewers ask for?

### Takeaway

The closest works appeared at ICML, NeurIPS, AISTATS, AAAI, in Science
Advances and in engineering journals; the two 2026 preprints have no venue
yet. Evaluation of formula-finding methods is by exact recovery on the Feynman
sets with noise levels up to 10% and thousands of points; the GP-side works
are evaluated on prediction and extrapolation. Public reviews exist for two of
them (NeurIPS 2020). Reviewers asked for failure cases, behaviour with few
points, what happens when the surrogate is wrong, more baselines, and proof
that the heuristic does what is claimed.

### Cited Findings

| Work | Venue | Benchmarks and data | Noise | Baselines | Compute |
|---|---|---|---|---|---|
| Compositional kernel search | ICML 2013 | Time series; kernels recovered from 300 points drawn from a GP; airline data; multi-input regression sets (§5–7) | Several signal-to-noise ratios (§6) | Linear regression, fixed-kernel GPs, additive models (§7) | Search to depth 10 (§5) |
| Zhu et al. | Mech. Syst. Signal Process. 2022 | Toy regression with 100 points; simulated and measured oscillators (§7) | 10% of the output in §7.1 | Relevance vector machine without the GP (§7) | Not compared |
| AI Feynman / 2.0 | Science Advances 2020 / NeurIPS 2020 | 100 + 20 Feynman equations, 10⁵ rows each (§III of the first paper) | Largest tolerated noise per equation (§III.D; Table 3 of 2.0) | Eureqa with 300 points, 2 CPU hours per equation (§III) | 2 hours per equation |
| NestyNet-SR | arXiv preprint, Aug 2026 | 120 SRBench Feynman equations, 2,000 + 2,000 points, units given; two astrophysics cases (§11) | 0, 10⁻³, 10⁻², 10⁻¹ of σy (Table 4) | Earlier published recovery rates; no rerun baselines in §11.1 as far as read | Median 6 min, one core (§11.1) |
| SHAREs | AISTATS 2024 | A risk-score GAM, Feynman I.18.12 with 100 rows, a water-heating problem with 2,000 samples, real tabular sets (§5, App. C) | One noise study (App. C.4, Fig. 10) | GAMs (EBM, PyGAM), PySR, XGBoost (§5, App. C) | About 15 hours for all experiments (App. C.5) |
| NEAR | NeurIPS 2020 | Three behaviour-classification sets (§4.1) | — | Enumeration, Monte Carlo sampling, MCTS, genetic search, an RNN (§4.2) | Search time reported (§4) |
| Kahlmeyer et al. | AAAI 2025 | Feynman and eponymous equations, 1,000 points (§5, supplement) | Several levels (Fig. 3) | PySR, Operon, DSR, transformer, UDFS, each with and without the method (Table 4) | Milliseconds per test (Table 1) |
| Evolutionary GPs | J. Mech. Des. 2021 | Eight analytical functions, one materials problem (§4) | Small and large (Table 2) | Ordinary GP, Eureqa (§4) | Not the focus |

Sources for the table: [arXiv:1302.4922](https://arxiv.org/abs/1302.4922); [doi:10.1016/j.ymssp.2021.108717](https://doi.org/10.1016/j.ymssp.2021.108717); [arXiv:1905.11481](https://arxiv.org/abs/1905.11481); [arXiv:2006.10782](https://arxiv.org/abs/2006.10782); [arXiv:2608.21051](https://arxiv.org/abs/2608.21051); [arXiv:2404.09788](https://arxiv.org/abs/2404.09788); [arXiv:2007.12101](https://arxiv.org/abs/2007.12101); [arXiv:2506.19537](https://arxiv.org/abs/2506.19537); [doi:10.1115/1.4050746](https://doi.org/10.1115/1.4050746)

Reviews of AI Feynman 2.0 (NeurIPS 2020, three reviewers, all positive):

- Review 1 called the description-length objective "highly principled" and
  the gradient-based detection of modularity novel, and criticised: "The
  system itself is seemingly quite complex, full of heuristics and relatively
  slow"; "It would be more interesting to see some failure cases and areas for
  improvement"; "any published formulas are likely to already be expressed in
  the 'right' coordinates for the problem being solved and thus present an
  easier task than what might be encountered in practice"; a later paper
  should show a system that "actually discovers a novel and useful result from
  empirical data". On the surrogate: "how is overfitting addressed? Does the
  system inherently require thousands of data points in order to get a
  reasonable fit?" and "Do the authors expect the system to function
  reasonably on small training sets (say, tens of points)?" — [Reviews](https://proceedings.neurips.cc/paper/2020/file/33a854e247155d590883b93bca53848a-Review.html)
- Review 2: the paper "leans on the ideas of its predecessor. This makes the
  improvements somewhat incremental"; the differences from the first AI
  Feynman "could be a bit better explained"; "how is f_NN trained?" — [Reviews](https://proceedings.neurips.cc/paper/2020/file/33a854e247155d590883b93bca53848a-Review.html)
- Review 3: "If the neural network perfectly approximate the observed y due to
  overfitting on noise, will the proposed method be misled by the overfitting
  NNs?" — [Reviews](https://proceedings.neurips.cc/paper/2020/file/33a854e247155d590883b93bca53848a-Review.html)
- Meta-review: accepted on condition that the promised revisions are made. In
  their reply the authors promised code on GitHub and PyPI and a fuller
  account of the difference from the first paper, and described its use of the
  network as "unprincipled and ad hoc" with "arbitrary
  hyperparameter-determined thresholds", which the second paper replaces. — [Meta-review](https://papers.neurips.cc/paper_files/paper/2020/file/33a854e247155d590883b93bca53848a-MetaReview.html); [Author feedback](https://papers.neurips.cc/paper_files/paper/2020/file/33a854e247155d590883b93bca53848a-AuthorFeedback.pdf)

Reviews of NEAR (NeurIPS 2020, four reviewers):

- Review 1: "The benchmark datasets are relatively toyish"; asked for MCTS as
  a baseline and "a synthetic experiment where there are ground truth programs
  for reference … it would be interesting to see if the algorithm can recover
  the ground truth program. The authors can also empirically verify the
  epsilon-admissibility of the neural net heuristics." — [Reviews](https://papers.nips.cc/paper/2020/file/342285bb2a8cadef22f667eeb6a63732-Review.html)
- Review 2: "The particular data sets that they apply this work to seem
  nonstandard/unusual"; asked for example programs to be shown; and: "If you
  want a model that is both accurate and interpretable, a simple thing you
  could do is to learn a small neural network (or even a logistic
  regressor/etc.) that predicts the residual after running your interpretable
  model. Did you consider this?" — [Reviews](https://papers.nips.cc/paper/2020/file/342285bb2a8cadef22f667eeb6a63732-Review.html)
- Review 3: the grammar, the library functions and the network classes used
  as relaxations were not described well enough (my summary of a fetched
  summary); the three data sets "depict a similar scenario", so "I doubt about
  whether these similar datasets can validate performance improvement". — [Reviews](https://papers.nips.cc/paper/2020/file/342285bb2a8cadef22f667eeb6a63732-Review.html)
- Review 4: "This work doesn't seem very novel … ultimately seems to be yet
  another heuristic for performing a search over program space. The
  evaluation is only against baseline methods and not even some of the related
  work the authors mention themselves", and asked for "some tests on
  generalisation". — [Reviews](https://papers.nips.cc/paper/2020/file/342285bb2a8cadef22f667eeb6a63732-Review.html)

Self-stated limits that reviewers of a new paper can be expected to know:

- SHAREs: the implementation does not scale and "is not a main contribution"
  (App. D.2). — [arXiv:2404.09788](https://arxiv.org/abs/2404.09788)
- NestyNet-SR: "The Stage B rewrite rules are heuristic in nature and may miss
  optimal analytical forms"; cost "scales exponentially with the number of
  variables for exhaustive separability search"; benchmarks beyond AI Feynman
  are future work (§12). — [arXiv:2608.21051](https://arxiv.org/abs/2608.21051)
- DDRSR: needs accurate values and derivatives (Limitations). — [arXiv:2608.02628](https://arxiv.org/abs/2608.02628)
- SRBench 2.0 (GECCO 2025 workshop) calls for standard tuning, execution
  limits and compute budgets across methods (search summary only). — [arXiv:2505.03977](https://arxiv.org/abs/2505.03977)

### Inferences

What a paper on the method would be expected to show, judging from the above:

1. **Exact recovery on a standard set with standard noise levels.** Feynman
   equations in the SRBench or SRSD form, noise 0, 10⁻³, 10⁻², 10⁻¹, the
   fraction recovered as the headline number, with a stated rule for "right"
   (NestyNet-SR Table 4, Kahlmeyer Table 4, AI Feynman 2.0 Table 3). Results
   that are approximations must not count as recoveries; NestyNet-SR checks
   algebraic identity up to fitted constants.
2. **The data regime as the claim.** A curve of recovery against the number of
   points, from tens to thousands, beside methods that need a neural
   surrogate. AI Feynman reports this (§III.C), and a reviewer of 2.0 asked
   for tens of points. This is where a GP surrogate can be shown to matter.
3. **Baselines a reviewer accepts.** A genetic programming method at its
   published budget (PySR, Operon), a reinforcement-learning or transformer
   method, a divide-and-conquer method (AI Feynman 2.0; NestyNet-SR if its
   code runs at 100 points), a Bayesian symbolic regression method that
   scores complete expressions, plain GP regression, and compositional kernel
   search as the nearest method on the GP side. Kahlmeyer et al. and DDRSR
   show their step in front of several solvers; the equivalent here is the
   search with single rules removed.
4. **The surrogate is right when it speaks.** NEAR's reviewer asked for a
   check of admissibility; AI Feynman 2.0's reviewers asked what happens when
   the network overfits. The equivalent: how often the evidence prefers a
   wrong refinement over the hole it came from, at each noise level, and how
   often a hole that should have been resolved remains.
5. **Removal of parts and sensitivity to settings.** Each rule and each cost
   left out in turn (Kahlmeyer Tables 2–3; NestyNet-SR's three acceptance
   modes), and the hand-set numbers varied. The authors of AI Feynman 2.0
   presented the removal of "arbitrary hyperparameter-determined thresholds"
   as its advance over the first paper (author feedback); fixed penalties per
   symbol may be questioned in the same way.
6. **Failure cases and limits stated** (Review 1 of AI Feynman 2.0), with run
   time on stated hardware (most works in the table report it).
7. **Something beyond chosen formulas.** A real or empirical data set, or
   targets without a closed form, where the partly symbolic answer is the
   point (SHAREs §5.3; NestyNet-SR §11.4; Review 1 of AI Feynman 2.0 asked
   for discovery from raw data). For partly symbolic answers a measure is
   needed; SHAREs reports the fit for each number of shape functions and
   plots them against the truth.
8. **The difference from the predecessors in plain terms.** "Somewhat
   incremental" (Review 2 of AI Feynman 2.0) and "yet another heuristic"
   (Review 4 of NEAR) are the criticisms to expect. The paper has to say
   early what the evidence of a partial expression gives that a surrogate with
   tests does not, and show it on data.
9. **Code that can be run.** Reviewers 1 and 2 of AI Feynman 2.0 raised
   reproducibility; the authors answered with a promise to release the code.

### Gaps

- No OpenReview reviews could be read (site not reachable from here), so
  nothing is known about how SHAREs, KAN, uDSR, the tree-search papers or any
  2026 submission were reviewed. AISTATS, AAAI, ICML 2013 and journal reviews
  are not public.
- NestyNet-SR and DDRSR are unreviewed preprints; their claims are the
  authors'.
- The SRBench 2.0 protocol was not read; another researcher may have covered
  benchmarks.
- The baselines of NestyNet-SR beyond §11.1–11.2 were not checked line by
  line.
