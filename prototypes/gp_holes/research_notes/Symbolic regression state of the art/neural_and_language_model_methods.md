# Symbolic regression with pretrained neural networks and with large language models (state as of November 2026)

Reading basis. Read in full text (arXiv HTML, converted to text, tables included): NeSymReS 2106.06427, E2E 2204.10532, TPSR 2303.06833, LLM-SR 2404.18400, LaSR 2409.09359, ICSR 2404.19094, LLM-SRBench 2504.10415 (HTML has no appendix; appendix read from the PDF), SR-Scientist 2510.11661v2, EML 2603.21852 (Sec. 4.3 and Table 4 only, as asked). Read in the relevant parts only (abstract, method, result tables, limitations): KeplerAgent 2602.12259, STRIDE 2605.17790, IdeaSearchFitter 2510.08317, "How AI agents discover scientific equations" 2610.00435, Sharlin and Josephson 2410.17448v3, SRSD 2206.10540, SRBench 2025 2505.03977, 2022 SR competition 2304.01117, MDLformer 2411.03753, univariate skeleton prediction 2406.17834, NSRwH 2304.10336, "Evaluating LLMs in scientific discovery" 2512.15567 (SR part only). No paper failed to open. Numbers that exist only inside figures could not be read and are marked as such. Nothing below is filled in from memory unless marked "(from memory)".

## 1. Per paper: method, pretraining, inputs, cost of a run

### Takeaway
The pretrained transformers (NeSymReS, E2E, TPSR) are small models (26M to 86M parameters) with hard input limits (3 or 10 variables, 200 to 500 points) that answer in seconds to about 100 s; the language-model methods have no formal input limit but spend about 1,000 or more model calls per problem and run for minutes to hours. None of the listed methods was designed or tested for about 100 noisy points as its main regime.

### Cited Findings

NeSymReS (Biggio et al. 2021)
- Method: Set Transformer encoder plus Transformer decoder predicts an equation skeleton (constants replaced by a placeholder) in prefix notation; at test time beam search gives candidate skeletons and BFGS fits the constants; default beam 32, 4 BFGS restarts, best candidate chosen by in-sample loss with a 1e-14 per-token penalty — [NeSymReS Sec. 3.2, 4.3](https://arxiv.org/abs/2106.06427)
- Size: encoder 11M and decoder 13M parameters (Sec. 4.1; App. A.1 says the decoder has 15M; Sec. 5 says "26-million parameters") — [NeSymReS Sec. 4.1, App. A.1](https://arxiv.org/abs/2106.06427)
- Pretraining: 10M pre-sampled skeletons (about 1.2M unique), trees with at most 5 non-leaf nodes, at most 3 input variables, at most 3 constants drawn from U(1,5), supports drawn inside [-10,10], up to 500 points, |y| above 1000 dropped, no noise; 1.5M steps at batch 150, about 225M equations; one RTX 2080 for 3 days — [NeSymReS Sec. 4.2, App. A.1, App. B.1](https://arxiv.org/abs/2106.06427)
- Operator prior used for generation (unnormalised weights): + 10, × 10, − 5, ÷ 5, sqrt 4, pow 4, ln 4, exp 4, sin 4, cos 4, tan 4, arcsin 1; leaf is a variable with probability 0.8, an integer with 0.2 — [NeSymReS Table 6, App. B.1](https://arxiv.org/abs/2106.06427)
- Inputs at test: 128 points by default, at most 3 variables; all 52 AI-Feynman equations with up to 3 variables are present as skeletons in the pretraining set (only constants and supports differ) — [NeSymReS Sec. 4.4](https://arxiv.org/abs/2106.06427)
- Cost: all timing experiments on a single CPU; fixed test budget of about 100 s in Fig. 2; on AI-Feynman "more than three orders of magnitudes faster at reaching the same maximum accuracy" than gplearn — [NeSymReS Sec. 5(ii), Fig. 2 to 4](https://arxiv.org/abs/2106.06427)

E2E (Kamienny et al. 2022)
- Method: one Transformer (86M parameters, 4 encoder and 16 decoder layers) outputs the full expression including numeric constants (constants as sign, mantissa, exponent tokens); constants are then refined by BFGS started from the predicted values; inputs are whitened and the formula is unscaled afterwards — [E2E Sec. 1.3, 2.1, 2.2](https://arxiv.org/abs/2204.10532)
- Pretraining: random trees with up to 10 input variables, binary operators +, −, × (uniform), up to 5 unary operators with weights inv 5, sqr 3, sqrt 3, exp 1, sin 1, cos 1, abs 1, tan 0.2, atan 0.2, log 0.2 ("selected to match the natural frequencies appearing in the Feynman dataset"), random affine constants everywhere, 10·D to 200 points from mixtures of Gaussians and uniforms; about 50 epochs of 3M examples; one epoch takes about half an hour on 32 GPUs of 32 GB — [E2E Sec. 1.1, 1.2, 2.1, Table 3, App. A](https://arxiv.org/abs/2204.10532)
- Inputs at test: at most 10 variables; "does not perform satisfactorily" with more than 200 points, so larger data are split into bags of 200; settings used on SRBench were 100 bags, 10 candidates per bag, 10 refined candidates, refinement on at most 1024 points; forward pass on one GPU, refinement on CPU — [E2E Sec. 2.2, footnote 10](https://arxiv.org/abs/2204.10532)
- Cost in an independent table: 4.0 s (Strogatz) and 4.6 s (Feynman) average search time, against 24.8 s for NeSymReS and hundreds to thousands of seconds for search methods — [MDLformer Table 1](https://arxiv.org/abs/2411.03753)

TPSR (Shojaee et al. 2023)
- Method: no new model; Monte Carlo tree search over the token sequence of a pretrained transformer (E2E as the main backbone, NeSymReS also shown). Selection uses P-UCB(s,a) = Q(s,a) + β·P_θ(a|s)·sqrt(ln N(s) / (1 + N(s'))), with P_θ the transformer's next-token probability; expansion is limited to the top-k tokens (k_max = 3); defaults are 3 rollouts, simulation beam size 1, β = 1 — [TPSR Sec. 3.2, App. B Table 2, App. C.1](https://arxiv.org/abs/2303.06833)
- Inputs: inherited from E2E (at most 10 variables, 200 points per bag, up to 10 bags used one after another until R² > 0.99) — [TPSR Sec. 4.3, App. B](https://arxiv.org/abs/2303.06833)
- Cost: "typically in the order of 10² equations, taking approximately 10² seconds"; for 9 input variables on black-box data up to about 10³ s (about 30 min) against 3 hours for Operon; two caches cut inference time by about 28%; experiments on four Quadro RTX 8000 GPUs — [TPSR App. C.2, App. D.4, Sec. 4.5, App. B](https://arxiv.org/abs/2303.06833)

LLM-SR (Shojaee et al. 2024)
- Method: the language model writes Python function bodies with a free parameter vector (at most 10 parameters); parameters are fitted by scipy BFGS or by Adam in PyTorch (30 s timeout); score is −MSE; a FunSearch-style buffer with 10 islands supplies 2 earlier programs per prompt; 4 samples per prompt at temperature 0.8 — [LLM-SR Sec. 2.2 to 2.4, App. B](https://arxiv.org/abs/2404.18400)
- Pretraining: none of its own; backbones GPT-3.5-turbo (API) or Mixtral-8x7B (on 4 RTX 8000 GPUs); the prompt carries a natural-language problem description and variable meanings — [LLM-SR Sec. 2.2, App. B](https://arxiv.org/abs/2404.18400)
- Cost: about 2,500 iterations of 4 samples each (about 10,000 sampled programs) per problem; islands are reset "every 4 hrs", which implies runs of many hours (the total run time is not stated); no money figure given — [LLM-SR Sec. 3.2, App. B](https://arxiv.org/abs/2404.18400)

LaSR (Grayeli et al. 2024)
- Method: PySR in which a fraction p of initialisation, mutation and crossover operations is replaced by language-model calls conditioned on a library of natural-language "concepts"; after each iteration the model summarises the best and worst expressions into a new concept and evolves the concept library — [LaSR Sec. 3, Algorithm 1](https://arxiv.org/abs/2409.09359)
- Cost: about 60,000 calls per iteration at p = 100%, each just under 1,000 tokens; at p = 1% and 40 iterations "just under 25M tokens for each equation (around 4 hours on a single A100)"; experiments ran on 8 A100 GPUs — [LaSR App. A.3.1](https://arxiv.org/abs/2409.09359)
- Inputs: Feynman data regenerated from the equations (the data set provides 100,000 samples per equation), target noise 0.001, extra irrelevant noise variables added; variable names are given to the model — [LaSR App. A.4.1](https://arxiv.org/abs/2409.09359)

ICSR (Merler et al. 2024)
- Method: OPRO-style loop; the prompt contains up to 40 data points written as text plus the 5 best earlier functions with their errors; the model (Llama 3 8B) returns 5 functional forms per call; coefficients are fitted by scipy non-linear least squares with 5 restarts; error is the TPSR reward (NMSE plus complexity term, λ = 0.05); 10 seed prompts plus up to 50 loop iterations, so about 60 calls per problem — [ICSR Sec. 4, Sec. 5.1.1, Sec. 5.5, App. B](https://arxiv.org/abs/2404.19094)
- Inputs: only 1 or 2 input variables were tested; one-input benchmarks use 20 training points, two-input ones 100; no noise experiments appear in the paper — [ICSR Sec. 5.1, App. D Table 6](https://arxiv.org/abs/2404.19094)

LLM-SRBench (Shojaee et al. 2025)
- A benchmark, not a method: 239 problems; LSR-Transform (111 Feynman equations solved for a different variable, with a newly written description) and LSR-Synth (128 problems mixing known terms with invented terms in chemistry 36, biology 24, physics 43, materials 25); every method gets 1,000 language-model calls per problem — [LLM-SRBench Sec. 2.1, 2.2, 3.1](https://arxiv.org/abs/2504.10415)
- LSR-Synth data: 5,000 evenly spaced samples per problem, the last 500 held out as out-of-domain test — [LLM-SRBench appendix on LSR-Synth data generation, PDF p. 13](https://arxiv.org/pdf/2504.10415)

SR-Scientist (Xia et al. 2025, ICLR 2026)
- Method: a ReAct agent with a code interpreter wrapped as two tools, a free-form data analyser and an equation evaluator that fits constants by BFGS; up to 25 turns per iteration and 40 iterations, with the best equations carried over between iterations; optional reinforcement-learning fine-tuning of a 30B model — [SR-Scientist Sec. 3.2, 3.3, 4.1, App. B](https://arxiv.org/abs/2510.11661)
- Cost: about 1,000 model calls per problem; estimated API cost $0.25 per problem with GPT-OSS-120B and $0.10 with GPT-OSS-20B; all 129 problems batched on 2 H100 GPUs finished within 5 hours — [SR-Scientist Sec. 4.1, App. D.2, Table 8](https://arxiv.org/abs/2510.11661)

EML operator (Odrzywolek 2026), Sec. 4.3 only
- Method: eml(x,y) = exp(x) − ln(y) with the constant 1 generates the elementary functions; a full binary tree of depth n in which each input is a soft choice α + βx + γf between 1, x and the child output is a "master formula" with 5·2^n − 6 parameters; it is trained with Adam on complex128, then weights are pushed to 0 or 1 and snapped — [EML Sec. 4.3](https://arxiv.org/abs/2603.21852)
- Result: over more than 1,000 runs, recovery from random initialisation succeeds in 100% of runs at depth 2, about 25% at depths 3 to 4, below 1% at depth 5, and 0 of 448 at depth 6; started from the perturbed correct weights it converges back in 100% of runs even at depths 5 and 6 — [EML Sec. 4.3](https://arxiv.org/abs/2603.21852)
- Ordinary operations are large in this basis: shortest known expression sizes K (direct search) are 17 for x·y, 19 for x + y, 15 for 1/x, 17 for x², 43 for sqrt(x) (lower bound above 35), against 3 for exp(x) and 7 for ln(x) — [EML Table 4](https://arxiv.org/abs/2603.21852)
- Targets were noise-free composed-EML functions of two variables; overflow from nested exponentials and NaNs had to be handled by clamping — [EML Sec. 4.3](https://arxiv.org/abs/2603.21852)

### Inferences
- With 4 CPU cores, no GPU and no network, only NeSymReS (evaluated by its authors on one CPU) and possibly E2E and the 24M-parameter univariate model of Sec. 6 are usable at all inside a 10 to 60 s budget; every language-model method needs either network access or a local GPU and far more wall time.
- The EML result says that gradient descent over a universal tree is not a practical search for expressions of realistic size: by my count from the sizes in Table 4, a product or a sum already needs a tree of depth 4 or more, where blind recovery is about 25% or lower on noise-free data.

### Gaps
- No paper states CPU-only inference time for E2E or TPSR.
- LLM-SR gives no cost in money or tokens; only iteration counts.
- NeSymReS accuracies exist only as curves (Fig. 2 to 5); no table.

## 2. Comparison with search-based methods (PySR, Operon, uDSR), noise and few points

### Takeaway
On their own R²-based metrics the pretrained transformers look close to genetic programming, but every independent table that counts exact symbolic recovery puts NeSymReS and E2E near the bottom (0% to about 10%), well below PySR, uDSR and AI Feynman. Language-model methods beat PySR in the authors' own tables on problems with described variables, but PySR's reported score on the same benchmark varies between 8% and 38% depending on who configured it.

### Cited Findings

Authors' own comparisons
- NeSymReS compares only with DSR, gplearn and a Gaussian-process regressor, not with PySR, Operon or uDSR; it claims to outperform them "by a large margin on most budgets"; performance is lower on constant-free equations because the model has a strong prior on constants — [NeSymReS Sec. 4.5, 5(ii)](https://arxiv.org/abs/2106.06427)
- E2E on 119 Feynman problems (SRBench, three target-noise levels, 10 seeds): "the average accuracy of our model is only ranked fourth", accuracy being the fraction of problems with test R² > 0.99; Operon and SBP-GP are the top two; no symbolic solution rate is reported; on the 14 Strogatz problems it "performs somewhat less well" than GP — [E2E Sec. 3.2, Fig. 5, App. E](https://arxiv.org/abs/2204.10532)
- TPSR on SRBench, fraction with R² > 0.99: Feynman 0.848 (E2E with sampling) to 0.949 (TPSR, λ = 0.1); Strogatz 0.357 to 0.785; black-box mean R² 0.864 to 0.945; average complexity (expression-tree nodes) on Feynman 50.7 (E2E) and 57.2 (TPSR, λ = 0.1); again no symbolic solution rate — [TPSR Table 1](https://arxiv.org/abs/2303.06833)
- TPSR on NeSymReS (52 Feynman equations with up to 3 variables): fraction with R² > 0.99 rises from 0.635 to 0.808 at equal average complexity 9.98 — [TPSR App. D.3, Table 3](https://arxiv.org/abs/2303.06833)
- LaSR, exact-match solve rate on 100 Feynman equations at noise 0.001: LaSR 72, PySR 59 (62 after 10 hours and 10^6 iterations), uDSR 40, AI Feynman 38, AFP-FE 26, AFP 24, DSR 23, gplearn 20; only LaSR and PySR were run by the authors, the others are copied from SRBench and from the uDSR paper — [LaSR Table 1, Sec. 4.1, App. A.5.1 Table 5](https://arxiv.org/abs/2409.09359)
- LLM-SR on its four new problems (normalised MSE, in-domain / out-of-domain): Oscillation 1: PySR 0.0009 / 0.3106, uDSR 0.0003 / 0.0007, LLM-SR(GPT-3.5) 4.65e-7 / 0.0005; Oscillation 2: PySR 0.0002 / 0.0098, uDSR 0.0032 / 0.0015, LLM-SR(GPT-3.5) 2.12e-7 / 3.81e-5, LLM-SR(Mixtral) 0.0030 / 0.0291; E. coli: PySR 0.0376 / 1.0141, LLM-SR(Mixtral) 0.0026 / 0.0037; baselines ran "over 2M iterations", best of 5 replications — [LLM-SR Table 1, App. A.2](https://arxiv.org/abs/2404.18400)
- The text of LLM-SR says both backbones "consistently outperform baselines", but in its own Table 1 the Mixtral backbone is worse than PySR on Oscillation 2 in both columns — [LLM-SR Sec. 3.3, Table 1](https://arxiv.org/abs/2404.18400)
- ICSR on one- and two-input benchmarks (test R² / complexity, ground-truth complexity about 4 to 8): overall ICSR 0.993 / 6.7, uDSR 0.993 / 20.0, TPSR 0.979 / 14.0, E2E 0.87 / 19.4, NeSymReS 0.93 / 6.2, DSR 0.93 / 6.2, gplearn 0.6 / 7.7; PySR and Operon were not included — [ICSR Table 2](https://arxiv.org/abs/2404.19094)

Independent tables that include these methods
- SRSD-Feynman (Matsubara et al.), solution rate Easy / Medium / Hard: E2E 0.00% / 0.00% / 0.00%; uDSR 50.0% / 17.5% / 4.00%; PySR 60.0% / 30.0% / 4.00%; DSR 46.7% / 10.0% / 2.00%; AI Feynman 30.0% / 2.50% / 2.00%. Accuracy R² > 0.999: E2E 26.7% / 17.5% / 14.0%; uDSR 100% / 75.0% / 20.0%; PySR 66.7% / 45.0% / 38.0%. With one to three irrelevant variables added, 100% of E2E's predictions used at least one of them — [SRSD Tables 4 to 6](https://arxiv.org/abs/2206.10540)
- SRBench ground-truth sets averaged over four noise levels (0, 0.001, 0.01, 0.1) and 10 seeds, recovery rate Strogatz / Feynman: NeSymReS 1.79% / 2.44%; E2E 3.78% / 10.40%; SNIP 6.79% / 1.60%; Operon 4.29% / 7.97%; gplearn 9.21% / 16.89%; DSR 18.05% / 18.60%; AI Feynman 2 15.27% / 27.24%; SR4MDL 66.78% / 33.93%. The authors comment that the generative methods aim at fitting the data and "cannot" improve by running longer. PySR and uDSR are not in this table — [MDLformer Table 1, Sec. 5.1](https://arxiv.org/abs/2411.03753)
- 2022 SRBench competition: on exact rediscovery only pysr, uDSR, operon and Bingo found exact solutions, and only at the two easiest levels; the pretrained transformer entry (E2ET) could not keep R² above 0.9 at the harder levels; real-world track ranking: uDSR 5.75, QLattice 5.21, geneticengine 4.99, Operon 4.80, Bingo 4.66, pySR 4.17, PS-Tree 3.15, E2ET 2.72 (last of 8) — [2022 competition Sec. IV-B1, Table III](https://arxiv.org/abs/2304.01117)
- LLM-SR's new problems: "Transformer SR models (NeSymReS, E2E) perform poorly"; for example Oscillation 2 in-domain NMSE NeSymReS 0.2488, E2E 0.1401, PySR 0.0002; NeSymReS cannot run the 4-variable problem at all — [LLM-SR Table 1, Sec. 3.3](https://arxiv.org/abs/2404.18400)
- LSR-Synth (129 problems), overall Acc_0.01 / Acc_0.001 in %: gplearn 0.00 / 0.00; E2E 0.26 / 0.00; NeSymReS 3.10 / 0.78; DSR 0.00 / 0.00; uDSR 29.46 / 12.40; PySR 29.46 / 14.47; LLM-SR 24.55 to 41.08 / 7.24 to 18.09 depending on backbone; LaSR 11.89 to 16.02 / 7.49 to 10.08; SR-Scientist 32.30 to 63.57 / 16.02 to 49.35. Non-LLM methods had 100,000 candidate equations, LLM methods 1,000 calls — [SR-Scientist Table 1, Sec. 4.1](https://arxiv.org/abs/2510.11661)
- Symbolic accuracy on the same 129 problems is very low for all: uDSR 0.77%, PySR 4.65%, LLM-SR 4.65 to 5.43%, SR-Scientist 7.00 to 7.75% — [SR-Scientist Table 2](https://arxiv.org/abs/2510.11661)
- SRBench 2025 includes E2E, NeSymReS and TPSR among its benchmarked methods (almost twice as many as in SRBench 1.0) but reports per-method results only in figures; in the text the highest black-box scores are attributed to GP methods (Brush, GP-GOMEA, GPZGD), TPSR is noted for long execution times and NeSymReS for stopping far before its time limit — [SRBench 2025 Sec. 5.1, 5.2, 5.5](https://arxiv.org/abs/2505.03977)

PySR against language-model methods: conflicting numbers
- LLM-SRBench's own appendix: PySR (default configuration, "same computational budget") symbolic accuracy / Acc_0.1 on LSR-Transform 8.11% / 56.76% against 31.53% / 39.64% for the best LLM-SR; on LSR-Synth PySR has 0% symbolic accuracy in chemistry, biology and materials and 4.54% in physics, with Acc_0.1 of 41.67, 25.0, 68.0 and 29.55% — [LLM-SRBench App. F, Table 3, PDF p. 23](https://arxiv.org/pdf/2504.10415)
- KeplerAgent paper, same LSR-Transform set: PySR 37.84% symbolic accuracy (2,440 s per problem), LLM-SR 31.53% (2,118 s, 209k tokens), KeplerAgent 35.14% in one run (238 s, 42k tokens) and 42.34% best of three; the authors call it "surprising" that PySR with many iterations beats LLM-SR — [KeplerAgent Table 1, Sec. 4.4](https://arxiv.org/abs/2602.12259); this contradicts the 8.11% in [LLM-SRBench App. F](https://arxiv.org/pdf/2504.10415)
- Feynman set with variable names and units given to the model, recovery rate at noise 0 / 0.001 / 0.01 / 0.1: IdeaSearchFitter 82.5 / 80.0 / 79.2 / 71.7%; PySR 48.3 / 34.2 / 31.7 / 25.8%; Operon 20.8 / 7.5 / 0.0 / 0.0%; AI Feynman 53.3 / 33.3 / 13.3 / 0.8% (AI Feynman copied from SRBench) — [IdeaSearchFitter Table 1](https://arxiv.org/abs/2510.08317)
- Nonlinear-dynamics subset of LLM-SRBench, Acc_0.1 in-domain / out-of-domain: PySR (default) 13.79 / 3.44%; LLM-SR loop with gpt-5 55.17 / 51.72%, claude-sonnet-4.5 55.17 / 48.28%, deepseek-R1 58.62 / 51.72% — [Evaluating LLMs in scientific discovery, Table 6](https://arxiv.org/abs/2512.15567)
- On LLM-SR's oscillator tasks PySR reaches Acc_0.001 of 98.53% on Oscillator 2 but 10.29% on Oscillator 1, where LLM-SR(GPT-5.1) has 43.16% and 95.64%; uDSR and DSR are far behind there — [STRIDE Table 1](https://arxiv.org/abs/2605.17790)

Noise and few points
- NeSymReS: no noise experiment at all; the only mention is that a validation set "might offer better performances with noisy data"; accuracy against number of points (1 to 1024) is a curve only, described as growing with the number of points — [NeSymReS footnote 4, Sec. 5(iii), Fig. 5](https://arxiv.org/abs/2106.06427)
- E2E, in-domain: accuracy rises with the number of points between 50 and 200, more strongly than for a skeleton model; under multiplicative noise y(1+ξ) the end-to-end model without refinement "is not robust to noise, and actually performs worse than the skeleton model at high noise" — [E2E Sec. 3.1, Fig. 4D, 4F](https://arxiv.org/abs/2204.10532)
- TPSR: "considerably better low-resource performance for N < 100" than E2E and "slightly better" under multiplicative noise; all models deteriorate with noise — [TPSR App. D.5, Fig. 15(d), 15(f), Fig. 6(b)](https://arxiv.org/abs/2303.06833)
- A NeSymReS-type model conditioned on side information (allowed sub-expressions, symmetries, complexity, constants) loses less accuracy than the unconditioned one as noise grows and as the number of points falls; numbers are in figures only — [NSRwH Sec. 3.3.1, 3.3.2, Fig. 4, 5](https://arxiv.org/abs/2304.10336)
- LLM-SR against PySR on Oscillation 2 with Gaussian noise σ in {0, 0.01, 0.05, 0.1}: both degrade, LLM-SR stays lower in NMSE, mostly out-of-domain; figure only — [LLM-SR App. F.1, Fig. 18](https://arxiv.org/abs/2404.18400)
- SR-Scientist under Gaussian noise σ in {0.01, 0.05, 0.1}: all methods drop; stated limitation: "in noisy scenarios, although SR-Scientist performs best, it still suffers from a significant performance drop" — [SR-Scientist Sec. 4.2, Sec. 5, Fig. 3](https://arxiv.org/abs/2510.11661)
- Ten two-equation ODE/PDE systems, symbolic accuracy clean / noisy: PySR 40 / 15%, LLM-SR 30 / 10%, KeplerAgent 75 / 45% — [KeplerAgent Table 2](https://arxiv.org/abs/2602.12259)
- With GPT-4 proposing forms for small experimental data sets, "including noisy data in the context sometimes undermined the search" — [Sharlin and Josephson Sec. 4](https://arxiv.org/abs/2410.17448)
- Small real data sets (the "phenomenological and first-principles" track): for almost every problem no method found an expression that dominates the accepted one; only ITEA retrieved one first-principles equation (Leavitt); with noisy data a higher R² from a more complex model "often indicates that the model is fitting the noise" — [SRBench 2025 Sec. 5.4, Table 3](https://arxiv.org/abs/2505.03977)

### Inferences
- R² > 0.99 accuracy and exact recovery rank the transformer methods very differently: E2E is "fourth" on the first and 0% to 10% on the second. For a goal of recovering structure, the R²-based claims in the E2E and TPSR papers carry little weight; their outputs are long (about 50 tree nodes on Feynman).
- The spread of PySR's symbolic accuracy on one benchmark (8.11% against 37.84%) shows that "language-model method beats PySR" depends on PySR's configuration, run time and on which model judges equivalence (GPT-4o in one paper, GPT-4o-mini in the other). Claims of that kind should be read with the PySR settings next to them.
- None of the comparisons is in the regime of about 100 noisy points with a tens-of-seconds budget. The nearest are ICSR (20 to 100 noise-free points), NSRwH's small-data curves, and SRBench 2025's small real data sets.

### Gaps
- No independent benchmark reports TPSR's exact-recovery rate in text or table (SRBench 2025 has it only in figures).
- No paper compares the pretrained transformers with PySR at equal wall time of 10 to 60 s.
- No table gives accuracy of any of these methods as a function of the number of points below 100 together with noise.

## 3. Language-model methods: what the model contributes, who fits constants, how much is recall

### Takeaway
In every working system the language model only proposes structure (program skeletons, mutated expressions, concepts, or templates) and an ordinary numerical optimiser fits the constants; letting the model produce constants fails. On the Feynman equations much of the success is recall, shown by perplexity, by one-shot recitation and by data-blind prompting; on LLM-SRBench, built to block recall, the best symbolic accuracy is 31.5%, and later papers put symbolic accuracy on its synthetic half at 5% to 8%.

### Cited Findings

Contribution of the model and of the optimiser
- LLM-SR: the model generates "equation skeletons and the core logic of the program"; constants are placeholders fitted by BFGS or Adam. Removing the split (model writes constants too) raises in-domain NMSE on Oscillation 2 from 2.12e-7 to 3.78e-1; removing the iterative loop gives 1.01e-1; removing the problem description gives 4.65e-5 and removing program form 3.22e-5 — [LLM-SR Sec. 2.3, Sec. 3.5, App. F.2 Table 4](https://arxiv.org/abs/2404.18400)
- FunSearch and LMX adapted to the same tasks do worse than PySR, uDSR and DSR (for example Oscillation 1 in-domain NMSE 0.48 and 0.50 against 0.0009 for PySR) — [LLM-SR App. F.2 Table 4](https://arxiv.org/abs/2404.18400)
- LaSR: the model acts in 1% of genetic operations and writes the concept library; PySR does the search and the constant fitting. Removing variable names "results in a substantial performance drop"; hints derived from the Feynman chapter title speed up solving — [LaSR Sec. 3, 4.3, 4.4, Fig. 3](https://arxiv.org/abs/2409.09359)
- ICSR: the model gives functional forms only; non-linear least squares fits coefficients. The 10 seed prompts alone, with no refinement, already reach test R² 0.95 to 0.99 on three of four benchmarks — [ICSR Sec. 4, Table 3](https://arxiv.org/abs/2404.19094)
- GPT-3.5 and GPT-4 asked directly for an expression with coefficients "produced expressions while hallucinating arbitrary coefficients"; when asked to show steps they wrote correct fitting code but reported coefficients that differ from what the code yields — [Sharlin and Josephson Sec. 2.1, Fig. 1](https://arxiv.org/abs/2410.17448)
- SR-Scientist: about 80% of tool calls are equation evaluations and 20% data analysis for Qwen and GLM models; removing the data-analysis tool drops Acc_0.01 from 63.57% to 35.66% with GPT-OSS-120B — [SR-Scientist Sec. 4.3, Table 3, Fig. 6](https://arxiv.org/abs/2510.11661)
- KeplerAgent: the model does not propose the final equation; it runs exploratory analysis, a plot-reading sub-agent and a symmetry detector, then passes PySR a template such as f(x1,x2) + g(x3) or sin(f(x)) and an operator list — [KeplerAgent Sec. 3.3](https://arxiv.org/abs/2602.12259)
- STRIDE fits linear and non-linear parameters separately ("mixed fitting") and has a critic propose repairs from the fitted parameter values — [STRIDE Sec. 1, Limitations](https://arxiv.org/abs/2605.17790)

Evidence of recall
- LLM-SR "rapidly achieves low data-driven errors within few iterations (<20) on Feynman problems, suggesting a recitation of memorized information"; Mixtral's perplexity for the equation given the context is much lower for Feynman problems than for the new ones; one-pass answers "often recite the exact form" (examples I.37.4, I.26.2) — [LLM-SR Sec. 3.1, App. C, Fig. 9, 11 to 13](https://arxiv.org/abs/2404.18400)
- Kepler's law: GPT-4 and GPT-4o associate variable names and the data values themselves with the law and name it in the first iteration; Hubble, Kepler and Bode results were moved to the supplement because of leakage — [Sharlin and Josephson Sec. 3, Sec. 4](https://arxiv.org/abs/2410.17448)
- ICSR's "random guessing" control (60 prompts, no data shown, constants fitted afterwards) reaches test R² 0.960 on Nguyen, 0.971 on Constant, 0.91 on R and 0.77 on Keijzer, against 0.996, 0.9991, 0.996 and 0.981 with data — [ICSR Sec. 5.5, Table 3](https://arxiv.org/abs/2404.19094)
- LaSR's defence: two runs give different algebraic forms for one Feynman equation, and on 41 synthetic equations LaSR reaches test R² 0.913 against 0.070 for PySR; these 41 were selected as the ones on which PySR did badly (MSE above 1 after 400 iterations) — [LaSR Sec. 4.5, Table 3, App. A.4.2, App. A.5.4](https://arxiv.org/abs/2409.09359)
- IdeaSearchFitter's 82.5% on Feynman is obtained with variable names and units given; the paper contains no discussion of memorisation (searched for the terms; none found) — [IdeaSearchFitter Sec. 3.1, Table 1](https://arxiv.org/abs/2510.08317)

What LLM-SRBench does about it, and the rates
- Controls: (i) LSR-Transform re-solves each Feynman equation for another variable with SymPy, keeps complexity inside the original range, and excludes "transformed problems that LLM can solve through direct sampling without requiring access to data"; (ii) LSR-Synth adds invented terms, checked for novelty by GPT-4o and for plausibility by two experts; (iii) a data-blind direct-prompting baseline measures recall — [LLM-SRBench Sec. 2.1, 2.2, 3.1](https://arxiv.org/abs/2504.10415)
- Symbolic accuracy is judged by GPT-4o comparing forms "after removing parameters and constants"; 94.6% agreement with two authors on 130 problems — [LLM-SRBench Sec. 2.3](https://arxiv.org/abs/2504.10415)
- LSR-Transform, symbolic accuracy for Llama-3.1-8B / GPT-3.5 / GPT-4o-mini: data-blind 3.61 / 2.10 / 7.21%; SGA 2.70 / 0.0 / 9.91%; LaSR 5.41 / 12.61 / 6.31%; LLM-SR 30.63 / 10.81 / 31.53%. Acc_0.1: LaSR 45.94 / 47.74 / 50.45%; LLM-SR 38.55 / 10.81 / 39.64% — [LLM-SRBench Table 1](https://arxiv.org/abs/2504.10415)
- LSR-Synth, best symbolic accuracy per domain: chemistry 11.11% (LLM-SR, GPT-4o-mini), biology 25.30% (LLM-SR, Llama), physics 9.91% (LLM-SR and LaSR, GPT-4o-mini), materials 28.12% (LaSR, GPT-4o-mini); data-blind prompting gives 0% everywhere except 4.54% in physics with GPT-4o-mini — [LLM-SRBench Table 1](https://arxiv.org/abs/2504.10415)
- Averaged over methods, LSR-Transform problems are solved much less often than original Feynman problems of the same complexity, also for the simplest ones (0 to 15 nodes) — [LLM-SRBench Sec. 3.3, Fig. 4](https://arxiv.org/abs/2504.10415)
- A later paper judges that LSR-Transform "might still have memorization issues, as nearly 50% of the problems achieved a sufficiently small error after the first iteration", and uses only LSR-Synth — [SR-Scientist footnote 4](https://arxiv.org/abs/2510.11661)

### Inferences
- The pattern across LLM-SR, ICSR, LaSR, SR-Scientist and Sharlin is the same division of labour: language model for structure, classical optimiser for constants, external code for scoring. Nothing in these papers suggests a language model can replace the numerical part.
- Recall inflates Feynman results by a large but unquantified amount: the drop from "low error within a few iterations" on Feynman to 31.5% on re-solved Feynman equations is the best available measure, and the ICSR control shows that even small local models "know" the standard one-input benchmark functions without seeing data.
- Because symbolic accuracy on LLM-SRBench ignores constants and is judged by a language model, it is a looser criterion than SRBench's SymPy-based solution rate; the two are not comparable.

### Gaps
- No paper separates, on one benchmark, the gain from variable names and descriptions from the gain from the model's general prior over formula shapes, beyond single-problem ablations (LLM-SR "w/o Prior", LaSR without variable names).
- The LLM-SRBench HTML lacks the appendix; noise handling and the number of points in LSR-Transform were not found in the parts read.

## 4. Stated limitations

### Takeaway
The pretrained models are bounded by their training distribution (number of variables, number of points, operator set, expression length, constant ranges) and cannot use feedback from evaluating a candidate; the language-model methods are bounded by cost, context length, input dimension and by what the model already knows.

### Cited Findings
- NeSymReS: cannot be used when the number of variables exceeds the pretraining maximum (3); BFGS may fail with several local minima; the network "does not directly interact with the function evaluator", so after a wrong first guess it "cannot adjust its posterior", only sample again; primitives absent from pretraining (x^6, x^y) cannot be produced — [NeSymReS Sec. 6 Limitations, Sec. 4.4, App. A.2](https://arxiv.org/abs/2106.06427)
- E2E: at most 10 variables, at most 200 points per forward pass; time-ordered trajectory data (Strogatz) are out of distribution; "closing the gap [to GP] also remains a challenge"; beam search gives near-duplicate candidates differing only in constants — [E2E Sec. 2.2, Conclusion, App. E](https://arxiv.org/abs/2204.10532)
- TPSR: slower than plain decoding; bound by the backbone's "input dimensionality, expression length, and vocabulary definition"; "still limited" for data far from the pretraining distribution — [TPSR App. E](https://arxiv.org/abs/2303.06833)
- Neural skeleton predictors in general: output complexity is bounded by the expressions seen in pretraining (example: a skeleton with eight operators cannot be found by a model trained with up to seven) — [Univariate skeleton prediction Sec. 5](https://arxiv.org/abs/2406.17834)
- LLM-SR: performance "inherently tied to the quality and breadth of the LLM's training data"; the cost of iterative queries "could be prohibitive for large-scale problems"; at most 10 parameters per equation — [LLM-SR Sec. 5, App. B](https://arxiv.org/abs/2404.18400)
- LaSR: "cannot guarantee that the concepts it learns are correct or insightful"; a concept may help for reasons that are "quirks of the model and data" and mislead a scientist; concepts are not checked for mutual consistency — [LaSR Sec. 6 Limitations](https://arxiv.org/abs/2409.09359)
- ICSR: 8k-token context limits points and history; with too many points in the prompt "the model often continues generating points, especially with two-dimensional functions", hence the cap of 40 points; higher dimensions are untested and expected to be harder; adding plots through a vision-language model (LLaVA-NeXT) left results unchanged (overall R² 0.989 with plots, 0.990 without) — [ICSR Sec. 6.1, App. A.3, Table 4](https://arxiv.org/abs/2404.19094)
- Natural language is "a somewhat clumsy interface for controlling expression length"; GPT-4 could neither measure the complexity of an expression correctly nor produce one of a requested length — [Sharlin and Josephson Sec. 4](https://arxiv.org/abs/2410.17448)
- SR-Scientist: text-only models, no plots; strong drop under noise; only top equations are remembered, so poor ones are re-explored — [SR-Scientist Sec. 5](https://arxiv.org/abs/2510.11661)
- STRIDE: evaluated on controlled benchmarks; noisier real data "may require additional uncertainty estimation and domain validation" — [STRIDE Limitations](https://arxiv.org/abs/2605.17790)
- LLM-SRBench: some transformed equations lack physical meaning; evaluation relies on GPT-4o as judge — [LLM-SRBench Sec. 2.3, appendix discussion, PDF p. 22](https://arxiv.org/pdf/2504.10415)

### Inferences
- Constants are a shared weak point: NeSymReS was trained with at most 3 constants in [1,5] and needs input normalisation in practice ("data normalization is crucial" for both transformers, per [LLM-SR App. A.2](https://arxiv.org/abs/2404.18400)); E2E handles scale by whitening but returns long expressions with an affine constant at every node.

### Gaps
- None of the pretrained-model papers quantifies failure on formulas outside the generator's distribution beyond the SOOSE-FC set (figure only) in NeSymReS.

## 5. Is there a published comparison where a general-purpose agent or chat model is simply given the data?

### Takeaway
Only fragments exist. A chat model given raw data hallucinates coefficients or just writes code that calls a library; with an external fitting loop it recovers simple or famous laws and loses to dedicated search on a hard real data set; a purpose-built code-interpreter agent reaches about 7% symbolic accuracy on LSR-Synth at about $0.25 per problem; and in the one controlled 2026 study of frontier coding agents on an unpublished formula, 4 of 18 single-prompt runs found the complete answer, all four with a correct hint. No paper was found that benchmarks a frontier general-purpose agent, free to use any tools, against PySR on SRBench, Feynman or SRSD data.

### Cited Findings
- ChatGPT-4 given an SRSD Easy data set (checked 6 February 2024): "not a strong baseline as ... it provides Python code to train linear regression models in scikit-learn or SR models in gplearn"; the authors exclude chat services from the benchmark because their training data are unknown — [SRSD Sec. 5.1, footnotes 16 and 17](https://arxiv.org/abs/2206.10540)
- GPT-4 and GPT-4o given the data as text, with constants fitted outside and results fed back: Langmuir's isotherm found in all settings but one; for the dual-site Langmuir data shorter expressions fit almost as well and the true form was sometimes found and then dropped; on Nikuradse's pipe-friction data (over 350 points, of which 10% or 20% were shown to the model) the best expressions had MAE 0.01086 (GPT-4) and 0.00924 (GPT-4o), "at least two to three times worse" than the Bayesian machine scientist's 0.00392; five runs of 153 expressions each cost about $27 in GPT-4 calls, and the authors judge the Bayesian method "far more efficient in terms of compute" — [Sharlin and Josephson Sec. 3.2, 3.3, 3.5, Table 1](https://arxiv.org/abs/2410.17448)
- Llama 3 8B shown 20 to 40 points and asked once for candidate forms (seed prompts only, no loop): test R² 0.95 (Nguyen), 0.982 (Constant), 0.990 (R), 0.93 (Keijzer) with 10 seed prompts; with a single prompt 5% to 13% of runs produced no valid function — [ICSR Table 3](https://arxiv.org/abs/2404.19094)
- Data-blind prompting (description only, no data) solves 2% to 7% of LSR-Transform and essentially none of LSR-Synth symbolically — [LLM-SRBench Table 1](https://arxiv.org/abs/2504.10415)
- A code-interpreter agent (SR-Scientist) on LSR-Synth: Acc_0.01 32% to 64% depending on backbone against 29.46% for PySR and uDSR; symbolic accuracy 7.00% to 7.75% against 4.65% for PySR — [SR-Scientist Tables 1, 2](https://arxiv.org/abs/2510.11661)
- Frontier coding agents on a formula that was unpublished at the time (the "hydrotope", a sum of max(x,0) terms over frequency regions; exact, noise-free values from supplied code): 18 single-prompt runs (Claude Opus 4.8 max and ultra, Codex 5.4 and 5.5 xhigh, DeepSeek v4 pro, Fugu ultra; three prompt conditions): complete formula in 4 runs, all with the true hint; partial sum 4; one region only 7; hint rejected 1; incorrect 2. No complete result without a hint or with a false hint — [How AI agents discover scientific equations Sec. 3.1, 4.1, Table 2](https://arxiv.org/abs/2610.00435)
- Reasons the agent runs stopped short: a failed attempt to combine regions led back to a restricted answer; the formula was tested only on the configurations used to build it; under a false hint "the prompt outweighs the numerical evidence" — [How AI agents discover scientific equations Sec. 4.1](https://arxiv.org/abs/2610.00435)
- On a fixed table of the same data (750 training rows), PySR, gplearn, Operon, Bingo, PS-Tree, LaSR and LLM-SR (with GPT-5.6-Terra), a KAN, and "SciExplorer, an agent that searches a fixed dataset" all kept non-zero error and none recovered the formula; Operon had the lowest median held-out MAE (1.36). A three-agent team that chose its own evaluation points recovered it exactly, using fewer than 100 points. The authors state that compute was not equal and that this is "a task-specific result rather than a general ranking" — [How AI agents discover scientific equations Sec. 3.2, Fig. 3, App. C](https://arxiv.org/abs/2610.00435)
- The authors' contrast: fitted methods "select among [candidates] by average error on sampled data", while an agent can choose a counterexample, test both sides of a boundary and change the kind of formula; but an agent "can follow an unproductive idea or accept a formula after too few tests, so every proposal must be checked" — [How AI agents discover scientific equations Sec. 3.2](https://arxiv.org/abs/2610.00435)

### Inferences
- What the literature shows a plain agent lacks, in its own authors' words, is a reliable acceptance test: runs stop on too few checks, follow the prompt over the numbers, and return one formula without a statement of how well the data decide between alternatives. An algorithm with an exact marginal likelihood for every partial expression supplies exactly that kind of calibrated comparison, and can say where it stopped resolving; no agent result above does so.
- The cost comparison is also clear in these papers: agents spend 10^5 tokens or more and minutes to hours per problem; the cheapest reported figure is $0.10 to $0.25 per problem with open-weight models on a GPU server.
- The agent successes come from exact data and the freedom to query new points; neither holds for 100 fixed noisy points.

### Gaps
- No published head-to-head of a general-purpose frontier agent (with code execution and freedom to install or call SR libraries) against PySR, Operon or uDSR on SRBench ground-truth problems, the Feynman set or SRSD was found; this may exist outside the papers searched.
- No published test of any agent in the regime of about 100 noisy points with 2 to 5 inputs.
- Two further 2025 systems seen only as baselines in other papers (DrSR, SR-LLM) and one fine-tuned-model paper (Symbolic-R1, 2508.09897) were not read.

## 6. What could be reused

### Takeaway
Three things are reusable without network access or training: the published operator priors, the P-UCB planning rule with a policy prior, and small released skeleton models (NeSymReS; a multi-set univariate model) as proposal sources for a one-input function. TPSR never scores a partial expression directly: it completes it with the transformer and scores the completion, which is the step an exact marginal likelihood of a partial expression would replace.

### Cited Findings

How TPSR scores a partial expression
- A tree node is a prefix-notation token sequence, possibly incomplete. To evaluate a newly expanded node the pretrained transformer is called to complete the sequence by beam search of size b (default 1, so greedy); constants of the completed equation are optimised; the node receives the highest reward among the completions; the value is backed up by Q(s,a) ← max(Q(s,a), r) — [TPSR Sec. 3.2 Evaluation and Backpropagation, App. C.1 Algorithm 1](https://arxiv.org/abs/2303.06833)
- Reward of a completed equation: r = 1/(1 + NMSE) + λ·exp(−l/L), with l the length in prefix tokens, L = 200 the maximum length, λ = 0.1 recommended; λ = 0 gives over-long equations that extrapolate worse — [TPSR Sec. 3.3, App. B, Sec. 4.4, App. D.1](https://arxiv.org/abs/2303.06833)
- Without the transformer, standard tree search would evaluate a partial state by a random rollout, which "exhibits significant variance" and often yields invalid equations; that is the stated reason for using the transformer for both expansion and evaluation — [TPSR App. C.2](https://arxiv.org/abs/2303.06833)
- Because greedy completion from a prefix is deterministic, a completed equation is cached and reused for every later state that is a prefix of it — [TPSR Sec. 3.4, App. C.3](https://arxiv.org/abs/2303.06833)
- More rollouts help most; β between 0.1 and 10 works, β = 0 (no exploration) and β = 100 are worse — [TPSR Sec. 4.5, App. D.6, Fig. 7(b), Fig. 16](https://arxiv.org/abs/2303.06833)

A learned score for partial expressions
- MDLformer (31.9M parameters; inputs at most 200 points and 10 variables; trained 41 hours on 8 A100 GPUs on 131M formula-data pairs) estimates, from data (x, y) alone, the length of the shortest formula linking them. In search, a candidate is a tuple of sub-formulas f; the network applied to (f(x), y) estimates how much formula is still missing; candidates with the smallest estimate are kept; the remaining map is finally fitted as a linear combination — [MDLformer Sec. 3.1, 4.1, 4.2, App. C](https://arxiv.org/abs/2411.03753)
- Motivation given: prediction error "does not decrease monotonously as the search approaches the target formula", while remaining description length does — [MDLformer abstract, Sec. 4](https://arxiv.org/abs/2411.03753)
- Result: recovery 33.93% on Feynman and 66.78% on Strogatz averaged over four noise levels, in 660 s and 338 s — [MDLformer Table 1](https://arxiv.org/abs/2411.03753)

Pretrained proposal sources for a function of one input
- NeSymReS covers 1 to 3 variables, runs on one CPU, and returns a beam of skeletons with constants left to an external fitter; weights and code are public — [NeSymReS Sec. 1, 4.3, 5(ii)](https://arxiv.org/abs/2106.06427)
- A Multi-Set Transformer (24.2M parameters) is trained for exactly this task: given several curves of the response against one variable, each with the other variables held fixed at different values, it outputs one univariate skeleton common to all curves. It was trained on 1M pre-generated one-variable expressions with 10 sets of 3,000 points each, constants from U(−10,10), and vocabulary +, ×, /, abs, acos, asin, atan, cos, cosh, exp, log, pow2 to pow5, sin, sinh, sqrt, tan, tanh. The curves are produced by a neural-network surrogate fitted to the data — [Univariate skeleton prediction Sec. 3.2, Sec. 4, Sec. 5](https://arxiv.org/abs/2406.17834)
- In that paper the model's univariate skeletons matched or were equivalent to the target for all variables of all test problems, better than PySR, TaylorGP, NeSymReS and E2E; an instructive failure of E2E was atan returned for a tanh target with low error; the paper leaves merging univariate skeletons into a multivariate formula as future work — [Univariate skeleton prediction Sec. 5, Sec. 6](https://arxiv.org/abs/2406.17834)
- A NeSymReS-type model can be conditioned at test time on known sub-expressions, absent sub-expressions, symmetry, complexity and constants (trained on 200M expressions with up to 5 variables; three RTX 3090 for five days); conditioning raises recovery most with noise and few points — [NSRwH Sec. 1, 2.3, 3.3, App. B](https://arxiv.org/abs/2304.10336)
- A language model as proposer for one-input curves: seed proposals from Llama 3 8B with points as text are already good on textbook functions, but a control without data is nearly as good on the easiest set, and plots as image input gave no gain — [ICSR Table 3, App. A.3](https://arxiv.org/abs/2404.19094)

Priors over operators and structure
- Two published generator priors: NeSymReS's operator weights and leaf probabilities, and E2E's unary weights tuned to Feynman frequencies with division replaced by a unary inverse because division "appears much less frequently than additions and multiplications" — [NeSymReS Table 6](https://arxiv.org/abs/2106.06427); [E2E Sec. 1.1 footnote 2, Table 3](https://arxiv.org/abs/2204.10532)
- Structural restrictions used when PySR is run as a baseline: no nesting of sin or cos inside sin or cos, of exp or log inside exp or log, of sqrt inside sqrt — [SR-Scientist App. A, footnote 10](https://arxiv.org/abs/2510.11661)
- Training-set construction for univariate skeletons forbids certain operator combinations and caps the nesting depth of unary operators — [Univariate skeleton prediction Sec. 3.2, App. B](https://arxiv.org/abs/2406.17834)

Other reusable mechanics
- Predicted constants as starting values make BFGS succeed where random starts fail (in-domain Acc_0.001 0.29 with model initialisation against 0.19 with random initialisation) — [E2E Table 2](https://arxiv.org/abs/2204.10532)
- Residual analysis and summary statistics by code are what the stronger agents use before proposing forms — [SR-Scientist Sec. 4.3, Fig. 6](https://arxiv.org/abs/2510.11661)
- A language model can usefully hand a search engine a template with unresolved sub-functions (f(x1,x2) + g(x3), sin(f(x))) rather than a full formula — [KeplerAgent Sec. 3.3](https://arxiv.org/abs/2602.12259)

### Inferences
- TPSR's lookahead exists to manufacture a score for a partial expression. Where the partial expression already has an exact marginal likelihood, the rollout is unnecessary and the planning rule reduces to: value = best evidence seen below a node, prior = a proposal distribution over refinements, selection by P-UCB. The max-backup and the caching carry over directly. This is an inference; no paper tests it.
- MDLformer is the closest published analogue to scoring a partial expression by what remains unexplained; it does so with a learned point estimate, without uncertainty, and from noise-free training data.
- For naming a one-input function from its curve, the candidates in order of fit to the constraints are: (1) a fixed dictionary with the published operator priors, at no cost; (2) NeSymReS or the Multi-Set Transformer run on CPU on points sampled from the posterior mean of the unknown function, which sidesteps their noise-free training and their appetite for many points, since any number of smooth points can be generated; the multi-set design matches the case where the same unknown function appears under several values of the other inputs; (3) a local language model shown about 40 points as text, which per ICSR mainly returns textbook forms. Option (2) would need a check that released weights load without network access and that inference fits the time budget; neither is stated in the papers.
- Conditioning a skeleton model on a partial structure (NSRwH) is the published form of "propose a completion given what is already fixed", but it needs its own 5-GPU-day pretraining.

### Gaps
- No paper feeds a pretrained skeleton model the posterior mean of a Gaussian-process component; the suggestion above is untested.
- No reported accuracy of NeSymReS or the Multi-Set Transformer on one-input curves sampled with about 100 points or fewer under noise.
- Licence terms and offline availability of the released weights were not checked.

## 7. Do any of these methods keep an unresolved part in the answer, or report that the data do not decide?

### Takeaway
No. Every method read returns a complete closed-form expression (or a Pareto front of complete expressions) chosen by a point score; none returns a formula with a part left as an unknown function, and none reports a probability that the data fail to discriminate between structures. The nearest things are intermediate objects (partial token sequences, sub-formula sets, templates) and remarks by authors that the data did not decide.

### Cited Findings
- NeSymReS and E2E select the single best candidate by in-sample loss or error on the input points — [NeSymReS Sec. 4.3](https://arxiv.org/abs/2106.06427); [E2E Sec. 2.2](https://arxiv.org/abs/2204.10532)
- TPSR's partial sequences exist only inside the search; "completed equations only exist at the terminal nodes" and the output is the completed equation with the highest reward — [TPSR App. C.1, C.2](https://arxiv.org/abs/2303.06833)
- MDLformer's search keeps sets of sub-formulas, but the remainder is always closed by a linear fit and a complete formula is returned — [MDLformer Sec. 4.1, 4.2](https://arxiv.org/abs/2411.03753)
- KeplerAgent's templates with free sub-functions are inputs to PySR, which fills them; the output is a full equation — [KeplerAgent Sec. 3.3](https://arxiv.org/abs/2602.12259)
- The univariate-skeleton method stops at one skeleton per variable and does not give the joint formula; this is a stated limitation, not a report of undecidability — [Univariate skeleton prediction Sec. 5](https://arxiv.org/abs/2406.17834)
- EML trees, when snapping does not give an exact formula, are described as "recovering closed-form elementary subexpressions alongside approximations" — [EML Sec. 5](https://arxiv.org/abs/2603.21852)
- Authors noting that data do not decide: for dual-site Langmuir data the target "does not significantly fit the data much better than many other shorter expressions", producing a plateau on the Pareto front, as also seen with BMS, PySR and a mixed-integer method — [Sharlin and Josephson Sec. 3.3](https://arxiv.org/abs/2410.17448)
- On small noisy real data sets the accepted equations do not reach R² = 1 and more complex, better-fitting results are judged to fit noise; the authors propose using the data's uncertainty information in future — [SRBench 2025 Sec. 5.4](https://arxiv.org/abs/2505.03977)
- Agents sometimes narrow their claim themselves: several runs "restrict their claim to the principal chamber" after failing to find the global formula; others accepted an incomplete formula as complete — [How AI agents discover scientific equations Sec. 4, 4.1](https://arxiv.org/abs/2610.00435)
- LaSR's concept library is an interpretable by-product, with the warning that concepts may be wrong or misleading — [LaSR Sec. 4.4, Sec. 6](https://arxiv.org/abs/2409.09359)

### Inferences
- Returning a partly resolved model with a calibrated statement about the unresolved part appears to be unoccupied ground among pretrained and language-model methods as of the papers read. The closest behaviours are informal (a Pareto plateau noticed by a human; an agent restricting its claim in prose).
- The benchmarks reward this poorly: solution rate and LLM-judged symbolic accuracy count only complete matches, so a correct partial answer scores zero. Any comparison on those benchmarks would need an additional measure.

### Gaps
- Bayesian symbolic regression methods (for example the Bayesian machine scientist used as a baseline by Sharlin and Josephson) were outside this reading list and may report posterior uncertainty over structures; not checked here.
