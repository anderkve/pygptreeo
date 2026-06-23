# Agentic Kernel Construction for GP Regression

## Core Idea

Standard GP regression with a generic kernel (e.g. RBF) is data-inefficient because the prior is uninformative: the GP must learn all relevant structure from scratch. This project explores a strategy for constructing **problem-specific, highly informative kernels** that exploit prior knowledge about the function being emulated, while retaining the safety of a universal fallback.

## Background: Kernels and Feature Maps

A kernel function k(x, x') is an inner product in a feature space:

```
k(x, x') = φ(x)ᵀ φ(x')
```

where φ: X → ℝᵈ is a feature map. The coordinates of the feature space act as basis functions in which the latent function is expressed:

```
f(x) = w₁φ₁(x) + w₂φ₂(x) + ... + wᵈφᵈ(x)
```

The GP prior is then a distribution over the weights **w**. Designing the feature map is therefore equivalent to choosing the basis in which the GP attempts to represent the function.

A low-dimensional, problem-specific feature map can yield a highly data-efficient GP: if the true function has a compact representation in the chosen basis, very few training points are needed to identify the weights.

## The Additive Kernel Architecture

Kernels can be added: if k₁ and k₂ are valid kernels, so is k₁ + k₂. In terms of feature maps, this corresponds to concatenating the two feature vectors. This motivates the following architecture:

```
k_total(x, x') = α²_specific · k_specific(x, x') + α²_RBF · k_RBF(x, x')
```

where:

- **k_specific** is built from problem-specific basis functions, encoding domain knowledge
- **k_RBF** is a standard RBF kernel, serving as a universal safety net
- **α_specific, α_RBF** are amplitude hyperparameters learned by marginal likelihood optimisation

This combination is:

- **Data-efficient**: k_specific gives the GP a strong head start when the basis is good
- **Safe**: k_RBF ensures the GP can still fit any smooth function, even if k_specific is imperfect
- **Self-diagnosing**: the posterior ratio α_specific / α_RBF indicates how much the structured component is contributing

A multiplicative combination k_specific · k_RBF is also valid and may be appropriate in some settings (locally modulated structure).

## Agentic Kernel Construction

### The Vision

An AI agent receives a description of the function to be emulated, in one or more of these forms:

- **Natural language**: e.g. "total NLO cross section for pp → χ̃₁⁺χ̃₁⁻ as a function of neutralino mass, M₂ and μ"
- **Statistical description**: e.g. "a 4-dimensional log-likelihood, smoothly varying, with at most 3 modes"
- **Code**: e.g. a C++ or Python file implementing the function numerically

The agent then:

1. Draws on pre-existing knowledge, literature search, and/or code inspection
1. Identifies relevant analytic structure (symmetries, threshold behaviour, known scaling laws, factorisation properties, expected smoothness)
1. Proposes a concrete set of basis functions φ₁(x), …, φᵈ(x) with explicit physical/mathematical justification
1. Constructs k_specific and returns k_total = k_specific + k_RBF as the recommended kernel

### Example: Neutralino Pair-Production Cross Section

Given the description "total cross section for pp → χ̃⁺χ̃⁻ as a function of (m_χ, M₂, μ)", the agent might reason as follows:

- Threshold behaviour near √s = 2m_χ → basis function: 1/(m_χ - m_threshold) or sqrt(1 - 4m_χ²/s)
- Phase space scaling → basis function: m_χ⁻²
- Mixing angle dependence from gauge couplings → basis function: sin²(2θ_W)
- Logarithmic QCD corrections → basis function: log(m_χ / m_ref)

These become the columns of the feature map φ(x), and k_specific = φ(x)ᵀ φ(x').

### Key Challenges

1. **Basis function construction**: translating qualitative physical insight into concrete, numerically well-conditioned basis functions is non-trivial and requires careful reasoning
1. **Hallucination risk**: a confidently wrong basis function is silently baked into the prior; the k_RBF safety net mitigates but does not eliminate this risk; the agent should flag uncertainty
1. **Code opacity**: numerical integration routines (e.g. Monte Carlo in C++) may obscure the analytic structure of the integrand
1. **Redundancy**: agent-suggested basis functions may be nearly linearly dependent, causing ill-conditioning; pruning or Gram-Schmidt orthogonalisation may be needed
1. **Validation**: a diagnostic pipeline is needed to verify that k_specific is genuinely contributing; leave-one-out cross-validation and the α_specific / α_RBF ratio are natural tools

### Relation to Other Methods

- **Symbolic regression (SR)**: discovers basis functions empirically from pilot data, complementary to the agent's deductive approach; the two can be combined
- **Deep kernel learning**: uses a neural network as feature map, jointly learned with GP hyperparameters; less interpretable but more flexible
- **Automatic kernel discovery** (Duvenaud): searches over kernel compositions in kernel space, without going via explicit basis functions
- **Physics-informed kernels**: encode known symmetries and conservation laws manually — the agent automates this process

-----

## Synthetic Demonstration (Non-Agentic)

### Purpose

Before involving an agent, demonstrate the plausibility and data efficiency gain of the approach on a synthetic function with known structure. The basis functions are chosen manually (simulating perfect agent output), and the comparison is:

1. **k_specific only**: the structured kernel alone, no safety net
1. **k_total = k_specific + k_RBF**: the full recommended architecture
1. **k_RBF only**: the standard baseline

### True Function

A 2-dimensional synthetic function with known analytic structure:

```
f(m, θ) = sin²(2θ) / (m² + 1) + 0.3 · exp(-m)
```

where m ∈ [0, 5] and θ ∈ [0, π/2]. This mimics a cross-section-like function with:

- A mixing-angle dependence via sin²(2θ)
- A mass-dependent suppression via 1/(m² + 1)
- An exponential correction term exp(-m)

Optionally add small Gaussian noise to simulate a noisy simulator.

### Basis Functions (Manually Chosen)

The feature map is:

```
φ(m, θ) = [sin²(2θ),  1/(m² + 1),  exp(-m)]
```

These are exactly the "correct" basis functions for the structured component.

The specific kernel is then:

```
k_specific(x, x') = φ(x)ᵀ φ(x')
```

### Implementation Notes

- Use **scikit-learn** for all GP machinery
- Implement k_specific as a subclass of `sklearn.gaussian_process.kernels.Kernel`
- The combined kernel k_total = k_specific + RBF can use scikit-learn's built-in kernel addition (`k_specific + RBF(...)`)
- Amplitude hyperparameters α_specific and α_RBF are implemented as `ConstantKernel` prefactors: `ConstantKernel() * k_specific + ConstantKernel() * RBF()`
- All hyperparameters (amplitudes, RBF length-scale) are optimised by marginal likelihood via `GaussianProcessRegressor(optimizer='fmin_l_bfgs_b')`

### Benchmark Protocol

For each N_train in [5, 10, 15, 20, 30, 50, 75, 100]:

- Sample N_train training points uniformly at random from the domain
- Fit all three GP variants on the same training set
- Evaluate RMSE and mean log predictive density (MLPD) on a fixed dense test grid (e.g. 50×50 = 2500 points)
- Repeat for n_trials = 20 random training sets and report mean ± std

### Outputs

1. **Learning curve plot**: RMSE vs. N_train for all three kernels (with error bands across trials)
1. **MLPD plot**: mean log predictive density vs. N_train for all three kernels
1. **Amplitude ratio plot**: posterior α_specific / α_RBF vs. N_train for k_total, showing how much the structured component contributes
1. **Predictive surface plot**: for a fixed N_train (e.g. 10), show the predicted mean and std for all three kernels alongside the ground truth

### Expected Results

- k_specific alone should perform well where the basis is correct but may underfit residuals
- k_total should reach low RMSE with substantially fewer training points than k_RBF alone
- k_RBF should eventually catch up at large N_train, but require many more points to do so
- The amplitude ratio plot should show α_specific dominating at small N_train and remaining large whenever the structured component is genuinely useful

### File Structure (suggested)

```
informed_kernel_gp/
├── README.md
├── informed_kernel_gp/
│   ├── __init__.py
│   ├── kernels.py          # DotProductFeatureKernel subclassing sklearn Kernel
│   └── benchmark.py        # benchmark loop and metrics
├── scripts/
│   └── run_synthetic_demo.py   # main script: defines true function, basis, runs benchmark, saves plots
├── tests/
│   └── test_kernels.py
└── requirements.txt
```

### Requirements

```
numpy
scipy
scikit-learn
matplotlib
```
