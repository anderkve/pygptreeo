"""Feature suggestion engine for informed-kernel GP.

Given a qualitative, natural-language description of the target function,
this module produces a concrete set of basis functions (and human-readable
rationale) that can be fed directly into the ARDDotProductFeatureKernel.

Design philosophy
-----------------
The user may know very little about the mathematical form of their function
but can often describe *qualitative* properties: dimensionality, smoothness,
number of modes, symmetry, input domain, etc.  From these clues we select
basis functions whose span is likely to overlap the true function's dominant
structure.  Because the kernel uses ARD (automatic relevance determination),
it is safe to be *generous* — irrelevant features will be downweighted
during GP hyperparameter optimisation.

This module hard-codes the reasoning rules (no LLM call required).  A future
version could wrap an LLM to parse free-text descriptions into the structured
``FunctionTraits`` used here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Structured description of the target function
# ---------------------------------------------------------------------------

@dataclass
class FunctionTraits:
    """Structured qualitative description of the target function.

    Parameters
    ----------
    n_dims : int
        Number of input dimensions.
    domain : list of (float, float)
        Bounding box for each input dimension.
    smoothness : str
        One of ``"smooth"``, ``"moderate"``, ``"rough"``.
    n_optima : int or tuple of int
        Number of local optima, or (min, max) range.
    optima_spread : str
        How the optima are distributed: ``"clustered"``, ``"spread"``,
        ``"unknown"``.
    is_log_likelihood : bool
        Whether the function is a log-likelihood (implies concave envelope
        and peaked optima).
    has_symmetry : Optional[str]
        ``"full"``, ``"partial"``, or ``None``.
    extra_notes : str
        Free-text notes for anything not captured above.
    """

    n_dims: int = 2
    domain: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 1.0), (0.0, 1.0)]
    )
    smoothness: str = "smooth"
    n_optima: int | Tuple[int, int] = (3, 4)
    optima_spread: str = "spread"
    is_log_likelihood: bool = False
    has_symmetry: Optional[str] = None
    function_class: Optional[str] = None  # e.g. "decay_chain", None = generic
    extra_notes: str = ""


# ---------------------------------------------------------------------------
# Feature suggestion result
# ---------------------------------------------------------------------------

@dataclass
class SuggestedFeature:
    """One suggested basis function with rationale."""

    name: str
    formula: str
    rationale: str
    category: str          # e.g. "envelope", "oscillatory", "interaction"
    func: Callable         # callable(X) -> (n,)


@dataclass
class FeatureSuggestion:
    """Complete feature suggestion with overall rationale."""

    traits: FunctionTraits
    features: List[SuggestedFeature]
    overall_rationale: str

    def basis_functions(self) -> List[Callable]:
        """Return just the callable basis functions, ready for the kernel."""
        return [f.func for f in self.features]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [self.overall_rationale, ""]
        for i, f in enumerate(self.features, 1):
            lines.append(
                f"  {i:>2}. [{f.category:>12}]  {f.name:30s}  "
                f"{f.formula}"
            )
            lines.append(f"      Rationale: {f.rationale}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: rescale inputs from [lo, hi] to [0, 1]
# ---------------------------------------------------------------------------

def _make_rescaler(
    dim: int, lo: float, hi: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that extracts column *dim* and maps [lo, hi] → [0, 1]."""
    span = hi - lo
    if span <= 0:
        raise ValueError(f"Empty domain for dim {dim}: [{lo}, {hi}]")

    def rescale(X: np.ndarray) -> np.ndarray:
        return (X[:, dim] - lo) / span

    return rescale


# ---------------------------------------------------------------------------
# Core suggestion logic
# ---------------------------------------------------------------------------

def _suggest_decay_chain_features(traits: FunctionTraits) -> FeatureSuggestion:
    """Suggest basis functions informed by decay-chain (Bateman) physics.

    Knowledge used
    --------------
    A radioactive decay chain  A → B → C  with rate constants λ₁, λ₂
    produces concentrations governed by the Bateman equations:

        N_A(t) = N₀ exp(-λ₁ t)
        N_B(t) = N₀ λ₁/(λ₂ - λ₁) [exp(-λ₁ t) - exp(-λ₂ t)]

    When fitting such a model to data, the log-likelihood as a function of
    (λ₁, λ₂) is dominated by:
      • Exponential decays in each rate parameter
      • Products of exponentials (from cross-terms in the Bateman solution)
      • Rational pre-factors like λ₁/(λ₂ - λ₁) — smooth on most of the
        domain but with a ridge near λ₁ ≈ λ₂
      • A log-likelihood envelope that falls off toward extreme parameter
        values

    These features differ substantially from generic Fourier/polynomial
    features and should provide a strong advantage when the true function
    is indeed a decay-chain log-likelihood.
    """
    features: List[SuggestedFeature] = []

    (x_lo, x_hi) = traits.domain[0]
    (y_lo, y_hi) = traits.domain[1]
    rescale_x = _make_rescaler(0, x_lo, x_hi)
    rescale_y = _make_rescaler(1, y_lo, y_hi)

    # ------------------------------------------------------------------
    # 1. Exponential decay features (from Bateman solution structure)
    # ------------------------------------------------------------------
    # The solution involves exp(-λ t) terms.  When fitting λ values,
    # the log-likelihood inherits exponential dependence on the rate
    # parameters.  We include several decay scales.

    for alpha, label in [(1.0, "slow"), (3.0, "medium"), (6.0, "fast")]:
        features.append(SuggestedFeature(
            name=f"exp_decay_x_{label}",
            formula=f"exp(-{alpha} u)  where u = rescaled x",
            rationale=(
                f"Exponential decay in x with rate {alpha}; captures the "
                f"exp(-λ₁ t) structure from the Bateman equations."
            ),
            category="exponential",
            func=lambda X, _r=rescale_x, _a=alpha: np.exp(-_a * _r(X)),
        ))
        features.append(SuggestedFeature(
            name=f"exp_decay_y_{label}",
            formula=f"exp(-{alpha} v)  where v = rescaled y",
            rationale=(
                f"Exponential decay in y with rate {alpha}; captures "
                f"exp(-λ₂ t) structure."
            ),
            category="exponential",
            func=lambda X, _r=rescale_y, _a=alpha: np.exp(-_a * _r(X)),
        ))

    # ------------------------------------------------------------------
    # 2. Exponential interaction features
    # ------------------------------------------------------------------
    # The Bateman solution for the intermediate species involves
    # differences of exponentials: exp(-λ₁ t) - exp(-λ₂ t).
    # In the log-likelihood as a function of (λ₁, λ₂), this creates
    # cross-terms.

    features.append(SuggestedFeature(
        name="exp_product_slow",
        formula="exp(-u) · exp(-v)",
        rationale=(
            "Product of slow exponentials; captures joint decay structure "
            "from cross-terms in the Bateman solution."
        ),
        category="exponential_interaction",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: (
            np.exp(-_rx(X)) * np.exp(-_ry(X))
        ),
    ))
    features.append(SuggestedFeature(
        name="exp_product_fast",
        formula="exp(-3u) · exp(-3v)",
        rationale=(
            "Product of faster exponentials; captures sharper joint decay."
        ),
        category="exponential_interaction",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: (
            np.exp(-3.0 * _rx(X)) * np.exp(-3.0 * _ry(X))
        ),
    ))
    features.append(SuggestedFeature(
        name="exp_diff",
        formula="exp(-3u) - exp(-3v)",
        rationale=(
            "Difference of exponentials mirrors the N_B(t) ∝ "
            "[exp(-λ₁ t) - exp(-λ₂ t)] structure in the Bateman solution."
        ),
        category="exponential_interaction",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: (
            np.exp(-3.0 * _rx(X)) - np.exp(-3.0 * _ry(X))
        ),
    ))

    # ------------------------------------------------------------------
    # 3. Rational / ridge features
    # ------------------------------------------------------------------
    # The Bateman prefactor λ₁/(λ₂ - λ₁) creates a ridge near λ₁ ≈ λ₂.
    # We approximate this structure with smooth features that vary along
    # and across the diagonal.

    features.append(SuggestedFeature(
        name="diagonal_diff",
        formula="u - v",
        rationale=(
            "Linear contrast between the two rate parameters; captures "
            "the sign structure of the λ₁/(λ₂ - λ₁) prefactor across "
            "the λ₁ = λ₂ diagonal."
        ),
        category="rational",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: _rx(X) - _ry(X),
    ))
    features.append(SuggestedFeature(
        name="inv_diff_smooth",
        formula="1 / (|u - v| + 0.1)",
        rationale=(
            "Smoothed reciprocal of the rate difference; approximates the "
            "ridge in the Bateman prefactor λ₁/(λ₂ - λ₁) near the diagonal."
        ),
        category="rational",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: (
            1.0 / (np.abs(_rx(X) - _ry(X)) + 0.1)
        ),
    ))

    # ------------------------------------------------------------------
    # 4. Envelope features (log-likelihood boundary fall-off)
    # ------------------------------------------------------------------
    if traits.is_log_likelihood or traits.smoothness == "smooth":
        features.append(SuggestedFeature(
            name="quadratic_envelope_x",
            formula="u(1-u)",
            rationale=(
                "Concave fall-off toward x-boundaries; log-likelihoods "
                "typically penalise extreme parameter values."
            ),
            category="envelope",
            func=lambda X, _r=rescale_x: _r(X) * (1 - _r(X)),
        ))
        features.append(SuggestedFeature(
            name="quadratic_envelope_y",
            formula="v(1-v)",
            rationale="Concave fall-off toward y-boundaries.",
            category="envelope",
            func=lambda X, _r=rescale_y: _r(X) * (1 - _r(X)),
        ))

    # ------------------------------------------------------------------
    # Build overall rationale
    # ------------------------------------------------------------------
    overall = (
        f"Suggested {len(features)} basis functions informed by "
        f"radioactive decay-chain (Bateman equation) physics.\n"
        f"\n"
        f"Strategy:\n"
        f"  • Exponential decay features at multiple rates capture the\n"
        f"    exp(-λ t) structure fundamental to the Bateman solution.\n"
        f"  • Exponential interaction features (products, differences)\n"
        f"    capture cross-terms from the intermediate-species\n"
        f"    concentration N_B ∝ [exp(-λ₁t) − exp(-λ₂t)].\n"
        f"  • Rational/ridge features approximate the λ₁/(λ₂ − λ₁)\n"
        f"    prefactor that creates a ridge near λ₁ ≈ λ₂.\n"
        f"  • Envelope features capture log-likelihood boundary fall-off.\n"
        f"  • With ARD, irrelevant features will be downweighted\n"
        f"    automatically during GP optimisation."
    )

    return FeatureSuggestion(
        traits=traits,
        features=features,
        overall_rationale=overall,
    )


def _suggest_kepler_orbital_features(traits: FunctionTraits) -> FeatureSuggestion:
    """Suggest basis functions informed by Keplerian orbital mechanics.

    Knowledge used
    --------------
    An exoplanet radial-velocity (RV) model has the form

        v_r(t) = K [cos(ω + ν(t)) + e cos(ω)]

    where K is the RV semi-amplitude, ν is the true anomaly, e the
    eccentricity, and ω the argument of periapsis.  Key relationships:

      • Kepler's third law:  P² ∝ a³   →  K ∝ P^{-1/3}
      • RV semi-amplitude:   K ∝ 1/√(1 − e²)
      • Orbital energy:      E ∝ −1/(2a) ∝ −P^{-2/3}

    When fitting orbital parameters (P, e) to RV data, the log-likelihood
    inherits structure from these power-law and algebraic relationships.
    Features capturing this structure should outperform generic bases.
    """
    features: List[SuggestedFeature] = []

    (x_lo, x_hi) = traits.domain[0]
    (y_lo, y_hi) = traits.domain[1]
    rescale_x = _make_rescaler(0, x_lo, x_hi)
    rescale_y = _make_rescaler(1, y_lo, y_hi)

    # ------------------------------------------------------------------
    # 1. Power-law features (Kepler's third law: P² ∝ a³)
    # ------------------------------------------------------------------
    # The RV signal scales as K ∝ P^{-1/3}, and orbital energy as
    # P^{-2/3}.  We include several power-law indices.  A small offset
    # (0.05) avoids singularities at u = 0.

    features.append(SuggestedFeature(
        name="power_neg_third_x",
        formula="(u + 0.05)^{-1/3}",
        rationale=(
            "Velocity scaling from Kepler's third law: K ∝ P^{-1/3}. "
            "Captures how the RV semi-amplitude decreases with period."
        ),
        category="power_law",
        func=lambda X, _r=rescale_x: (_r(X) + 0.05) ** (-1.0 / 3.0),
    ))
    features.append(SuggestedFeature(
        name="power_two_thirds_x",
        formula="(u + 0.05)^{2/3}",
        rationale=(
            "Semi-major axis scaling: a ∝ P^{2/3}. Captures the "
            "period-to-semimajor-axis relationship."
        ),
        category="power_law",
        func=lambda X, _r=rescale_x: (_r(X) + 0.05) ** (2.0 / 3.0),
    ))
    features.append(SuggestedFeature(
        name="power_neg_one_x",
        formula="(u + 0.05)^{-1}",
        rationale=(
            "Orbital frequency scaling: f = 1/P. Steeper power law "
            "that captures short-period sensitivity."
        ),
        category="power_law",
        func=lambda X, _r=rescale_x: (_r(X) + 0.05) ** (-1.0),
    ))

    # ------------------------------------------------------------------
    # 2. Eccentricity geometry features
    # ------------------------------------------------------------------
    # Orbital geometry introduces factors like √(1−e²) (angular
    # momentum, semi-latus rectum) and 1/√(1−e²) (RV amplitude).

    features.append(SuggestedFeature(
        name="sqrt_one_minus_v_sq",
        formula="√(1 − v²)",
        rationale=(
            "Angular momentum / semi-latus rectum factor: h ∝ √(1−e²). "
            "Captures how circular orbits differ from eccentric ones."
        ),
        category="eccentricity",
        func=lambda X, _r=rescale_y: np.sqrt(1.0 - _r(X) ** 2),
    ))
    features.append(SuggestedFeature(
        name="inv_sqrt_one_minus_v_sq",
        formula="1 / √(1 − 0.81 v²)",
        rationale=(
            "RV semi-amplitude scaling: K ∝ 1/√(1−e²). Regularised "
            "(0.81 = 0.9²) to avoid divergence at v = 1."
        ),
        category="eccentricity",
        func=lambda X, _r=rescale_y: 1.0 / np.sqrt(1.0 - 0.81 * _r(X) ** 2),
    ))
    features.append(SuggestedFeature(
        name="one_minus_v_sq",
        formula="1 − v²",
        rationale=(
            "Quadratic eccentricity factor: (1−e²) appears in the "
            "semi-latus rectum p = a(1−e²) and orbital energy."
        ),
        category="eccentricity",
        func=lambda X, _r=rescale_y: 1.0 - _r(X) ** 2,
    ))

    # ------------------------------------------------------------------
    # 3. Interaction features (K ∝ P^{-1/3} / √(1−e²))
    # ------------------------------------------------------------------
    # The most important composite: the RV semi-amplitude couples
    # period and eccentricity.

    features.append(SuggestedFeature(
        name="rv_amplitude_proxy",
        formula="(u + 0.05)^{-1/3} / √(1 − 0.81 v²)",
        rationale=(
            "Proxy for the RV semi-amplitude K ∝ P^{-1/3}/√(1−e²). "
            "The single most important compound feature for RV fitting."
        ),
        category="interaction",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: (
            (_rx(X) + 0.05) ** (-1.0 / 3.0)
            / np.sqrt(1.0 - 0.81 * _ry(X) ** 2)
        ),
    ))
    features.append(SuggestedFeature(
        name="period_ecc_product",
        formula="(u + 0.05)^{2/3} · (1 − v²)",
        rationale=(
            "Captures the semi-latus rectum p = a(1−e²) ∝ P^{2/3}(1−e²), "
            "which sets the orbital shape."
        ),
        category="interaction",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: (
            (_rx(X) + 0.05) ** (2.0 / 3.0)
            * (1.0 - _ry(X) ** 2)
        ),
    ))

    # ------------------------------------------------------------------
    # 4. Linear and diagonal features
    # ------------------------------------------------------------------
    features.append(SuggestedFeature(
        name="linear_x",
        formula="u",
        rationale="Linear trend in period parameter.",
        category="linear",
        func=lambda X, _r=rescale_x: _r(X),
    ))
    features.append(SuggestedFeature(
        name="linear_y",
        formula="v",
        rationale="Linear trend in eccentricity parameter.",
        category="linear",
        func=lambda X, _r=rescale_y: _r(X),
    ))

    # ------------------------------------------------------------------
    # 5. Envelope features (log-likelihood boundary fall-off)
    # ------------------------------------------------------------------
    if traits.is_log_likelihood or traits.smoothness == "smooth":
        features.append(SuggestedFeature(
            name="quadratic_envelope_x",
            formula="u(1−u)",
            rationale=(
                "Concave fall-off toward period boundaries; "
                "log-likelihoods penalise extreme parameter values."
            ),
            category="envelope",
            func=lambda X, _r=rescale_x: _r(X) * (1 - _r(X)),
        ))
        features.append(SuggestedFeature(
            name="quadratic_envelope_y",
            formula="v(1−v)",
            rationale="Concave fall-off toward eccentricity boundaries.",
            category="envelope",
            func=lambda X, _r=rescale_y: _r(X) * (1 - _r(X)),
        ))

    # ------------------------------------------------------------------
    # Build overall rationale
    # ------------------------------------------------------------------
    overall = (
        f"Suggested {len(features)} basis functions informed by "
        f"Keplerian orbital mechanics.\n"
        f"\n"
        f"Strategy:\n"
        f"  • Power-law features capture Kepler's third law (P² ∝ a³)\n"
        f"    and velocity scalings (K ∝ P^{{-1/3}}).\n"
        f"  • Eccentricity features capture the √(1−e²) and 1/√(1−e²)\n"
        f"    factors from orbital geometry and RV amplitude.\n"
        f"  • Interaction features capture the coupled dependence\n"
        f"    K ∝ P^{{-1/3}}/√(1−e²) of the radial velocity signal.\n"
        f"  • Envelope features capture log-likelihood boundary fall-off.\n"
        f"  • With ARD, irrelevant features will be downweighted\n"
        f"    automatically during GP optimisation."
    )

    return FeatureSuggestion(
        traits=traits,
        features=features,
        overall_rationale=overall,
    )


def suggest_features(traits: FunctionTraits) -> FeatureSuggestion:
    """Suggest basis functions from qualitative function traits.

    Currently supports 2-D inputs.  Higher-dimensional support is planned.
    """
    if traits.n_dims != 2:
        raise NotImplementedError(
            f"Feature suggestion currently supports 2-D inputs, "
            f"got n_dims={traits.n_dims}"
        )

    # Dispatch to specialised suggestion logic if a function class is known
    if traits.function_class == "decay_chain":
        return _suggest_decay_chain_features(traits)
    if traits.function_class == "kepler_orbital":
        return _suggest_kepler_orbital_features(traits)

    features: List[SuggestedFeature] = []

    # Unpack domain
    (x_lo, x_hi) = traits.domain[0]
    (y_lo, y_hi) = traits.domain[1]
    rescale_x = _make_rescaler(0, x_lo, x_hi)
    rescale_y = _make_rescaler(1, y_lo, y_hi)

    # Parse n_optima
    if isinstance(traits.n_optima, tuple):
        n_opt_min, n_opt_max = traits.n_optima
    else:
        n_opt_min = n_opt_max = traits.n_optima

    # ------------------------------------------------------------------
    # 1. Envelope / trend features
    # ------------------------------------------------------------------
    # Log-likelihoods and similar functions typically fall off toward
    # domain boundaries.  Quadratic "bump" functions capture this.
    if traits.is_log_likelihood or traits.smoothness == "smooth":
        # 1-D parabolic humps centred at domain midpoint
        features.append(SuggestedFeature(
            name="quadratic_envelope_x",
            formula="u(1-u)  where u = (x - x_lo) / (x_hi - x_lo)",
            rationale=(
                "Captures concave fall-off toward the x-boundaries, "
                "typical of smooth log-likelihoods."
            ),
            category="envelope",
            func=lambda X, _r=rescale_x: _r(X) * (1 - _r(X)),
        ))
        features.append(SuggestedFeature(
            name="quadratic_envelope_y",
            formula="v(1-v)  where v = (y - y_lo) / (y_hi - y_lo)",
            rationale=(
                "Captures concave fall-off toward the y-boundaries."
            ),
            category="envelope",
            func=lambda X, _r=rescale_y: _r(X) * (1 - _r(X)),
        ))
        features.append(SuggestedFeature(
            name="quadratic_envelope_2d",
            formula="u(1-u) · v(1-v)",
            rationale=(
                "2-D concave hump peaked at domain centre; "
                "captures joint boundary fall-off."
            ),
            category="envelope",
            func=lambda X, _rx=rescale_x, _ry=rescale_y: (
                _rx(X) * (1 - _rx(X)) * _ry(X) * (1 - _ry(X))
            ),
        ))

    # ------------------------------------------------------------------
    # 2. First-harmonic Fourier modes
    # ------------------------------------------------------------------
    # With k optima well-spread in [0,1], the dominant wavelength is
    # roughly 1/ceil(k/2) in each coordinate.  For 3-4 optima in 2-D
    # this means ~1 full cycle per coordinate (frequency ω = 2π).
    features.append(SuggestedFeature(
        name="cos_2pi_x",
        formula="cos(2π u)",
        rationale=(
            "First cosine harmonic in x; one full oscillation across the "
            "domain creates 2 extrema per coordinate — combined with other "
            "terms this yields 3-4 optima in 2-D."
        ),
        category="oscillatory",
        func=lambda X, _r=rescale_x: np.cos(2 * np.pi * _r(X)),
    ))
    features.append(SuggestedFeature(
        name="sin_2pi_x",
        formula="sin(2π u)",
        rationale="Phase-shifted first harmonic in x; allows asymmetric peak placement.",
        category="oscillatory",
        func=lambda X, _r=rescale_x: np.sin(2 * np.pi * _r(X)),
    ))
    features.append(SuggestedFeature(
        name="cos_2pi_y",
        formula="cos(2π v)",
        rationale="First cosine harmonic in y.",
        category="oscillatory",
        func=lambda X, _r=rescale_y: np.cos(2 * np.pi * _r(X)),
    ))
    features.append(SuggestedFeature(
        name="sin_2pi_y",
        formula="sin(2π v)",
        rationale="Phase-shifted first harmonic in y.",
        category="oscillatory",
        func=lambda X, _r=rescale_y: np.sin(2 * np.pi * _r(X)),
    ))

    # ------------------------------------------------------------------
    # 3. Interaction harmonics
    # ------------------------------------------------------------------
    # Optima spread in 2-D (not just along axes) require x-y coupling.
    features.append(SuggestedFeature(
        name="cos_cos_interaction",
        formula="cos(2π u) · cos(2π v)",
        rationale=(
            "Creates a 2×2 grid of peaks; the dominant interaction mode "
            "for well-spread optima."
        ),
        category="interaction",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: (
            np.cos(2 * np.pi * _rx(X)) * np.cos(2 * np.pi * _ry(X))
        ),
    ))
    features.append(SuggestedFeature(
        name="sin_sin_interaction",
        formula="sin(2π u) · sin(2π v)",
        rationale=(
            "Phase-shifted 2-D interaction; together with cos·cos this "
            "allows arbitrary placement of the 2-D peak grid."
        ),
        category="interaction",
        func=lambda X, _rx=rescale_x, _ry=rescale_y: (
            np.sin(2 * np.pi * _rx(X)) * np.sin(2 * np.pi * _ry(X))
        ),
    ))

    # ------------------------------------------------------------------
    # 4. Second harmonics (distinguishing 3 vs 4 optima)
    # ------------------------------------------------------------------
    # When n_optima might be 3 or 4, second harmonics add the asymmetry
    # needed to break from a simple 2×2 grid into 3 peaks.
    if n_opt_max >= 3:
        features.append(SuggestedFeature(
            name="cos_4pi_x",
            formula="cos(4π u)",
            rationale=(
                "Second harmonic in x; enables odd numbers of optima "
                "and asymmetric arrangements."
            ),
            category="oscillatory",
            func=lambda X, _r=rescale_x: np.cos(4 * np.pi * _r(X)),
        ))
        features.append(SuggestedFeature(
            name="cos_4pi_y",
            formula="cos(4π v)",
            rationale="Second harmonic in y; same role as cos_4pi_x.",
            category="oscillatory",
            func=lambda X, _r=rescale_y: np.cos(4 * np.pi * _r(X)),
        ))

    # ------------------------------------------------------------------
    # Build overall rationale
    # ------------------------------------------------------------------
    overall = (
        f"Suggested {len(features)} basis functions for a "
        f"{'log-likelihood' if traits.is_log_likelihood else 'smooth'} "
        f"function on [{x_lo}, {x_hi}] × [{y_lo}, {y_hi}] with "
        f"{n_opt_min}–{n_opt_max} well-spread local optima.\n"
        f"\n"
        f"Strategy:\n"
        f"  • Quadratic envelope terms capture boundary fall-off typical\n"
        f"    of log-likelihoods (concave overall shape).\n"
        f"  • First-harmonic Fourier modes (ω = 2π) provide the 1-cycle-\n"
        f"    per-coordinate oscillations that create multi-modal structure.\n"
        f"  • Interaction harmonics (cos·cos, sin·sin) couple the two\n"
        f"    coordinates so that optima can appear off-axis.\n"
        f"  • Second harmonics (ω = 4π) allow asymmetric peak counts\n"
        f"    (3 vs 4) and non-grid arrangements.\n"
        f"  • With ARD, irrelevant features will be downweighted\n"
        f"    automatically during GP optimisation."
    )

    return FeatureSuggestion(
        traits=traits,
        features=features,
        overall_rationale=overall,
    )


# ---------------------------------------------------------------------------
# Convenience: parse a vague description into FunctionTraits
# ---------------------------------------------------------------------------

def parse_vague_description(description: str) -> FunctionTraits:
    """Rule-based parser for vague natural-language function descriptions.

    This is a simple keyword-based parser.  A future version could use an
    LLM to extract traits from arbitrary free text.

    Parameters
    ----------
    description : str
        Natural-language description of the target function.

    Returns
    -------
    FunctionTraits
        Extracted traits with sensible defaults for anything not mentioned.
    """
    desc = description.lower()
    traits = FunctionTraits()

    # --- Dimensionality ---
    if "two" in desc or "2d" in desc or "two-dimensional" in desc:
        traits.n_dims = 2
    for phrase in ["two input parameters", "two parameters", "two inputs",
                   "2 parameters", "2 inputs"]:
        if phrase in desc:
            traits.n_dims = 2

    # --- Domain ---
    if "[0,1]" in desc or "[0, 1]" in desc:
        traits.domain = [(0.0, 1.0)] * traits.n_dims
    elif "[-1,1]" in desc or "[-1, 1]" in desc:
        traits.domain = [(-1.0, 1.0)] * traits.n_dims

    # --- Smoothness ---
    if "smooth" in desc:
        traits.smoothness = "smooth"
    elif "rough" in desc or "noisy" in desc or "jagged" in desc:
        traits.smoothness = "rough"

    # --- Number of optima ---
    import re
    # Look for patterns like "three or four", "3-4", "3 to 4", etc.
    m = re.search(r"(\d+)\s*(?:or|to|-|–)\s*(\d+)\s*(?:local\s+)?optim", desc)
    if m:
        traits.n_optima = (int(m.group(1)), int(m.group(2)))
    else:
        # Try word forms
        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4,
                       "five": 5, "six": 6, "seven": 7, "eight": 8}
        m = re.search(
            r"(one|two|three|four|five|six|seven|eight)\s+"
            r"(?:or\s+(one|two|three|four|five|six|seven|eight)\s+)?"
            r"(?:local\s+)?optim",
            desc,
        )
        if m:
            lo = word_to_num[m.group(1)]
            hi = word_to_num[m.group(2)] if m.group(2) else lo
            traits.n_optima = (lo, hi) if lo != hi else lo

    # --- Optima spread ---
    if "well-spread" in desc or "well spread" in desc or "spread out" in desc:
        traits.optima_spread = "spread"
    elif "cluster" in desc:
        traits.optima_spread = "clustered"

    # --- Log-likelihood ---
    if "log-likelihood" in desc or "loglikelihood" in desc or "log likelihood" in desc:
        traits.is_log_likelihood = True

    # --- Symmetry ---
    if "symmetric" in desc or "symmetry" in desc:
        traits.has_symmetry = "partial"

    # --- Function class: decay chain ---
    decay_keywords = [
        "decay chain", "radioactive decay", "bateman",
        "decay model", "decay stages", "decay components",
        "exponential decay", "nuclear decay",
    ]
    if any(kw in desc for kw in decay_keywords):
        traits.function_class = "decay_chain"

    # --- Function class: Keplerian / orbital dynamics ---
    kepler_keywords = [
        "orbital", "kepler", "keplerian", "exoplanet",
        "radial velocity", "orbit fitting", "orbit parameter",
        "planetary orbit", "planet orbit", "rv fitting",
        "transit fitting", "semi-major axis", "eccentricity",
    ]
    if any(kw in desc for kw in kepler_keywords):
        traits.function_class = "kepler_orbital"

    traits.extra_notes = description

    return traits
