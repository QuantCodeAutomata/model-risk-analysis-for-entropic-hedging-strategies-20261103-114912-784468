"""Case study: bond-and-swap incomplete market formulas.

Paper: "Model Risk Analysis for Entropic Hedging Strategies", Paul McCloud (McCloud Research), 2026.

Implements the explicit scalar formulas from the CASE STUDY section (pp. 6-9):
  - Eq (41): derivative final price P = η + ε(Q + ρσ_q Z) + (1/2)γ(Q + ρσ_q Z)²
  - Eq (42): underlying-only entropy-adjusted mean, optimal ω* = 0
  - Eq (43-45): P&L decomposition into normal and non-central chi-squared
  - Eq (46): total incremental entropy-adjusted mean
  - Eq (47): penalties χ^γ, χ^σ, χ^s, χ^δ
  - Eq (48): optimal hedge ratio δ*
  - Eq (52): ζ for indifference pricing
  - p* formula derived from Eq (39) with r=0, δ^s·q = 1/σ_b²
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple


# ── Scalar penalty and hedge formulas ─────────────────────────────────────────


def cs_convexity_penalty(al: float, gamma: float, rho: float, sigma_q: float) -> float:
    """Convexity penalty χ^γ (Eq 47).

    χ^γ = (αλγ(1+ρ²)σ_q² - log[1+αλγ(1+ρ²)σ_q²]) / (αλ)²

    For αλ → 0 the limit is (1/2) γ²(1+ρ²)²σ_q⁴.
    Validity: αλγ(1+ρ²)σ_q² > -1.
    """
    u = al * gamma * (1.0 + rho ** 2) * sigma_q ** 2
    if abs(al) < 1e-10:
        # Custom implementation (paper Eq 47, limit αλ→0)
        return 0.5 * (gamma * (1.0 + rho ** 2) * sigma_q ** 2) ** 2
    assert 1.0 + u > 0, f"αλγ(1+ρ²)σ_q² = {u:.4f} ≤ -1; outside domain"
    return float((u - np.log1p(u)) / al ** 2)


def cs_dimensionality_penalty(al: float, epsilon: float, gamma: float,
                               rho: float, sigma_q: float) -> float:
    """Dimensionality penalty χ^σ (Eq 47).

    χ^σ = ε²ρ²σ_q² / (1 + αλγρ²σ_q²)

    Validity: 1 + αλγρ²σ_q² > 0 (stricter condition than the full domain).
    """
    denom = 1.0 + al * gamma * rho ** 2 * sigma_q ** 2
    assert denom > 0, f"1 + αλγρ²σ_q² = {denom:.4f} ≤ 0; outside domain"
    return float(epsilon ** 2 * rho ** 2 * sigma_q ** 2 / denom)


def cs_funding_penalty(sigma_b: float, p: float) -> float:
    """Funding penalty χ^s (Eq 47).

    χ^s = σ_b² p²
    """
    return float(sigma_b ** 2 * p ** 2)


def cs_strategy_penalty(al: float, gamma: float, rho: float, sigma_q: float,
                         delta: float, delta_star: float) -> float:
    """Strategy penalty χ^δ (Eq 47).

    χ^δ = (1 + αλγρ²σ_q²) / (1 + αλγ(1+ρ²)σ_q²) · σ_q² · (δ - δ*)²
    """
    num = 1.0 + al * gamma * rho ** 2 * sigma_q ** 2
    den = 1.0 + al * gamma * (1.0 + rho ** 2) * sigma_q ** 2
    assert den > 0, f"1 + αλγ(1+ρ²)σ_q² = {den:.4f} ≤ 0; outside domain"
    return float(num / den * sigma_q ** 2 * (delta - delta_star) ** 2)


def cs_optimal_hedge_ratio(al: float, epsilon: float, gamma: float,
                            rho: float, sigma_q: float) -> float:
    """Optimal hedge ratio δ* (Eq 48).

    δ* = ε / (1 + αλγρ²σ_q²)
    """
    denom = 1.0 + al * gamma * rho ** 2 * sigma_q ** 2
    assert denom > 0, f"1 + αλγρ²σ_q² = {denom:.4f} ≤ 0; outside domain"
    return float(epsilon / denom)


def cs_zeta(eta: float, gamma: float, rho: float, sigma_q: float,
             chi_gamma: float, chi_sigma: float, al: float) -> float:
    """ζ contribution to indifference price (Eq 52).

    ζ = η + (1/2)γ(1+ρ²)σ_q² - (1/2)αλ(χ^γ + χ^σ)
    """
    return float(eta + 0.5 * gamma * (1.0 + rho ** 2) * sigma_q ** 2
                 - 0.5 * al * (chi_gamma + chi_sigma))


def cs_indifference_price(zeta: float, sigma_b: float, al: float) -> float:
    """Indifference price p* (derived from Eq 39 with r=0, δ^s·q = 1/σ_b²).

    For σ_b > 0:
      p* = (sqrt(1 + 2αλζσ_b²) - 1) / (αλσ_b²)
         = 2ζ / (1 + sqrt(1 + 2αλζσ_b²))   [equivalent forms]

    For σ_b = 0 (no funding): p* = ζ/(1+r) = ζ (since r=0).
    For αλ = 0 (mid case):    p* = ζ (limit of the formula).
    """
    if sigma_b == 0.0:
        return float(zeta)
    if abs(al) < 1e-10:
        # Mid case: p* → ζ as αλ → 0
        return float(zeta)
    disc = 1.0 + 2.0 * al * zeta * sigma_b ** 2
    assert disc >= 0, f"Discriminant {disc:.4f} < 0; αλζσ_b² = {al*zeta*sigma_b**2:.4f} < -1/2"
    return float((np.sqrt(disc) - 1.0) / (al * sigma_b ** 2))


# ── Incremental entropy-adjusted mean ─────────────────────────────────────────


def cs_incremental_eam(eta: float, gamma: float, rho: float, sigma_q: float,
                        p: float, al: float,
                        chi_gamma: float, chi_sigma: float,
                        chi_s: float, chi_delta: float) -> float:
    """Total incremental entropy-adjusted mean per unit notional (Eq 46).

    = η + (1/2)γ(1+ρ²)σ_q² - p - (1/2)αλ(χ^γ + χ^σ + χ^s + χ^δ)
    """
    return float(eta + 0.5 * gamma * (1.0 + rho ** 2) * sigma_q ** 2
                 - p - 0.5 * al * (chi_gamma + chi_sigma + chi_s + chi_delta))


# ── Case study result container ────────────────────────────────────────────────


class CaseStudyResult(NamedTuple):
    """All scalar outputs for the bond-and-swap case study."""
    al: float          # αλ (effective notional × risk aversion)
    chi_gamma: float   # convexity penalty χ^γ
    chi_sigma: float   # dimensionality penalty χ^σ
    chi_s: float       # funding penalty χ^s (at optimal or given p)
    chi_delta: float   # strategy penalty χ^δ (at given δ vs δ*)
    delta_star: float  # optimal hedge ratio δ*
    zeta: float        # ζ (pricing contribution)
    p_star: float      # indifference price p*
    incremental_eam: float  # incremental entropy-adjusted mean at (p*, δ*)


def compute_case_study(sigma_q: float, sigma_b: float,
                        eta: float, epsilon: float, gamma: float,
                        rho: float, alpha: float, lam: float,
                        p: float | None = None,
                        delta: float | None = None) -> CaseStudyResult:
    """Compute full case study outputs for given parameters.

    Args:
        sigma_q, sigma_b: swap and bond volatilities
        eta, epsilon, gamma, rho: derivative coefficients
        alpha, lam: risk aversion and notional
        p:     initial derivative price (None → use p*)
        delta: hedge ratio (None → use δ*)

    Returns:
        CaseStudyResult
    """
    al = alpha * lam

    chi_gamma = cs_convexity_penalty(al, gamma, rho, sigma_q)
    chi_sigma = cs_dimensionality_penalty(al, epsilon, gamma, rho, sigma_q)
    delta_star = cs_optimal_hedge_ratio(al, epsilon, gamma, rho, sigma_q)

    # Compute p* iteratively: χ^s depends on p, but ζ in pricing does not
    # (for the case study χ^s = σ_b²p², and the pricing formula gives p* directly)
    zeta = cs_zeta(eta, gamma, rho, sigma_q, chi_gamma, chi_sigma, al)
    p_star = cs_indifference_price(zeta, sigma_b, al)

    if p is None:
        p_use = p_star
    else:
        p_use = p

    if delta is None:
        delta_use = delta_star
    else:
        delta_use = delta

    chi_s = cs_funding_penalty(sigma_b, p_use)
    chi_delta = cs_strategy_penalty(al, gamma, rho, sigma_q, delta_use, delta_star)
    ieam = cs_incremental_eam(eta, gamma, rho, sigma_q, p_use, al,
                               chi_gamma, chi_sigma, chi_s, chi_delta)

    return CaseStudyResult(
        al=al, chi_gamma=chi_gamma, chi_sigma=chi_sigma,
        chi_s=chi_s, chi_delta=chi_delta,
        delta_star=delta_star, zeta=zeta, p_star=p_star,
        incremental_eam=ieam,
    )


# ── Monte Carlo simulation for case study ─────────────────────────────────────


def simulate_case_study_paths(n_paths: int, sigma_q: float, sigma_b: float,
                               eta: float, epsilon: float, gamma: float,
                               rho: float, rng: np.random.Generator) -> dict:
    """Simulate paths for the bond-and-swap case study.

    Draws Z_b, Z_q, Z ~ N(0,1) independently and returns final prices.
    """
    Z_b = rng.standard_normal(n_paths)
    Z_q = rng.standard_normal(n_paths)
    Z = rng.standard_normal(n_paths)

    B = 1.0 + sigma_b * Z_b                    # bond final price
    Q = sigma_q * Z_q                           # swap final price
    P = (eta + epsilon * (Q + rho * sigma_q * Z)
         + 0.5 * gamma * (Q + rho * sigma_q * Z) ** 2)  # derivative Eq (41)

    return dict(Z_b=Z_b, Z_q=Z_q, Z=Z, B=B, Q=Q, P=P)


def compute_pl_distribution(paths: dict, delta: float, p: float) -> np.ndarray:
    """Compute portfolio P&L = P - δQ - pB for given hedge and initial price."""
    return paths['P'] - delta * paths['Q'] - p * paths['B']
