"""Unit tests for src.case_study."""

import numpy as np
import pytest

from src.case_study import (
    compute_case_study,
    compute_pl_distribution,
    cs_convexity_penalty,
    cs_dimensionality_penalty,
    cs_funding_penalty,
    cs_indifference_price,
    cs_optimal_hedge_ratio,
    cs_strategy_penalty,
    cs_zeta,
    simulate_case_study_paths,
)

# Base parameters (paper case study, p.6-9)
BASE = dict(
    sigma_q=0.2, sigma_b=0.1, eta=0.5, epsilon=1.0,
    gamma=2.0, rho=0.5, alpha=1.0, lam=0.0,
)


def test_base_case_p_star_positive() -> None:
    """p* > 0 with base parameters (mid case lam=0)."""
    result = compute_case_study(**BASE)
    assert result.p_star > 0, f"p* = {result.p_star:.4f}"


def test_delta_star_mid_case() -> None:
    """For lam=0, δ* = ε (since αλγρ²σ_q² = 0; denom = 1)."""
    result = compute_case_study(**BASE)
    assert abs(result.delta_star - BASE["epsilon"]) < 1e-10, (
        f"δ* = {result.delta_star:.6f}, expected ε = {BASE['epsilon']}"
    )


def test_convexity_penalty_zero_when_gamma_zero() -> None:
    """χ^γ = 0 when γ = 0 (Eq 47)."""
    params = {**BASE, "gamma": 0.0}
    result = compute_case_study(**params)
    assert abs(result.chi_gamma) < 1e-12, f"χ^γ = {result.chi_gamma:.2e}"


def test_dimensionality_penalty_zero_when_rho_zero() -> None:
    """χ^σ = 0 when ρ = 0 (Eq 47: ε²ρ²σ_q²/... = 0)."""
    params = {**BASE, "rho": 0.0}
    result = compute_case_study(**params)
    assert abs(result.chi_sigma) < 1e-12, f"χ^σ = {result.chi_sigma:.2e}"


def test_funding_penalty_zero_when_sigmab_zero() -> None:
    """χ^s = 0 when σ_b = 0 (Eq 47: χ^s = σ_b²p*²)."""
    params = {**BASE, "sigma_b": 0.0}
    result = compute_case_study(**params)
    assert abs(result.chi_s) < 1e-12, f"χ^s = {result.chi_s:.2e}"


def test_strategy_penalty_zero_at_optimum() -> None:
    """χ^δ = 0 when δ = δ* (Eq 47)."""
    result = compute_case_study(**BASE)
    chi_d = cs_strategy_penalty(
        result.al, BASE["gamma"], BASE["rho"], BASE["sigma_q"],
        result.delta_star, result.delta_star,
    )
    assert abs(chi_d) < 1e-12, f"χ^δ at δ* = {chi_d:.2e}"


def test_strategy_penalty_positive_off_optimum() -> None:
    """χ^δ > 0 when δ ≠ δ* (Eq 47)."""
    result = compute_case_study(**BASE)
    chi_d = cs_strategy_penalty(
        result.al, BASE["gamma"], BASE["rho"], BASE["sigma_q"],
        result.delta_star + 0.5, result.delta_star,
    )
    assert chi_d > 0, f"χ^δ off-optimum = {chi_d:.4e}"


def test_p_star_changes_with_gamma() -> None:
    """p* changes as γ varies (holding other parameters fixed)."""
    result_lo = compute_case_study(**{**BASE, "gamma": 0.5})
    result_hi = compute_case_study(**{**BASE, "gamma": 5.0})
    assert result_lo.p_star != result_hi.p_star


def test_domain_validity_enforced() -> None:
    """αλγ(1+ρ²)σ_q² ≤ -1 should raise (outside domain)."""
    # with lam>0, gamma<0 large: αλγ(1+ρ²)σ_q² < -1
    params = {**BASE, "lam": 1.0, "gamma": -100.0}
    with pytest.raises((AssertionError, Exception)):
        compute_case_study(**params)


def test_all_penalties_non_negative() -> None:
    """χ^γ, χ^σ, χ^s ≥ 0 (all incompleteness penalties non-negative)."""
    result = compute_case_study(**BASE)
    assert result.chi_gamma >= 0
    assert result.chi_sigma >= 0
    assert result.chi_s >= 0


def test_simulate_paths_shapes() -> None:
    """simulate_case_study_paths returns arrays of correct shape."""
    rng = np.random.default_rng(42)
    paths = simulate_case_study_paths(
        1000, BASE["sigma_q"], BASE["sigma_b"],
        BASE["eta"], BASE["epsilon"], BASE["gamma"], BASE["rho"], rng,
    )
    for key in ("Z_b", "Z_q", "Z", "B", "Q", "P"):
        assert paths[key].shape == (1000,), f"{key} shape {paths[key].shape}"


def test_pl_distribution_mean_near_indifference_price() -> None:
    """At αλ=0 (mid case), E[P - δ*Q - p*B] ≈ 0 under risk-neutral measure."""
    result = compute_case_study(**BASE)
    rng = np.random.default_rng(0)
    paths = simulate_case_study_paths(
        100_000, BASE["sigma_q"], BASE["sigma_b"],
        BASE["eta"], BASE["epsilon"], BASE["gamma"], BASE["rho"], rng,
    )
    pl = compute_pl_distribution(paths, result.delta_star, result.p_star)
    # For lam=0 mid case, E[P - δ*Q - p*B] ≈ 0 (by construction of p*)
    assert abs(pl.mean()) < 0.02, f"E[PL] = {pl.mean():.4f} too large"


def test_convexity_penalty_explicit() -> None:
    """Cross-check cs_convexity_penalty against direct formula."""
    al, gamma, rho, sigma_q = 0.5, 2.0, 0.5, 0.2
    u = al * gamma * (1.0 + rho ** 2) * sigma_q ** 2
    expected = (u - np.log1p(u)) / al ** 2
    result = cs_convexity_penalty(al, gamma, rho, sigma_q)
    assert abs(result - expected) < 1e-12


def test_dimensionality_penalty_explicit() -> None:
    """Cross-check cs_dimensionality_penalty against direct formula."""
    al, epsilon, gamma, rho, sigma_q = 0.5, 1.0, 2.0, 0.5, 0.2
    denom = 1.0 + al * gamma * rho ** 2 * sigma_q ** 2
    expected = epsilon ** 2 * rho ** 2 * sigma_q ** 2 / denom
    result = cs_dimensionality_penalty(al, epsilon, gamma, rho, sigma_q)
    assert abs(result - expected) < 1e-12


def test_mid_case_indifference_price_limit() -> None:
    """For αλ→0, cs_indifference_price converges to ζ."""
    zeta = 0.5
    sigma_b = 0.1
    # At αλ=0, p* = ζ (limit formula)
    p0 = cs_indifference_price(zeta, sigma_b, al=0.0)
    assert abs(p0 - zeta) < 1e-10

    # For small αλ, should be close to ζ
    p_small = cs_indifference_price(zeta, sigma_b, al=1e-6)
    assert abs(p_small - zeta) < 1e-4
