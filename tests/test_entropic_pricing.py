"""Unit tests for src.entropic_pricing."""

import numpy as np
import pytest
import scipy.linalg as la

from src.entropic_pricing import (
    SingleIntervalModel,
    calibrate,
    compute_adjusted_coefficients,
    compute_hedge,
    compute_indifference_price,
    compute_strategy_penalty,
    entropy_adjusted_mean,
    pl_attributions,
    solve_single_interval_entropic_hedge,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_model(**kwargs) -> SingleIntervalModel:
    defaults = dict(
        q=np.array([1.0]),
        mu=np.array([1.05]),
        sigma=np.array([[0.20]]),
        eta=0.5,
        epsilon=np.array([0.30]),
        gamma=np.array([[0.10]]),
        alpha=1.0,
        lam=0.5,
    )
    defaults.update(kwargs)
    return SingleIntervalModel(**defaults)


# ── entropy_adjusted_mean ─────────────────────────────────────────────────────

def test_entropy_adjusted_mean_gamma_zero() -> None:
    """For γ=0, E^[α][X] = η (Eq 9; no quadratic term, ε·(I)^{-1}ε term yields αε²/2 correction,
    but with γ=0: E^[α][X] = η + 0 - (1/2)α(ε·ε + 0) = η - (1/2)α||ε||²."""
    eta, epsilon, alpha = 1.0, np.array([0.5]), 2.0
    gamma = np.array([[0.0]])
    result = entropy_adjusted_mean(eta, epsilon, gamma, alpha)
    expected = eta - 0.5 * alpha * float(np.dot(epsilon, epsilon))
    assert abs(result - expected) < 1e-12, f"{result} ≠ {expected}"


def test_entropy_adjusted_mean_positive_definite_check() -> None:
    """With I + αγ singular or indefinite, must raise."""
    # γ = -1/α makes I + αγ = 0
    alpha = 2.0
    gamma = np.array([[-1.0 / alpha]])
    epsilon = np.array([0.0])
    with pytest.raises(AssertionError):
        entropy_adjusted_mean(0.0, epsilon, gamma, alpha)


def test_entropy_adjusted_mean_scalar_known_value() -> None:
    """Verify against direct formula for scalar (dim=1)."""
    eta, alpha = 1.0, 1.0
    gamma_val = 0.4
    epsilon_val = 0.3
    g = np.array([[gamma_val]])
    e = np.array([epsilon_val])
    # E^[α][X] = η + (1/2)tr[γ] - (1/2)α(ε·(I+αγ)^{-1}ε + tr[(αγ - log(1+αγ))/α²])
    I_plus_ag = 1.0 + alpha * gamma_val
    quad = epsilon_val ** 2 / I_plus_ag
    ag = alpha * gamma_val
    log_term = (ag - np.log1p(ag)) / alpha ** 2
    expected = eta + 0.5 * gamma_val - 0.5 * alpha * (quad + log_term)
    result = entropy_adjusted_mean(eta, e, g, alpha)
    assert abs(result - expected) < 1e-12, f"{result} ≠ {expected}"


# ── calibrate ─────────────────────────────────────────────────────────────────

def test_calibrate_self_funding(small_model_params) -> None:
    """φ·q must vanish (self-funding condition, Eq 18)."""
    model = _make_model(**small_model_params)
    cal = calibrate(model)
    assert abs(float(cal.phi @ model.q)) < 1e-10, f"φ·q = {cal.phi @ model.q:.2e}"


def test_calibrate_funding_rate_positive(small_model_params) -> None:
    """With μ > q (upward drift), r > 0 when q·(σσ†)^{-1}μ > 0."""
    model = _make_model(**small_model_params)
    assert model.mu[0] > model.q[0], "test requires μ > q"
    cal = calibrate(model)
    assert cal.r > 0, f"r = {cal.r:.4f} should be > 0"


def test_calibrate_omega_star_equals_phi_over_alpha(small_model_params) -> None:
    """ω* = φ/α."""
    model = _make_model(**small_model_params)
    cal = calibrate(model)
    assert np.allclose(cal.omega_star, cal.phi / model.alpha)


def test_calibrate_max_eam_positive(small_model_params) -> None:
    """max_eam = (1/2α)||σ†φ||² ≥ 0."""
    model = _make_model(**small_model_params)
    cal = calibrate(model)
    assert cal.max_eam >= 0


# ── compute_hedge ──────────────────────────────────────────────────────────────

def test_hedge_delta_star_self_funding(small_model_params) -> None:
    """δ*·q = p (self-funding condition, below Eq 23)."""
    model = _make_model(**small_model_params)
    p = 0.5
    hedge, adj, cal = compute_hedge(model, p=p)
    assert abs(float(hedge.delta_star @ model.q) - p) < 1e-10


def test_strategy_penalty_zero_at_optimum(small_model_params) -> None:
    """χ^δ = 0 when δ = δ* (Eq 34)."""
    model = _make_model(**small_model_params)
    hedge, adj, cal = compute_hedge(model, p=0.5)
    chi_delta = compute_strategy_penalty(model, hedge.delta_star, hedge.delta_star, adj)
    assert abs(chi_delta) < 1e-12, f"χ^δ at δ* = {chi_delta:.2e}"


def test_strategy_penalty_positive_off_optimum(small_model_params) -> None:
    """χ^δ > 0 when δ ≠ δ* (Eq 34)."""
    model = _make_model(**small_model_params)
    hedge, adj, cal = compute_hedge(model, p=0.5)
    off_opt = hedge.delta_star + np.array([0.1])
    chi_delta = compute_strategy_penalty(model, off_opt, hedge.delta_star, adj)
    assert chi_delta > 0, f"χ^δ off-optimum = {chi_delta:.4e}"


def test_convexity_penalty_zero_when_gamma_zero() -> None:
    """χ^γ = 0 when γ = 0 (Eq 25)."""
    model = _make_model(gamma=np.array([[0.0]]), lam=0.5)
    hedge, adj, cal = compute_hedge(model, p=0.5)
    assert abs(hedge.chi_gamma) < 1e-12, f"χ^γ = {hedge.chi_gamma:.2e} should be 0"


def test_dimensionality_penalty_zero_when_hedgeable() -> None:
    """χ^σ ≈ 0 when ε̌ is in the span of σ̌ (Eq 28)."""
    # With 1 factor and 1 underlying, the hedgeable subspace is full-rank → χ^σ = 0
    model = _make_model(
        q=np.array([1.0]),
        mu=np.array([1.05]),
        sigma=np.array([[0.20]]),
        epsilon=np.array([0.30]),
        gamma=np.array([[0.10]]),
    )
    hedge, adj, cal = compute_hedge(model, p=0.5)
    # 1-factor 1-underlying: check_sigma is 1×1, proj = I, so (1-proj)ε̌ = 0
    assert abs(hedge.chi_sigma) < 1e-10, f"χ^σ = {hedge.chi_sigma:.2e} should be ≈ 0"


def test_all_penalties_non_negative(small_model_params) -> None:
    """χ^γ, χ^σ, χ^s ≥ 0 (paper: all incompleteness penalties are non-negative)."""
    model = _make_model(**small_model_params)
    hedge, adj, cal = compute_hedge(model, p=0.5)
    assert hedge.chi_gamma >= 0, f"χ^γ = {hedge.chi_gamma}"
    assert hedge.chi_sigma >= 0, f"χ^σ = {hedge.chi_sigma}"
    assert hedge.chi_s >= 0, f"χ^s = {hedge.chi_s}"


def test_funding_spread_zero_when_p_equals_delta_sigma_q(small_model_params) -> None:
    """When p = δ^σ·q, the funding spread s = 0 and χ^s = 0 (Eq 23, Eq 31)."""
    model = _make_model(**small_model_params)
    # First get δ^σ
    hedge0, adj0, cal0 = compute_hedge(model, p=0.0)
    p_hedge = float(hedge0.delta_sigma @ model.q)
    hedge, adj, cal = compute_hedge(model, p=p_hedge)
    assert abs(hedge.s) < 1e-10, f"s = {hedge.s:.2e}"
    assert abs(hedge.chi_s) < 1e-10, f"χ^s = {hedge.chi_s:.2e}"


# ── pl_attributions ───────────────────────────────────────────────────────────

def test_pl_decomposition_identity(small_model_params) -> None:
    """Pathwise: portfolio_pl - const ≈ X_γ + X_σ + X_s + X_δ (Eq 24)."""
    model = _make_model(**small_model_params)
    p = 0.5
    hedge, adj, cal = compute_hedge(model, p=p)
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((10_000, model.sigma.shape[1]))
    attrs = pl_attributions(model, Z, hedge, adj, cal, hedge.delta_star)
    const = adj.check_eta - (1.0 + cal.r) * p
    lhs = attrs["portfolio_pl"] - const
    rhs = attrs["X_gamma"] + attrs["X_sigma"] + attrs["X_s"] + attrs["X_delta"]
    residual_std = float(np.std(lhs - rhs))
    assert residual_std < 1e-12, f"P&L decomposition residual std = {residual_std:.2e}"


# ── indifference pricing ───────────────────────────────────────────────────────

def test_indifference_price_mid_case() -> None:
    """For αλ → 0, p* → η̌/(1+r) (first-order limit, Eq 39)."""
    model_tiny = _make_model(lam=1e-8)
    hedge, adj, cal = compute_hedge(model_tiny, p=0.0)
    pricing = compute_indifference_price(model_tiny, hedge, adj, cal)
    mid_expected = adj.check_eta / (1.0 + cal.r)
    assert abs(pricing.p_star - mid_expected) < 1e-4, (
        f"p* = {pricing.p_star:.6f}, mid limit η̌/(1+r) = {mid_expected:.6f}"
    )


def test_solve_returns_complete_dict(small_model_params) -> None:
    """solve_single_interval_entropic_hedge returns all expected keys."""
    result = solve_single_interval_entropic_hedge(**{
        k: v for k, v in small_model_params.items()
    })
    required = ["r", "phi", "delta_star", "chi_gamma", "chi_sigma", "chi_s",
                "s_star", "p_star", "admissible"]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_penalties_two_factor(two_factor_model_params) -> None:
    """With full-rank 2-factor model, all penalties are non-negative."""
    result = solve_single_interval_entropic_hedge(**two_factor_model_params)
    assert result["chi_gamma"] >= 0
    assert result["chi_sigma"] >= 0
    assert result["chi_s"] >= 0
    assert result["admissible"]
