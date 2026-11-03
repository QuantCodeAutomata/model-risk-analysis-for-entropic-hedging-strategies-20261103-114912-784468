"""Unit tests for src.gaussian_process."""

import numpy as np
import pytest

from src.gaussian_process import (
    GaussianProcessDriver,
    IntervalCoefficients,
    IntervalResult,
    estimate_scaling_powers,
    solve_interval,
    solve_multi_interval,
)


def _brownian_driver(n: int = 5, dim: int = 1, dt: float = 0.1) -> GaussianProcessDriver:
    """Helper: build a Brownian GP (ν_{ij} = min(i,j)*dt)."""
    beta = np.zeros((n, dim))
    nu = np.zeros((n, n, dim, dim))
    for i in range(n):
        for j in range(n):
            nu[i, j] = np.eye(dim) * min(i + 1, j + 1) * dt
    driver = GaussianProcessDriver(n=n, dim=dim, beta=beta, nu=nu)
    driver.build_filtered_cholesky()
    return driver


def _make_coeffs(n: int = 5, dim: int = 1, q0: float = 1.0, dt: float = 0.1):
    """Helper: build uniform IntervalCoefficients."""
    coeffs = []
    for i in range(n):
        coeffs.append(IntervalCoefficients(
            q_prev=np.array([q0 + 0.001 * i]),
            mu_i=np.array([q0 + 0.001 * i + 0.001 * dt]),
            sigma_i=np.array([[0.2]]),
            eta_i=0.05 * dt,
            epsilon_i=np.array([1.0]),
            gamma_i=np.array([[0.5]]),
        ))
    return coeffs


def test_filtered_cholesky_reproduces_covariance() -> None:
    """Reconstruct ν_{ij} = Σ_k ς_{ik}ς_{jk}^† and verify against input."""
    n, dim = 4, 1
    driver = _brownian_driver(n=n, dim=dim)
    for i in range(n):
        for j in range(n):
            reconstructed = np.zeros((dim, dim))
            for k in range(min(i, j) + 1):
                reconstructed += driver.varsigma[i, k] @ driver.varsigma[j, k].T
            np.testing.assert_allclose(
                reconstructed, driver.nu[i, j], atol=1e-10,
                err_msg=f"ν[{i},{j}] mismatch",
            )


def test_simulate_path_shape() -> None:
    """Simulated path has shape (n, dim)."""
    n, dim = 5, 1
    driver = _brownian_driver(n=n, dim=dim)
    rng = np.random.default_rng(0)
    x, z = driver.simulate_path(rng)
    assert x.shape == (n, dim)
    assert z.shape == (n, dim)


def test_simulate_path_mean_near_beta() -> None:
    """Mean of many paths ≈ beta (zero for our test driver)."""
    n, dim = 4, 1
    driver = _brownian_driver(n=n, dim=dim)
    rng = np.random.default_rng(42)
    means = np.zeros((n, dim))
    n_paths = 5000
    for _ in range(n_paths):
        x, _ = driver.simulate_path(rng)
        means += x / n_paths
    # Mean should be close to beta (all zeros)
    np.testing.assert_allclose(means, driver.beta, atol=0.05)


def test_conditional_increment_variance() -> None:
    """Conditional increment ς_{ii}ς_{ii}^† equals the computed conditional covariance."""
    n, dim = 5, 1
    dt = 0.1
    driver = _brownian_driver(n=n, dim=dim, dt=dt)
    z_path = np.zeros((3, dim))  # history of 3 steps
    x_bar, varsigma_ii = driver.conditional_increment(3, z_path)
    cond_cov = varsigma_ii @ varsigma_ii.T
    # For Brownian: ν̄_i = dt * I (each increment has variance dt)
    np.testing.assert_allclose(cond_cov, np.eye(dim) * dt, atol=1e-10)


def test_interval_result_non_negative_penalties() -> None:
    """All interval penalties χ^γ_i, χ^σ_i, χ^s_i ≥ 0."""
    n, dim = 4, 1
    driver = _brownian_driver(n=n, dim=dim)
    coeffs = _make_coeffs(n=n, dim=dim)
    for i in range(n):
        z_path = np.zeros((i, dim))
        x_bar, varsigma_ii = driver.conditional_increment(i, z_path)
        res = solve_interval(coeffs[i], x_bar, varsigma_ii, alpha=1.0, lam=0.5)
        assert res.chi_gamma_i >= -1e-12, f"chi_gamma_i[{i}] = {res.chi_gamma_i}"
        assert res.chi_sigma_i >= -1e-12, f"chi_sigma_i[{i}] = {res.chi_sigma_i}"
        assert res.chi_s_i >= -1e-12, f"chi_s_i[{i}] = {res.chi_s_i}"


def test_multi_interval_p_stars_shape() -> None:
    """output p_stars has shape (n,)."""
    n, dim = 5, 1
    driver = _brownian_driver(n=n, dim=dim)
    coeffs = _make_coeffs(n=n, dim=dim)
    result = solve_multi_interval(driver, coeffs, alpha=1.0, lam=0.5)
    assert result["p_stars"].shape == (n,)
    assert result["penalties_gamma"].shape == (n,)
    assert result["penalties_sigma"].shape == (n,)
    assert result["penalties_s"].shape == (n,)


def test_estimate_scaling_powers_quadratic() -> None:
    """For quantity = dt^2, estimate_scaling_powers returns ≈ 2.0."""
    dt_values = np.logspace(-2, -0.5, 10)
    quantity = 3.0 * dt_values ** 2.0
    slope = estimate_scaling_powers(dt_values, quantity)
    assert abs(slope - 2.0) < 0.05, f"slope = {slope:.4f}, expected ≈ 2.0"


def test_estimate_scaling_powers_linear() -> None:
    """For quantity = dt^1, estimate_scaling_powers returns ≈ 1.0."""
    dt_values = np.logspace(-2, -0.5, 10)
    quantity = 5.0 * dt_values
    slope = estimate_scaling_powers(dt_values, quantity)
    assert abs(slope - 1.0) < 0.05, f"slope = {slope:.4f}, expected ≈ 1.0"


def test_solve_interval_produces_namedtuple() -> None:
    """solve_interval returns an IntervalResult with all expected fields."""
    n, dim = 4, 1
    driver = _brownian_driver(n=n, dim=dim)
    coeffs = _make_coeffs(n=n, dim=dim)
    x_bar, varsigma_ii = driver.conditional_increment(0, np.zeros((0, dim)))
    res = solve_interval(coeffs[0], x_bar, varsigma_ii, alpha=1.0, lam=0.5)
    assert isinstance(res, IntervalResult)
    for field in ("r_i", "phi_i", "delta_sigma_i", "delta_s_i", "s_i",
                  "delta_star_i", "chi_gamma_i", "chi_sigma_i", "chi_s_i",
                  "p_star_i", "x_bar_i", "nu_bar_i"):
        assert hasattr(res, field), f"Missing field: {field}"
