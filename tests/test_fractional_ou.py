"""Unit tests for src.fractional_ou."""

import numpy as np
import pytest

from src.fractional_ou import (
    FractionalOU,
    build_fbm_cholesky,
    build_fbm_covariance_matrix,
    conditional_coefficient_bound,
    fbm_covariance,
    xi_diagonal_bounds,
)


def test_fbm_covariance_H_half() -> None:
    """For H=0.5, f_{1/2}[i,j] = min(i,j)."""
    H = 0.5
    for i in range(1, 6):
        for j in range(1, 6):
            result = fbm_covariance(i, j, H)
            expected = min(i, j)
            assert abs(result - expected) < 1e-12, f"f_{{1/2}}[{i},{j}] = {result} ≠ {expected}"


def test_fbm_covariance_symmetric() -> None:
    """f_H is symmetric: f_H[i,j] = f_H[j,i]."""
    H = 0.3
    for i in range(1, 6):
        for j in range(1, 6):
            assert abs(fbm_covariance(i, j, H) - fbm_covariance(j, i, H)) < 1e-12


def test_fbm_covariance_positive_definite() -> None:
    """n×n fBM covariance matrix is positive semi-definite."""
    for H in [0.2, 0.5, 0.8]:
        mat = build_fbm_covariance_matrix(6, H)
        vals = np.linalg.eigvalsh(mat)
        assert np.all(vals >= -1e-10), f"H={H}: min eigenvalue = {vals.min():.4e}"


def test_fbm_cholesky_lower_triangular() -> None:
    """ξ matrix is lower-triangular (zero above diagonal)."""
    H = 0.4
    xi = build_fbm_cholesky(6, H)
    for i in range(6):
        for j in range(i + 1, 6):
            assert xi[i, j] == 0.0, f"ξ[{i},{j}] = {xi[i,j]} ≠ 0"


def test_fbm_cholesky_reproduces_covariance() -> None:
    """ξ ξ^† = f_H matrix (Cholesky factorisation)."""
    H = 0.4
    n = 6
    xi = build_fbm_cholesky(n, H)
    f_mat = build_fbm_covariance_matrix(n, H)
    reconstructed = xi @ xi.T
    np.testing.assert_allclose(reconstructed, f_mat, atol=1e-10,
                                err_msg=f"Cholesky reconstruction failed for H={H}")


def test_fbm_cholesky_brownian_limit() -> None:
    """For H=0.5, the Cholesky of min(i,j) should give ξ_{ii}=1, ξ_{ij}=1 for j<i."""
    H = 0.5
    n = 4
    xi = build_fbm_cholesky(n, H)
    f_mat = build_fbm_covariance_matrix(n, H)
    reconstructed = xi @ xi.T
    np.testing.assert_allclose(reconstructed, f_mat, atol=1e-10)
    # Diagonal entries: ξ_{11} should be sqrt(f[0,0]) = sqrt(1) = 1
    assert abs(xi[0, 0] - 1.0) < 1e-10


def test_xi_diagonal_bounds_satisfied() -> None:
    """ξ_{ii}^2 satisfies the bounds in Eq 74 for all i>1."""
    for H in [0.2, 0.3, 0.7]:
        lo, hi, xi_diag_sq = xi_diagonal_bounds(H, n=15)
        assert lo >= 0, f"H={H}: lower bound {lo:.4f} < 0"
        assert hi <= 1.0 + 1e-10, f"H={H}: upper bound {hi:.4f} > 1"
        assert np.all(xi_diag_sq >= lo - 1e-6), (
            f"H={H}: ξ_diag² below lower bound: {xi_diag_sq.min():.4f} < {lo:.4f}"
        )
        assert np.all(xi_diag_sq <= hi + 1e-6), (
            f"H={H}: ξ_diag² above upper bound: {xi_diag_sq.max():.4f} > {hi:.4f}"
        )


def test_conditional_coefficient_identity() -> None:
    """Σ_{k<i}(ξ_{ik}-ξ_{i-1,k})^2 + ξ_{ii}^2 = 1 for all i > 1 (Eq 75)."""
    for H in [0.3, 0.5, 0.7]:
        lhs_sum, rhs = conditional_coefficient_bound(H, n=12)
        # lhs_sum = Σ(ξ_{ik}-ξ_{i-1,k})^2, rhs = 1 - ξ_{ii}^2
        # So lhs_sum + ξ_{ii}^2 = 1
        combined = lhs_sum + rhs  # = Σ(...) + (1 - ξ_{ii}^2)
        # Check: lhs_sum = rhs means Σ(Δξ)^2 = 1 - ξ_{ii}^2
        np.testing.assert_allclose(
            lhs_sum, rhs, atol=1e-10,
            err_msg=f"H={H}: conditional coefficient identity failed",
        )


def test_fou_simulate_path_shape() -> None:
    """FOU simulate_path returns arrays of shape (n, dim)."""
    fou = FractionalOU(
        n=5, dt=0.1, H=0.5,
        beta=np.array([0.0]), kappa=np.array([[0.5]]),
        varsigma=np.array([[1.0]]), x0=np.array([0.0]),
    )
    rng = np.random.default_rng(0)
    x, z = fou.simulate_path(rng)
    assert x.shape == (5, 1)
    assert z.shape == (5, 1)


def test_fou_conditional_increment_shape() -> None:
    """FOU conditional_increment returns correct shapes."""
    n, dim = 8, 1
    fou = FractionalOU(
        n=n, dt=0.05, H=0.3,
        beta=np.array([0.0]), kappa=np.array([[0.5]]),
        varsigma=np.array([[1.0]]), x0=np.array([0.0]),
    )
    i = 5
    z_path = np.zeros((i, dim))
    x_bar, varsigma_ii = fou.conditional_increment(i, z_path)
    assert x_bar.shape == (dim,), f"x_bar shape {x_bar.shape}"
    assert varsigma_ii.shape == (dim, dim), f"varsigma_ii shape {varsigma_ii.shape}"


def test_fou_conditional_variance_positive_definite() -> None:
    """ν̄_i = ς_{ii}ς_{ii}^† is positive semi-definite."""
    for H in [0.2, 0.5, 0.8]:
        fou = FractionalOU(
            n=8, dt=0.05, H=H,
            beta=np.array([0.0]), kappa=np.array([[0.5]]),
            varsigma=np.array([[1.0]]), x0=np.array([0.0]),
        )
        z_path = np.zeros((5, 1))
        _, varsigma_ii = fou.conditional_increment(5, z_path)
        cond_cov = varsigma_ii @ varsigma_ii.T
        vals = np.linalg.eigvalsh(cond_cov)
        assert np.all(vals >= -1e-12), f"H={H}: conditional covariance not PSD, min val={vals.min()}"


def test_fou_brownian_matches_expected_variance() -> None:
    """For H=0.5, ξ_{ii}=1 so conditional variance = ςς† dt^{2H} = ςς† dt."""
    H = 0.5
    dt = 0.1
    varsigma_val = 1.0
    n = 8
    fou = FractionalOU(
        n=n, dt=dt, H=H,
        beta=np.array([0.0]), kappa=np.array([[0.5]]),
        varsigma=np.array([[varsigma_val]]), x0=np.array([0.0]),
    )
    # At i=1 (first interval, i_target=1 in 0-indexed → paper i=2):
    z_path = np.zeros((1, 1))
    _, varsigma_ii = fou.conditional_increment(1, z_path)
    cond_cov = varsigma_ii @ varsigma_ii.T
    # Expected: ξ_{22}^2 * varsigma^2 * dt = 1 * 1 * dt (for H=0.5 Brownian, ξ_{22}^2=1)
    xi = build_fbm_cholesky(n, H)
    xi_22_sq = xi[1, 1] ** 2
    expected_cov = xi_22_sq * varsigma_val ** 2 * dt ** (2 * H)
    assert abs(float(cond_cov[0, 0]) - expected_cov) < 1e-10, (
        f"cond_cov = {float(cond_cov[0,0]):.6f}, expected {expected_cov:.6f}"
    )
