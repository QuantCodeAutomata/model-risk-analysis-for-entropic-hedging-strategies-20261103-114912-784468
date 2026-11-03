"""Multi-interval Gaussian-process entropic hedging.

Paper: "Model Risk Analysis for Entropic Hedging Strategies", Paul McCloud (McCloud Research), 2026.

Implements the GAUSSIAN PROCESSES section:
  - Filtered Cholesky representation of the Gaussian process (Eq 54-55)
  - Conditional increment mean and covariance (derived from Eq 54-55)
  - Intervalwise entropic hedging by translating single-interval formulas (Eq 57-63)
  - Asymptotic scaling analysis

The key insight: at each interval i, the conditional distribution of dx_i given
the prior path is N(x̄_i, ν̄_i) where x̄_i and ν̄_i are derived from the
filtered Cholesky coefficients ς_{ik}. The single-interval formulas apply
with effective parameters that incorporate this conditional distribution.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
from dataclasses import dataclass, field
from typing import NamedTuple

from src.entropic_pricing import (
    SingleIntervalModel, calibrate, compute_adjusted_coefficients,
    compute_hedge, compute_indifference_price, compute_strategy_penalty,
)


# ── Gaussian process driver ────────────────────────────────────────────────────


@dataclass
class GaussianProcessDriver:
    """Gaussian process on a hedging schedule with filtered Cholesky representation.

    The process x_1,...,x_n has joint Gaussian distribution with means β_i and
    covariance blocks ν_{ij}.  It is represented as:
      x_i = β_i + Σ_{k=1}^i ς_{ik} z_k    (Eq 54)
    where z_1,...,z_n are independent standard normal vectors.  The coefficient
    matrices ς_{ik} are built by the filtered Cholesky scheme (Eq 55).
    """

    n: int               # number of hedging intervals
    dim: int             # factor dimension
    beta: np.ndarray     # means β_i, shape (n, dim)
    nu: np.ndarray       # covariance blocks ν_{ij}, shape (n, n, dim, dim)
    # Filled by build_filtered_cholesky():
    varsigma: np.ndarray = field(init=False)  # ς_{ik}, shape (n, n, dim, dim)

    def __post_init__(self) -> None:
        self.beta = np.asarray(self.beta, dtype=float)
        self.nu = np.asarray(self.nu, dtype=float)
        assert self.beta.shape == (self.n, self.dim)
        assert self.nu.shape == (self.n, self.n, self.dim, self.dim)
        self.varsigma = np.zeros((self.n, self.n, self.dim, self.dim))

    def build_filtered_cholesky(self) -> None:
        """Build the filtered Cholesky coefficients ς_{ik} (Eq 55).

        ς_{ij} = (ν_{ij} - Σ_{k=1}^{j-1} ς_{ik} ς_{jk}^†) ς_{jj}^{-†}
        ς_{ii} ς_{ii}^† = ν_{ii} - Σ_{k=1}^{i-1} ς_{ik} ς_{ik}^†
        """
        for i in range(self.n):          # i = 0..n-1 (0-indexed)
            # Conditional covariance ν̄_i = ν_{ii} - Σ_{k<i} ς_{ik}ς_{ik}^†
            cond_cov = self.nu[i, i].copy()
            for k in range(i):
                cond_cov -= self.varsigma[i, k] @ self.varsigma[i, k].T

            # ς_{ii}: Cholesky factor of conditional covariance (Custom: Eq 55)
            try:
                L = la.cholesky(cond_cov, lower=True)
            except la.LinAlgError:
                # Near-singular: use symmetric square root via eigh
                vals, vecs = la.eigh(cond_cov)
                vals = np.maximum(vals, 0.0)
                L = vecs @ np.diag(np.sqrt(vals))
            self.varsigma[i, i] = L

            # Off-diagonal: ς_{j,i} for j > i (lower triangle in first two indices)
            for j in range(i + 1, self.n):
                # ς_{ji} = (ν_{ji} - Σ_{k<i} ς_{jk}ς_{ik}^†) ς_{ii}^{-†}
                cross = self.nu[j, i].copy()
                for k in range(i):
                    cross -= self.varsigma[j, k] @ self.varsigma[i, k].T
                # Solve: ς_{ji} L^† = cross  →  ς_{ji} = cross (L^†)^{-1} = cross L^{-T}
                self.varsigma[j, i] = np.linalg.solve(L.T, cross.T).T

    def conditional_increment(self, i: int,
                               z_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conditional mean x̄_i and Cholesky factor ς_{ii} of conditional cov at interval i.

        i:       0-indexed interval index (1..n in paper notation → 0..n-1 here)
        z_path:  realised innovations z_1,...,z_{i-1}, shape (i, dim)

        Conditional distribution: dx_i | path = x̄_i + ς_{ii} z_i
        Conditional covariance:   ν̄_i = ς_{ii} ς_{ii}^†

        Derivation from Eq 54:
          x_i = β_i + Σ_{k≤i} ς_{ik} z_k
          x_{i-1} = β_{i-1} + Σ_{k<i} ς_{i-1,k} z_k
          dx_i = x_i - x_{i-1} = (β_i-β_{i-1}) + ς_{ii}z_i + Σ_{k<i}(ς_{ik}-ς_{i-1,k})z_k
          E[dx_i|past] = (β_i-β_{i-1}) + Σ_{k<i}(ς_{ik}-ς_{i-1,k})z_k
        """
        beta_diff = self.beta[i] - (self.beta[i - 1] if i > 0 else np.zeros(self.dim))
        x_bar = beta_diff.copy()
        for k in range(i):
            # Cholesky coefficients for interval i and j: ς_{i,k} and ς_{i-1,k}
            delta_varsigma = self.varsigma[i, k] - (self.varsigma[i - 1, k] if i > 0 else 0.0)
            x_bar = x_bar + delta_varsigma @ z_path[k]
        return x_bar, self.varsigma[i, i]  # (x̄_i, ς_{ii})

    def simulate_path(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Simulate one Gaussian process path.

        Returns:
            x:      factor values x_1,...,x_n, shape (n, dim)
            z:      innovations z_1,...,z_n, shape (n, dim)
        """
        z = rng.standard_normal((self.n, self.dim))
        x = np.zeros((self.n, self.dim))
        for i in range(self.n):
            x[i] = self.beta[i].copy()
            for k in range(i + 1):
                x[i] += self.varsigma[i, k] @ z[k]
        return x, z


# ── Intervalwise pricing coefficients ─────────────────────────────────────────


@dataclass
class IntervalCoefficients:
    """Pricing expansion coefficients at interval i.

    Underlying: q_i = μ_i + σ_i dx_i  →  initial underlying price q_{i-1}
    Derivative: p_i = η_i + ε_i·dx_i + (1/2)dx_i·γ_i dx_i
    (Eq 57)
    """
    q_prev: np.ndarray   # initial underlying price q_{i-1}, shape (N_m,)
    mu_i: np.ndarray     # underlying mean coefficient, shape (N_m,)
    sigma_i: np.ndarray  # underlying factor loading, shape (N_m, dim)
    eta_i: float         # derivative constant
    epsilon_i: np.ndarray  # derivative linear coefficient, shape (dim,)
    gamma_i: np.ndarray  # derivative quadratic coefficient, shape (dim, dim)


# ── Intervalwise entropic hedging solver ──────────────────────────────────────


class IntervalResult(NamedTuple):
    """Result of the single-interval entropic hedge applied at interval i."""
    r_i: float           # funding rate r_i
    phi_i: np.ndarray    # optimal portfolio direction φ_i
    delta_sigma_i: np.ndarray  # hedging leg δ^σ_i
    delta_s_i: np.ndarray      # funding leg δ^s_i
    s_i: float           # funding spread s_i
    delta_star_i: np.ndarray   # optimal hedge δ*_i
    chi_gamma_i: float   # convexity penalty χ^γ_i
    chi_sigma_i: float   # dimensionality penalty χ^σ_i
    chi_s_i: float       # funding penalty χ^s_i
    p_star_i: float      # indifference price p^*_{i-1}
    x_bar_i: np.ndarray  # conditional mean of dx_i
    nu_bar_i: np.ndarray # conditional covariance of dx_i


def solve_interval(coeffs: IntervalCoefficients,
                   x_bar_i: np.ndarray,
                   varsigma_ii: np.ndarray,
                   alpha: float, lam: float) -> IntervalResult:
    """Apply single-interval quadratic Gaussian formulas at interval i.

    The conditional distribution dx_i | history = x̄_i + ς_{ii} z_i
    translates to effective parameters for the single-interval solver:
      μ_eff = μ_i + σ_i x̄_i         (shift by conditional mean)
      σ_eff = σ_i ς_{ii}             (factor loading under conditional cov)
      η_eff = η_i + ε_i·x̄_i + (1/2)x̄_i·γ_i x̄_i
      ε_eff = ς_{ii}^† (ε_i + γ_i x̄_i)
      γ_eff = ς_{ii}^† γ_i ς_{ii}
    """
    q_prev = coeffs.q_prev
    mu_i, sigma_i = coeffs.mu_i, coeffs.sigma_i
    eta_i, epsilon_i, gamma_i = coeffs.eta_i, coeffs.epsilon_i, coeffs.gamma_i

    # Effective parameters incorporating conditional distribution
    mu_eff = mu_i + sigma_i @ x_bar_i
    sigma_eff = sigma_i @ varsigma_ii      # shape (N_m, dim)

    eta_eff = float(eta_i
                    + float(epsilon_i @ x_bar_i)
                    + 0.5 * float(x_bar_i @ gamma_i @ x_bar_i))
    epsilon_eff = varsigma_ii.T @ (epsilon_i + gamma_i @ x_bar_i)
    gamma_eff = varsigma_ii.T @ gamma_i @ varsigma_ii

    model_i = SingleIntervalModel(
        q=q_prev, mu=mu_eff, sigma=sigma_eff,
        eta=eta_eff, epsilon=epsilon_eff, gamma=gamma_eff,
        alpha=alpha, lam=lam,
    )

    cal = calibrate(model_i)
    adj = compute_adjusted_coefficients(model_i, cal.phi)
    hedge, _, _ = compute_hedge(model_i, p=0.0)  # probe at p=0 to get δ^σ, δ^s
    pricing = compute_indifference_price(model_i, hedge, adj, cal)

    nu_bar_i = varsigma_ii @ varsigma_ii.T

    return IntervalResult(
        r_i=cal.r, phi_i=cal.phi,
        delta_sigma_i=hedge.delta_sigma, delta_s_i=hedge.delta_s,
        s_i=pricing.s_star, delta_star_i=hedge.delta_sigma + pricing.s_star * hedge.delta_s,
        chi_gamma_i=hedge.chi_gamma, chi_sigma_i=hedge.chi_sigma,
        chi_s_i=pricing.s_star ** 2 * float(hedge.delta_s @ q_prev),
        p_star_i=pricing.p_star,
        x_bar_i=x_bar_i, nu_bar_i=nu_bar_i,
    )


# ── Full multi-interval solver ─────────────────────────────────────────────────


def solve_multi_interval(driver: GaussianProcessDriver,
                          coeff_list: list[IntervalCoefficients],
                          alpha: float, lam: float,
                          rng: np.random.Generator | None = None,
                          n_paths: int = 0) -> dict:
    """Solve entropic hedging over all n intervals along a single path.

    Args:
        driver:     Gaussian process driver with built Cholesky coefficients
        coeff_list: list of n IntervalCoefficients, one per interval
        alpha:      risk aversion
        lam:        derivative notional
        rng:        RNG for simulation (if n_paths > 0)
        n_paths:    number of Monte Carlo paths (0 = analytical only)

    Returns dict with:
        interval_results: list of n IntervalResult
        penalties_gamma, penalties_sigma, penalties_s: arrays shape (n,)
        p_stars: array of indifference prices, shape (n,)
    """
    assert len(coeff_list) == driver.n

    # Solve along a single representative path (all-zero innovations)
    z_path = np.zeros((driver.n, driver.dim))
    results = []
    for i in range(driver.n):
        x_bar_i, varsigma_ii = driver.conditional_increment(i, z_path[:i])
        res = solve_interval(coeff_list[i], x_bar_i, varsigma_ii, alpha, lam)
        results.append(res)

    penalties_gamma = np.array([r.chi_gamma_i for r in results])
    penalties_sigma = np.array([r.chi_sigma_i for r in results])
    penalties_s = np.array([r.chi_s_i for r in results])
    p_stars = np.array([r.p_star_i for r in results])

    return dict(
        interval_results=results,
        penalties_gamma=penalties_gamma,
        penalties_sigma=penalties_sigma,
        penalties_s=penalties_s,
        p_stars=p_stars,
    )


# ── Asymptotic scaling estimation ─────────────────────────────────────────────


def estimate_scaling_powers(dt_values: np.ndarray,
                             quantity_values: np.ndarray) -> float:
    """Estimate asymptotic power n such that quantity = O(dt^n) by log-log regression.

    Args:
        dt_values:       1-D array of dt values (must be positive)
        quantity_values: 1-D array of |quantity| values at each dt

    Returns:
        Estimated power (slope of log-log regression).
    """
    mask = quantity_values > 0
    if mask.sum() < 2:
        return float('nan')
    log_dt = np.log(dt_values[mask])
    log_q = np.log(quantity_values[mask])
    slope = np.polyfit(log_dt, log_q, 1)[0]
    return float(slope)
