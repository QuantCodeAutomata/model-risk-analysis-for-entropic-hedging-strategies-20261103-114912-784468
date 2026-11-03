"""Discrete fractional Ornstein-Uhlenbeck process and asymptotic scaling.

Paper: "Model Risk Analysis for Entropic Hedging Strategies", Paul McCloud (McCloud Research), 2026.

Implements the GAUSSIAN PROCESSES — Fractional Ornstein-Uhlenbeck subsection (Eq 67-84):
  - fBM covariance kernel f_H[i,j] (Eq 69)
  - Cholesky coefficients ξ_{ik} (Eq 70-71)
  - FOU factor mean β_i and Cholesky coefficients ς_{ik} (Eq 72)
  - Conditional mean x̄_i and covariance ν̄_i (Eq 73)
  - Diagonal bound on ξ_{ii}^2 (Eq 74-75)
  - Asymptotic powers n_x = H, n_ν = 2H for H ≠ 1/2 (Eq 76)
  - Scaling of r_i, φ_i, p*_{i-1}, χ^γ_i, χ^σ_i, χ^s_i (Eq 77-84)

Note (paper, Eq 80 vicinity): The paper writes ν̄_i = ςζ† dt which appears to
be a typographical inconsistency; from context (Eq 73) it should be
ν̄_i = ξ_{ii}^2 ςς† dt^{2H}, confirmed by the surrounding derivation.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.special as sp
from dataclasses import dataclass, field
from typing import NamedTuple


# ── fBM covariance kernel ──────────────────────────────────────────────────────


def fbm_covariance(i: int, j: int, H: float) -> float:
    """fBM covariance kernel f_H[i,j] (Eq 69).

    f_H[i,j] = (1/2)(i^{2H} + j^{2H} - |i-j|^{2H})
    """
    return 0.5 * (float(i) ** (2.0 * H) + float(j) ** (2.0 * H)
                  - abs(float(i) - float(j)) ** (2.0 * H))


def build_fbm_covariance_matrix(n: int, H: float) -> np.ndarray:
    """Build n×n fBM covariance matrix f_H[i,j] for i,j = 1,...,n."""
    mat = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            mat[i - 1, j - 1] = fbm_covariance(i, j, H)
    return mat


# ── Filtered Cholesky for fBM ──────────────────────────────────────────────────


def build_fbm_cholesky(n: int, H: float) -> np.ndarray:
    """Build n×n lower-triangular Cholesky coefficient matrix ξ (Eq 70-71).

    w_i = Σ_{k=1}^i ξ_{ik} z_k dt^H  (Eq 70)

    Recursion (Eq 71):
      ξ_{ij} = (f_H[i,j] - Σ_{k=1}^{j-1} ξ_{ik}ξ_{jk}) / ξ_{jj}  for j < i
      ξ_{ii}^2 = f_H[i,i] - Σ_{k=1}^{i-1} ξ_{ik}^2

    Returns ξ as a lower-triangular matrix of shape (n, n) with 1-indexing
    stored in 0-indexed array xi[i,k] → ξ_{i+1, k+1}.
    """
    xi = np.zeros((n, n))
    for i in range(n):        # i in 0..n-1 → paper index i+1
        # Diagonal: ξ_{ii}^2 = f_H[i,i] - Σ_{k<i} ξ_{ik}^2
        diag_sq = fbm_covariance(i + 1, i + 1, H) - float(np.sum(xi[i, :i] ** 2))
        diag_sq = max(diag_sq, 0.0)  # numerical safeguard
        xi[i, i] = np.sqrt(diag_sq)
        # Off-diagonal: ξ_{ji} for j > i
        for j in range(i + 1, n):   # j in i+1..n-1 → paper index j+1
            if xi[i, i] == 0.0:
                xi[j, i] = 0.0
            else:
                cross = fbm_covariance(j + 1, i + 1, H) - float(np.dot(xi[j, :i], xi[i, :i]))
                xi[j, i] = cross / xi[i, i]
    return xi


# ── FOU process ───────────────────────────────────────────────────────────────


@dataclass
class FractionalOU:
    """Discrete fractional Ornstein-Uhlenbeck process (Eq 67-73).

    Satisfies: dx_i = κ(β - x_{i-1}) dt + ς dw_i
    where w is a vector of fBMs with Hurst exponent H.
    Scalar version: dim = 1, κ and ς are scalars.
    """

    n: int            # number of intervals
    dt: float         # interval length (uniform schedule)
    H: float          # Hurst exponent (0, 1)
    beta: np.ndarray  # reversion level, shape (dim,)
    kappa: np.ndarray # reversion rate matrix, shape (dim, dim)
    varsigma: np.ndarray  # normal volatility matrix, shape (dim, dim)
    x0: np.ndarray    # initial factor, shape (dim,)

    xi: np.ndarray = field(init=False)        # fBM Cholesky coefficients (n, n)
    beta_i: np.ndarray = field(init=False)    # factor means β_i (n, dim)
    varsigma_ik: np.ndarray = field(init=False)  # ς_{ik} coefficients (n, n, dim, dim)

    def __post_init__(self) -> None:
        self.beta = np.asarray(self.beta, dtype=float)
        self.kappa = np.asarray(self.kappa, dtype=float)
        self.varsigma = np.asarray(self.varsigma, dtype=float)
        self.x0 = np.asarray(self.x0, dtype=float)
        self.dim = self.beta.shape[0]
        self._build()

    def _build(self) -> None:
        """Build fBM Cholesky coefficients and factor distribution parameters."""
        n, dt, H = self.n, self.dt, self.H
        kappa, varsigma, beta, x0 = self.kappa, self.varsigma, self.beta, self.x0

        # fBM Cholesky coefficients ξ_{ik} (Eq 71)
        self.xi = build_fbm_cholesky(n, H)

        # Factor means β_i = β + (I - κdt)^i (x_0 - β) (Eq 72)
        I_minus_kdt = np.eye(self.dim) - kappa * dt
        self.beta_i = np.zeros((n, self.dim))
        power = np.eye(self.dim)
        for i in range(n):
            power_i = np.linalg.matrix_power(I_minus_kdt, i + 1)
            self.beta_i[i] = beta + power_i @ (x0 - beta)

        # Factor Cholesky coefficients ς_{ik} (Eq 72)
        # ς_{ik} = (ξ_{ik} - κdt Σ_{l=k}^{i-1} (I-κdt)^{i-l-1} ξ_{lk}) ς dt^H
        self.varsigma_ik = np.zeros((n, n, self.dim, self.dim))
        I_minus_kdt = np.eye(self.dim) - kappa * dt
        for i in range(n):
            for k in range(i + 1):
                xi_ik = self.xi[i, k]
                # Mean-reversion correction: -κdt Σ_{l=k}^{i-1} (I-κdt)^{i-l-1} ξ_{lk}
                correction = 0.0
                for l in range(k, i):
                    power_il = np.linalg.matrix_power(I_minus_kdt, i - l - 1)
                    correction += power_il * self.xi[l, k]
                coeff_scalar = xi_ik - kappa * dt * correction
                self.varsigma_ik[i, k] = coeff_scalar * varsigma * dt ** H

    def conditional_increment(self, i: int,
                               z_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conditional mean x̄_i and covariance Cholesky factor ς_{ii} at interval i.

        i is 0-indexed (paper: 1-indexed).
        z_path: realised innovations z_1,...,z_{i-1}, shape (i, dim) or (i,) for dim=1.

        Conditional mean and covariance (Eq 73):
          x̄_i = κ(β - x_{i-1}) dt + ς Σ_{k<i} (ξ_{ik} - ξ_{i-1,k}) z_k dt^H
          ν̄_i = ξ_{ii}^2 ςς† dt^{2H}
        """
        dt, H = self.dt, self.H
        kappa, varsigma, beta = self.kappa, self.varsigma, self.beta

        # Reconstruct x_{i-1} from path
        if i == 0:
            x_prev = self.x0.copy()
        else:
            x_prev = self.beta_i[i - 1].copy()
            for k in range(i):
                x_prev += self.varsigma_ik[i - 1, k] @ z_path[k]

        # Conditional mean x̄_i (Eq 73)
        x_bar = kappa @ (beta - x_prev) * dt
        for k in range(i):
            # (ξ_{i,k} - ξ_{i-1,k})
            xi_diff = self.xi[i, k] - (self.xi[i - 1, k] if i > 0 else 0.0)
            x_bar = x_bar + varsigma @ (xi_diff * z_path[k]) * dt ** H

        # Conditional covariance Cholesky: ς_{ii} = ξ_{ii} ς dt^H  (Eq 73)
        varsigma_ii = self.varsigma_ik[i, i]   # already includes ξ_{ii} ς dt^H

        return x_bar, varsigma_ii

    def simulate_path(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Simulate one FOU process path.

        Returns:
            x:  factor values x_1,...,x_n, shape (n, dim)
            z:  innovations z_1,...,z_n, shape (n, dim)
        """
        z = rng.standard_normal((self.n, self.dim))
        x = np.zeros((self.n, self.dim))
        for i in range(self.n):
            x[i] = self.beta_i[i].copy()
            for k in range(i + 1):
                x[i] += self.varsigma_ik[i, k] @ z[k]
        return x, z


# ── Asymptotic bounds verification ────────────────────────────────────────────


def xi_diagonal_bounds(H: float, n: int = 20) -> tuple[float, float, np.ndarray]:
    """Compute theoretical bounds and measured values of ξ_{ii}^2 (Eq 74-75).

    Lower bound: Γ(2H+1)sin(πH) / (2H Γ(H+1/2)²)
    Upper bound: 1 - (2^{2H-1} - 1)²

    Returns (lower_bound, upper_bound, xi_diagonal_squared_values[2..n]).
    """
    lower = (sp.gamma(2.0 * H + 1.0) * np.sin(np.pi * H)
             / (2.0 * H * sp.gamma(H + 0.5) ** 2))
    upper = 1.0 - (2.0 ** (2.0 * H - 1.0) - 1.0) ** 2

    xi = build_fbm_cholesky(n, H)
    xi_diag_sq = xi[1:, :][:, :].diagonal() ** 2  # ξ_{ii}^2 for i = 2..n (0-indexed: 1..n-1)
    # Actually get the diagonal of xi directly
    xi_diag_sq = np.array([xi[i, i] ** 2 for i in range(1, n)])

    return lower, upper, xi_diag_sq


def conditional_coefficient_bound(H: float, n: int = 20) -> np.ndarray:
    """Verify Eq 75: Σ_{k<i} (ξ_{ik} - ξ_{i-1,k})^2 = 1 - ξ_{ii}^2 for i > 1."""
    xi = build_fbm_cholesky(n, H)
    values = np.zeros(n - 1)
    for i in range(1, n):
        total = 0.0
        for k in range(i):
            xi_diff = xi[i, k] - xi[i - 1, k]
            total += xi_diff ** 2
        values[i - 1] = total
    rhs = np.array([1.0 - xi[i, i] ** 2 for i in range(1, n)])
    return values, rhs
