"""Single-interval quadratic Gaussian entropic hedging and pricing.

Paper: "Model Risk Analysis for Entropic Hedging Strategies", Paul McCloud (McCloud Research), 2026.

Implements the core formulas from the QUADRATIC GAUSSIAN PRICING section:
  - entropy-adjusted mean (Eq 9)
  - calibration: funding rate and optimal portfolio (Eq 18)
  - adjusted coefficients for measure/convexity (Eq 21-22)
  - hedge and funding legs, funding spread (Eq 23)
  - risk penalties and P&L attributions (Eq 25, 28, 31, 34)
  - indifference pricing (Eq 38-39)
  - P&L decomposition identity (Eq 24)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import scipy.linalg as la


# ── Model dataclass ────────────────────────────────────────────────────────────


@dataclass
class SingleIntervalModel:
    """Parameters for a single-interval quadratic Gaussian market.

    Underlying final prices: Q = μ + σ Z  (Z ~ N(0, I_{N_f}))
    Derivative final price:  P = η + ε·Z + (1/2) Z·γ Z
    """

    q: np.ndarray       # initial underlying prices, shape (N_m,)
    mu: np.ndarray      # underlying final-price means, shape (N_m,)
    sigma: np.ndarray   # underlying factor loadings, shape (N_m, N_f)
    eta: float          # derivative constant coefficient
    epsilon: np.ndarray # derivative linear factor coefficients, shape (N_f,)
    gamma: np.ndarray   # derivative quadratic factor matrix, shape (N_f, N_f); symmetric
    alpha: float        # investor risk aversion α > 0
    lam: float          # derivative notional λ

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        self.epsilon = np.asarray(self.epsilon, dtype=float)
        self.gamma = np.asarray(self.gamma, dtype=float)
        assert self.sigma.ndim == 2, "sigma must be 2-D"
        assert np.allclose(self.gamma, self.gamma.T), "gamma must be symmetric"


# ── Matrix-function helpers ────────────────────────────────────────────────────


def _sym_func(A: np.ndarray, f) -> np.ndarray:
    """Apply scalar function f to eigenvalues of symmetric matrix A.

    Uses scipy.linalg.eigh for numerical stability (paper uses spectral decomposition).
    """
    vals, vecs = la.eigh(A)
    return vecs @ np.diag(f(vals)) @ vecs.T


def _sym_inv_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric matrix inverse square root A^{-1/2} via eigendecomposition."""
    return _sym_func(A, lambda v: 1.0 / np.sqrt(v))


def _sym_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric matrix square root A^{1/2} via eigendecomposition."""
    return _sym_func(A, np.sqrt)


def _tr_log_term(gamma: np.ndarray, al: float) -> float:
    """Compute tr[(αλγ - log(I+αλγ)) / (αλ)²] for the convexity penalty (Eq 25).

    For αλ → 0 the limit is (1/2) tr[γ²] (second-order expansion).
    """
    if abs(al) < 1e-10:
        # Custom implementation (paper Eq 25, limit αλ→0)
        return 0.5 * float(np.trace(gamma @ gamma))
    vals = la.eigvalsh(gamma)
    al_vals = al * vals
    terms = (al_vals - np.log1p(al_vals)) / (al ** 2)
    return float(np.sum(terms))


# ── Entropy-adjusted mean ──────────────────────────────────────────────────────


def entropy_adjusted_mean(eta: float, epsilon: np.ndarray, gamma: np.ndarray,
                          alpha: float) -> float:
    """Entropy-adjusted mean E^[α][X] for a quadratic Gaussian variable (Eq 9).

    X = η + ε·Z + (1/2) Z·γ Z,  Z ~ N(0, I).

    The formula is:
      E^[α][X] = η + (1/2) tr[γ]
                 - (1/2) α ( ε·(I+αγ)^{-1}ε + tr[(αγ - log(I+αγ))/α²] )

    Valid when all eigenvalues of I+αγ are positive (i.e., eigenvalues of γ > -1/α).
    """
    gamma = np.asarray(gamma, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    n = gamma.shape[0]
    I_plus_ag = np.eye(n) + alpha * gamma
    # Check positive definiteness
    vals = la.eigvalsh(I_plus_ag)
    assert np.all(vals > 0), "I + αγ must be positive definite for the entropy-adjusted mean"

    # ε·(I+αγ)^{-1}ε
    quad_term = float(epsilon @ la.solve(I_plus_ag, epsilon))

    # tr[(αγ - log(I+αγ)) / α²]
    log_term = _tr_log_term(gamma, alpha)

    return float(eta + 0.5 * np.trace(gamma) - 0.5 * alpha * (quad_term + log_term))


# ── Calibration ───────────────────────────────────────────────────────────────


class CalibrationResult(NamedTuple):
    """Calibration output: funding rate, optimal portfolio, max entropy-adjusted mean."""
    r: float          # funding rate r (Eq 18)
    phi: np.ndarray   # optimal portfolio direction φ (Eq 18)
    omega_star: np.ndarray  # optimal self-funded portfolio ω* = φ/α (below Eq 20)
    max_eam: float    # maximum entropy-adjusted mean (1/2α)||σ^†φ||² (Eq 20)


def calibrate(model: SingleIntervalModel) -> CalibrationResult:
    """Calibrate the underlying market: funding rate and optimal portfolio (Eq 18).

    r = [q·(σσ^†)^{-1}(μ-q)] / [q·(σσ^†)^{-1}q]
    φ = (σσ^†)^{-1}(μ-(1+r)q)
    ω* = φ/α (self-funded maximiser of entropy-adjusted mean)
    max_eam = (1/2α)||σ^†φ||²  (Eq 20)
    """
    q, mu, sigma, alpha = model.q, model.mu, model.sigma, model.alpha
    sigma_cov = sigma @ sigma.T   # σσ^†, shape (N_m, N_m)

    # Solve (σσ^†) x = q, (σσ^†) y = (μ-q) for x, y
    x = la.solve(sigma_cov, q)      # (σσ^†)^{-1} q
    y = la.solve(sigma_cov, mu - q) # (σσ^†)^{-1}(μ-q)

    r = float(q @ y / (q @ x))
    phi = la.solve(sigma_cov, mu - (1.0 + r) * q)
    omega_star = phi / alpha

    # Verify self-funding: φ·q = 0
    assert abs(float(phi @ q)) < 1e-10 * (1 + abs(float(phi @ q)) + la.norm(phi)), \
        f"φ·q = {phi @ q:.2e} should be 0 (self-funding violated)"

    max_eam = 0.5 / alpha * float(np.dot(sigma.T @ phi, sigma.T @ phi))
    return CalibrationResult(r=r, phi=phi, omega_star=omega_star, max_eam=max_eam)


# ── Adjusted coefficients ──────────────────────────────────────────────────────


class AdjustedCoefficients(NamedTuple):
    """Coefficients adjusted for measure (Girsanov) and convexity (Eq 21-22)."""
    check_sigma: np.ndarray   # σ̌ = σ(I+αλγ)^{-1/2}, shape (N_m, N_f)
    check_eta: float          # η̌ = η - ε·σ^†φ + (1/2)σ^†φ·γσ^†φ + (1/2)tr[γ]
    check_epsilon: np.ndarray # ε̌ = (I+αλγ)^{-1/2}(ε - γσ^†φ), shape (N_f,)
    proj: np.ndarray          # 1_{σ̌} = σ̌^†(σ̌σ̌^†)^{-1}σ̌, shape (N_f, N_f)
    I_plus_alg: np.ndarray    # I + αλγ, used for domain validation
    I_alg_inv_sqrt: np.ndarray  # (I+αλγ)^{-1/2}


def compute_adjusted_coefficients(model: SingleIntervalModel,
                                   phi: np.ndarray) -> AdjustedCoefficients:
    """Compute measure- and convexity-adjusted coefficients (Eq 21-22).

    check_sigma := σ(I + αλγ)^{-1/2}
    check_eta   := η - ε·σ^†φ + (1/2)σ^†φ·γσ^†φ + (1/2)tr[γ]
    check_eps   := (I + αλγ)^{-1/2}(ε - γσ^†φ)
    proj        := σ̌^†(σ̌σ̌^†)^{-1}σ̌   (projection onto hedgeable subspace, Eq 22)
    """
    sigma, gamma = model.sigma, model.gamma
    eta, epsilon = model.eta, model.epsilon
    alpha, lam = model.alpha, model.lam
    al = alpha * lam

    n_f = sigma.shape[1]
    I_plus_alg = np.eye(n_f) + al * gamma

    # Check positive definiteness of I + αλγ
    vals = la.eigvalsh(I_plus_alg)
    assert np.all(vals > 0), \
        f"I + αλγ must be positive definite; eigenvalues: {vals}"

    I_alg_inv_sqrt = _sym_inv_sqrt(I_plus_alg)

    # σ^†φ
    sigma_t_phi = sigma.T @ phi    # shape (N_f,)

    # check_sigma = σ * (I+αλγ)^{-1/2}
    check_sigma = sigma @ I_alg_inv_sqrt   # shape (N_m, N_f)

    # check_eta = η - ε·σ^†φ + (1/2)σ^†φ·γσ^†φ + (1/2)tr[γ]
    check_eta = (eta
                 - float(epsilon @ sigma_t_phi)
                 + 0.5 * float(sigma_t_phi @ gamma @ sigma_t_phi)
                 + 0.5 * float(np.trace(gamma)))

    # check_epsilon = (I+αλγ)^{-1/2}(ε - γσ^†φ)
    check_epsilon = I_alg_inv_sqrt @ (epsilon - gamma @ sigma_t_phi)

    # Projection 1_{σ̌} = σ̌^†(σ̌σ̌^†)^{-1}σ̌
    check_sig_cov = check_sigma @ check_sigma.T   # shape (N_m, N_m)
    proj = check_sigma.T @ la.solve(check_sig_cov, check_sigma)  # shape (N_f, N_f)

    return AdjustedCoefficients(
        check_sigma=check_sigma,
        check_eta=check_eta,
        check_epsilon=check_epsilon,
        proj=proj,
        I_plus_alg=I_plus_alg,
        I_alg_inv_sqrt=I_alg_inv_sqrt,
    )


# ── Hedge portfolio ────────────────────────────────────────────────────────────


class HedgeResult(NamedTuple):
    """Hedge and funding legs, penalties, and optimal portfolio (Eq 23, 25, 28, 31, 34)."""
    delta_sigma: np.ndarray   # hedging leg δ^σ (Eq 23)
    delta_s: np.ndarray       # funding leg δ^s (Eq 23)
    s: float                  # funding spread s (Eq 23)
    delta_star: np.ndarray    # optimal hedge δ* = δ^σ + s·δ^s (below Eq 23)
    chi_gamma: float          # convexity penalty χ^γ ≥ 0 (Eq 25)
    chi_sigma: float          # dimensionality penalty χ^σ ≥ 0 (Eq 28)
    chi_s: float              # funding penalty χ^s ≥ 0 (Eq 31)


def compute_hedge(model: SingleIntervalModel, p: float,
                  delta: np.ndarray | None = None) -> tuple[HedgeResult, AdjustedCoefficients, CalibrationResult]:
    """Compute hedge legs, funding spread, and risk penalties (Eq 23-31).

    Args:
        model: single-interval model parameters
        p:     derivative initial price
        delta: candidate hedge portfolio (N_m,); if None, uses δ*

    Returns:
        (HedgeResult, AdjustedCoefficients, CalibrationResult)
    """
    cal = calibrate(model)
    adj = compute_adjusted_coefficients(model, cal.phi)

    check_sigma, check_eta, check_epsilon, proj = (
        adj.check_sigma, adj.check_eta, adj.check_epsilon, adj.proj)
    q, r = model.q, cal.r
    al = model.alpha * model.lam

    # Hedging leg: δ^σ = (σ̌σ̌^†)^{-1}σ̌ε̌  (Eq 23)
    check_sig_cov = check_sigma @ check_sigma.T
    delta_sigma = la.solve(check_sig_cov, check_sigma @ check_epsilon)

    # Funding leg: δ^s = (σ̌σ̌^†)^{-1}q  (Eq 23)
    delta_s = la.solve(check_sig_cov, q)

    # Funding spread: s = (p - δ^σ·q) / (δ^s·q)  (Eq 23)
    delta_s_q = float(delta_s @ q)
    s = float((p - float(delta_sigma @ q)) / delta_s_q)

    # Optimal hedge: δ* = δ^σ + s·δ^s  (below Eq 23)
    delta_star = delta_sigma + s * delta_s
    # Self-funding check: δ*·q = p
    assert abs(float(delta_star @ q) - p) < 1e-10 * (1.0 + abs(p)), \
        f"δ*·q = {delta_star @ q:.6f} ≠ p = {p:.6f}"

    # Convexity penalty: χ^γ = tr[(αλγ - log(I+αλγ))/(αλ)²]  (Eq 25)
    chi_gamma = _tr_log_term(model.gamma, al)

    # Dimensionality penalty: χ^σ = ||(1-1_{σ̌})ε̌||²  (Eq 28)
    residual = check_epsilon - proj @ check_epsilon  # (1-1_{σ̌})ε̌
    chi_sigma = float(np.dot(residual, residual))

    # Funding penalty: χ^s = s²δ^s·q  (Eq 31)
    chi_s = s ** 2 * delta_s_q

    return HedgeResult(
        delta_sigma=delta_sigma,
        delta_s=delta_s,
        s=s,
        delta_star=delta_star,
        chi_gamma=chi_gamma,
        chi_sigma=chi_sigma,
        chi_s=chi_s,
    ), adj, cal


def compute_strategy_penalty(model: SingleIntervalModel,
                              delta: np.ndarray,
                              delta_star: np.ndarray,
                              adj: AdjustedCoefficients) -> float:
    """Strategy penalty χ^δ = ||σ̌^†(δ - δ*)||² (Eq 34)."""
    diff = adj.check_sigma.T @ (delta - delta_star)
    return float(np.dot(diff, diff))


# ── P&L attribution terms ──────────────────────────────────────────────────────


def pl_attributions(model: SingleIntervalModel, Z: np.ndarray,
                    hedge: HedgeResult, adj: AdjustedCoefficients,
                    cal: CalibrationResult,
                    delta: np.ndarray) -> dict[str, np.ndarray]:
    """Compute pathwise P&L attribution terms (Eq 25, 28, 31, 34).

    Args:
        Z:     Monte Carlo paths of standard normal factors, shape (N_paths, N_f)
        hedge: HedgeResult from compute_hedge
        adj:   AdjustedCoefficients
        cal:   CalibrationResult
        delta: actual hedge portfolio used (N_m,)

    Returns:
        dict with keys X_gamma, X_sigma, X_s, X_delta, Q, P, portfolio_pl
    """
    sigma, gamma = model.sigma, model.gamma
    epsilon = model.epsilon
    phi, r = cal.phi, cal.r
    delta_sigma, delta_s, s, delta_star = (
        hedge.delta_sigma, hedge.delta_s, hedge.s, hedge.delta_star)
    q = model.q
    al = model.alpha * model.lam

    # shifted factor: Z + σ^†φ
    sigma_t_phi = sigma.T @ phi           # shape (N_f,)
    Z_shifted = Z + sigma_t_phi[None, :]  # shape (N_paths, N_f)

    # Q - (1+r)q, shape (N_paths, N_m)
    Q = model.mu[None, :] + (Z @ sigma.T)          # Q = μ + σZ
    Q_excess = Q - (1.0 + r) * q[None, :]          # Q - (1+r)q

    # P = η + ε·Z + (1/2)Z·γZ
    P = (model.eta
         + Z @ epsilon
         + 0.5 * np.einsum('pi,ij,pj->p', Z, gamma, Z))

    # X^γ = (1/2)((Z+σ^†φ)·γ(Z+σ^†φ) - tr[γ])  (Eq 25)
    X_gamma = 0.5 * (np.einsum('pi,ij,pj->p', Z_shifted, gamma, Z_shifted)
                     - np.trace(gamma))

    # X^σ = (ε - γσ^†φ)·(Z+σ^†φ) - δ^σ·(Q-(1+r)q)  (Eq 28)
    eps_shifted = epsilon - gamma @ sigma_t_phi
    X_sigma = (Z_shifted @ eps_shifted
               - Q_excess @ delta_sigma)

    # X^s = -s·δ^s·(Q-(1+r)q)  (Eq 31)
    X_s = -s * (Q_excess @ delta_s)

    # X^δ = -(δ - δ*)·Q  (Eq 34)
    X_delta = -((delta - delta_star) @ Q.T)

    # Portfolio P&L: P - δ·Q
    pl = P - Q @ delta

    return dict(
        X_gamma=X_gamma, X_sigma=X_sigma, X_s=X_s, X_delta=X_delta,
        Q=Q, P=P, portfolio_pl=pl,
    )


# ── Indifference pricing ───────────────────────────────────────────────────────


class PricingResult(NamedTuple):
    """Indifference pricing output (Eq 38-39)."""
    zeta: float           # ζ = (η̌ - (1+r)δ^σ·q) - (1/2)αλ(χ^γ+χ^σ)  (Eq 38)
    s_star: float         # indifference funding spread s*  (Eq 39)
    p_star: float         # indifference price p*  (Eq 39)
    admissible: bool      # whether αλζ ≥ -(1/2)(1+r)²δ^s·q  (Eq 40)


def compute_indifference_price(model: SingleIntervalModel,
                                hedge: HedgeResult,
                                adj: AdjustedCoefficients,
                                cal: CalibrationResult) -> PricingResult:
    """Compute indifference price p* and funding spread s* (Eq 38-39).

    zeta := (η̌ - (1+r)δ^σ·q) - (1/2)αλ(χ^γ + χ^σ)            (Eq 38)
    s*   := (1+r)/(αλ) * (sqrt(1 + 2αλζ/((1+r)²δ^s·q)) - 1)  (Eq 39)
    p*   := 1/(1+r) * (η̌ - (1/2)αλ(χ^γ + χ^σ + s*²δ^s·q))    (Eq 39)

    For αλ → 0 the limit gives the mid price p* = (η̌ - δ^σ·q)/(1+r).
    Admissibility (Eq 40): αλζ ≥ -(1/2)(1+r)²δ^s·q.
    """
    r = cal.r
    check_eta = adj.check_eta
    chi_gamma, chi_sigma = hedge.chi_gamma, hedge.chi_sigma
    delta_sigma, delta_s = hedge.delta_sigma, hedge.delta_s
    al = model.alpha * model.lam
    delta_s_q = float(delta_s @ model.q)
    delta_sigma_q = float(delta_sigma @ model.q)

    # ζ (Eq 38)
    zeta = float((check_eta - (1.0 + r) * delta_sigma_q)
                 - 0.5 * al * (chi_gamma + chi_sigma))

    admissible = zeta * al >= -0.5 * (1.0 + r) ** 2 * delta_s_q

    if abs(al) < 1e-10:
        # Mid case αλ → 0: s* → ζ/((1+r)δ^s·q) and p* → η̌/(1+r) (first-order limit, Eq 39)
        s_star = float(zeta / ((1.0 + r) * delta_s_q))
        p_star = float(check_eta / (1.0 + r))
    else:
        disc = 1.0 + 2.0 * al * zeta / ((1.0 + r) ** 2 * delta_s_q)
        assert disc >= 0, f"Discriminant {disc:.4f} < 0; solution outside domain"
        s_star = float((1.0 + r) / al * (np.sqrt(disc) - 1.0))
        p_star = float(1.0 / (1.0 + r) * (check_eta
                                            - 0.5 * al * (chi_gamma + chi_sigma
                                                           + s_star ** 2 * delta_s_q)))

    return PricingResult(zeta=zeta, s_star=s_star, p_star=p_star, admissible=admissible)


# ── Full single-interval solver ────────────────────────────────────────────────


def solve_single_interval_entropic_hedge(
    q: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eta: float,
    epsilon: np.ndarray,
    gamma: np.ndarray,
    alpha: float,
    lam: float,
    p: float | None = None,
) -> dict:
    """Solve the complete single-interval entropic hedging problem.

    Returns a dictionary with:
      r, phi, omega_star, max_eam: calibration outputs
      check_sigma, check_eta, check_epsilon, proj: adjusted coefficients
      delta_sigma, delta_s, s, delta_star: hedge portfolio components
      chi_gamma, chi_sigma, chi_s: incompleteness risk penalties
      zeta, s_star, p_star, admissible: indifference pricing outputs
      adj, cal, hedge: full result objects for downstream use
    """
    model = SingleIntervalModel(q=q, mu=mu, sigma=sigma, eta=eta,
                                 epsilon=epsilon, gamma=gamma,
                                 alpha=alpha, lam=lam)

    cal = calibrate(model)

    # Use mid price for p if not provided
    if p is None:
        p_probe = 0.0  # initial value; replaced by p_star below
        hedge, adj, _ = compute_hedge(model, p=p_probe)
        pricing = compute_indifference_price(model, hedge, adj, cal)
        p_final = pricing.p_star
        hedge, adj, _ = compute_hedge(model, p=p_final)
        pricing = compute_indifference_price(model, hedge, adj, cal)
    else:
        p_final = p
        hedge, adj, _ = compute_hedge(model, p=p_final)
        pricing = compute_indifference_price(model, hedge, adj, cal)

    return dict(
        r=cal.r, phi=cal.phi, omega_star=cal.omega_star, max_eam=cal.max_eam,
        check_sigma=adj.check_sigma, check_eta=adj.check_eta,
        check_epsilon=adj.check_epsilon, proj=adj.proj,
        delta_sigma=hedge.delta_sigma, delta_s=hedge.delta_s,
        s=hedge.s, delta_star=hedge.delta_star,
        chi_gamma=hedge.chi_gamma, chi_sigma=hedge.chi_sigma, chi_s=hedge.chi_s,
        zeta=pricing.zeta, s_star=pricing.s_star, p_star=pricing.p_star,
        admissible=pricing.admissible,
        adj=adj, cal=cal, hedge=hedge, pricing=pricing, model=model,
        p=p_final,
    )
