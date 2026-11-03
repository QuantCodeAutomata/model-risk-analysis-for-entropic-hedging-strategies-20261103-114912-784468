"""Synthetic data generation for experiments (no market data required).

Paper: "Model Risk Analysis for Entropic Hedging Strategies", Paul McCloud (McCloud Research), 2026.

All experiments use synthetic model inputs only; no market data is needed.
"""

from __future__ import annotations

import numpy as np
from src.config import (
    EXP1_N_FACTORS, EXP1_N_UNDERLYINGS, EXP1_Q, EXP1_MU, EXP1_SIGMA,
    EXP1_ETA, EXP1_EPSILON, EXP1_GAMMA, EXP1_ALPHA, EXP1_LAMBDA,
    EXP2_SIGMA_Q, EXP2_SIGMA_B, EXP2_ETA, EXP2_EPSILON, EXP2_GAMMA,
    EXP2_RHO, EXP2_ALPHA, EXP2_LAMBDA,
    EXP4_H_ROUGH, EXP4_H_BROWNIAN, EXP4_H_PERSISTENT,
    EXP4_BETA, EXP4_KAPPA, EXP4_VARSIGMA, EXP4_X0,
    EXP4_N_INTERVALS, EXP4_DT_GRID,
)


def make_exp1_model_params() -> dict:
    """Return model parameters for Experiment 1 (general single-interval model)."""
    return dict(
        q=EXP1_Q.copy(),
        mu=EXP1_MU.copy(),
        sigma=EXP1_SIGMA.copy(),
        eta=EXP1_ETA,
        epsilon=EXP1_EPSILON.copy(),
        gamma=EXP1_GAMMA.copy(),
        alpha=EXP1_ALPHA,
        lam=EXP1_LAMBDA,
    )


def make_exp2_base_params() -> dict:
    """Return base parameters for Experiment 2 (bond-and-swap case study)."""
    return dict(
        sigma_q=EXP2_SIGMA_Q,
        sigma_b=EXP2_SIGMA_B,
        eta=EXP2_ETA,
        epsilon=EXP2_EPSILON,
        gamma=EXP2_GAMMA,
        rho=EXP2_RHO,
        alpha=EXP2_ALPHA,
        lam=EXP2_LAMBDA,
    )


def make_fou_params(H: float, n_intervals: int, dt: float) -> dict:
    """Return FOU process parameters for Experiment 4."""
    return dict(
        n=n_intervals,
        dt=dt,
        H=H,
        beta=EXP4_BETA.copy(),
        kappa=np.array([[EXP4_KAPPA]]),
        varsigma=np.array([[EXP4_VARSIGMA]]),
        x0=EXP4_X0.copy(),
    )
