"""Configuration: all experiment parameters from the paper as module-level variables.

Paper: "Model Risk Analysis for Entropic Hedging Strategies", Paul McCloud (McCloud Research), 2026.
"""

# ── Exp 1: Single-interval quadratic Gaussian (general model) ─────────────────
# Paper does not specify concrete numerical values for the general model;
# these are representative parameters satisfying the paper's assumptions.

# Number of factors and underlyings (not specified in the paper)
EXP1_N_FACTORS: int = 2      # N_f: number of standard normal factors
EXP1_N_UNDERLYINGS: int = 2  # N_m: number of underlyings

# Underlying initial prices (not specified in the paper; must satisfy q ≠ 0)
import numpy as np
EXP1_Q = np.array([1.0, 0.8])          # initial underlying prices
EXP1_MU = np.array([1.05, 0.84])       # underlying final-price linear means
EXP1_SIGMA = np.array([[0.15, 0.05],   # underlying final-price factor loadings
                        [0.08, 0.20]])  # shape (N_m, N_f)

# Derivative coefficients (not specified in the paper)
EXP1_ETA: float = 0.5          # derivative constant coefficient
EXP1_EPSILON = np.array([0.3, 0.4])    # derivative linear factor coefficients
EXP1_GAMMA = np.array([[0.2, 0.05],    # derivative quadratic factor matrix
                        [0.05, 0.1]])   # must be symmetric

# Investor and derivative parameters
EXP1_ALPHA: float = 1.0   # risk aversion α > 0
EXP1_LAMBDA: float = 0.5  # derivative notional λ

# Monte Carlo paths for validation (not specified in the paper)
EXP1_N_PATHS: int = 100_000
EXP1_SEED: int = 42

# ── Exp 2: Case study — bond-and-swap market ──────────────────────────────────
# Base parameters from the paper (p. 9, case study section)
EXP2_SIGMA_Q: float = 0.2    # swap normal volatility σ_q
EXP2_SIGMA_B: float = 0.1    # bond lognormal volatility σ_b
EXP2_ETA: float = 0.5        # derivative constant coefficient η
EXP2_EPSILON: float = 1.0    # derivative linear coefficient ε
EXP2_GAMMA: float = 2.0      # derivative quadratic coefficient γ
EXP2_RHO: float = 0.5        # derivative noise parameter ρ
EXP2_ALPHA: float = 1.0      # risk aversion α
EXP2_LAMBDA: float = 0.0     # derivative notional λ (mid case)

# Parameter sweep ranges (paper Figs 4, 5)
EXP2_GAMMA_SWEEP = np.linspace(-10.0, 10.0, 41)     # γ sweep, paper Fig 4/5
EXP2_RHO_SWEEP = np.linspace(0.0, 1.0, 21)          # ρ sweep, paper Fig 4/5
EXP2_SIGMA_B_SWEEP = np.linspace(0.0, 0.5, 21)      # σ_b sweep, paper Fig 4/5
EXP2_ALPHA_LAMBDA_SWEEP = np.linspace(-1.0, 1.0, 41) # αλ sweep, paper Fig 4
EXP2_DELTA_SWEEP = np.linspace(-1.0, 3.0, 81)       # hedge ratio scan, paper Fig 3

# Monte Carlo paths (not specified in the paper)
EXP2_N_PATHS: int = 100_000
EXP2_SEED: int = 42

# ── Exp 3: Multi-interval Gaussian-process entropic hedging ───────────────────
# Hedging schedule and process parameters (not specified in the paper)
EXP3_N_INTERVALS: int = 20      # number of hedging intervals n
EXP3_DT: float = 0.05           # interval length dt (total horizon = 1.0)
EXP3_N_FACTORS: int = 1         # factor dimension
EXP3_N_UNDERLYINGS: int = 1     # number of underlyings

# GP process parameters (not specified in the paper)
EXP3_BETA = np.array([0.0])     # reversion level
EXP3_KAPPA = np.array([[0.5]])  # reversion rate matrix
EXP3_VARSIGMA = np.array([[1.0]])  # normal volatility matrix
EXP3_X0 = np.array([0.0])      # initial factor

# Intervalwise pricing coefficients (not specified in the paper; fixed across intervals)
EXP3_MU_COEFF: float = 1.0     # underlying mean shift
EXP3_SIGMA_COEFF: float = 0.2  # underlying factor loading
EXP3_Q0: float = 1.0           # initial underlying price
EXP3_ETA_COEFF: float = 0.5    # derivative constant
EXP3_EPSILON_COEFF: float = 1.0  # derivative linear coefficient
EXP3_GAMMA_COEFF: float = 0.5  # derivative quadratic coefficient
EXP3_ALPHA: float = 1.0        # risk aversion
EXP3_LAMBDA: float = 0.5       # derivative notional

# DT grid for asymptotic scaling study (not specified in the paper)
EXP3_DT_GRID = np.logspace(-2, -0.5, 10)   # refined schedule dt values
EXP3_N_PATHS: int = 10_000
EXP3_SEED: int = 42

# ── Exp 4: Discrete fractional Ornstein-Uhlenbeck case ───────────────────────
# FOU process parameters (not specified in the paper)
EXP4_HURST_VALUES = [0.2, 0.3, 0.5, 0.7, 0.8]  # Hurst exponent H values to test
EXP4_H_ROUGH: float = 0.2       # rough regime H < 0.5
EXP4_H_BROWNIAN: float = 0.5    # Brownian benchmark H = 0.5
EXP4_H_PERSISTENT: float = 0.8  # persistent regime H > 0.5

EXP4_BETA = np.array([0.0])     # reversion level
EXP4_KAPPA: float = 0.5         # reversion rate (scalar, 1D)
EXP4_VARSIGMA: float = 1.0      # normal volatility (scalar, 1D)
EXP4_X0: float = 0.0            # initial factor value

# DT grid for asymptotic scaling (not specified in the paper)
EXP4_DT_GRID = np.logspace(-2, -0.3, 10)  # dt values for log-log regression
EXP4_N_INTERVALS: int = 50               # number of intervals per path
EXP4_N_PATHS: int = 5_000
EXP4_SEED: int = 42

# Intervalwise pricing coefficients for FOU scaling study (not specified in the paper)
EXP4_Q0: float = 1.0           # initial underlying price
EXP4_MU_COEFF: float = 1.0     # underlying mean shift
EXP4_SIGMA_COEFF: float = 0.2  # underlying factor loading
EXP4_ETA_COEFF: float = 0.5    # derivative constant
EXP4_EPSILON_COEFF: float = 1.0  # derivative linear coefficient
EXP4_GAMMA_COEFF: float = 0.5  # derivative quadratic coefficient
EXP4_ALPHA: float = 1.0        # risk aversion
EXP4_LAMBDA: float = 0.5       # derivative notional

# ── Shared ────────────────────────────────────────────────────────────────────
RESULTS_DIR: str = "results"
