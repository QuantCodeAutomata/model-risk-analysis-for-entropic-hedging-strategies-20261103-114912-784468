"""Experiment 3: Multi-interval Gaussian-process entropic hedging and asymptotics.

Paper: "Model Risk Analysis for Entropic Hedging Strategies", Paul McCloud (McCloud Research), 2026.

Implements Eq 57-63 (intervalwise GP formulas) and the asymptotic scaling table (Fig 6).
Run as: python -m exp.exp3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP3_N_INTERVALS, EXP3_DT, EXP3_N_FACTORS, EXP3_N_UNDERLYINGS,
    EXP3_BETA, EXP3_KAPPA, EXP3_VARSIGMA, EXP3_X0,
    EXP3_MU_COEFF, EXP3_SIGMA_COEFF, EXP3_Q0,
    EXP3_ETA_COEFF, EXP3_EPSILON_COEFF, EXP3_GAMMA_COEFF,
    EXP3_ALPHA, EXP3_LAMBDA,
    EXP3_DT_GRID, EXP3_N_PATHS, EXP3_SEED,
    RESULTS_DIR,
)
from src.gaussian_process import (
    GaussianProcessDriver,
    IntervalCoefficients,
    solve_interval,
    solve_multi_interval,
    estimate_scaling_powers,
)

RESULTS = Path(RESULTS_DIR)
RESULTS.mkdir(exist_ok=True)


def build_brownian_gp(n: int, dt: float, dim: int,
                       sigma_sq: float = 1.0) -> GaussianProcessDriver:
    """Build a Gaussian process with Brownian covariance ν_{ij} = min(i,j)*dt*sigma_sq."""
    # 0-indexed: interval 0 corresponds to paper interval 1
    beta = np.zeros((n, dim))
    nu = np.zeros((n, n, dim, dim))
    for i in range(n):
        for j in range(n):
            nu[i, j] = np.eye(dim) * min(i + 1, j + 1) * dt * sigma_sq
    driver = GaussianProcessDriver(n=n, dim=dim, beta=beta, nu=nu)
    driver.build_filtered_cholesky()
    return driver


def make_interval_coefficients(n: int, q0: float, dim: int,
                                 mu_coeff: float, sigma_coeff: float, dt: float,
                                 eta_coeff: float, epsilon_coeff: float,
                                 gamma_coeff: float) -> list[IntervalCoefficients]:
    """Build uniform pricing coefficients for each interval."""
    coeffs = []
    for i in range(n):
        q_prev = np.array([q0 + 0.001 * i])
        mu_i = np.array([q_prev[0] + mu_coeff * dt])
        sigma_i = np.array([[sigma_coeff]])
        eta_i = eta_coeff * dt
        epsilon_i = np.array([epsilon_coeff])
        gamma_i = np.array([[gamma_coeff]])
        coeffs.append(IntervalCoefficients(
            q_prev=q_prev, mu_i=mu_i, sigma_i=sigma_i,
            eta_i=eta_i, epsilon_i=epsilon_i, gamma_i=gamma_i,
        ))
    return coeffs


def run_multi_interval_base() -> dict:
    """Run multi-interval entropic hedge on the base Brownian GP schedule."""
    n = EXP3_N_INTERVALS
    dt = EXP3_DT
    dim = EXP3_N_FACTORS

    driver = build_brownian_gp(n, dt, dim)
    coeffs = make_interval_coefficients(
        n, EXP3_Q0, dim, EXP3_MU_COEFF, EXP3_SIGMA_COEFF, dt,
        EXP3_ETA_COEFF, EXP3_EPSILON_COEFF, EXP3_GAMMA_COEFF,
    )
    result = solve_multi_interval(driver, coeffs, EXP3_ALPHA, EXP3_LAMBDA)

    print("\n=== Exp 3: Multi-interval GP (Brownian, base) ===")
    print(f"{'Interval':>9}  {'chi_gamma':>12}  {'chi_sigma':>12}  {'chi_s':>12}  {'p_star':>12}")
    for i in range(n):
        r = result["interval_results"][i]
        print(f"{i+1:>9d}  {r.chi_gamma_i:>12.6f}  {r.chi_sigma_i:>12.6f}  "
              f"{r.chi_s_i:>12.6f}  {r.p_star_i:>12.6f}")

    # Verify non-negativity (paper: all penalties ≥ 0)
    assert np.all(result["penalties_gamma"] >= -1e-12), "Convexity penalties must be ≥ 0"
    assert np.all(result["penalties_sigma"] >= -1e-12), "Dimensionality penalties must be ≥ 0"
    assert np.all(result["penalties_s"] >= -1e-12), "Funding penalties must be ≥ 0"
    print("✓ All interval penalties non-negative")

    return result


def run_asymptotic_scaling() -> dict:
    """Asymptotic scaling study: estimate powers χ = O(dt^n) for Brownian GP.

    Expected (paper, Brownian H=1/2 case):
      χ^γ = O(dt^2) → slope ≈ 2
      χ^σ = O(dt^1) → slope ≈ 1
      χ^s = O(dt^1) → slope ≈ 1
    """
    dt_grid = EXP3_DT_GRID
    dim = EXP3_N_FACTORS
    n_intervals = 10  # use fewer intervals for speed in scaling study

    chi_gamma_means = []
    chi_sigma_means = []
    chi_s_means = []
    p_star_means = []

    for dt in dt_grid:
        driver = build_brownian_gp(n_intervals, dt, dim)
        coeffs = make_interval_coefficients(
            n_intervals, EXP3_Q0, dim, EXP3_MU_COEFF, EXP3_SIGMA_COEFF, dt,
            EXP3_ETA_COEFF, EXP3_EPSILON_COEFF, EXP3_GAMMA_COEFF,
        )
        try:
            result = solve_multi_interval(driver, coeffs, EXP3_ALPHA, EXP3_LAMBDA)
            cg = float(np.mean(np.abs(result["penalties_gamma"])))
            cs = float(np.mean(np.abs(result["penalties_sigma"])))
            cf = float(np.mean(np.abs(result["penalties_s"])))
            ps = float(np.mean(np.abs(result["p_stars"])))
        except Exception as e:
            print(f"  dt={dt:.4f}: skipped ({e})")
            cg = cs = cf = ps = float("nan")
        chi_gamma_means.append(cg)
        chi_sigma_means.append(cs)
        chi_s_means.append(cf)
        p_star_means.append(ps)

    chi_gamma_means = np.array(chi_gamma_means)
    chi_sigma_means = np.array(chi_sigma_means)
    chi_s_means = np.array(chi_s_means)

    slope_gamma = estimate_scaling_powers(dt_grid, chi_gamma_means)
    slope_sigma = estimate_scaling_powers(dt_grid, chi_sigma_means)
    slope_s = estimate_scaling_powers(dt_grid, chi_s_means)

    print("\n=== Asymptotic Scaling (Brownian GP, expected n_γ≈2, n_σ≈1, n_s≈1) ===")
    print(f"{'dt':>10}  {'chi_gamma':>12}  {'chi_sigma':>12}  {'chi_s':>12}")
    for i, dt in enumerate(dt_grid):
        print(f"{dt:>10.4f}  {chi_gamma_means[i]:>12.6e}  "
              f"{chi_sigma_means[i]:>12.6e}  {chi_s_means[i]:>12.6e}")
    print(f"\nEstimated slopes: gamma={slope_gamma:.3f}, sigma={slope_sigma:.3f}, s={slope_s:.3f}")
    print(f"Expected (Brownian H=1/2): gamma≈2, sigma≈1, s≈1")

    # Scaling plot
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = chi_gamma_means > 0
    if mask.any():
        ax.loglog(dt_grid[mask], chi_gamma_means[mask], "o-", label=f"χ^γ (slope≈{slope_gamma:.2f})")
    mask = chi_sigma_means > 0
    if mask.any():
        ax.loglog(dt_grid[mask], chi_sigma_means[mask], "s-", label=f"χ^σ (slope≈{slope_sigma:.2f})")
    mask = chi_s_means > 0
    if mask.any():
        ax.loglog(dt_grid[mask], chi_s_means[mask], "^-", label=f"χ^s (slope≈{slope_s:.2f})")
    # Reference lines
    dt_ref = np.array([dt_grid[0], dt_grid[-1]])
    ax.loglog(dt_ref, 0.1 * dt_ref ** 2, "k--", alpha=0.4, label="O(dt²)")
    ax.loglog(dt_ref, 0.05 * dt_ref, "k:", alpha=0.4, label="O(dt)")
    ax.set_xlabel("dt")
    ax.set_ylabel("Mean penalty")
    ax.set_title("Asymptotic scaling of incompleteness penalties (Brownian GP, H=1/2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp3_scaling.png", dpi=120)
    plt.close()

    return dict(
        dt_grid=dt_grid.tolist(),
        chi_gamma_means=chi_gamma_means.tolist(),
        chi_sigma_means=chi_sigma_means.tolist(),
        chi_s_means=chi_s_means.tolist(),
        p_star_means=p_star_means,
        slope_gamma=slope_gamma,
        slope_sigma=slope_sigma,
        slope_s=slope_s,
    )


def run_mc_paths(base_result: dict) -> dict:
    """Simulate multiple paths and compute pathwise penalty statistics."""
    n = EXP3_N_INTERVALS
    dt = EXP3_DT
    dim = EXP3_N_FACTORS
    rng = np.random.default_rng(EXP3_SEED)

    driver = build_brownian_gp(n, dt, dim)
    coeffs = make_interval_coefficients(
        n, EXP3_Q0, dim, EXP3_MU_COEFF, EXP3_SIGMA_COEFF, dt,
        EXP3_ETA_COEFF, EXP3_EPSILON_COEFF, EXP3_GAMMA_COEFF,
    )

    # Collect penalties across paths at a fixed interval (i=5)
    i_target = min(5, n - 1)
    chi_gamma_paths = []
    chi_sigma_paths = []
    chi_s_paths = []

    n_paths = min(EXP3_N_PATHS, 1000)  # use 1000 for speed
    for _ in range(n_paths):
        _, z = driver.simulate_path(rng)
        x_bar_i, varsigma_ii = driver.conditional_increment(i_target, z[:i_target])
        try:
            res = solve_interval(coeffs[i_target], x_bar_i, varsigma_ii,
                                  EXP3_ALPHA, EXP3_LAMBDA)
            chi_gamma_paths.append(res.chi_gamma_i)
            chi_sigma_paths.append(res.chi_sigma_i)
            chi_s_paths.append(res.chi_s_i)
        except Exception:
            pass

    chi_g = np.array(chi_gamma_paths)
    chi_s_arr = np.array(chi_sigma_paths)
    chi_sf = np.array(chi_s_paths)

    print(f"\n=== MC path statistics at interval {i_target+1} (n={len(chi_g)} paths) ===")
    print(f"  χ^γ: mean={chi_g.mean():.6f}, std={chi_g.std():.6f}")
    print(f"  χ^σ: mean={chi_s_arr.mean():.6f}, std={chi_s_arr.std():.6f}")
    print(f"  χ^s: mean={chi_sf.mean():.6f}, std={chi_sf.std():.6f}")

    return dict(
        mc_chi_gamma_mean=float(chi_g.mean()),
        mc_chi_sigma_mean=float(chi_s_arr.mean()),
        mc_chi_s_mean=float(chi_sf.mean()),
        mc_n_paths=len(chi_g),
    )


def main() -> None:
    base_result = run_multi_interval_base()
    scaling = run_asymptotic_scaling()
    mc = run_mc_paths(base_result)

    output = dict(
        interval_penalties_gamma=base_result["penalties_gamma"].tolist(),
        interval_penalties_sigma=base_result["penalties_sigma"].tolist(),
        interval_penalties_s=base_result["penalties_s"].tolist(),
        interval_p_stars=base_result["p_stars"].tolist(),
        scaling_dt_grid=scaling["dt_grid"],
        scaling_chi_gamma_means=scaling["chi_gamma_means"],
        scaling_chi_sigma_means=scaling["chi_sigma_means"],
        scaling_chi_s_means=scaling["chi_s_means"],
        estimated_slope_gamma=scaling["slope_gamma"],
        estimated_slope_sigma=scaling["slope_sigma"],
        estimated_slope_s=scaling["slope_s"],
        mc_chi_gamma_mean=mc["mc_chi_gamma_mean"],
        mc_chi_sigma_mean=mc["mc_chi_sigma_mean"],
        mc_chi_s_mean=mc["mc_chi_s_mean"],
    )

    (RESULTS / "exp3_results.json").write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS}/exp3_results.json and exp3_scaling.png")


if __name__ == "__main__":
    main()
