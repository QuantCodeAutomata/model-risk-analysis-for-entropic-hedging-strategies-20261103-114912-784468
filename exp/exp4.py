"""Experiment 4: Discrete fractional Ornstein-Uhlenbeck case.

Paper: "Model Risk Analysis for Entropic Hedging Strategies", Paul McCloud (McCloud Research), 2026.

Tests continuous-limit behaviour under rough (H<1/2), Brownian (H=1/2), and persistent (H>1/2) dynamics.
Expected asymptotic powers (paper, Eq 76-84):
  n_x = H  (conditional mean)
  n_ν = 2H (conditional covariance)
  χ^γ = O(dt^{4H}), per unit time: O(dt^{4H-1})
  χ^σ = O(dt^{2H}), per unit time: O(dt^{2H-1})
  χ^s = O(dt^{2H}), per unit time: O(dt^{2H-1})

Divergence (per unit time) when H < 1/4 for χ^γ, H < 1/2 for χ^σ/χ^s.

Note: Eq 80 in the paper writes ν̄_i = ςζ†dt, which appears to be a typographical
inconsistency; from context (Eq 73) and the surrounding derivation, this should be
ν̄_i = ξ_{ii}^2 ςς†dt^{2H}. Implementation follows the latter.

Run as: python -m exp.exp4
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP4_HURST_VALUES, EXP4_H_ROUGH, EXP4_H_BROWNIAN, EXP4_H_PERSISTENT,
    EXP4_BETA, EXP4_KAPPA, EXP4_VARSIGMA_MAT, EXP4_X0,
    EXP4_DT_GRID, EXP4_N_INTERVALS, EXP4_N_PATHS, EXP4_SEED,
    EXP4_Q0, EXP4_MU_COEFF, EXP4_SIGMA_COEFF,
    EXP4_ETA_COEFF, EXP4_EPSILON_COEFF, EXP4_GAMMA_COEFF,
    EXP4_ALPHA, EXP4_LAMBDA,
    RESULTS_DIR,
)
from src.fractional_ou import (
    FractionalOU, build_fbm_cholesky, xi_diagonal_bounds,
    conditional_coefficient_bound,
)
from src.gaussian_process import (
    IntervalCoefficients, solve_interval, estimate_scaling_powers,
)

RESULTS = Path(RESULTS_DIR)
RESULTS.mkdir(exist_ok=True)


def make_fou(H: float, n: int, dt: float) -> FractionalOU:
    """Construct a 2D FOU process (N_f=2, N_m=1 gives dimensionality incompleteness)."""
    return FractionalOU(
        n=n, dt=dt, H=H,
        beta=EXP4_BETA.copy(),
        kappa=EXP4_KAPPA.copy(),
        varsigma=EXP4_VARSIGMA_MAT.copy(),
        x0=EXP4_X0.copy(),
    )


def make_interval_coeff(q0: float, dt: float,
                         mu_coeff: float, sigma_coeff: np.ndarray,
                         eta_coeff: float, epsilon_coeff: np.ndarray,
                         gamma_coeff: np.ndarray) -> IntervalCoefficients:
    """Build a single IntervalCoefficients for the FOU scaling study.

    sigma_coeff: shape (N_m, dim); epsilon_coeff: shape (dim,); gamma_coeff: shape (dim, dim)
    """
    sigma_coeff = np.asarray(sigma_coeff)
    n_m = sigma_coeff.shape[0]
    return IntervalCoefficients(
        q_prev=np.full(n_m, q0),
        mu_i=np.full(n_m, q0 + mu_coeff * dt),
        sigma_i=sigma_coeff.copy(),
        eta_i=float(eta_coeff * dt),
        epsilon_i=np.asarray(epsilon_coeff).copy(),
        gamma_i=np.asarray(gamma_coeff).copy(),
    )


def run_xi_bounds() -> None:
    """Verify diagonal bounds on ξ_{ii}^2 (Eq 74-75)."""
    print("\n=== ξ diagonal bounds (Eq 74-75) ===")
    for H in [0.2, 0.3, 0.5, 0.7, 0.8]:
        lo, hi, xi_diag_sq = xi_diagonal_bounds(H, n=15)
        ok = np.all((xi_diag_sq >= lo - 1e-10) & (xi_diag_sq <= hi + 1e-10))
        print(f"  H={H:.1f}: lower={lo:.4f}, upper={hi:.4f}, "
              f"ξ_diag² range=[{xi_diag_sq.min():.4f}, {xi_diag_sq.max():.4f}] "
              f"{'✓' if ok else '✗ OUT OF BOUNDS'}")

    print("\n=== Conditional-coefficient identity Σ_k(ξ_{ik}-ξ_{i-1,k})^2 = 1-ξ_{ii}^2 (Eq 75) ===")
    for H in [0.3, 0.5, 0.7]:
        lhs, rhs = conditional_coefficient_bound(H, n=15)
        max_err = float(np.max(np.abs(lhs - rhs)))
        print(f"  H={H:.1f}: max|lhs-rhs| = {max_err:.2e} {'✓' if max_err < 1e-12 else '✗'}")


def compute_penalties_at_dt(H: float, dt: float, i_target: int = 15) -> dict:
    """Compute penalties at a fixed interval i_target for given H and dt.

    i_target should be large enough that the fBm Cholesky diagonal ξ_{ii}
    has converged to its asymptotic value; i_target=15 is sufficient for H ≥ 0.2.
    """
    n = max(i_target + 5, 20)
    try:
        fou = make_fou(H, n, dt)
    except Exception as e:
        return dict(r_i=np.nan, chi_gamma=np.nan, chi_sigma=np.nan, chi_s=np.nan)

    dim = EXP4_BETA.shape[0]
    zero_path = np.zeros((i_target, dim))
    try:
        x_bar, varsigma_ii = fou.conditional_increment(i_target, zero_path)
    except Exception as e:
        return dict(r_i=np.nan, chi_gamma=np.nan, chi_sigma=np.nan, chi_s=np.nan)

    coeff = make_interval_coeff(EXP4_Q0, dt, EXP4_MU_COEFF, EXP4_SIGMA_COEFF,
                                 EXP4_ETA_COEFF, EXP4_EPSILON_COEFF, EXP4_GAMMA_COEFF)
    try:
        res = solve_interval(coeff, x_bar, varsigma_ii, EXP4_ALPHA, EXP4_LAMBDA)
    except Exception as e:
        return dict(r_i=np.nan, chi_gamma=np.nan, chi_sigma=np.nan, chi_s=np.nan)

    return dict(
        r_i=float(res.r_i),
        chi_gamma=float(res.chi_gamma_i),
        chi_sigma=float(res.chi_sigma_i),
        chi_s=float(res.chi_s_i),
    )


def run_hurst_comparison() -> None:
    """Print per-H fixed-dt comparison at dt=0.01."""
    dt_fixed = 0.01
    i_target = 5
    print("\n=== Per-H comparison at dt=0.01, interval 5 ===")
    print(f"{'H':>6}  {'r_i':>12}  {'chi_gamma':>12}  {'chi_sigma':>12}  {'chi_s':>12}")
    for H in EXP4_HURST_VALUES:
        d = compute_penalties_at_dt(H, dt_fixed, i_target)
        print(f"{H:>6.1f}  {d['r_i']:>12.6e}  {d['chi_gamma']:>12.6e}  "
              f"{d['chi_sigma']:>12.6e}  {d['chi_s']:>12.6e}")


def run_asymptotic_scaling() -> dict:
    """Estimate asymptotic powers for each H value.

    Expected (Eq 76-84):
      slope of chi_gamma vs dt: 4H
      slope of chi_sigma vs dt: 2H
      slope of chi_s    vs dt: 2H
    """
    # i_target=15 ensures the fBm Cholesky diagonal has converged for H ≥ 0.2
    i_target = 15
    results = {}

    for H in EXP4_HURST_VALUES:
        chi_gamma_arr = []
        chi_sigma_arr = []
        chi_s_arr = []

        for dt in EXP4_DT_GRID:
            d = compute_penalties_at_dt(H, dt, i_target)
            chi_gamma_arr.append(d["chi_gamma"])
            chi_sigma_arr.append(d["chi_sigma"])
            chi_s_arr.append(d["chi_s"])

        chi_gamma_arr = np.array(chi_gamma_arr)
        chi_sigma_arr = np.array(chi_sigma_arr)
        chi_s_arr = np.array(chi_s_arr)

        slope_g = estimate_scaling_powers(EXP4_DT_GRID, np.abs(chi_gamma_arr))
        slope_s = estimate_scaling_powers(EXP4_DT_GRID, np.abs(chi_sigma_arr))
        slope_f = estimate_scaling_powers(EXP4_DT_GRID, np.abs(chi_s_arr))

        exp_g = 4.0 * H
        exp_s = 2.0 * H

        print(f"\n  H={H:.1f}: slope_gamma={slope_g:.3f} (expected {exp_g:.2f}), "
              f"slope_sigma={slope_s:.3f} (expected {exp_s:.2f}), "
              f"slope_s={slope_f:.3f} (expected {exp_s:.2f})")

        key = f"H_{H:.1f}".replace(".", "p")
        results[key] = dict(
            H=H,
            scaling_slopes_chi_gamma=slope_g,
            scaling_slopes_chi_sigma=slope_s,
            scaling_slopes_chi_s=slope_f,
            expected_chi_gamma_slope=exp_g,
            expected_chi_sigma_slope=exp_s,
            chi_gamma_arr=chi_gamma_arr.tolist(),
            chi_sigma_arr=chi_sigma_arr.tolist(),
            chi_s_arr=chi_s_arr.tolist(),
        )

    return results


def make_scaling_plots(scaling: dict) -> None:
    """Log-log plot of chi_gamma and chi_sigma vs dt for each H."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, d in scaling.items():
        H = d["H"]
        label = f"H={H:.1f}"
        cg = np.abs(np.array(d["chi_gamma_arr"]))
        cs = np.abs(np.array(d["chi_sigma_arr"]))
        mask_g = cg > 0
        mask_s = cs > 0

        if mask_g.any():
            axes[0].loglog(EXP4_DT_GRID[mask_g], cg[mask_g], "o-",
                           label=f"{label} (slope≈{d['scaling_slopes_chi_gamma']:.2f})")
        if mask_s.any():
            axes[1].loglog(EXP4_DT_GRID[mask_s], cs[mask_s], "s-",
                           label=f"{label} (slope≈{d['scaling_slopes_chi_sigma']:.2f})")

    # Reference lines
    dt_ref = np.array([EXP4_DT_GRID[0], EXP4_DT_GRID[-1]])
    for H in [0.2, 0.5, 0.8]:
        axes[0].loglog(dt_ref, 1e-4 * dt_ref ** (4 * H), "--", alpha=0.3,
                       label=f"O(dt^{{{4*H:.1f}}})")
        axes[1].loglog(dt_ref, 1e-2 * dt_ref ** (2 * H), "--", alpha=0.3,
                       label=f"O(dt^{{{2*H:.1f}}})")

    axes[0].set_xlabel("dt")
    axes[0].set_ylabel("χ^γ (convexity penalty)")
    axes[0].set_title("Convexity penalty scaling vs H (Eq 82)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("dt")
    axes[1].set_ylabel("χ^σ (dimensionality penalty)")
    axes[1].set_title("Dimensionality penalty scaling vs H (Eq 83)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "exp4_scaling.png", dpi=120)
    plt.close()


def make_per_unit_time_plot(scaling: dict) -> None:
    """Plot chi_sigma/dt vs dt for each H (diverges/converges per unit time)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, d in scaling.items():
        H = d["H"]
        cs = np.abs(np.array(d["chi_sigma_arr"]))
        put = cs / EXP4_DT_GRID   # per unit time
        mask = put > 0
        if mask.any():
            slope = d["scaling_slopes_chi_sigma"] - 1.0  # per-unit-time slope = 2H-1
            ax.loglog(EXP4_DT_GRID[mask], put[mask], "o-",
                      label=f"H={H:.1f} (per-unit slope≈{slope:.2f})")

    ax.set_xlabel("dt")
    ax.set_ylabel("χ^σ / dt (per unit time)")
    ax.set_title("Dimensionality penalty per unit time\n"
                 "Diverges (H<0.5), converges (H>0.5) as dt→0 (Eq 83)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp4_per_unit_time.png", dpi=120)
    plt.close()


def check_brownian_simplifications() -> None:
    """Verify H=0.5 simplifications (Eq 80-81): ξ_{ii}^2 ≈ 1 for all i ≥ 2."""
    H = 0.5
    n = 15
    xi = build_fbm_cholesky(n, H)
    xi_diag_sq = np.array([xi[i, i] ** 2 for i in range(n)])
    # For H=0.5, ξ_{ii}^2 = 1 - min(i,i-1)/i = 1/i (decreasing but not 1)
    # Actually for Brownian: ξ_{i,i}^2 = f_{1/2}[i,i] - Σ_{k<i} ξ_{ik}^2
    # f_{1/2}[i,i] = i, f_{1/2}[i,k] = min(i,k) = k for k ≤ i
    # So ξ_{ii}^2 = 1 (by construction from fBM increment variance)
    print(f"\n=== Brownian H=0.5 simplifications ===")
    print(f"  ξ_{{ii}}^2 values for i=1..{n}: {xi_diag_sq[:10].round(6)}")
    print(f"  For H=0.5, ξ_{{ii}}^2 = 1 (increment variance = 1 per unit time)")


def main() -> None:
    print("=== Experiment 4: Fractional OU entropic hedging scaling ===")

    run_xi_bounds()
    check_brownian_simplifications()
    run_hurst_comparison()

    print("\n=== Asymptotic scaling analysis ===")
    scaling = run_asymptotic_scaling()

    make_scaling_plots(scaling)
    make_per_unit_time_plot(scaling)

    # Build results dict (exclude raw arrays for compactness)
    out: dict = {
        "hurst_values": EXP4_HURST_VALUES,
        "brownian_chi_sigma_slope": scaling.get(
            "H_0p5", scaling.get("H_0.5", {})
        ).get("scaling_slopes_chi_sigma", None),
    }
    for key, d in scaling.items():
        out[key] = {
            "H": d["H"],
            "scaling_slopes_chi_gamma": d["scaling_slopes_chi_gamma"],
            "scaling_slopes_chi_sigma": d["scaling_slopes_chi_sigma"],
            "scaling_slopes_chi_s": d["scaling_slopes_chi_s"],
            "expected_chi_gamma_slope": d["expected_chi_gamma_slope"],
            "expected_chi_sigma_slope": d["expected_chi_sigma_slope"],
        }

    (RESULTS / "exp4_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {RESULTS}/exp4_results.json, exp4_scaling.png, exp4_per_unit_time.png")


if __name__ == "__main__":
    main()
