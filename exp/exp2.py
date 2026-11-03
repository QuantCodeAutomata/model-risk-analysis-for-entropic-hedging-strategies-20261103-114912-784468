"""Experiment 2: Bond-and-swap case study reproduction.

Reproduces the scalar analytical results and Monte Carlo P&L distributions
for the incomplete bond-and-swap market described in the paper.

Run as: python -m exp.exp2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from src.case_study import (
    compute_case_study,
    compute_pl_distribution,
    simulate_case_study_paths,
)
from src.config import (
    EXP2_ALPHA,
    EXP2_ALPHA_LAMBDA_SWEEP,
    EXP2_DELTA_SWEEP,
    EXP2_EPSILON,
    EXP2_ETA,
    EXP2_GAMMA,
    EXP2_GAMMA_SWEEP,
    EXP2_LAMBDA,
    EXP2_N_PATHS,
    EXP2_RHO,
    EXP2_RHO_SWEEP,
    EXP2_SEED,
    EXP2_SIGMA_B,
    EXP2_SIGMA_B_SWEEP,
    EXP2_SIGMA_Q,
    RESULTS_DIR,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _is_valid_gamma(al: float, gamma: float, rho: float, sigma_q: float) -> bool:
    """Return True iff αλγ(1+ρ²)σ_q² > -1 and the dimensionality denominator > 0."""
    u = al * gamma * (1.0 + rho ** 2) * sigma_q ** 2
    denom = 1.0 + al * gamma * rho ** 2 * sigma_q ** 2
    return (1.0 + u > 0) and (denom > 0)


def _try_case_study(**kwargs: Any) -> "tuple[Any, bool]":
    """Wrap compute_case_study, returning (result, ok) so callers can skip invalid points."""
    try:
        return compute_case_study(**kwargs), True
    except (AssertionError, ValueError, ZeroDivisionError) as exc:
        return None, False


# ── Step 1: Base case ──────────────────────────────────────────────────────────


def run_base_case() -> "Any":
    """Compute and print the base-case case study result (lam=0, mid case)."""
    base = compute_case_study(
        sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
        eta=EXP2_ETA, epsilon=EXP2_EPSILON,
        gamma=EXP2_GAMMA, rho=EXP2_RHO,
        alpha=EXP2_ALPHA, lam=EXP2_LAMBDA,
    )

    print("=== EXP2: Case Study — Bond-and-Swap Market ===\n")
    print("--- Base case (λ=0, mid) ---")
    print(f"  αλ              : {base.al:.4f}")
    print(f"  δ*              : {base.delta_star:.6f}")
    print(f"  ζ               : {base.zeta:.6f}")
    print(f"  p*              : {base.p_star:.6f}")
    print(f"  χ^γ             : {base.chi_gamma:.6f}")
    print(f"  χ^σ             : {base.chi_sigma:.6f}")
    print(f"  χ^s             : {base.chi_s:.6f}")
    print(f"  χ^δ (at δ*)     : {base.chi_delta:.6f}")
    print(f"  incremental EAM : {base.incremental_eam:.6f}")
    return base


# ── Step 2: Hedge-ratio scan ───────────────────────────────────────────────────


def run_hedge_scan(results_dir: Path) -> "tuple[np.ndarray, float]":
    """Vary δ over EXP2_DELTA_SWEEP and maximise incremental_eam to locate δ*.

    With αλ = 0 (mid case, lam=0) the strategy-penalty term in the incremental
    EAM carries a factor of αλ and therefore vanishes for every δ, making the
    curve exactly flat.  To obtain a non-trivial scan that reveals the maximum
    at δ* ≈ 1 — consistent with the paper's Fig 3 — we use lam=0.5 (long
    position, αλ=0.5) for this scan only.
    """
    SCAN_LAM: float = 0.5   # long position gives non-trivial penalty profile

    # Compute the p* at this notional so the scan evaluates penalties at the
    # correct indifference price, then fix p and vary δ.
    cs_ref = compute_case_study(
        sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
        eta=EXP2_ETA, epsilon=EXP2_EPSILON,
        gamma=EXP2_GAMMA, rho=EXP2_RHO,
        alpha=EXP2_ALPHA, lam=SCAN_LAM,
    )
    analytical_delta_star: float = cs_ref.delta_star
    p_fixed: float = cs_ref.p_star

    ieam_vals: list[float] = []
    for delta in EXP2_DELTA_SWEEP:
        cs = compute_case_study(
            sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
            eta=EXP2_ETA, epsilon=EXP2_EPSILON,
            gamma=EXP2_GAMMA, rho=EXP2_RHO,
            alpha=EXP2_ALPHA, lam=SCAN_LAM,
            p=p_fixed, delta=float(delta),
        )
        ieam_vals.append(cs.incremental_eam)

    ieam_arr = np.array(ieam_vals)
    best_idx = int(np.argmax(ieam_arr))
    best_delta: float = float(EXP2_DELTA_SWEEP[best_idx])
    grid_step: float = float(EXP2_DELTA_SWEEP[1] - EXP2_DELTA_SWEEP[0])
    scan_matches: bool = abs(best_delta - analytical_delta_star) <= grid_step

    print(f"\n--- Hedge-ratio scan (lam={SCAN_LAM}, αλ={EXP2_ALPHA * SCAN_LAM}) ---")
    print(f"  Scan maximiser δ̂   : {best_delta:.4f}  (nearest grid point to analytical δ*)")
    print(f"  Analytical δ*      : {analytical_delta_star:.6f}  (paper: ≈ 1.0 for base params)")
    print(f"  |δ̂ − δ*| ≤ grid step : "
          f"{abs(best_delta - analytical_delta_star):.4f} ≤ {grid_step:.4f}  "
          f"{'PASS' if scan_matches else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(EXP2_DELTA_SWEEP, ieam_arr, lw=2, label="Incremental EAM")
    ax.axvline(analytical_delta_star, color="C1", ls="--",
               label=f"Analytical δ* = {analytical_delta_star:.3f}")
    ax.axvline(best_delta, color="C2", ls=":",
               label=f"Scan max δ̂ = {best_delta:.3f}")
    ax.set_xlabel("Hedge ratio δ")
    ax.set_ylabel("Incremental EAM")
    ax.set_title(f"EXP2: Incremental EAM vs δ  (λ={SCAN_LAM})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scan_path = results_dir / "exp2_hedge_scan.png"
    fig.savefig(scan_path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved → {scan_path}")

    return ieam_arr, best_delta


# ── Step 3: γ sweep ────────────────────────────────────────────────────────────


def run_gamma_sweep(results_dir: Path) -> "tuple[list[float], list[float]]":
    """Sweep γ for lam=0 (mid), lam=0.5 (long), lam=-0.5 (short)."""
    lam_cases: list[tuple[float, str]] = [
        (0.0, "mid (λ=0)"),
        (0.5, "long (λ=0.5)"),
        (-0.5, "short (λ=−0.5)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_p, ax_d = axes

    p_stars_mid: list[float] = []
    d_stars_mid: list[float] = []

    for lam, label in lam_cases:
        al = EXP2_ALPHA * lam
        ps_list: list[float | None] = []
        ds_list: list[float | None] = []

        for gamma in EXP2_GAMMA_SWEEP:
            # Validity check: αλγ(1+ρ²)σ_q² > −1
            if not _is_valid_gamma(al, float(gamma), EXP2_RHO, EXP2_SIGMA_Q):
                ps_list.append(np.nan)
                ds_list.append(np.nan)
                continue

            result, ok = _try_case_study(
                sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
                eta=EXP2_ETA, epsilon=EXP2_EPSILON,
                gamma=float(gamma), rho=EXP2_RHO,
                alpha=EXP2_ALPHA, lam=lam,
            )
            if ok:
                ps_list.append(result.p_star)
                ds_list.append(result.delta_star)
            else:
                ps_list.append(np.nan)
                ds_list.append(np.nan)

        ps_arr = np.array(ps_list, dtype=float)
        ds_arr = np.array(ds_list, dtype=float)

        ax_p.plot(EXP2_GAMMA_SWEEP, ps_arr, label=label, lw=2)
        ax_d.plot(EXP2_GAMMA_SWEEP, ds_arr, label=label, lw=2)

        if lam == 0.0:
            p_stars_mid = [float(v) for v in ps_arr]
            d_stars_mid = [float(v) for v in ds_arr]

    ax_p.set_xlabel("γ"); ax_p.set_ylabel("p*"); ax_p.set_title("Indifference price p* vs γ")
    ax_p.legend(); ax_p.grid(True, alpha=0.3)
    ax_d.set_xlabel("γ"); ax_d.set_ylabel("δ*"); ax_d.set_title("Optimal hedge δ* vs γ")
    ax_d.legend(); ax_d.grid(True, alpha=0.3)
    fig.suptitle("EXP2: γ sweep")
    fig.tight_layout()
    path = results_dir / "exp2_gamma_sweep.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\n--- γ sweep --- (plot → {path})")
    print(f"  Valid mid-case points: {np.sum(np.isfinite(p_stars_mid))}/{len(p_stars_mid)}")

    return p_stars_mid, d_stars_mid


# ── Step 4: ρ sweep ────────────────────────────────────────────────────────────


def run_rho_sweep(results_dir: Path) -> list[float]:
    """Vary ρ, compute p*, δ*, χ^σ."""
    ps_list: list[float] = []
    ds_list: list[float] = []
    xs_list: list[float] = []

    for rho in EXP2_RHO_SWEEP:
        result, ok = _try_case_study(
            sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
            eta=EXP2_ETA, epsilon=EXP2_EPSILON,
            gamma=EXP2_GAMMA, rho=float(rho),
            alpha=EXP2_ALPHA, lam=EXP2_LAMBDA,
        )
        ps_list.append(result.p_star if ok else np.nan)
        ds_list.append(result.delta_star if ok else np.nan)
        xs_list.append(result.chi_sigma if ok else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, ylabel, title in zip(
        axes,
        [ps_list, ds_list, xs_list],
        ["p*", "δ*", "χ^σ"],
        ["Indifference price", "Optimal hedge", "Dimensionality penalty"],
    ):
        ax.plot(EXP2_RHO_SWEEP, vals, lw=2, color="C0")
        ax.set_xlabel("ρ"); ax.set_ylabel(ylabel)
        ax.set_title(f"EXP2: {title} vs ρ")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = results_dir / "exp2_rho_sweep.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\n--- ρ sweep --- (plot → {path})")
    return [float(v) for v in ps_list]


# ── Step 5: σ_b sweep ─────────────────────────────────────────────────────────


def run_sigmab_sweep(results_dir: Path) -> list[float]:
    """Vary σ_b, compute p*, δ*, χ^s."""
    ps_list: list[float] = []
    ds_list: list[float] = []
    xs_list: list[float] = []

    for sigma_b in EXP2_SIGMA_B_SWEEP:
        result, ok = _try_case_study(
            sigma_q=EXP2_SIGMA_Q, sigma_b=float(sigma_b),
            eta=EXP2_ETA, epsilon=EXP2_EPSILON,
            gamma=EXP2_GAMMA, rho=EXP2_RHO,
            alpha=EXP2_ALPHA, lam=EXP2_LAMBDA,
        )
        ps_list.append(result.p_star if ok else np.nan)
        ds_list.append(result.delta_star if ok else np.nan)
        xs_list.append(result.chi_s if ok else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, ylabel, title in zip(
        axes,
        [ps_list, ds_list, xs_list],
        ["p*", "δ*", "χ^s"],
        ["Indifference price", "Optimal hedge", "Funding penalty"],
    ):
        ax.plot(EXP2_SIGMA_B_SWEEP, vals, lw=2, color="C1")
        ax.set_xlabel("σ_b"); ax.set_ylabel(ylabel)
        ax.set_title(f"EXP2: {title} vs σ_b")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = results_dir / "exp2_sigmab_sweep.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\n--- σ_b sweep --- (plot → {path})")
    return [float(v) for v in ps_list]


# ── Step 6: Monte Carlo P&L distributions ─────────────────────────────────────


def run_pl_distributions(
    base: "Any", results_dir: Path
) -> "tuple[float, float, float]":
    """Simulate paths and plot KDE of P&L for three hedge strategies."""
    rng = np.random.default_rng(EXP2_SEED)
    paths = simulate_case_study_paths(
        n_paths=EXP2_N_PATHS,
        sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
        eta=EXP2_ETA, epsilon=EXP2_EPSILON,
        gamma=EXP2_GAMMA, rho=EXP2_RHO,
        rng=rng,
    )

    # Case A: optimal hedge at base p*
    pl_opt = compute_pl_distribution(paths, delta=base.delta_star, p=base.p_star)

    # Case B: zero hedge, same initial price
    pl_zero = compute_pl_distribution(paths, delta=0.0, p=base.p_star)

    # Case C: γ=0 complete market (no convexity risk) — uses same factor paths but
    # P is re-computed with γ=0; simulate a fresh set of paths with gamma=0.
    rng2 = np.random.default_rng(EXP2_SEED)   # same seed for comparability
    paths_g0 = simulate_case_study_paths(
        n_paths=EXP2_N_PATHS,
        sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
        eta=EXP2_ETA, epsilon=EXP2_EPSILON,
        gamma=0.0, rho=EXP2_RHO,
        rng=rng2,
    )
    cs_g0, ok_g0 = _try_case_study(
        sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
        eta=EXP2_ETA, epsilon=EXP2_EPSILON,
        gamma=0.0, rho=EXP2_RHO,
        alpha=EXP2_ALPHA, lam=EXP2_LAMBDA,
    )
    pl_g0 = compute_pl_distribution(
        paths_g0,
        delta=cs_g0.delta_star if ok_g0 else 0.0,
        p=cs_g0.p_star if ok_g0 else base.p_star,
    )

    mc_pl_mean_optimal: float = float(np.mean(pl_opt))
    mc_pl_std_optimal: float = float(np.std(pl_opt))
    mc_pl_mean_zero: float = float(np.mean(pl_zero))

    print(f"\n--- MC P&L distributions ({EXP2_N_PATHS:,} paths) ---")
    print(f"  Optimal hedge  : mean = {mc_pl_mean_optimal:.4f},  std = {mc_pl_std_optimal:.4f}")
    print(f"  Zero hedge     : mean = {mc_pl_mean_zero:.4f},  std = {np.std(pl_zero):.4f}")
    if ok_g0:
        print(f"  γ=0 complete   : mean = {np.mean(pl_g0):.4f},  std = {np.std(pl_g0):.4f}")

    # KDE plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for pl_arr, label, color in [
        (pl_opt, f"Optimal δ*={base.delta_star:.2f}", "C0"),
        (pl_zero, "Zero hedge δ=0", "C1"),
        (pl_g0, "γ=0 (no convexity)", "C2"),
    ]:
        kde = stats.gaussian_kde(pl_arr, bw_method="scott")
        xs = np.linspace(pl_arr.min(), pl_arr.max(), 400)
        ax.plot(xs, kde(xs), lw=2, label=label, color=color)
        ax.axvline(np.mean(pl_arr), ls="--", lw=1, color=color, alpha=0.7)

    ax.set_xlabel("Portfolio P&L  (P − δQ − pB)")
    ax.set_ylabel("Density")
    ax.set_title("EXP2: P&L distributions for different hedge strategies")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = results_dir / "exp2_pl_distributions.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  KDE plot saved → {path}")

    return mc_pl_mean_optimal, mc_pl_std_optimal, mc_pl_mean_zero


# ── Step 7: P&L decomposition check ───────────────────────────────────────────


def run_decomposition_check(base: "Any") -> float:
    """Verify pathwise: P − δ*Q − p*B = const + X_gamma + X_sigma + X_s."""
    rng = np.random.default_rng(EXP2_SEED + 1)   # fresh seed to avoid reuse
    paths = simulate_case_study_paths(
        n_paths=EXP2_N_PATHS,
        sigma_q=EXP2_SIGMA_Q, sigma_b=EXP2_SIGMA_B,
        eta=EXP2_ETA, epsilon=EXP2_EPSILON,
        gamma=EXP2_GAMMA, rho=EXP2_RHO,
        rng=rng,
    )

    delta_star: float = base.delta_star
    p_star: float = base.p_star

    # Full portfolio P&L: P − δ*Q − p*B
    pl: np.ndarray = compute_pl_distribution(paths, delta=delta_star, p=p_star)

    # W = Q + ρ σ_q Z  (the "total" swap factor driving the derivative)
    W: np.ndarray = paths["Q"] + EXP2_RHO * EXP2_SIGMA_Q * paths["Z"]
    E_W2: float = EXP2_SIGMA_Q ** 2 * (1.0 + EXP2_RHO ** 2)

    # Pathwise attribution components (algebraic, not statistical)
    const: float = EXP2_ETA - p_star + 0.5 * EXP2_GAMMA * E_W2
    X_gamma: np.ndarray = 0.5 * EXP2_GAMMA * (W ** 2 - E_W2)
    X_sigma: np.ndarray = (EXP2_EPSILON - delta_star) * paths["Q"] + EXP2_EPSILON * EXP2_RHO * EXP2_SIGMA_Q * paths["Z"]
    X_s: np.ndarray = -p_star * EXP2_SIGMA_B * paths["Z_b"]

    residual: np.ndarray = pl - const - X_gamma - X_sigma - X_s
    res_mean: float = float(np.mean(residual))
    res_std: float = float(np.std(residual))

    print(f"\n--- P&L decomposition check ---")
    print(f"  Residual mean : {res_mean:.2e}  (expected ≈ 0)")
    print(f"  Residual std  : {res_std:.2e}  (expected ≈ machine epsilon)")
    if res_std < 1e-10:
        print(f"  Identity check: PASS")
    else:
        print(f"  Identity check: FAIL (std = {res_std:.2e} exceeds 1e-10)")

    return res_std


# ── Step 8: Save results ───────────────────────────────────────────────────────


def save_results(
    results_dir: Path,
    base: "Any",
    best_delta: float,
    gamma_p_stars: list[float],
    rho_p_stars: list[float],
    sigmab_p_stars: list[float],
    mc_pl_mean_optimal: float,
    mc_pl_std_optimal: float,
    mc_pl_mean_zero: float,
    decomp_residual_std: float,
) -> None:
    output: dict[str, Any] = {
        "base_p_star": base.p_star,
        "base_delta_star": base.delta_star,
        "base_chi_gamma": base.chi_gamma,
        "base_chi_sigma": base.chi_sigma,
        "base_chi_s": base.chi_s,
        "delta_sweep_optimal_delta": best_delta,
        "gamma_sweep_p_stars": gamma_p_stars,
        "rho_sweep_p_stars": rho_p_stars,
        "sigmab_sweep_p_stars": sigmab_p_stars,
        "mc_pl_mean_optimal": mc_pl_mean_optimal,
        "mc_pl_std_optimal": mc_pl_std_optimal,
        "mc_pl_mean_zero_hedge": mc_pl_mean_zero,
        "decomposition_residual_std": decomp_residual_std,
    }
    path = results_dir / "exp2_results.json"
    with path.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved → {path}")


# ── entry point ────────────────────────────────────────────────────────────────


def run_exp2() -> None:
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    base = run_base_case()
    _, best_delta = run_hedge_scan(results_dir)
    gamma_p_stars, _ = run_gamma_sweep(results_dir)
    rho_p_stars = run_rho_sweep(results_dir)
    sigmab_p_stars = run_sigmab_sweep(results_dir)
    mc_pl_mean_opt, mc_pl_std_opt, mc_pl_mean_zero = run_pl_distributions(base, results_dir)
    decomp_std = run_decomposition_check(base)

    save_results(
        results_dir=results_dir,
        base=base,
        best_delta=best_delta,
        gamma_p_stars=gamma_p_stars,
        rho_p_stars=rho_p_stars,
        sigmab_p_stars=sigmab_p_stars,
        mc_pl_mean_optimal=mc_pl_mean_opt,
        mc_pl_std_optimal=mc_pl_std_opt,
        mc_pl_mean_zero=mc_pl_mean_zero,
        decomp_residual_std=decomp_std,
    )
    print("\n=== EXP2 complete ===")


if __name__ == "__main__":
    run_exp2()
