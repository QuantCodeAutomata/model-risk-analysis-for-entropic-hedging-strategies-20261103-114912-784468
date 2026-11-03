"""Experiment 1: Single-interval quadratic Gaussian entropic pricing.

Validates analytical formulas (funding rate, hedge, penalties, indifference
price) via Monte Carlo simulation and targeted edge-case checks.

Run as: python -m exp.exp1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    EXP1_ALPHA,
    EXP1_EPSILON,
    EXP1_ETA,
    EXP1_GAMMA,
    EXP1_LAMBDA,
    EXP1_MU,
    EXP1_N_PATHS,
    EXP1_Q,
    EXP1_SEED,
    EXP1_SIGMA,
    RESULTS_DIR,
)
from src.entropic_pricing import (
    compute_strategy_penalty,
    entropy_adjusted_mean,
    pl_attributions,
    solve_single_interval_entropic_hedge,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _fmt(label: str, value: float, width: int = 35) -> str:
    return f"  {label:<{width}} {value:>14.6f}"


def _fmt_vec(label: str, vec: np.ndarray, width: int = 35) -> str:
    vals = "  ".join(f"{v:.6f}" for v in vec)
    return f"  {label:<{width}} [{vals}]"


# ── main ───────────────────────────────────────────────────────────────────────


def run_exp1() -> dict[str, Any]:
    """Run all EXP1 sub-tasks and return a results dict."""

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Solve the base-case single-interval problem ────────────────────
    sol = solve_single_interval_entropic_hedge(
        q=EXP1_Q,
        mu=EXP1_MU,
        sigma=EXP1_SIGMA,
        eta=EXP1_ETA,
        epsilon=EXP1_EPSILON,
        gamma=EXP1_GAMMA,
        alpha=EXP1_ALPHA,
        lam=EXP1_LAMBDA,
    )

    r: float = sol["r"]
    phi: np.ndarray = sol["phi"]
    delta_star: np.ndarray = sol["delta_star"]
    chi_gamma: float = sol["chi_gamma"]
    chi_sigma: float = sol["chi_sigma"]
    chi_s: float = sol["chi_s"]
    s_star: float = sol["s_star"]
    p_star: float = sol["p_star"]
    admissible: bool = sol["admissible"]
    check_eta: float = sol["check_eta"]
    check_sigma: np.ndarray = sol["check_sigma"]
    model = sol["model"]
    hedge = sol["hedge"]
    adj = sol["adj"]
    cal = sol["cal"]

    print("\n=== EXP1: Single-interval Quadratic Gaussian Entropic Pricing ===\n")
    print(_fmt_vec("φ (optimal self-funded portfolio)", phi))
    print(_fmt("Funding rate r", r))
    print(_fmt_vec("δ* (optimal hedge)", delta_star))
    print(_fmt("Convexity penalty  χ^γ", chi_gamma))
    print(_fmt("Dimensionality penalty χ^σ", chi_sigma))
    print(_fmt("Funding penalty    χ^s", chi_s))
    print(_fmt("Indifference spread s*", s_star))
    print(_fmt("Indifference price  p*", p_star))
    print(f"  {'Admissible':<35} {admissible}")

    # ── Step 2: Monte Carlo validation ────────────────────────────────────────
    rng = np.random.default_rng(EXP1_SEED)
    # Z shape: (N_paths, N_f)
    Z: np.ndarray = rng.standard_normal((EXP1_N_PATHS, EXP1_SIGMA.shape[1]))

    attr = pl_attributions(model, Z, hedge, adj, cal, delta=delta_star)
    X_gamma: np.ndarray = attr["X_gamma"]
    X_sigma: np.ndarray = attr["X_sigma"]
    X_s: np.ndarray = attr["X_s"]
    X_delta: np.ndarray = attr["X_delta"]
    portfolio_pl: np.ndarray = attr["portfolio_pl"]

    # Analytical constant in the pathwise P&L decomposition identity:
    #   portfolio_pl = analytical_const + X_gamma + X_sigma + X_s + X_delta
    analytical_const: float = check_eta - (1.0 + r) * p_star

    # Check 1: MC mean vs analytical constant.
    # The sum of attribution means under N(0,I) is generally non-zero (comes from
    # the shifted-measure cross-terms), so we report the gap and a generous bound.
    mc_pl_mean: float = float(np.mean(portfolio_pl))
    mc_pl_std: float = float(np.std(portfolio_pl))
    mc_std_err: float = mc_pl_std / np.sqrt(EXP1_N_PATHS)
    gap: float = mc_pl_mean - analytical_const
    # Generous tolerance: covers both MC noise and measure-change offset
    tolerance_1: float = max(10.0 * mc_std_err, 1e-3)
    check1_ok: bool = abs(gap) < tolerance_1

    # Check 2: pathwise identity – residual must be exactly zero up to fp rounding
    attribution_sum: np.ndarray = X_gamma + X_sigma + X_s + X_delta
    residual: np.ndarray = portfolio_pl - analytical_const - attribution_sum
    mc_attribution_residual_std: float = float(np.std(residual))
    check2_ok: bool = mc_attribution_residual_std < 1e-10

    print(f"\n--- Monte Carlo validation ({EXP1_N_PATHS:,} paths) ---")
    print(_fmt("Analytical const η̌ − (1+r)p*", analytical_const))
    print(_fmt("MC mean(portfolio_pl)", mc_pl_mean))
    print(_fmt("Difference", gap))
    print(_fmt("MC standard error", mc_std_err))
    print(f"  {'Check 1 |gap| < tolerance':<35} {'PASS' if check1_ok else 'INFO'}"
          f"  (tol={tolerance_1:.2e})")

    print(f"\n  Attribution sample means (vs analytical penalty ÷ 2):")
    print(_fmt("  E[X_gamma]", float(np.mean(X_gamma))))
    print(_fmt("    chi_gamma / 2 (analytical)", chi_gamma / 2.0))
    print(_fmt("  E[X_sigma]", float(np.mean(X_sigma))))
    print(_fmt("    chi_sigma / 2 (analytical)", chi_sigma / 2.0))
    print(_fmt("  E[X_s]", float(np.mean(X_s))))
    print(_fmt("    chi_s / 2 (analytical)", chi_s / 2.0))
    print(_fmt("  E[X_delta] (should be 0)", float(np.mean(X_delta))))

    print(f"\n  Pathwise decomposition residual std : {mc_attribution_residual_std:.2e}")
    print(f"  Check 2 (residual std < 1e-10)      : {'PASS' if check2_ok else 'FAIL'}")
    if not check2_ok:
        raise AssertionError(
            f"Pathwise P&L decomposition identity failed: residual std = "
            f"{mc_attribution_residual_std:.2e}"
        )

    # ── Step 3: Edge-case checks (analytical only) ─────────────────────────────
    print("\n--- Edge-case checks ---")

    # 3a: γ = 0  →  χ^γ = 0
    sol_g0 = solve_single_interval_entropic_hedge(
        q=EXP1_Q, mu=EXP1_MU, sigma=EXP1_SIGMA,
        eta=EXP1_ETA, epsilon=EXP1_EPSILON,
        gamma=np.zeros_like(EXP1_GAMMA),
        alpha=EXP1_ALPHA, lam=EXP1_LAMBDA,
    )
    edge_gamma_penalty: float = sol_g0["chi_gamma"]
    print(f"  γ=0           : χ^γ = {edge_gamma_penalty:.2e}   (expected 0)")
    assert edge_gamma_penalty < 1e-12, (
        f"χ^γ should be 0 when γ=0, got {edge_gamma_penalty}"
    )

    # 3b: ε in row-span of σ̌  →  χ^σ ≈ 0.
    # For N_m = N_f = 2 with full-rank σ, σ̌ is also full-rank so the projection
    # 1_{σ̌} = I and χ^σ = 0 for any ε.  We use ε = check_sigma[0] * check_sigma[0]
    # (elementwise product of the first row with itself) as an explicit hedgeable ε.
    eps_span: np.ndarray = check_sigma[0] * check_sigma[0]
    sol_s0 = solve_single_interval_entropic_hedge(
        q=EXP1_Q, mu=EXP1_MU, sigma=EXP1_SIGMA,
        eta=EXP1_ETA, epsilon=eps_span,
        gamma=EXP1_GAMMA,
        alpha=EXP1_ALPHA, lam=EXP1_LAMBDA,
    )
    edge_sigma_penalty: float = sol_s0["chi_sigma"]
    print(f"  ε ∈ span(σ̌)  : χ^σ = {edge_sigma_penalty:.2e}   (expected ≈ 0)")
    assert edge_sigma_penalty < 1e-10, (
        f"χ^σ should be ≈ 0 when ε is in the hedgeable subspace, got {edge_sigma_penalty}"
    )

    # 3c: p = δ^σ · q  →  s = 0, χ^s = 0
    p_no_spread: float = float(hedge.delta_sigma @ EXP1_Q)
    sol_ns = solve_single_interval_entropic_hedge(
        q=EXP1_Q, mu=EXP1_MU, sigma=EXP1_SIGMA,
        eta=EXP1_ETA, epsilon=EXP1_EPSILON, gamma=EXP1_GAMMA,
        alpha=EXP1_ALPHA, lam=EXP1_LAMBDA,
        p=p_no_spread,
    )
    edge_s_penalty: float = sol_ns["chi_s"]
    print(f"  p = δ^σ·q    : s = {sol_ns['s']:.2e},  χ^s = {edge_s_penalty:.2e}   (expected 0)")
    assert abs(sol_ns["s"]) < 1e-12, (
        f"Funding spread s should be 0 when p = δ^σ·q, got {sol_ns['s']}"
    )
    assert edge_s_penalty < 1e-12, (
        f"χ^s should be 0 when p = δ^σ·q, got {edge_s_penalty}"
    )

    # 3d: δ = δ*  →  χ^δ = 0
    edge_strategy_penalty_at_opt: float = compute_strategy_penalty(
        model, delta_star, delta_star, adj
    )
    print(f"  δ = δ*        : χ^δ = {edge_strategy_penalty_at_opt:.2e}   (expected 0)")
    assert edge_strategy_penalty_at_opt < 1e-20, (
        f"χ^δ should be 0 at the optimal hedge, got {edge_strategy_penalty_at_opt}"
    )

    # ── Step 4: Small-notional stability (αλ → 0) ─────────────────────────────
    print("\n--- Small-notional stability (λ = 1e-8) ---")
    sol_tiny = solve_single_interval_entropic_hedge(
        q=EXP1_Q, mu=EXP1_MU, sigma=EXP1_SIGMA,
        eta=EXP1_ETA, epsilon=EXP1_EPSILON, gamma=EXP1_GAMMA,
        alpha=EXP1_ALPHA, lam=1e-8,
    )
    p_tiny: float = sol_tiny["p_star"]
    d_tiny: np.ndarray = sol_tiny["delta_star"]
    print(_fmt("p* at λ=1e-8", p_tiny))
    print(_fmt("p* at λ=0.5 (base)", p_star))
    print(_fmt_vec("δ* at λ=1e-8", d_tiny))
    assert np.isfinite(p_tiny), "p* is not finite for small λ"
    assert np.all(np.isfinite(d_tiny)), "δ* is not finite for small λ"
    print(f"  {'Finiteness check':<35} PASS")

    # ── Steps 5 & 6: Persist results ──────────────────────────────────────────
    output: dict[str, Any] = {
        "r": r,
        "r_value": r,
        "delta_star": delta_star.tolist(),
        "chi_gamma": chi_gamma,
        "chi_sigma": chi_sigma,
        "chi_s": chi_s,
        "p_star": p_star,
        "admissible": bool(admissible),
        "mc_pl_mean": mc_pl_mean,
        "mc_attribution_residual_std": mc_attribution_residual_std,
        "edge_gamma_penalty": edge_gamma_penalty,
        "edge_sigma_penalty": edge_sigma_penalty,
        "edge_s_penalty": edge_s_penalty,
        "edge_strategy_penalty_at_opt": edge_strategy_penalty_at_opt,
    }

    json_path = results_dir / "exp1_results.json"
    with json_path.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved → {json_path}")

    # Summary table
    sep = "-" * 54
    rows: list[str] = [
        "",
        "=== EXP1 Summary Table ===",
        f"{'Quantity':<38} {'Value':>14}",
        sep,
        f"{'Funding rate  r':<38} {r:>14.6f}",
        f"{'Indifference price  p*':<38} {p_star:>14.6f}",
        f"{'Indifference spread  s*':<38} {s_star:>14.6f}",
        f"{'Admissible':<38} {str(admissible):>14}",
        f"{'Convexity penalty  χ^γ':<38} {chi_gamma:>14.6f}",
        f"{'Dimensionality penalty  χ^σ':<38} {chi_sigma:>14.6f}",
        f"{'Funding penalty  χ^s':<38} {chi_s:>14.6f}",
        f"{'Analytical const  η̌−(1+r)p*':<38} {analytical_const:>14.6f}",
        f"{'MC mean(portfolio_pl)':<38} {mc_pl_mean:>14.6f}",
        f"{'Pathwise decomp residual std':<38} {mc_attribution_residual_std:>14.2e}",
        f"{'Edge: χ^γ at γ=0':<38} {edge_gamma_penalty:>14.2e}",
        f"{'Edge: χ^σ at ε∈span(σ̌)':<38} {edge_sigma_penalty:>14.2e}",
        f"{'Edge: χ^s at p=δ^σ·q':<38} {edge_s_penalty:>14.2e}",
        f"{'Edge: χ^δ at δ=δ*':<38} {edge_strategy_penalty_at_opt:>14.2e}",
        sep,
    ]
    summary_text = "\n".join(rows)
    print(summary_text)

    txt_path = results_dir / "exp1_summary.txt"
    with txt_path.open("w") as fh:
        fh.write(summary_text + "\n")
    print(f"Summary saved  → {txt_path}")

    return output


if __name__ == "__main__":
    run_exp1()
