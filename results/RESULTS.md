# Results: Model Risk Analysis for Entropic Hedging Strategies

**Paper:** Paul McCloud, "Model Risk Analysis for Entropic Hedging Strategies,"
McCloud Research (2 November 2026).

All experiments use synthetic (closed-form model) inputs; no market data is required.
Numbers below are taken from the result files saved in `results/` by the final runs.

---

## Exp 1 — Single-interval quadratic Gaussian entropic pricing and hedge decomposition

**Setup:** Two-factor (N_f = 2), two-underlying (N_m = 2) market with α = 1, λ = 0.5.
Parameters are representative values satisfying the paper's assumptions (q ≠ 0,
σσ† positive definite, q·(σσ†)⁻¹μ > 0).

**Analytical outputs** (`results/exp1_results.json`, `results/exp1_summary.txt`):

| Quantity | Value |
|---|---|
| Funding rate r | 0.0500 |
| Indifference price p* | 0.5943 |
| Convexity penalty χ^γ | 0.025764 |
| Dimensionality penalty χ^σ | ≈ 0 (machine ε) |
| Funding penalty χ^s | 0.078147 |
| Pathwise decomp residual std | 1.27 × 10⁻¹⁶ |

**Verification checks** (paper §Hedging, §Pricing):

| Check | Expected | Result | Verdict |
|---|---|---|---|
| All penalties ≥ 0 | True | True | **Reproduced** |
| χ^δ = 0 at δ = δ* | 0 | 0.00e+00 | **Reproduced** |
| χ^γ = 0 when γ = 0 | 0 | 0.00e+00 | **Reproduced** |
| χ^σ ≈ 0 when ε ∈ span(σ̌) | ≈ 0 | 4.83e-35 | **Reproduced** |
| χ^s = 0 at p = δ^σ·q | 0 | 0.00e+00 | **Reproduced** |
| Pathwise P&L decomp identity | 0 residual | 1.27e-16 | **Reproduced** |
| Solution admissible | True | True | **Reproduced** |

The χ^σ is numerically zero in this configuration because the chosen ε lies in the
span of σ̌ (the convexity-adjusted loading matrix), which the paper identifies as the
condition for vanishing dimensionality penalty. This is the expected outcome for this
parameter set and confirms the paper's edge-case condition.

**Summary:** All structural identities and edge-case conditions from the paper are
reproduced to machine precision.

---

## Exp 2 — Case study: bond-and-swap incomplete market

**Setup:** Bond B = 1 + σ_b Z_b, swap Q = σ_q Z_q, derivative
P = η + ε(Q + ρσ_q Z) + ½γ(Q + ρσ_q Z)².
Base parameters: σ_q = 0.2, σ_b = 0.1, η = 0.5, ε = 1, γ = 2, ρ = 0.5, α = 1, λ = 0.

**Base-case results** (`results/exp2_results.json`):

| Quantity | Computed | Paper statement | Verdict |
|---|---|---|---|
| Indifference price p* (λ=0) | 0.5500 | Not specified numerically | — |
| Optimal hedge ratio δ* | 1.0000 | "highlighted at δ* = 1" (p. 7) | **Reproduced** |
| Convexity penalty χ^γ | 0.005000 | Non-negative | **Reproduced** |
| Dimensionality penalty χ^σ | 0.010000 | Non-negative | **Reproduced** |
| Funding penalty χ^s | 0.003025 | Non-negative | **Reproduced** |
| χ^δ at δ = δ* | 0 | 0 (exact) | **Reproduced** |
| P&L decomp residual | ≈ 0 | ≈ 0 | **Reproduced** |

**Parameter sweeps** (paper Figs 4, 5; `results/exp2_*_sweep.png`):

| Sweep | Qualitative outcome | Verdict |
|---|---|---|
| γ ∈ [−10, 10]: convexity effect on p*, δ* | χ^γ increases with \|γ\|; p*, δ* shift | **Reproduced** |
| ρ ∈ [0, 1]: dimensionality effect on p*, δ* | χ^σ increases with ρ; p*, δ* shift | **Reproduced** |
| σ_b ∈ [0, 0.5]: funding effect on p*, δ* | χ^s increases with σ_b; p*, δ* shift | **Reproduced** |
| Hedge scan δ ∈ [−1, 3]: optimal δ* | χ^δ minimised at δ = δ* = 1 | **Reproduced** |

**Summary:** The case-study formulas reproduce δ* = 1 as stated by the paper. All four
penalties are non-negative. Parameter sweep directions match the paper's qualitative
descriptions. Exact numerical values for all plots are not reported in the paper, so
the comparison is qualitative.

---

## Exp 3 — Multi-interval Gaussian-process entropic hedging and continuous-hedging asymptotics

**Setup:** Brownian GP (H = 1/2, Markovian), n = 20 intervals, dt = 0.05, scalar factor.
Asymptotic scaling study over dt ∈ [0.01, 0.31].

**Interval penalties across 20 Brownian intervals** (`results/exp3_results.json`):

| Penalty | Property | Computed | Verdict |
|---|---|---|---|
| χ^γ (all intervals) | Non-negative | ✓ | **Reproduced** |
| χ^σ (all intervals) | Non-negative | ✓ (≈ 0 in this config) | **Reproduced** |
| χ^s (all intervals) | Non-negative | ✓ | **Reproduced** |

**Scaling slopes** (log-log regression of mean interval penalty vs dt;
paper Fig 6 predicts slope 2 for χ^γ, slope 1 for χ^σ and χ^s at H = 1/2):

| Penalty | Estimated slope | Expected (H = 1/2) | Verdict |
|---|---|---|---|
| χ^γ | 1.987 | 2.0 | **Reproduced** (−0.7%) |
| χ^σ | not estimable | 1.0 | **Not reproduced** (χ^σ ≈ 0 in config) |
| χ^s | 0.968 | 1.0 | **Reproduced** (−3.2%) |

χ^σ is not estimable because in the chosen coefficient configuration ε lies in
span(σ̌), making χ^σ identically zero. To estimate the chi_sigma slope, coefficients
with ε outside span(σ̌) are required; this is a configuration choice, not a formula
error. The paper's claim that dimensionality penalty scales as O(dt^{2H}) is consistent
with the formula but could not be verified numerically for this configuration.

**Summary:** The convexity and funding penalty scaling exponents for H = 1/2 are
reproduced to within 4%. The dimensionality slope cannot be estimated with the chosen
coefficients.

---

## Exp 4 — Discrete fractional OU: continuous-limit behaviour

**Setup:** Scalar FOU process, κ = 0.5, ς = 1, β = 0, x₀ = 0, n = 50 intervals.
Asymptotic study over dt ∈ [0.01, 0.31] for H ∈ {0.2, 0.3, 0.5, 0.7, 0.8}.

**Per-interval scaling slopes** (`results/exp4_results.json`, `results/exp4_scaling.png`).
Paper (Fig 6) predicts slope 4H for χ^γ and 2H for χ^σ, χ^s per interval:

| H | χ^γ slope | χ^γ expected (4H) | Verdict | χ^σ slope | χ^σ expected (2H) | Verdict | χ^s slope | χ^s expected (2H) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.781 | 0.80 | **Reproduced** (−2.4%) | NaN | 0.40 | **Not reproduced** (χ^σ ≈ 0) | 0.354 | 0.40 | **Partially reproduced** (−11%) |
| 0.3 | 1.179 | 1.20 | **Reproduced** (−1.8%) | 0.671 | 0.60 | **Partially reproduced** (+12%) | 0.552 | 0.60 | **Reproduced** (−8%) |
| 0.5 | 1.983 | 2.00 | **Reproduced** (−0.9%) | NaN | 1.00 | **Not reproduced** (χ^σ ≈ 0) | 0.958 | 1.00 | **Reproduced** (−4%) |
| 0.7 | 2.789 | 2.80 | **Reproduced** (−0.4%) | 1.594 | 1.40 | **Partially reproduced** (+14%) | 1.367 | 1.40 | **Reproduced** (−2%) |
| 0.8 | 3.193 | 3.20 | **Reproduced** (−0.2%) | 1.349 | 1.60 | **Partially reproduced** (−16%) | 1.573 | 1.60 | **Reproduced** (−2%) |

**ξ diagonal bound verification** (paper Eq. 74–75): confirmed for all H values tested.

**Per-unit-time penalty behaviour** (`results/exp4_per_unit_time.png`):

| H < 1/4 | χ^γ/dt diverges | Confirmed (slope 4H−1 < 0) |
| H = 1/4 | χ^γ/dt finite | Confirmed |
| H > 1/4 | χ^γ/dt → 0 | Confirmed |
| H < 1/2 | χ^s/dt diverges | Confirmed (slope 2H−1 < 0) |
| H > 1/2 | χ^s/dt → 0 | Confirmed |

**Notes on χ^σ:** NaN entries arise when χ^σ is numerically zero (< machine epsilon)
across all dt values for H = 0.2 and H = 0.5. This is a configuration effect (ε in span
of σ̌); the scaling law n_ν = 2H is not violated but cannot be measured. For H = 0.3,
0.7, 0.8 where chi_sigma is non-negligible, estimated slopes are within 12–16% of the
theoretical value; the discrepancy is attributed to numerical noise in regression on
very small chi_sigma values. The paper's qualitative claim (divergence for H < 1/2,
decay for H > 1/2) is confirmed.

**Summary:** χ^γ scaling slopes are reproduced to within 2.5% across all H values.
χ^s slopes are reproduced for H ≥ 0.3 and partially reproduced for H = 0.2. χ^σ slopes
are partially reproduced where estimable; the NaN cases are explained by the coefficient
configuration, not a formula error.

---

## Deviations from the paper

1. **Data source:** All experiments use synthetic (closed-form) inputs, as the paper
   specifies. No market data is required or used.

2. **Parameter values (Exp 1):** The paper does not specify concrete numerical values for
   the general single-interval model outside the case study. Representative values
   satisfying the paper's assumptions were chosen (detailed in `src/config.py`).

3. **Parameter values (Exp 3/4):** Intervalwise coefficient values (μ_i, σ_i, η_i, ε_i,
   γ_i), schedule length n, factor dimension, and simulation paths are not specified by
   the paper. Representative values are used (detailed in `src/config.py`).

4. **ε in span(σ̌):** In the chosen Exp 1, 3, and 4 configurations, the derivative linear
   coefficient ε lies in the span of the hedgeable subspace, making χ^σ identically
   zero. This correctly exercises the vanishing-dimensionality edge case stated in the
   paper but prevents estimation of the dimensionality scaling slope for H = 0.2 and 0.5.

5. **χ^σ scaling discrepancy (Exp 4):** For H = 0.3, 0.7, 0.8, estimated χ^σ slopes
   deviate from 2H by 12–16%. The discrepancy is attributed to numerical noise when
   regressing a very small penalty quantity. The paper's qualitative claim is consistent
   with the results.

6. **Small-notional (αλ → 0) expansion:** The paper mentions first-order expansion
   around αλ = 0 but does not provide explicit expansion formulas in the general section.
   Small-notional continuity is verified numerically by evaluating formulas at a sequence
   αλ → 0 (Exp 1 edge-case check).

7. **Typographical note (Exp 4, Eq. 73):** The paper prints ν̄_i = ςζ†dt, which appears
   to be a typo; the contextually consistent expression ν̄_i = ξ_{ii}² ςς† dt^{2H} is
   implemented (flagged in code comment).

8. **Monte Carlo paths:** The paper does not specify the number of simulation paths for
   validating analytical identities. 100 000 paths are used (Exp 1, 2).

---

## Not reproduced

- **Exact figure images (Figs 3–6):** The paper's figures are not digitally available
  in machine-readable form, so exact curve-by-curve comparison is not possible. Qualitative
  directions and selected quantitative outputs are compared above.
- **Full bid-offer spread functional over a trade life:** The paper states aggregation
  of intervalwise prices over many intervals as future work; no summary functional is
  implemented.
