# Model Risk Analysis for Entropic Hedging Strategies

**Paper:** Paul McCloud, "Model Risk Analysis for Entropic Hedging Strategies,"
McCloud Research (2 November 2026).

This repository replicates the four core experiments of the paper, which derives
exact price and hedge ratios from entropic risk optimisation for quadratic
Gaussian prices, decomposes entropic risk into convexity, dimensionality, funding
and strategy penalties, and studies their behaviour in the continuous-hedging
limit for Gaussian processes including fractional Ornstein-Uhlenbeck processes.

---

## Repository layout

```
.
├── README.md
├── AGENTS.md
├── requirements.txt
├── reproduce.sh
├── .env.example          (not needed: experiments use synthetic data only)
├── src/                  # Library modules
│   ├── config.py         # All experiment parameters
│   ├── entropic_pricing.py  # Single-interval quadratic Gaussian formulas
│   ├── case_study.py        # Bond-and-swap case study (exp 2)
│   ├── gaussian_process.py  # Multi-interval Gaussian-process solver (exp 3)
│   └── fractional_ou.py     # Discrete fractional OU driver (exp 4)
├── exp/                  # Experiment scripts
│   ├── exp1.py           # Single-interval entropic pricing and decomposition
│   ├── exp2.py           # Bond-and-swap case study
│   ├── exp3.py           # Multi-interval GP hedging and asymptotics
│   └── exp4.py           # Fractional OU continuous-limit scaling
├── tests/                # pytest suite
│   ├── conftest.py
│   ├── test_entropic_pricing.py
│   ├── test_case_study.py
│   ├── test_gaussian_process.py
│   └── test_fractional_ou.py
├── data/
│   ├── __init__.py
│   └── generate_synthetic.py  # Synthetic-data helpers used in tests
└── results/              # All output artefacts (tables, plots, RESULTS.md)
```

All experiments use synthetic (closed-form model) inputs. No market data API
is required.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running experiments

Run from the repository root:

```bash
python -m exp.exp1          # Single-interval quadratic Gaussian
python -m exp.exp2          # Bond-and-swap case study
python -m exp.exp3          # Multi-interval Gaussian-process hedging
python -m exp.exp4          # Fractional OU continuous-limit scaling
```

Each experiment writes its results (JSON metrics + PNG plots) to `results/`.

---

## Tests

```bash
python -m pytest                          # full suite
python -m pytest -m "not market_data"     # skip (no) market-data tests
```

---

## Reproduce all results

```bash
bash reproduce.sh
```

---

## Results

Results and comparisons with the paper are in [`results/RESULTS.md`](results/RESULTS.md).
