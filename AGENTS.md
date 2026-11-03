# AGENTS.md — Guide for coding agents and contributors

## What this repository implements

Replication of: Paul McCloud, "Model Risk Analysis for Entropic Hedging Strategies,"
McCloud Research (2 November 2026).

The paper derives exact price and hedge ratios from entropic risk optimisation for
quadratic Gaussian prices. It decomposes entropic risk into convexity, dimensionality,
funding and strategy penalties and studies their behaviour in the continuous-hedging
limit for Gaussian processes, including discrete fractional Ornstein-Uhlenbeck processes.

**Experiments:**
- `exp1` — Single-interval quadratic Gaussian entropic pricing and hedge decomposition
- `exp2` — Bond-and-swap incomplete market case study
- `exp3` — Multi-interval Gaussian-process hedging and continuous-hedging asymptotics
- `exp4` — Fractional OU continuous-limit scaling under rough and persistent dynamics

## Layout

```
src/                  Flat library modules (no subdirectories)
  config.py           All experiment parameters as module-level UPPER_CASE variables
  entropic_pricing.py Single-interval quadratic Gaussian formulas (exp1, exp2)
  case_study.py       Bond-and-swap scalar case study helpers (exp2)
  gaussian_process.py Multi-interval GP filtered-Cholesky and interval solver (exp3)
  fractional_ou.py    Discrete fractional OU driver (exp4)
exp/                  Experiment scripts (python -m exp.<name>)
tests/                All pytest files
data/                 Synthetic-data generators; no downloaded data committed
results/              All output artefacts: JSON, PNG, RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest                       # full suite (no market-data tests exist)
python -m pytest -m "not market_data"  # same (no market-data marker used)
python -m pytest tests/test_entropic_pricing.py   # one module
```

## Running experiments

All experiments run from the repository root; they write to `results/`.

```bash
python -m exp.exp1    # Single-interval model — writes exp1_results.json, exp1_summary.txt
python -m exp.exp2    # Case study — writes exp2_*.json + *.png
python -m exp.exp3    # Multi-interval GP — writes exp3_results.json, exp3_scaling.png
python -m exp.exp4    # Fractional OU — writes exp4_results.json, exp4_scaling.png, exp4_per_unit_time.png
bash reproduce.sh     # Tests then all four experiments
```

## Conventions

- **Parameters only in `src/config.py`:** every tunable value lives there as an
  `UPPER_CASE` module-level variable with a comment giving its meaning, unit, and
  paper source (section/equation or "not specified in the paper").
- **Data only through `data/`:** `data/generate_synthetic.py` provides all synthetic
  inputs. No market-data loaders are needed; never write a private loader in `src/`
  or `exp/`.
- **Test rules:** unit tests use small synthetic fixtures; no live API calls. The
  `market_data` pytest marker is registered in `tests/conftest.py` but no test
  carries it (all data is synthetic).
- **API keys:** not required. `.env.example` is not needed for this replication.
- **Rerunning experiments:** `python -m exp.expN` overwrites the files in `results/`.
  After a rerun, check `results/RESULTS.md` and update numbers and verdicts by hand
  to match the new files.
- **No secrets in files:** API keys, tokens, and credentials must never appear in
  source code, configuration, logs, result files, or commits.
