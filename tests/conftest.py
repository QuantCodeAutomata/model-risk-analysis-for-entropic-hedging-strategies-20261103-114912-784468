"""Shared pytest fixtures and markers."""

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data: tests that require live API keys (DATALOADER_API_KEY, MASSIVE_API_KEY)",
    )


@pytest.fixture(scope="session")
def small_model_params() -> dict:
    """Minimal 1-factor, 1-underlying model satisfying paper assumptions."""
    return dict(
        q=np.array([1.0]),
        mu=np.array([1.05]),
        sigma=np.array([[0.20]]),   # σσ† = 0.04 > 0
        eta=0.5,
        epsilon=np.array([0.30]),
        gamma=np.array([[0.10]]),   # symmetric, I+αγ = 1.1 > 0
        alpha=1.0,
        lam=0.5,
    )


@pytest.fixture(scope="session")
def two_factor_model_params() -> dict:
    """2-factor, 2-underlying model from config (used by exp1)."""
    return dict(
        q=np.array([1.0, 0.8]),
        mu=np.array([1.05, 0.84]),
        sigma=np.array([[0.15, 0.05], [0.08, 0.20]]),
        eta=0.5,
        epsilon=np.array([0.3, 0.4]),
        gamma=np.array([[0.2, 0.05], [0.05, 0.1]]),
        alpha=1.0,
        lam=0.5,
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
