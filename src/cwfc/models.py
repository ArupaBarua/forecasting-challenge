"""Forecasters for the aggregate CPU series.

All functions use the same convention as ``cwfc.baselines``: they receive the
training frame, a ``template`` whose index defines the forecast rows, and the
target column name. They only ever see ``train``, so every value they use was
available at the forecast origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_K = 3
DEFAULT_TAU = 6.0


def _history(train: pd.DataFrame, target: str) -> np.ndarray:
    values = train[target].astype(float).dropna().to_numpy()
    if values.size == 0:
        raise ValueError(f"train has no valid observations of {target!r}")
    return values


def expanding_mean_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Constant forecast equal to the mean of the whole training history."""
    mu = float(_history(train, target).mean())
    return pd.Series(mu, index=template.index, dtype=float)


def shrinkage_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    k: int = DEFAULT_K,
    tau: float = DEFAULT_TAU,
) -> pd.Series:
    """Start at the recent level and decay toward the long-run mean.

    ``forecast(h) = mu + exp(-h / tau) * (recent - mu)`` for h = 1, 2, ...

    * ``mu``     mean of the whole training history (the anchor);
    * ``recent`` mean of the last ``k`` observations;
    * ``tau``    decay time constant in steps (6 steps = 30 minutes).

    Rationale: the aggregate is mostly noise around a slowly moving level, so
    the last observations help at short horizons and carry almost no
    information at two hours.
    """
    if k <= 0 or tau <= 0:
        raise ValueError("k and tau must be positive")
    history = _history(train, target)
    mu = float(history.mean())
    recent = float(history[-k:].mean())
    steps = np.arange(1, len(template) + 1)
    forecast = mu + np.exp(-steps / tau) * (recent - mu)
    return pd.Series(forecast, index=template.index, dtype=float)
