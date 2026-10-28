"""A small direct multi-horizon ridge model for the aggregate CPU series.

For every forecast step h = 1..H a separate ridge regression predicts how far
the future value will sit from the training mean::

    y[e + h - 1] - mu_e   ~   w_h . x_e

``e`` is a forecast origin inside the history, ``mu_e`` the mean of ``y[:e]``,
and ``x_e`` is computed from ``y[:e]`` only:

    recent3    mean of the last 3 values  - mu_e
    recent12   mean of the last 12 values - mu_e
    tail3      (optional) mean of the last 3 values of the share of VMs above
               50% CPU, minus the mean of that share over ``[:e]``

Every origin whose h-step target already lies inside the history is a training
example, and the model is refit from scratch at each evaluation origin, so it
never sees later data. Features are divided by their standard deviation but not
centred: with no intercept, "no deviation from the mean" means "forecast the
mean". The shrinkage model in ``cwfc.models`` is the special case that uses
``recent3`` alone with weights exp(-h / tau); this model learns the weights.

Two safeguards against extrapolation (found by testing: without them the model
produced forecasts like 17.7 and -4.2 for a series that sits near 9):

* the scale of each feature is computed ONCE from all origins in the history
  and shared by every horizon. At long horizons the usable training examples
  all precede the current regime, so a per-horizon scale would be tiny and a
  recent level shift would look like a 30-90 sigma event;
* forecasts are clipped to the range of values seen in the history
  (``clip_to_history``).

Only numpy is needed. The history is short, so a horizon with fewer than
``min_examples`` training examples falls back to the plain mean.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .data import TAIL50

SHORT = 3
LONG = 12
MIN_EXAMPLES = 8


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float, fit_intercept: bool) -> tuple[np.ndarray, float]:
    if fit_intercept:
        x_mean, y_mean = X.mean(axis=0), float(y.mean())
    else:
        x_mean, y_mean = np.zeros(X.shape[1]), 0.0
    Xc, yc = X - x_mean, y - y_mean
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ yc)
    return w, y_mean - float(x_mean @ w)


def _features(values: np.ndarray, tail: np.ndarray | None, e: int) -> tuple[float, list[float]]:
    mu = float(values[:e].mean())
    row = [values[e - SHORT : e].mean() - mu, values[e - LONG : e].mean() - mu]
    if tail is not None:
        row.append(tail[e - SHORT : e].mean() - tail[:e].mean())
    return mu, row


def ridge_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    alpha: float = 10.0,
    fit_intercept: bool = False,
    use_tail: bool = False,
    min_examples: int = MIN_EXAMPLES,
    clip_to_history: bool = True,
) -> pd.Series:
    """Direct multi-horizon ridge forecast; same signature as ``cwfc.baselines``."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if use_tail and TAIL50 not in train.columns:
        raise KeyError(f"column {TAIL50!r} is required when use_tail=True")
    columns = [target] + ([TAIL50] if use_tail else [])
    data = train[columns].astype(float).dropna()  # rows are treated as consecutive steps
    if data.empty:
        raise ValueError(f"train has no valid observations of {target!r}")
    values = data[target].to_numpy()
    tail = data[TAIL50].to_numpy() if use_tail else None
    t, horizon = len(values), len(template)
    if t < LONG:  # not enough history to build the features
        return pd.Series(values.mean(), index=template.index, dtype=float)

    ends = np.arange(LONG, t + 1)  # candidate origins; the last one (e = t) is "now"
    means, feats = zip(*(_features(values, tail, e) for e in ends))
    means, feats = np.asarray(means), np.asarray(feats)
    mu_now, x_now = means[-1], feats[-1]
    scale = feats.std(axis=0)  # one scale for all horizons, from history only
    scale[scale == 0] = 1.0

    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        usable = ends[ends <= t - h]  # the h-step target must lie inside the history
        if usable.size < min_examples:
            out[h - 1] = mu_now
            continue
        rows = usable - LONG
        X = feats[rows]
        y = values[usable + h - 1] - means[rows]
        w, b = _ridge(X / scale, y, alpha, fit_intercept)
        out[h - 1] = mu_now + b + float((x_now / scale) @ w)
    if clip_to_history:
        out = np.clip(out, values.min(), values.max())
    return pd.Series(out, index=template.index, dtype=float)


def ridge_models() -> dict:
    """Fixed set of variants (no search beyond this). ``ridge_a10`` is the declared candidate."""
    return {
        "ridge_a1": partial(ridge_forecast, alpha=1.0),
        "ridge_a10": partial(ridge_forecast, alpha=10.0),
        "ridge_a30": partial(ridge_forecast, alpha=30.0),
        "ridge_a10_icpt": partial(ridge_forecast, alpha=10.0, fit_intercept=True),
        "ridge_a10_tail": partial(ridge_forecast, alpha=10.0, use_tail=True),
    }
