"""Rolling-origin evaluation on the TRAINING window only.

    python -m cwfc.evaluate --data challenge_data --out reports/results

The final holdout is split off first and discarded here: no table below can
depend on it. Every model receives the history up to the forecast origin and
only the *timestamps* of the rows it must forecast; the realised targets are
used afterwards for scoring.

Conventions: ``bias`` is mean(actual - forecast), so a positive bias means the
model under-forecasts. ``nMAE_vs_x`` is model MAE divided by the MAE of
reference ``x`` (below 1 = better than the reference). "Peak" rows are those
whose realised aggregate CPU is at or above the 75th percentile of the
training series; this is a descriptive slice, not an input to any model.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import persistence_forecast, seasonal_naive_forecast
from .data import COUNT, CPU, HOLDOUT, build_target_frame, load_readings, split_train_holdout
from .metrics import mae, normalized_mae, rmse
from .ml import ridge_models
from .models import (
    DEFAULT_K,
    DEFAULT_TAU,
    expanding_mean_forecast,
    moving_average_forecast,
    shrinkage_forecast,
)
from .validation import rolling_origin_splits

HORIZON = HOLDOUT
MIN_TRAIN = 48
MIN_TRAIN_GRID = (36, 48, 60)
PEAK_QUANTILE = 0.75
HORIZON_STEPS = (1, 3, 6, 12, 18, 24)
REFERENCE = "last_value"
STRONG_BASELINE = "expanding_mean"
MAIN_MODEL = f"shrinkage_k{DEFAULT_K}_tau{int(DEFAULT_TAU)}"
ADOPTION_MARGIN = 0.95  # a ridge model must beat the shrinkage MAE by >= 5% at every min_train


def main_models() -> dict:
    return {
        REFERENCE: persistence_forecast,
        "moving_mean_12": partial(moving_average_forecast, window=12),
        "seasonal_naive_12": partial(seasonal_naive_forecast, season_length=12),
        STRONG_BASELINE: expanding_mean_forecast,
        MAIN_MODEL: partial(shrinkage_forecast, k=DEFAULT_K, tau=DEFAULT_TAU),
    }


def grid_models() -> dict:
    """Small, fixed sensitivity grid around the main model (no search beyond this)."""
    models = {REFERENCE: persistence_forecast, STRONG_BASELINE: expanding_mean_forecast}
    for k in (3, 6):
        for tau in (3, 6, 12):
            models[f"shrinkage_k{k}_tau{tau}"] = partial(shrinkage_forecast, k=k, tau=tau)
    return models


def ml_models() -> dict:
    """References plus the fixed ridge variants from ``cwfc.ml``."""
    models = {
        REFERENCE: persistence_forecast,
        STRONG_BASELINE: expanding_mean_forecast,
        MAIN_MODEL: partial(shrinkage_forecast, k=DEFAULT_K, tau=DEFAULT_TAU),
    }
    models.update(ridge_models())
    return models


def rolling_origin_predictions(
    train_frame: pd.DataFrame,
    models: dict,
    target: str,
    horizon: int = HORIZON,
    min_train: int = MIN_TRAIN,
    step: int = 1,
) -> tuple[np.ndarray, dict]:
    """Return ``y`` (origins x horizon) and ``{model: forecasts}`` of the same shape."""
    splits = rolling_origin_splits(
        train_frame, timestamp_col="time_utc", horizon=horizon, min_train_size=min_train, step=step
    )
    if not splits:
        raise ValueError(f"series of {len(train_frame)} rows is too short for horizon={horizon}, min_train={min_train}")
    time = train_frame["time_utc"]
    y: list = []
    preds: dict = {name: [] for name in models}
    for split in splits:
        history = train_frame[time <= split.train_end]
        window = train_frame[(time >= split.validation_start) & (time <= split.validation_end)]
        template = window[["timestamp"]]  # timestamps only; targets are used for scoring below
        y.append(window[target].to_numpy(dtype=float))
        for name, forecaster in models.items():
            preds[name].append(forecaster(history, template, target).to_numpy(dtype=float))
    return np.vstack(y), {name: np.vstack(rows) for name, rows in preds.items()}


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def score_table(y: np.ndarray, preds: dict, peak_threshold: float) -> pd.DataFrame:
    peak = y >= peak_threshold
    flat_y = y.ravel()
    rows = []
    for name, p in preds.items():
        err = y - p
        rows.append(
            {
                "model": name,
                "MAE": mae(flat_y, p.ravel()),
                "RMSE": rmse(flat_y, p.ravel()),
                "bias": float(err.mean()),
                "nMAE_vs_last": normalized_mae(flat_y, p.ravel(), preds[REFERENCE].ravel()),
                "nMAE_vs_expanding": normalized_mae(flat_y, p.ravel(), preds[STRONG_BASELINE].ravel()),
                "MAE_peak": _mean_or_nan(np.abs(err[peak])),
                "bias_peak": _mean_or_nan(err[peak]),
                "MAE_offpeak": _mean_or_nan(np.abs(err[~peak])),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def horizon_table(y: np.ndarray, preds: dict, names: list, steps=HORIZON_STEPS) -> pd.DataFrame:
    keep = [s for s in steps if s <= y.shape[1]]
    return pd.DataFrame(
        {name: np.abs(y - preds[name]).mean(axis=0)[[s - 1 for s in keep]] for name in names},
        index=pd.Index(keep, name="horizon_step"),
    )


def adoption_check(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered rule: adopt a ridge model only if its MAE is at most 95% of the
    shrinkage MAE at every minimum training size evaluated."""
    if sensitivity.empty:
        return pd.DataFrame(columns=["worst_MAE_ratio_vs_shrinkage", "adopt"])
    ratio = sensitivity.div(sensitivity.loc[MAIN_MODEL], axis=1)
    ridge_rows = [name for name in ratio.index if name.startswith("ridge")]
    worst = ratio.loc[ridge_rows].max(axis=1)
    return pd.DataFrame(
        {
            "worst_MAE_ratio_vs_shrinkage": worst,
            "adopt": np.where(worst <= ADOPTION_MARGIN, "yes", "no"),
        }
    )


def evaluate_count(train_frame: pd.DataFrame, horizon: int, min_train: int) -> pd.DataFrame:
    y, preds = rolling_origin_predictions(train_frame, {REFERENCE: persistence_forecast}, COUNT, horizon, min_train)
    p = preds[REFERENCE]
    return pd.DataFrame(
        {
            "MAE": [mae(y.ravel(), p.ravel())],
            "RMSE": [rmse(y.ravel(), p.ravel())],
            "series_is_constant": [bool(train_frame[COUNT].nunique() == 1)],
        },
        index=pd.Index([REFERENCE], name="model"),
    )


def evaluate(
    frame: pd.DataFrame,
    holdout: int = HOLDOUT,
    horizon: int = HORIZON,
    min_train: int = MIN_TRAIN,
    min_train_grid=MIN_TRAIN_GRID,
) -> dict:
    """Run every table. ``frame`` is the full series; the holdout rows are discarded first."""
    train_frame, _ = split_train_holdout(frame, holdout)
    peak_threshold = float(np.quantile(train_frame[CPU].dropna(), PEAK_QUANTILE))

    y, preds = rolling_origin_predictions(train_frame, main_models(), CPU, horizon, min_train)
    sensitivity = {}
    for mt in min_train_grid:
        try:
            y_mt, p_mt = rolling_origin_predictions(train_frame, main_models(), CPU, horizon, mt)
        except ValueError:
            continue  # series too short for this setting (e.g. a tiny smoke dataset)
        sensitivity[f"min_train_{mt}"] = score_table(y_mt, p_mt, peak_threshold)["MAE"]
    y_g, p_g = rolling_origin_predictions(train_frame, grid_models(), CPU, horizon, min_train)

    y_m, p_m = rolling_origin_predictions(train_frame, ml_models(), CPU, horizon, min_train)
    ml_table = score_table(y_m, p_m, peak_threshold)
    ml_table["nMAE_vs_shrinkage"] = ml_table["MAE"] / ml_table.loc[MAIN_MODEL, "MAE"]
    ml_table = ml_table[["MAE", "RMSE", "bias", "nMAE_vs_expanding", "nMAE_vs_shrinkage", "MAE_peak", "MAE_offpeak"]]
    ml_sensitivity = {}
    for mt in min_train_grid:
        try:
            y_s, p_s = rolling_origin_predictions(train_frame, ml_models(), CPU, horizon, mt)
        except ValueError:
            continue
        ml_sensitivity[f"min_train_{mt}"] = score_table(y_s, p_s, peak_threshold)["MAE"]
    ml_sensitivity = pd.DataFrame(ml_sensitivity)

    return {
        "main": score_table(y, preds, peak_threshold),
        "by_horizon": horizon_table(y, preds, [REFERENCE, STRONG_BASELINE, MAIN_MODEL], steps=HORIZON_STEPS),
        "min_train_sensitivity": pd.DataFrame(sensitivity),
        "shrinkage_grid": score_table(y_g, p_g, peak_threshold)[["MAE", "RMSE", "bias"]],
        "active_vm_count": evaluate_count(train_frame, horizon, min_train),
        "ml_comparison": ml_table,
        "ml_min_train_sensitivity": ml_sensitivity,
        "ml_adoption_check": adoption_check(ml_sensitivity),
        "meta": pd.Series(
            {
                "train_rows": len(train_frame),
                "n_origins": y.shape[0],
                "horizon": y.shape[1],
                "min_train": min_train,
                "non_overlapping_windows": (len(train_frame) - min_train) // horizon,
                "peak_threshold": peak_threshold,
                "peak_share": float((y >= peak_threshold).mean()),
            }
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin evaluation on the training window only.")
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--out", default="reports/results", help="directory for CSV result tables")
    parser.add_argument("--holdout", type=int, default=HOLDOUT)
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN)
    args = parser.parse_args(argv)

    frame = build_target_frame(load_readings(args.data))
    results = evaluate(frame, holdout=args.holdout, min_train=args.min_train)

    meta = results["meta"]
    print(f"Training rows: {int(meta['train_rows'])} (holdout of {args.holdout} excluded)")
    print(
        f"{int(meta['n_origins'])} rolling origins, horizon {int(meta['horizon'])}, min_train {int(meta['min_train'])}; "
        f"only ~{int(meta['non_overlapping_windows'])} non-overlapping windows, so small differences are noise."
    )
    print(f"Peak = realised CPU >= {meta['peak_threshold']:.3f} ({meta['peak_share']:.0%} of evaluated points)\n")

    labels = {
        "main": "Aggregate CPU: rolling-origin scores (bias = actual - forecast)",
        "min_train_sensitivity": "MAE by minimum training size",
        "by_horizon": "MAE by forecast step (1 step = 5 min)",
        "shrinkage_grid": "Shrinkage sensitivity grid",
        "active_vm_count": "active_vm_count (persistence)",
        "ml_comparison": "Ridge comparison (candidate: ridge_a10)",
        "ml_min_train_sensitivity": "Ridge comparison: MAE by minimum training size",
        "ml_adoption_check": "Adoption rule: ridge must beat shrinkage MAE by >= 5% at every min_train",
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, title in labels.items():
        table = results[key]
        print(f"== {title} ==")
        print(table.round(4).to_string(), "\n")
        table.to_csv(out_dir / f"{key}.csv")
    print(f"tables written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
