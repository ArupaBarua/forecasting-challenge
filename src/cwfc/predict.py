"""Generate ``predictions.csv`` for the final two-hour holdout.

    python -m cwfc.predict --data challenge_data --output predictions.csv

Models (fixed; chosen by rolling-origin evaluation inside the training window
only, never by looking at the holdout):

* ``active_vm_count``           persistence (last training value);
* ``aggregate_cpu_utilization`` shrinkage forecast, see ``cwfc.models``.

The models receive the training frame and only the holdout *timestamps*, so
the holdout targets cannot reach them. ``tests/test_predict.py`` checks this.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .baselines import persistence_forecast
from .data import COUNT, CPU, HOLDOUT, build_target_frame, grid_report, load_readings, split_train_holdout
from .models import DEFAULT_K, DEFAULT_TAU, shrinkage_forecast

COLUMNS = ["timestamp", COUNT, CPU]


def validate_predictions(pred: pd.DataFrame, holdout_timestamps) -> None:
    """Raise ValueError unless ``pred`` satisfies SUBMISSION.md."""
    if list(pred.columns) != COLUMNS:
        raise ValueError(f"columns must be exactly {COLUMNS}, got {list(pred.columns)}")
    expected = np.asarray(holdout_timestamps)
    if len(pred) != len(expected) or not np.array_equal(pred["timestamp"].to_numpy(), expected):
        raise ValueError("timestamps must match the holdout timestamps exactly, in order")
    if pred["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps")
    values = pred[[COUNT, CPU]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    if (pred[COUNT] < 0).any():
        raise ValueError(f"{COUNT} must be >= 0")
    if not pred[CPU].between(0.0, 100.0).all():
        raise ValueError(f"{CPU} must be in [0, 100]")


def make_predictions(
    frame: pd.DataFrame,
    holdout: int = HOLDOUT,
    k: int = DEFAULT_K,
    tau: float = DEFAULT_TAU,
) -> pd.DataFrame:
    """Forecast the last ``holdout`` rows of ``frame`` from the rows before them."""
    train, test = split_train_holdout(frame, holdout)
    template = test[["timestamp"]]  # timestamps only: no holdout targets are passed on
    count = persistence_forecast(train, template, COUNT)
    cpu = shrinkage_forecast(train, template, CPU, k=k, tau=tau)
    pred = pd.DataFrame(
        {
            "timestamp": template["timestamp"].to_numpy(),
            COUNT: np.maximum(count.to_numpy(), 0.0),
            CPU: np.clip(cpu.to_numpy(), 0.0, 100.0),
        }
    )
    validate_predictions(pred, template["timestamp"])
    return pred


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write predictions.csv for the final holdout.")
    parser.add_argument("--data", default="challenge_data", help="directory containing vm_cpu_readings.csv.gz")
    parser.add_argument("--output", default="predictions.csv", help="output CSV path")
    parser.add_argument("--holdout", type=int, default=HOLDOUT, help="number of final timestamps to forecast")
    args = parser.parse_args(argv)

    frame = build_target_frame(load_readings(args.data))
    report = grid_report(frame)
    if report["n_irregular_steps"]:
        print(f"WARNING: {report['n_irregular_steps']} steps are not 300 s apart; rows are treated as consecutive.")
    if report["n_timestamps_without_valid_cpu"]:
        print(f"WARNING: {report['n_timestamps_without_valid_cpu']} timestamps have no valid CPU reading.")

    pred = make_predictions(frame, holdout=args.holdout)
    pred.to_csv(args.output, index=False)
    print(f"{report['n_timestamps']} timestamps; trained on {len(frame) - args.holdout}, forecast {len(pred)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
