"""Load the challenge release and build the two required aggregate target series.

Treatment of irregularities (documented in the final report):

* A reading with a missing ``avg_cpu`` is not a valid observation and is dropped.
* A VM is counted at most once per timestamp: duplicate ``(timestamp, vm_id)``
  rows are collapsed (first kept) before counting or averaging.
* Every timestamp present in the source stays in the series, even if it has no
  valid reading (count 0, CPU NaN). Nothing is interpolated or invented.
* Rows are treated as consecutive five-minute steps. ``grid_report`` counts
  irregular steps so a violation is visible instead of silent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

READINGS_FILE = "vm_cpu_readings.csv.gz"
STEP_SECONDS = 300
COUNT = "active_vm_count"
CPU = "aggregate_cpu_utilization"
HOLDOUT = 24


def load_readings(data_dir: str | Path) -> pd.DataFrame:
    """Read the VM-level five-minute CPU readings (timestamp, vm_id, avg_cpu)."""
    path = Path(data_dir) / READINGS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Extract data-v1.tar.gz so that the data directory "
            f"contains {READINGS_FILE}."
        )
    return pd.read_csv(path, usecols=["timestamp", "vm_id", "avg_cpu"])


def build_target_frame(readings: pd.DataFrame) -> pd.DataFrame:
    """One row per source timestamp, in chronological order, with both targets.

    ``timestamp`` keeps the source value (seconds). ``time_utc`` is a datetime
    version of it, used by ``cwfc.validation.rolling_origin_splits``.
    """
    all_ts = pd.Index(np.sort(readings["timestamp"].unique()), name="timestamp")
    valid = readings.dropna(subset=["avg_cpu"]).drop_duplicates(["timestamp", "vm_id"])
    grouped = valid.groupby("timestamp")
    frame = pd.DataFrame(
        {
            COUNT: grouped["vm_id"].nunique(),
            CPU: grouped["avg_cpu"].mean(),
        }
    ).reindex(all_ts)
    frame[COUNT] = frame[COUNT].fillna(0).astype(int)
    frame = frame.reset_index()
    frame["time_utc"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame


def grid_report(frame: pd.DataFrame, step: int = STEP_SECONDS) -> dict:
    """Cadence and coverage checks on the target frame."""
    steps = np.diff(frame["timestamp"].to_numpy())
    return {
        "n_timestamps": int(len(frame)),
        "n_irregular_steps": int((steps != step).sum()),
        "n_timestamps_without_valid_cpu": int(frame[CPU].isna().sum()),
    }


def split_train_holdout(frame: pd.DataFrame, holdout: int = HOLDOUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: everything before the last ``holdout`` rows is training.

    On the challenge release (134 timestamps) this is 110 training rows and a
    24-row (two-hour) holdout. The series is built in full before splitting;
    no statistic is computed across the boundary.
    """
    if holdout <= 0:
        raise ValueError("holdout must be positive")
    if len(frame) <= holdout:
        raise ValueError(f"need more than {holdout} timestamps, got {len(frame)}")
    train = frame.iloc[:-holdout].reset_index(drop=True)
    test = frame.iloc[-holdout:].reset_index(drop=True)
    return train, test
