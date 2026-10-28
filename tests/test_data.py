import numpy as np
import pandas as pd
import pytest

from cwfc.data import build_target_frame, grid_report, split_train_holdout


def test_each_vm_counted_once_per_timestamp_and_nulls_dropped():
    readings = pd.DataFrame(
        {
            "timestamp": [0, 0, 0, 300, 300],
            "vm_id": ["a", "a", "b", "a", "b"],  # (0, "a") appears twice
            "avg_cpu": [10.0, 10.0, 30.0, 20.0, np.nan],
        }
    )
    frame = build_target_frame(readings)
    assert frame["active_vm_count"].tolist() == [2, 1]
    assert frame["aggregate_cpu_utilization"].tolist() == [20.0, 20.0]


def test_timestamp_without_valid_readings_is_kept_not_invented():
    readings = pd.DataFrame({"timestamp": [0, 300], "vm_id": ["a", "a"], "avg_cpu": [5.0, np.nan]})
    frame = build_target_frame(readings)
    assert frame["active_vm_count"].tolist() == [1, 0]
    assert np.isnan(frame["aggregate_cpu_utilization"].iloc[1])
    assert grid_report(frame)["n_timestamps_without_valid_cpu"] == 1


def test_frame_is_chronological_even_if_input_is_shuffled(make_readings):
    frame = build_target_frame(make_readings())
    assert frame["timestamp"].is_monotonic_increasing
    assert len(frame) == 134


def test_grid_report_flags_irregular_steps():
    regular = pd.DataFrame({"timestamp": [0, 300, 600], "aggregate_cpu_utilization": [1.0, 2.0, 3.0]})
    gappy = pd.DataFrame({"timestamp": [0, 300, 900], "aggregate_cpu_utilization": [1.0, 2.0, 3.0]})
    assert grid_report(regular)["n_irregular_steps"] == 0
    assert grid_report(gappy)["n_irregular_steps"] == 1


def test_split_is_chronological(make_readings):
    frame = build_target_frame(make_readings())
    train, holdout = split_train_holdout(frame, 24)
    assert (len(train), len(holdout)) == (110, 24)
    assert train["timestamp"].max() < holdout["timestamp"].min()
    assert holdout["timestamp"].tolist() == frame["timestamp"].tolist()[-24:]


def test_split_rejects_series_not_longer_than_holdout(make_readings):
    frame = build_target_frame(make_readings(n_ts=24))
    with pytest.raises(ValueError):
        split_train_holdout(frame, 24)


def test_tail_shares_use_valid_vms_only():
    readings = pd.DataFrame(
        {
            "timestamp": [0, 0, 0, 0],
            "vm_id": ["a", "b", "c", "d"],
            "avg_cpu": [60.0, 90.0, 10.0, np.nan],  # d is not a valid observation
        }
    )
    frame = build_target_frame(readings)
    assert frame["share_cpu_gt50"].iloc[0] == pytest.approx(2 / 3)
    assert frame["share_cpu_gt80"].iloc[0] == pytest.approx(1 / 3)
