import numpy as np
import pandas as pd
import pytest

from cwfc.data import build_target_frame
from cwfc.predict import COLUMNS, main, make_predictions, validate_predictions


def test_predictions_follow_submission_contract(make_readings):
    frame = build_target_frame(make_readings())
    pred = make_predictions(frame)
    assert list(pred.columns) == COLUMNS
    assert len(pred) == 24
    assert pred["timestamp"].tolist() == frame["timestamp"].tolist()[-24:]
    assert np.isfinite(pred[COLUMNS[1:]].to_numpy()).all()
    assert (pred["active_vm_count"] >= 0).all()
    assert pred["aggregate_cpu_utilization"].between(0, 100).all()


def test_predictions_do_not_depend_on_holdout_targets(make_readings):
    readings = make_readings()
    last24 = np.sort(readings["timestamp"].unique())[-24:]
    in_holdout = readings["timestamp"].isin(last24)

    changed_cpu = readings.copy()
    changed_cpu.loc[in_holdout, "avg_cpu"] = 99.0  # wildly different CPU in the holdout

    dropped_vms = readings[~(in_holdout & readings["vm_id"].isin(["vm0", "vm1", "vm2"]))]  # different counts

    baseline = make_predictions(build_target_frame(readings))
    pd.testing.assert_frame_equal(baseline, make_predictions(build_target_frame(changed_cpu)))
    pd.testing.assert_frame_equal(baseline, make_predictions(build_target_frame(dropped_vms)))


def test_validate_predictions_rejects_bad_files(make_readings):
    frame = build_target_frame(make_readings())
    pred = make_predictions(frame)
    ts = pred["timestamp"]
    bad_cpu = pred.assign(aggregate_cpu_utilization=101.0)
    with pytest.raises(ValueError):
        validate_predictions(bad_cpu, ts)
    with pytest.raises(ValueError):
        validate_predictions(pred.iloc[:-1], ts)
    with pytest.raises(ValueError):
        validate_predictions(pred.assign(active_vm_count=-1.0), ts)


def test_cli_writes_valid_csv(tmp_path, make_readings):
    make_readings().to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    out = tmp_path / "predictions.csv"
    assert main(["--data", str(tmp_path), "--output", str(out)]) == 0
    written = pd.read_csv(out)
    assert list(written.columns) == COLUMNS  # no index column
    assert len(written) == 24
    assert written["timestamp"].is_monotonic_increasing


def test_pipeline_is_not_hard_coded_to_the_release(tmp_path, make_readings):
    make_readings(n_vms=3, n_ts=40).to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    out = tmp_path / "predictions.csv"
    assert main(["--data", str(tmp_path), "--output", str(out)]) == 0
    written = pd.read_csv(out)
    assert len(written) == 24
    assert (written["active_vm_count"] == 3).all()  # not 10,000


def test_missing_data_directory_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--data", str(tmp_path / "nope"), "--output", str(tmp_path / "p.csv")])
