import numpy as np
import pandas as pd
import pytest

from cwfc.data import CPU, build_target_frame, split_train_holdout
from cwfc.evaluate import (
    MAIN_MODEL,
    evaluate,
    main,
    main_models,
    rolling_origin_predictions,
    score_table,
)
from cwfc.models import moving_average_forecast


def test_moving_average_uses_only_the_last_window():
    train = pd.DataFrame({"y": [100.0, 1.0, 2.0, 3.0]})
    forecast = moving_average_forecast(train, pd.DataFrame(index=range(3)), "y", window=3)
    assert forecast.tolist() == [2.0] * 3
    with pytest.raises(ValueError):
        moving_average_forecast(train, pd.DataFrame(index=range(3)), "y", window=0)


def test_rolling_origins_are_aligned_with_the_series(make_readings):
    train_frame, _ = split_train_holdout(build_target_frame(make_readings()), 24)
    y, preds = rolling_origin_predictions(train_frame, main_models(), CPU, horizon=24, min_train=48)
    values = train_frame[CPU].to_numpy()
    assert y.shape == (39, 24)  # origins 48..86 on 110 rows
    assert np.allclose(y[0], values[48:72])
    assert np.allclose(y[-1], values[86:110])
    assert np.allclose(preds["last_value"][0], values[47])  # persistence = value just before the origin
    assert all(p.shape == y.shape for p in preds.values())


def test_forecasts_only_see_data_before_the_origin(make_readings):
    train_frame, _ = split_train_holdout(build_target_frame(make_readings()), 24)
    y1, p1 = rolling_origin_predictions(train_frame, main_models(), CPU, horizon=24, min_train=48)
    altered = train_frame.copy()
    altered.loc[altered.index >= 72, CPU] = 99.0  # change everything from row 72 on
    y2, p2 = rolling_origin_predictions(altered, main_models(), CPU, horizon=24, min_train=48)
    first = 0  # origin at row 48: forecasts may only use rows 0..47
    for name in p1:
        assert np.allclose(p1[name][first], p2[name][first])


def test_score_table_arithmetic():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    preds = {"last_value": y + 1.0, "expanding_mean": y + 2.0, "m": y - 0.5}
    table = score_table(y, preds, peak_threshold=3.0)
    assert table.loc["m", "MAE"] == pytest.approx(0.5)
    assert table.loc["m", "bias"] == pytest.approx(0.5)  # actual - forecast > 0: under-forecast
    assert table.loc["m", "nMAE_vs_last"] == pytest.approx(0.5)
    assert table.loc["m", "nMAE_vs_expanding"] == pytest.approx(0.25)
    assert table.loc["m", "MAE_peak"] == pytest.approx(0.5)


def test_evaluation_does_not_depend_on_the_holdout(make_readings):
    readings = make_readings()
    last24 = np.sort(readings["timestamp"].unique())[-24:]
    tampered = readings.copy()
    tampered.loc[tampered["timestamp"].isin(last24), "avg_cpu"] = 99.0
    a = evaluate(build_target_frame(readings))
    b = evaluate(build_target_frame(tampered))
    for key in (
        "main", "by_horizon", "min_train_sensitivity", "shrinkage_grid", "active_vm_count",
        "ml_comparison", "ml_min_train_sensitivity", "ml_adoption_check",
    ):
        pd.testing.assert_frame_equal(a[key], b[key])
    assert MAIN_MODEL in a["main"].index


def test_evaluate_cli_writes_tables(tmp_path, make_readings):
    make_readings().to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    out = tmp_path / "results"
    assert main(["--data", str(tmp_path), "--out", str(out)]) == 0
    for name in (
        "main", "by_horizon", "min_train_sensitivity", "shrinkage_grid", "active_vm_count",
        "ml_comparison", "ml_min_train_sensitivity", "ml_adoption_check",
    ):
        assert (out / f"{name}.csv").is_file()
    assert MAIN_MODEL in pd.read_csv(out / "main.csv")["model"].tolist()


def test_sensitivity_skips_settings_the_series_is_too_short_for(make_readings):
    results = evaluate(build_target_frame(make_readings(n_vms=3, n_ts=104)))  # 80 training rows
    columns = results["min_train_sensitivity"].columns
    assert "min_train_36" in columns and "min_train_48" in columns
    assert "min_train_60" not in columns  # 60 + 24 > 80


def test_too_short_series_raises_a_clear_error(make_readings):
    with pytest.raises(ValueError, match="too short"):
        evaluate(build_target_frame(make_readings(n_vms=3, n_ts=70)))  # 46 training rows < 48 + 24


def test_adoption_rule_is_applied_mechanically():
    from cwfc.evaluate import adoption_check

    sens = pd.DataFrame(
        {"min_train_36": [1.0, 0.90, 0.99], "min_train_48": [1.0, 0.94, 0.80]},
        index=[MAIN_MODEL, "ridge_good", "ridge_unstable"],
    )
    check = adoption_check(sens)
    assert check.loc["ridge_good", "adopt"] == "yes"  # worst ratio 0.94 <= 0.95
    assert check.loc["ridge_unstable", "adopt"] == "no"  # worst ratio 0.99 at one setting
    assert MAIN_MODEL not in check.index
