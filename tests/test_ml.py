import numpy as np
import pandas as pd
import pytest

from cwfc.data import CPU, TAIL50, build_target_frame, split_train_holdout
from cwfc.evaluate import ml_models, rolling_origin_predictions
from cwfc.ml import ridge_forecast, ridge_models
from cwfc.models import expanding_mean_forecast


def _frame(values, tail=None):
    data = {"y": list(values)}
    if tail is not None:
        data[TAIL50] = list(tail)
    return pd.DataFrame(data)


def _ar1(n=200, phi=0.95, sigma=0.3, seed=1):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + rng.normal(0, sigma)
    return 50.0 + y


def test_huge_alpha_reverts_to_the_expanding_mean():
    train = _frame(_ar1(60))
    template = pd.DataFrame(index=range(24))
    ridge = ridge_forecast(train, template, "y", alpha=1e12)
    mean = expanding_mean_forecast(train, template, "y")
    assert np.allclose(ridge, mean, atol=1e-6)


def test_short_history_falls_back_to_the_mean():
    train = _frame([1.0, 2.0, 3.0, 4.0, 5.0])
    forecast = ridge_forecast(train, pd.DataFrame(index=range(6)), "y")
    assert np.allclose(forecast, 3.0)


def test_too_few_training_examples_fall_back_to_the_mean_at_long_horizons():
    train = _frame(_ar1(30))  # at h=24 only a handful of origins qualify (< 8)
    forecast = ridge_forecast(train, pd.DataFrame(index=range(24)), "y")
    assert forecast.iloc[-1] == pytest.approx(train["y"].mean())


def test_persistent_series_is_forecast_better_than_the_mean_at_step_one():
    frame = pd.DataFrame({"y": _ar1(200)})
    frame["time_utc"] = pd.to_datetime(np.arange(200) * 300, unit="s", utc=True)
    frame["timestamp"] = np.arange(200) * 300
    y, preds = rolling_origin_predictions(
        frame, {"expanding_mean": expanding_mean_forecast, "ridge": ridge_models()["ridge_a1"]}, "y", horizon=24, min_train=100
    )
    assert np.abs(y[:, 0] - preds["ridge"][:, 0]).mean() < np.abs(y[:, 0] - preds["expanding_mean"][:, 0]).mean()


def test_tail_feature_requires_its_column():
    with pytest.raises(KeyError):
        ridge_forecast(_frame(_ar1(60)), pd.DataFrame(index=range(5)), "y", use_tail=True)


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        ridge_forecast(_frame(_ar1(60)), pd.DataFrame(index=range(5)), "y", alpha=0.0)


def test_tail_variant_uses_the_share_column():
    base = _ar1(80)
    flat = ridge_forecast(_frame(base, tail=np.full(80, 0.03)), pd.DataFrame(index=range(12)), "y", use_tail=True)
    swing = np.where(np.arange(80) % 10 < 5, 0.02, 0.06)
    varied = ridge_forecast(_frame(base, tail=swing), pd.DataFrame(index=range(12)), "y", use_tail=True)
    assert not np.allclose(flat, varied)  # a constant feature adds nothing; a varying one changes the forecast


def test_ridge_forecasts_only_use_the_history_before_the_origin(make_readings):
    train_frame, _ = split_train_holdout(build_target_frame(make_readings()), 24)
    _, p1 = rolling_origin_predictions(train_frame, ml_models(), CPU, horizon=24, min_train=48)
    altered = train_frame.copy()
    altered.loc[altered.index >= 72, [CPU, TAIL50]] = [99.0, 0.9]
    _, p2 = rolling_origin_predictions(altered, ml_models(), CPU, horizon=24, min_train=48)
    for name in p1:
        assert np.allclose(p1[name][0], p2[name][0])  # first origin (row 48) cannot see rows >= 72


def test_all_ridge_variants_return_one_finite_forecast_per_row(make_readings):
    train_frame, _ = split_train_holdout(build_target_frame(make_readings()), 24)
    template = pd.DataFrame(index=range(24))
    for name, model in ridge_models().items():
        forecast = model(train_frame, template, CPU)
        assert len(forecast) == 24 and np.isfinite(forecast).all(), name


def test_a_level_shift_does_not_cause_wild_extrapolation():
    rng = np.random.default_rng(0)
    y = 9.15 + rng.normal(0, 0.05, 110)
    y[50:70] += 0.75  # a sudden hump, as in the real aggregate
    history = y[:62]  # the hump has just begun to show; long-horizon examples all precede it
    spread = history.max() - history.min()
    template = pd.DataFrame(index=range(24))

    raw = ridge_forecast(_frame(history), template, "y", alpha=10.0, clip_to_history=False)
    assert raw.min() > history.min() - spread and raw.max() < history.max() + spread

    clipped = ridge_forecast(_frame(history), template, "y", alpha=10.0)
    assert clipped.min() >= history.min() and clipped.max() <= history.max()
