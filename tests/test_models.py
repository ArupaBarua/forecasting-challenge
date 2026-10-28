import numpy as np
import pandas as pd
import pytest

from cwfc.models import expanding_mean_forecast, shrinkage_forecast


def _frame(values):
    return pd.DataFrame({"y": list(values)})


def test_expanding_mean_ignores_missing_values():
    forecast = expanding_mean_forecast(_frame([1.0, np.nan, 3.0]), _frame(range(4)), "y")
    assert forecast.tolist() == [2.0] * 4


def test_shrinkage_decays_from_recent_level_to_long_run_mean():
    train = _frame([0, 0, 0, 10, 10, 10])  # mean 5, mean of last 3 = 10
    forecast = shrinkage_forecast(train, _frame(range(24)), "y", k=3, tau=6)
    assert forecast.iloc[0] == pytest.approx(5 + 5 * np.exp(-1 / 6))
    assert (np.diff(forecast.to_numpy()) < 0).all()  # decays monotonically
    assert forecast.iloc[-1] > 5  # approaches the mean, never crosses it


def test_shrinkage_is_flat_when_history_is_flat():
    forecast = shrinkage_forecast(_frame([4.0] * 10), _frame(range(5)), "y")
    assert np.allclose(forecast, 4.0)


def test_models_reject_empty_history_and_bad_parameters():
    with pytest.raises(ValueError):
        shrinkage_forecast(_frame([np.nan]), _frame(range(3)), "y")
    with pytest.raises(ValueError):
        shrinkage_forecast(_frame([1.0, 2.0]), _frame(range(3)), "y", k=0)
