import pandas as pd
import numpy as np
import pytest
from return_classifier import compute_features, finalize_features, split_features_and_labels, naive_baseline

def _make_ramp_prices(n=60, start=100):
    """Simple, hand-verifiable synthetic price series: a straight +1/day ramp."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series([start + i for i in range(n)], index=dates, dtype=float)

def test_daily_and_cumulative_returns():
    prices = _make_ramp_prices()
    data = compute_features(prices)

    # hand formula for a +1/day ramp: N-day return at row i = N / price[i-N]
    i = 20  # a row with full lookback for all three windows
    assert data['Daily Return'].iloc[i] == pytest.approx(1 / (99 + i))
    assert data['5 Day Return'].iloc[i] == pytest.approx(5 / (95 + i))
    assert data['10 Day Return'].iloc[i] == pytest.approx(10 / (90 + i))


def test_nan_boundaries_not_dropped():
    prices = _make_ramp_prices()
    data = compute_features(prices)

    # compute_features must NOT drop NaNs
    assert pd.isna(data['Daily Return'].iloc[0])
    assert pd.isna(data['5 Day Return'].iloc[:5]).all()
    assert pd.isna(data['50 Day Moving Average'].iloc[:49]).all()
    assert pd.isna(data['Label'].iloc[-1])  # no "tomorrow" for the last row


def test_signal_on_rising_trend():
    prices = _make_ramp_prices()
    data = compute_features(prices)

    # for a steadily rising price, the short-term average always sits above the long-term average once both are defined -- Signal should be 1 throughout
    valid = data.iloc[49:]  # first row where the 50-day MA exists
    assert (valid['Signal'] == 1).all()


def test_label_matches_next_day_direction():
    prices = _make_ramp_prices()
    data = compute_features(prices)

    # every "tomorrow" is also up, since the whole series rises monotonically -- every row except the last (no tomorrow to check) should be labeled 1
    assert (data['Label'].iloc[:-1] == 1).all()
    assert pd.isna(data['Label'].iloc[-1])

def test_finalize_features_drops_nan_and_casts_label():
    prices = _make_ramp_prices()
    data = compute_features(prices)
    finalized = finalize_features(data)

    assert not finalized.isna().any().any()
    assert finalized['Label'].dtype == int
    # 49 warm-up rows (50-day MA) + 1 trailing row (no "tomorrow") should be gone
    assert len(finalized) == len(prices) - 49 - 1

def test_split_is_chronological_not_shuffled():
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    data = pd.DataFrame({
        'Daily Return': range(20),
        '5 Day Return': range(20),
        '10 Day Return': range(20),
        'Volatility': range(20),
        'Signal': [0, 1] * 10,
        'Label': [0, 1] * 10,
    }, index=dates)

    X_train, X_test, y_train, y_test = split_features_and_labels(data, test_size=0.2)

    assert len(X_train) == 16
    assert len(X_test) == 4
    # no shuffle -> test set must be exactly the last 4 rows, in order
    assert list(X_test.index) == list(dates[-4:])

def test_naive_baseline_picks_majority_and_aligns_by_date():
    dates_train = pd.date_range("2024-01-01", periods=10, freq="B")
    dates_test = pd.date_range("2024-01-15", periods=5, freq="B")

    y_train = pd.Series([1, 1, 1, 1, 0, 1, 1, 0, 1, 1], index=dates_train)  # majority = 1
    y_test = pd.Series([1, 0, 1, 1, 0], index=dates_test)

    majority_position, y_pred_naive, naive_accuracy = naive_baseline(y_train, y_test)

    assert majority_position == 1
    assert list(y_pred_naive.index) == list(dates_test)   # dates, not label values
    assert (y_pred_naive == 1).all()
    assert naive_accuracy == pytest.approx(3 / 5)          # 3 of 5 test days were actually 1