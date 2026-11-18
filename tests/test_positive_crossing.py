from restock_predictor import estimate_days_until_threshold


def test_stock_above_threshold_always_requires_at_least_one_day():
    assert estimate_days_until_threshold(0.1, -1e14, 0) == 1
    assert estimate_days_until_threshold(0, -1, 0) == 0
    assert estimate_days_until_threshold(3.0000000000000004, -1, 0) == 3
