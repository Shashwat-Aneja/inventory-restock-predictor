from restock_predictor import estimate_days_until_threshold


def test_large_fractional_crossing_is_never_rounded_down():
    assert estimate_days_until_threshold(1_000_000_000_000.25, -1, 0) == 1_000_000_000_001


def test_small_regression_noise_still_preserves_integer_boundary():
    assert estimate_days_until_threshold(3.0000000000000004, -1, 0) == 3
