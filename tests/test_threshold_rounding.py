import math

from restock_predictor import estimate_days_until_threshold


def test_exact_integer_crossing_is_not_rounded_up_by_regression_noise():
    assert estimate_days_until_threshold(800, -99.99999999999997, 500) == 3


def test_fractional_crossing_still_rounds_up_to_a_full_day():
    assert estimate_days_until_threshold(800, -95, 500) == math.ceil(300 / 95)
