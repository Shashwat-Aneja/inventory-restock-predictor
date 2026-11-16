import pytest
from restock_predictor import estimate_days_until_threshold


def test_tiny_decline_reports_unrepresentable_crossing():
    with pytest.raises(ValueError, match="numeric range"):
        estimate_days_until_threshold(1e308, -1e-308, 0)
