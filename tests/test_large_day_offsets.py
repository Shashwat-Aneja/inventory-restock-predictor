import pandas as pd
import pytest
from restock_predictor import predict_restock


def test_stock_forecast_is_invariant_to_large_day_offset(tmp_path):
    normal = pd.DataFrame({"day": [1, 2, 3, 4], "stock_left": [0.4, 0.3, 0.2, 0.1]})
    shifted = normal.copy()
    shifted["day"] += 10**15
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    normal.to_csv(first, index=False)
    shifted.to_csv(second, index=False)
    a, b = predict_restock(first), predict_restock(second)
    assert b["predicted_stock"] == pytest.approx(a["predicted_stock"], abs=1e-12)
    assert b["daily_stock_change"] == pytest.approx(a["daily_stock_change"], abs=1e-12)
