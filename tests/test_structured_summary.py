import pandas as pd
import pytest

from restock_predictor import predict_restock


def test_successful_forecast_returns_machine_readable_summary(tmp_path, capsys):
    source = tmp_path / "inventory.csv"
    pd.DataFrame(
        {"day": [1, 2], "stock_left": [900, 800]}
    ).to_csv(source, index=False)

    result = predict_restock(str(source), threshold=500)

    assert result is not None
    assert result["target_day"] == pytest.approx(3.0)
    assert result["predicted_stock"] == pytest.approx(700.0)
    assert result["daily_stock_change"] == pytest.approx(-100.0)
    assert result["days_until_threshold"] == 3
    assert result["restock_threshold"] == pytest.approx(500.0)
    assert result["status"] == "healthy"
    assert result["history_points"] == 2
    assert "Stock risk: healthy" in capsys.readouterr().out


def test_invalid_forecast_returns_none_for_callers(tmp_path, capsys):
    source = tmp_path / "invalid.csv"
    pd.DataFrame(
        {"day": [1, 2], "stock_left": [100, -5]}
    ).to_csv(source, index=False)

    assert predict_restock(str(source)) is None
    assert "negative inventory counts" in capsys.readouterr().out
