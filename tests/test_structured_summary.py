import pandas as pd

from restock_predictor import predict_restock


def test_successful_forecast_returns_machine_readable_summary(tmp_path, capsys):
    source = tmp_path / "inventory.csv"
    pd.DataFrame(
        {"day": [1, 2], "stock_left": [900, 800]}
    ).to_csv(source, index=False)

    result = predict_restock(str(source), threshold=500)

    assert result == {
        "target_day": 3.0,
        "predicted_stock": 700.0,
        "daily_stock_change": -100.0,
        "days_until_threshold": 3,
        "restock_threshold": 500.0,
        "status": "healthy",
        "history_points": 2,
    }
    assert "Stock risk: healthy" in capsys.readouterr().out


def test_invalid_forecast_returns_none_for_callers(tmp_path, capsys):
    source = tmp_path / "invalid.csv"
    pd.DataFrame(
        {"day": [1, 2], "stock_left": [100, -5]}
    ).to_csv(source, index=False)

    assert predict_restock(str(source)) is None
    assert "negative inventory counts" in capsys.readouterr().out
