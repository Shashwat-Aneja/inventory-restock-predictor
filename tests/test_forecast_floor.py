import pandas as pd

from restock_predictor import predict_restock


def test_forecast_never_reports_negative_physical_inventory(tmp_path, capsys):
    csv_path = tmp_path / "declining-inventory.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [10, 0]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path), threshold=1)

    output = capsys.readouterr().out
    assert "Predicted stock for next day: 0.00" in output
    assert "Warning: Stock will fall below threshold" in output
