import pandas as pd

from restock_predictor import predict_restock


def test_predictor_sorts_history_before_forecasting(tmp_path, capsys):
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame(
        {"day": [3, 1, 2], "stock_left": [70, 100, 85]}
    ).to_csv(csv_path, index=False)

    predict_restock(str(csv_path))

    output = capsys.readouterr().out
    assert "Predicted stock for next day" in output
    assert "Warning: Stock will fall below threshold" in output


def test_predictor_rejects_empty_history(tmp_path, capsys):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["day", "stock_left"]).to_csv(csv_path, index=False)

    predict_restock(str(csv_path))

    assert "CSV contains no inventory records" in capsys.readouterr().out


def test_predictor_rejects_non_numeric_values(tmp_path, capsys):
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [100, "unknown"]}).to_csv(csv_path, index=False)

    predict_restock(str(csv_path))

    assert "finite numeric values" in capsys.readouterr().out


def test_predictor_requires_two_distinct_days(tmp_path, capsys):
    csv_path = tmp_path / "one-day.csv"
    pd.DataFrame({"day": [1, 1], "stock_left": [100, 90]}).to_csv(csv_path, index=False)

    predict_restock(str(csv_path))

    assert "two distinct day values" in capsys.readouterr().out

def test_predictor_rejects_negative_stock_counts(tmp_path, capsys):
    csv_path = tmp_path / "negative-stock.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [100, -5]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path))

    assert "stock_left" in capsys.readouterr().out
    assert "negative inventory counts" in capsys.readouterr().out


def test_predictor_rejects_missing_required_columns(tmp_path, capsys):
    csv_path = tmp_path / "missing-columns.csv"
    pd.DataFrame({"day": [1, 2], "quantity": [100, 90]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path))

    assert "must include 'day' and 'stock_left' columns" in capsys.readouterr().out
