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

    output = capsys.readouterr().out
    assert "stock_left" in output
    assert "negative inventory counts" in output


def test_predictor_rejects_missing_required_columns(tmp_path, capsys):
    csv_path = tmp_path / "missing-columns.csv"
    pd.DataFrame({"day": [1, 2], "quantity": [100, 90]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path))

    assert "must include 'day' and 'stock_left' columns" in capsys.readouterr().out

def test_predictor_averages_duplicate_day_observations(tmp_path, capsys):
    csv_path = tmp_path / "duplicate-days.csv"
    pd.DataFrame(
        {"day": [1, 1, 2], "stock_left": [100, 200, 150]}
    ).to_csv(csv_path, index=False)

    predict_restock(str(csv_path))

    output = capsys.readouterr().out
    assert "Predicted stock for next day: 150.00" in output

def test_predictor_uses_custom_restock_threshold(tmp_path, capsys):
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [900, 800]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path), threshold=850)

    output = capsys.readouterr().out
    assert "Predicted stock for next day: 700.00" in output
    assert "Warning: Stock will fall below threshold" in output



def test_predictor_rejects_infinite_inventory_values(tmp_path, capsys):
    csv_path = tmp_path / "infinite.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [100, float("inf")]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path))

    assert "finite numeric values" in capsys.readouterr().out


def test_predictor_rejects_non_finite_threshold(tmp_path, capsys):
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [900, 800]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path), threshold=float("nan"))

    assert "finite, non-negative number" in capsys.readouterr().out


def test_predictor_rejects_negative_threshold(tmp_path, capsys):
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [900, 800]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path), threshold=-1)

    assert "finite, non-negative number" in capsys.readouterr().out


def test_predictor_supports_a_multi_day_forecast(tmp_path, capsys):
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [900, 800]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path), threshold=500, days_ahead=2)

    output = capsys.readouterr().out
    assert "Predicted stock for day 4 (2 days ahead): 600.00" in output
    assert "Stock levels are safe for now" in output


def test_predictor_rejects_non_positive_forecast_horizon(tmp_path, capsys):
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [900, 800]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path), days_ahead=0)

    assert "Forecast horizon must be a positive whole number" in capsys.readouterr().out




def test_predictor_clamps_long_horizon_below_zero(tmp_path, capsys):
    csv_path = tmp_path / "declining-inventory.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [10, 5]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path), threshold=1, days_ahead=10)

    output = capsys.readouterr().out
    assert "Predicted stock for day 12 (10 days ahead): 0.00" in output
    assert "Warning: Stock will fall below threshold" in output


def test_predictor_rejects_boolean_forecast_horizon(tmp_path, capsys):
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame({"day": [1, 2], "stock_left": [90, 80]}).to_csv(
        csv_path, index=False
    )

    predict_restock(str(csv_path), days_ahead=True)

    assert "positive whole number of days" in capsys.readouterr().out
