import numpy as np
import pytest
import restock_predictor


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_prediction_cannot_be_reported_as_zero_stock(tmp_path, monkeypatch, capsys, value):
    source = tmp_path / "inventory.csv"
    source.write_text("day,stock_left\n1,100\n2,80\n")
    monkeypatch.setattr(restock_predictor.LinearRegression, "predict", lambda self, values: [value])
    assert restock_predictor.predict_restock(source) is None
    output = capsys.readouterr().out
    assert "Forecast stock must be a finite number" in output
    assert "Stock risk:" not in output
