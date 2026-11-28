import pytest
from restock_predictor import predict_restock

def test_last_source_observation_and_rejection(tmp_path):
    src=tmp_path/"a.csv";src.write_text("day,stock_left\n1,100\n2,70\n1,80\n")
    assert predict_restock(src,duplicate_days="last")["predicted_stock"]==pytest.approx(60)
    assert predict_restock(src)["predicted_stock"]==pytest.approx(50)
    assert predict_restock(src,duplicate_days="error") is None
