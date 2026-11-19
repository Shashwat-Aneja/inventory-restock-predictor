import pytest
from restock_predictor import predict_restock
def test_large_duplicate_daily_stock_has_finite_mean(tmp_path):
    source=tmp_path/'stock.csv';source.write_text('day,stock_left\n1,1e308\n1,1e308\n2,1e308\n2,1e308\n')
    result=predict_restock(source)
    assert result is not None
    assert result['predicted_stock']==pytest.approx(1e308)
    assert result['history_points']==2
