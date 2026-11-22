import pytest
from restock_predictor import predict_restock
def test_lookback_fits_latest_aggregated_observed_days(tmp_path):
    p=tmp_path/'stock.csv';p.write_text('day,stock_left\n1,1000\n2,90\n3,80\n4,70\n4,70\n')
    result=predict_restock(p,lookback_points=3)
    assert result['predicted_stock']==pytest.approx(60) and result['daily_stock_change']==pytest.approx(-10)
    assert result['history_points']==3 and result['available_history_points']==4
def test_invalid_lookback_is_rejected_before_reading():
    assert predict_restock('missing.csv',lookback_points=1) is None
