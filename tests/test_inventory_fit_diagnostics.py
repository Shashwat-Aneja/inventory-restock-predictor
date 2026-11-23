import pytest
from restock_predictor import predict_restock
def test_exact_linear_inventory_has_near_zero_fit_error(tmp_path):
    p=tmp_path/'stock.csv';p.write_text('day,stock_left\n1,100\n2,90\n3,80\n')
    assert predict_restock(p)['historical_fit_mae']==pytest.approx(0,abs=1e-10)
def test_large_constant_inventory_fit_metric_remains_finite(tmp_path):
    p=tmp_path/'stock.csv';p.write_text('day,stock_left\n1,1e308\n2,1e308\n3,1e308\n')
    assert predict_restock(p)['historical_fit_mae']==0
