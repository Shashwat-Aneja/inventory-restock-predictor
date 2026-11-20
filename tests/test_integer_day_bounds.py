from restock_predictor import predict_restock,MAX_SAFE_DAY
def test_int64_minimum_cannot_bypass_absolute_day_bound(tmp_path,capsys):
    p=tmp_path/'stock.csv';p.write_text('day,stock_left\n-9223372036854775808,10\n0,9\n')
    assert predict_restock(p) is None
    assert 'exact float64 integer range' in capsys.readouterr().out
def test_unbounded_integer_horizon_is_rejected_without_float_overflow(tmp_path):
    p=tmp_path/'stock.csv';p.write_text('day,stock_left\n1,10\n2,9\n')
    assert predict_restock(p,days_ahead=10**1000) is None
