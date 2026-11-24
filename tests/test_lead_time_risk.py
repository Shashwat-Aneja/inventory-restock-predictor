from restock_predictor import predict_restock

def test_threshold_crossing_within_supplier_lead_time(tmp_path):
    p=tmp_path/'in.csv';p.write_text('day,stock_left\n1,100\n2,80\n')
    result=predict_restock(p,threshold=50,lead_time_days=2)
    assert result['reorder_before_delivery'] is True and result['lead_time_days']==2
    assert predict_restock(p,threshold=50,lead_time_days=0)['reorder_before_delivery'] is False

def test_rising_inventory_has_no_projected_lead_time_crossing(tmp_path):
    p=tmp_path/'in.csv';p.write_text('day,stock_left\n1,80\n2,100\n')
    assert predict_restock(p,threshold=50,lead_time_days=3)['reorder_before_delivery'] is False
    assert predict_restock(p,lead_time_days=-1) is None
