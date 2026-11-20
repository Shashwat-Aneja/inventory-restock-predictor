import json
from restock_predictor import main
def test_json_mode_outputs_one_complete_structured_summary(tmp_path,capsys):
    p=tmp_path/'stock.csv';p.write_text('day,stock_left\n1,100\n2,90\n3,80\n')
    assert main([str(p),'--threshold','50','--days-ahead','2','--json'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['target_day']==5 and data['history_points']==3
    assert data['predicted_stock']<61 and data['days_until_threshold']==3
