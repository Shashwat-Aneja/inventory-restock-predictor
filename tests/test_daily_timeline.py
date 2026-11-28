import pytest
from restock_predictor import predict_restock

def test_daily_timeline_matches_point_forecast(tmp_path):
    src=tmp_path/"a.csv";src.write_text("day,stock_left\n1,100\n2,80\n3,60\n")
    result=predict_restock(src,days_ahead=3,timeline_days=3)
    assert [item["target_day"] for item in result["timeline"]]==[4,5,6]
    assert result["timeline"][-1]["predicted_stock"]==pytest.approx(result["predicted_stock"])
    assert all(item["predicted_stock"]>=0 for item in result["timeline"])
    assert predict_restock("missing",timeline_days=3661) is None
