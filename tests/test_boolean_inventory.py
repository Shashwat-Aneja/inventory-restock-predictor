import pandas as pd
import pytest
import restock_predictor as app
@pytest.mark.parametrize('column',['day','stock_left'])
def test_boolean_inventory_rejected(monkeypatch,column):
    data=pd.DataFrame({'day':[0,1],'stock_left':[True,False]} if column=='stock_left' else {'day':[False,True],'stock_left':[100,90]})
    monkeypatch.setattr(app.pd,'read_csv',lambda _:data)
    assert app.predict_restock('ignored') is None
