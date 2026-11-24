import json,pytest
from restock_predictor import main,export_summary

def test_cli_exports_the_same_structured_forecast(tmp_path,capsys):
    p=tmp_path/'in.csv';p.write_text('day,stock_left\n1,100\n2,90\n');out=tmp_path/'out.json'
    assert main([str(p),'--json','--output',str(out)])==0
    assert json.loads(out.read_text())==json.loads(capsys.readouterr().out)
    before=p.read_bytes()
    with pytest.raises(ValueError):export_summary({},p,p)
    assert p.read_bytes()==before

def test_failed_replace_keeps_prior_summary_and_cleans_temporary(tmp_path,monkeypatch):
    import restock_predictor as app
    source=tmp_path/'in.csv';source.write_text('source');out=tmp_path/'out.json';out.write_text('keep')
    def fail(*args):raise OSError('disk error')
    monkeypatch.setattr(app.os,'replace',fail)
    with pytest.raises(OSError):export_summary({'value':1},out,source)
    assert out.read_text()=='keep' and sorted(p.name for p in tmp_path.iterdir())==['in.csv','out.json']
