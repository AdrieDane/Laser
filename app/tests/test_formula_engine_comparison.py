"""Engine failures and unsupported queries remain in fixed denominators."""
import json
from pathlib import Path

import pandas as pd


def test_missing_engine_output_is_not_an_exclusion(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools'))
    from formula_engine_comparison import evaluate
    prepared=tmp_path/'prepared'
    prepared.mkdir()
    pd.DataFrame([
        {'benchmark_id':'q1','stream':'RP_Pos','feature_id':'1','lipid_class':'PC','formula_query':True,
         'neutral_formula':'C10H22O2','adduct':'[M+H]+','library_known':True},
        {'benchmark_id':'q2','stream':'RP_Neg','feature_id':'2','lipid_class':'CL','formula_query':True,
         'neutral_formula':'C70H130O17P2','adduct':'[M-2H]2-','library_known':False},
        {'benchmark_id':'q3','stream':'NP_Neg','feature_id':'3','lipid_class':'PS','formula_query':True,
         'neutral_formula':'C80H140NO10P','adduct':'[M-H]-','library_known':False},
    ]).to_csv(prepared/'reference_ledger.csv',index=False)
    predictions=tmp_path/'predictions.jsonl'
    predictions.write_text(json.dumps({'query_id':'q1','status':'complete','candidates':[
        {'formula':'C10H22O2','adduct':'[M+H]+','rank':1}]})+'\n'+json.dumps({
        'query_id':'q2','status':'unsupported charge','candidates':[]})+'\n')
    scored=evaluate(prepared,predictions,tmp_path/'scored')
    assert len(scored)==3
    assert scored.pool_recall.sum()==1
    assert scored.top1.sum()==1
    assert scored.status.ne('complete').sum()==2
    assert scored.seconds.isna().all()  # Missing native timing is not zero runtime.
    metrics=json.loads((tmp_path/'scored/metrics.json').read_text())
    all_modes=[r for r in metrics if r['scope']=='all']
    assert sum(r['N'] for r in all_modes)==3
    assert sum(r['unavailable_or_failed'] for r in all_modes)==2
    assert next(r for r in all_modes if r['mode']=='NP_Pos')['top1']['percent'] is None
