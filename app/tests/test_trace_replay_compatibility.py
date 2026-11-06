"""A legacy numeric-key checkpoint must not break blind trace endpoint joins."""
import importlib.util
from pathlib import Path

import pandas as pd


def test_numeric_checkpoint_ids_join_text_query_keys_without_charge_or_gold_hints():
    path=Path(__file__).resolve().parents[2]/'tools/evaluate_trace_evidence.py'
    spec=importlib.util.spec_from_file_location('trace_replay',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    queries=pd.DataFrame({'stream':['RP_Pos','NP_Pos'],'feature_id':['123','123'],'mz':[500.,500.]})
    tested=pd.DataFrame({'stream':['RP_Pos','NP_Pos'],'feature_id':[123,123],
        'isotope_m_plus_1_feature_id':[124.,125.], 'isotope_charge_hypothesis':[2,1],
        'adduct':['private_A','private_B'],'elemental_formula_candidates':['private_C','private_D']})
    snapshot=tested.copy(deep=True)
    actual=module.isotope_trace_queries(queries,tested)
    assert actual.isotope_m_plus_1_feature_id.tolist()==[124.,125.]
    assert actual.columns.tolist()==[*queries.columns,'isotope_m_plus_1_feature_id']
    pd.testing.assert_frame_equal(tested,snapshot,check_exact=True)
