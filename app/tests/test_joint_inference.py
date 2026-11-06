"""Conflicting hypotheses, ambiguity, physical bounds and additive persistence."""
from dataclasses import asdict, replace
import json

import pandas as pd
import pytest

from standard_lipidomics.joint_inference import select_components, SETTINGS, JointEngine, apply_joint_inference
from standard_lipidomics.msbuddy_runtime import freeze_runtime, verify_runtime
from standard_lipidomics.untargeted import UntargetedComparisonCriteria
from standard_lipidomics.acquisition_method import default_acquisition_method
from test_complete_isotopes import fixture, INSTRUMENT

CRITERIA = UntargetedComparisonCriteria('A', 'B', isotope_model='complete', formula_model='lipid_aware', joint_model='joint_ms1')


def option(root, formula, cost, members=None):
    return {'formula': formula, 'adduct': '[M+H]+', 'joint_cost': cost,
            'members': {root: (formula, '[M+H]+'), **(members or {})}}


def test_component_solver_chooses_consistent_joint_explanation():
    a, b = ('RP_Pos', 'a'), ('RP_Pos', 'b')
    options = {a: [option(a, 'F1', 0., {b: ('F1', '[M+H]+')}), option(a, 'F2', .5)],
               b: [option(b, 'F2', 0.), option(b, 'F1', 2.)]}
    selected, audit = select_components(options)
    assert selected[a]['label'] == ('F2', '[M+H]+')
    assert selected[b]['label'] == ('F2', '[M+H]+')
    assert len(audit) == 1 and audit.roots.iloc[0] == 2


def test_equal_scores_abstain_instead_of_lexicographic_confidence():
    a = ('RP_Pos', 'a')
    selected, _ = select_components({a: [option(a, 'F1', .1), option(a, 'F2', .1)]})
    assert selected[a]['label'] is None
    assert selected[a]['margin'] == 0.


def test_weak_and_oversized_components_are_not_silently_split():
    a = ('RP_Pos', 'a')
    selected, _ = select_components({a: [option(a, 'F1', 9.)]})
    assert selected[a]['label'] is None
    members = {('RP_Pos', str(i)): ('F', '[M+H]+') for i in range(3)}
    selected, audit = select_components({r: [option(r, 'F', 0., members)] for r in members},
                                       {**SETTINGS, 'component_root_limit': 2})
    assert all(r['label'] is None for r in selected.values())
    assert audit.status.iloc[0] == 'component budget exceeded'


def test_runtime_freeze_preserves_venv_path_and_detects_drift(tmp_path, monkeypatch):
    monkeypatch.setattr('standard_lipidomics.msbuddy_runtime.MODELS', {})
    monkeypatch.setattr('standard_lipidomics.msbuddy_runtime.native_environment', lambda python: {'test': 'synthetic environment'})
    python = tmp_path/'python'
    python.write_text('synthetic executable')
    python.chmod(0o755)
    venv = tmp_path/'venv-python'
    venv.symlink_to(python)
    runtime = freeze_runtime({'schema_version': 1, 'python': str(venv), 'data': str(tmp_path), 'libomp': ''})
    assert runtime['available'] and runtime['python'] == str(venv)
    assert verify_runtime(runtime)
    python.write_text('changed executable')
    assert not verify_runtime(runtime)


def test_current_default_and_dependency_checks_remain_explicit():
    assert UntargetedComparisonCriteria('A', 'B').joint_model == 'current'
    assert not UntargetedComparisonCriteria('A', 'B').use_msbuddy
    with pytest.raises(ValueError, match='requires lipid-aware'):
        replace(CRITERIA, formula_model='current')
    with pytest.raises(ValueError, match='requires joint'):
        replace(CRITERIA, joint_model='current', use_msbuddy=True)


def example():
    frame = fixture()
    method = default_acquisition_method()
    method['methods']['RP']['instrument'].update(INSTRUMENT)
    row = {'stream': 'RP_Pos', 'feature_id': '0', 'mz': frame.mz.iloc[0], 'rt': 5.,
           'p_value': .0001, 'log2_fold_change_b_vs_a': 2., 'raw_p_significant': True, 'significant': True,
           'compound_id': 'compound_0001', 'adduct': 'legacy adduct', 'predicted_chemical_formula': 'legacy formula'}
    original = pd.DataFrame([row])
    analysis = {'features': original, 'raw_p_features': original.copy(), 'significant_features': original.copy(),
                'compound_features': original.copy(), 'criteria': asdict(CRITERIA), 'acquisition_method': method}
    result = {'streams': {'RP_Pos': {'features': frame}}, 'automatic_annotations': pd.DataFrame(), 'acquisition_method': method}
    return result, analysis


def test_joint_evidence_is_additive_and_roundtrips_with_unavailable_native(tmp_path, monkeypatch):
    from standard_lipidomics.project_store import save_untargeted_workflow, load_untargeted_workflow
    monkeypatch.setenv('LASER_MSBUDDY_CONFIG', str(tmp_path/'missing.json'))
    result, analysis = example()
    criteria = replace(CRITERIA, use_msbuddy=True, formula_scoring_limit=8)
    analysis['criteria'] = asdict(criteria)
    out = apply_joint_inference(result, analysis, criteria)
    for name in ['features', 'raw_p_features', 'significant_features', 'compound_features']:
        pd.testing.assert_frame_equal(out[name], analysis[name])
    assert out['msbuddy_provenance']['status'] == 'unavailable'
    assert out['msbuddy_cases'].native_status.str.contains('unavailable').all()
    assert len(out['joint_hypotheses']) > 8
    assert out['joint_search_audit'].source_budget_reached.iloc[0]
    save_untargeted_workflow(tmp_path, result, out)
    loaded, _ = load_untargeted_workflow(tmp_path)
    saved = loaded['untargeted_analysis']
    assert saved['criteria']['joint_model'] == 'joint_ms1'
    assert len(saved['joint_hypotheses']) == len(out['joint_hypotheses'])
    assert saved['joint_provenance'] == out['joint_provenance']


def test_RT_bounds_remain_separate_between_chromatographies():
    result, analysis = example()
    other = result['streams']['RP_Pos']['features'].copy()
    other['rt'] += 10.
    result['streams']['NP_Pos'] = {'features': other}
    frame = result['streams']['RP_Pos']['features']
    last = str(frame.feature_id.iloc[-1])
    frame.loc[frame.index[-1], 'rt'] = 6.
    engine = JointEngine(result, analysis, CRITERIA)
    assert engine.bounded({('RP_Pos', '0'): ('F', '[M+H]+'), ('NP_Pos', '0'): ('F', '[M+H]+')})
    assert not engine.bounded({('RP_Pos', '0'): ('F', '[M+H]+'), ('RP_Pos', last): ('F', '[M+Na]+')})


def test_viewer_uses_saved_joint_tables_and_limits_alternatives(monkeypatch):
    from presentation.untargeted_mode import build_mode_payload
    result, analysis = example()
    out = apply_joint_inference(result, analysis, replace(CRITERIA, formula_scoring_limit=8))
    out['lipid_grid_version'], out['compound_version'] = 0, 0
    monkeypatch.setattr('presentation.untargeted_mode.assign_compound_ids',
                        lambda *a, **k: pytest.fail('Saved joint results must not invoke the old selector'))
    monkeypatch.setattr('standard_lipidomics.joint_inference.apply_joint_inference',
                        lambda *a, **k: pytest.fail('Viewer must not rerun inference'))
    payload = build_mode_payload(result, out)
    assert payload['jointModel'] and payload['fullFeatureGrouping']
    assert len(payload['jointHypotheses']['RP_Pos\0'+'0']) <= 8
    assert payload['msbuddyStatus'] == 'not requested'
    assert not payload['links']  # Old mass-only links cannot reappear in this view.
    json.dumps(payload, allow_nan=False)


def test_background_job_freezes_optional_runtime_and_reports_missing_native(tmp_path, monkeypatch):
    from application.untargeted_jobs import start_untargeted_job
    from test_annotation_jobs import wait_job
    from standard_lipidomics.project_store import load_untargeted_workflow
    monkeypatch.setenv('LASER_MSBUDDY_CONFIG', str(tmp_path/'not-installed.json'))
    result, _ = example()
    frame = result['streams']['RP_Pos']['features']
    for i, scale in enumerate([1., 1.02, .98, 20., 20.2, 19.8]):
        frame[f'b{i}'] = frame.b0*scale
    job = start_untargeted_job(tmp_path, replace(CRITERIA, use_msbuddy=True, formula_scoring_limit=4), result=result)
    request = json.loads((job/'request.json').read_text())
    assert not request['msbuddy_runtime']['available']
    state = wait_job(job)
    assert state['status'] == 'complete', (job/'untargeted.log').read_text()
    assert state['warnings'] and 'unavailable' in state['message']
    loaded, _ = load_untargeted_workflow(tmp_path)
    assert loaded['untargeted_analysis']['joint_provenance']['version'] == 1


def phase_example(extra_isomer=False):
    from standard_lipidomics.untargeted_adducts import ADDUCTS
    from standard_lipidomics.isotope_patterns import fine_pattern
    ions = {a.name: a for a in ADDUCTS}
    neutral = ions['[M+H]+'].neutral(fine_pattern('C40H80O8', '[M+H]+')[0].mz)
    streams, rows = {}, []
    for stream, adduct, rt in [('RP_Pos', '[M+H]+', 5.), ('RP_Neg', '[M-H]-', 5.02),
                              ('NP_Pos', '[M+H]+', 8.), ('NP_Neg', '[M-H]-', 8.02)]:
        data = {'feature_id': ['a'], 'mz': [ions[adduct].mz(neutral)], 'rt': [rt]}
        for i, v in enumerate([10., 12., 11., 21., 20., 22.]):
            data[f'b{i}'] = [v]
        frame = pd.DataFrame(data)
        if extra_isomer and stream == 'RP_Pos':
            frame = pd.concat([frame, frame.assign(feature_id='b', rt=9.)], ignore_index=True)
        frame.attrs.update(sample_columns=[f'b{i}' for i in range(6)],
                           sample_group_map={f'b{i}': 'A' if i < 3 else 'B' for i in range(6)})
        streams[stream] = {'features': frame}
        for r in frame.itertuples():
            rows.append({'stream': stream, 'feature_id': r.feature_id, 'mz': r.mz, 'rt': r.rt,
                         'p_value': .0001, 'log2_fold_change_b_vs_a': 2.})
    frame = pd.DataFrame(rows)
    result = {'streams': streams, 'automatic_annotations': pd.DataFrame()}
    analysis = {'raw_p_features': frame, 'features': frame, 'criteria': asdict(CRITERIA),
                'acquisition_method': default_acquisition_method()}
    engine = JointEngine(result, analysis, CRITERIA)
    selected = {key: {'label': ('C40H80O8', '[M-H]-' if key[0].endswith('Neg') else '[M+H]+'),
                      'status': 'synthetic resolved', 'component': 'synthetic', 'margin': 1.}
                for key in engine.root_keys}
    for source, sfid, target in [('RP_Pos', 'a', 'RP_Neg'), ('RP_Pos', 'a', 'NP_Pos'), ('RP_Pos', 'a', 'NP_Neg'),
                                 *([('RP_Pos', 'b', 'NP_Pos'), ('RP_Pos', 'b', 'NP_Neg')] if extra_isomer else [])]:
        a = frame[frame.stream.eq(source) & frame.feature_id.eq(sfid)].iloc[0]
        b = frame[frame.stream.eq(target)].iloc[0]
        engine.support.append({'source_stream': source, 'source_feature_id': sfid, 'source_adduct': '[M+H]+',
            'target_stream': target, 'target_feature_id': 'a', 'target_adduct': '[M-H]-' if target.endswith('Neg') else '[M+H]+',
            'source_mz': a.mz, 'target_mz': b.mz, 'source_rt': a.rt, 'target_rt': b.rt,
            'rt_compared': source[:2] == target[:2], 'source_formula': 'C40H80O8', 'context_cost': .1,
            'context_selected': True, 'profile_cost': 0.})
    return engine, selected


def test_unique_four_mode_families_link_with_independent_RT_domains():
    engine, selected = phase_example()
    ledger, _, associations, _ = engine.assemble(selected)
    assert ledger.compound_id.nunique() == 1
    assert len(ledger) == 4
    assert associations.association_status.eq('selected').all()


def test_RP_isomers_keep_one_to_many_NP_associations_separate():
    engine, selected = phase_example(extra_isomer=True)
    ledger, _, associations, _ = engine.assemble(selected)
    assert ledger.compound_id.nunique() == 3
    assert len(associations) == 2
    assert associations.association_status.str.startswith('ambiguous').all()


def test_water_loss_cannot_borrow_an_intact_parent_from_another_mode():
    engine, selected = phase_example()
    root = ('RP_Pos', 'a')
    selected[root]['label'] = ('C40H80O8', '[M+H-H2O]+')
    for e in engine.support:
        if (e['source_stream'], e['source_feature_id']) == root:
            e['source_adduct'] = '[M+H-H2O]+'
    ledger, _, _, _ = engine.assemble(selected)
    row = ledger[ledger.stream.eq('RP_Pos') & ledger.feature_id.eq('a')].iloc[0]
    assert row.joint_adduct == ''
    assert 'without same-mode intact parent' in row.joint_status
