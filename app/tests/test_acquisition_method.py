"""Independent chemistry, persistence, worker and UI contract checks."""
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from standard_lipidomics.acquisition_method import (
    default_acquisition_method, normalize_acquisition_method, load_acquisition_method,
    save_acquisition_method, method_id, ion_state, chemistry_penalty, guard_homologous_edges,
)
from standard_lipidomics.untargeted_adducts import ADDUCTS, adduct_options, search_adduct_evidence
from standard_lipidomics.untargeted import UntargetedComparisonCriteria, analyse_standard_untargeted
from standard_lipidomics.untargeted_compounds import assign_compound_ids
from standard_lipidomics.project_store import save_untargeted_workflow, load_untargeted_workflow
from application.untargeted_jobs import start_untargeted_job
from test_standard_untargeted import StandardUntargetedTests
from test_annotation_jobs import wait_job

ROOT = Path(__file__).resolve().parents[2]


def recipe():
    method = json.loads((ROOT / 'docs/examples/acquisition_method.example.json').read_text())
    method['adduct_scoring'] = 'method_aware'
    return method


def test_unknown_chemistry_does_not_invent_additives_or_remove_alternatives():
    method = default_acquisition_method()
    assert method['adduct_scoring'] == 'legacy'
    assert ion_state(method, 'NP_Neg', '[M+CH3COO]-') == 'possible'
    assert ion_state(method, 'RP_Neg', '[M+HCOO]-') == 'possible'
    assert {x.name for x in adduct_options('neg', method=method, stream='RP_Neg')} == {x.name for x in ADDUCTS if x.polarity == 'neg'}
    assert '[M-2H]2-' in {x.name for x in adduct_options('neg', 2, method=method, stream='RP_Neg')}


def test_solvent_specific_ammonia_is_not_treated_as_constant_gradient_concentration():
    method = recipe()
    assert method['methods']['NP']['solvents']['A']['additives'][1]['concentration'] == 3.35
    assert len(method['methods']['NP']['solvents']['B']['additives']) == 1
    assert ion_state(method, 'RP_Pos', '[M+NH4]+') == 'common'
    assert ion_state(method, 'NP_Neg', '[M+HCOO]-') == 'common'
    assert ion_state(method, 'NP_Neg', '[M+CH3COO]-') == 'possible'
    assert method['methods']['RP']['instrument']['resolution_reference_mz'] is None


def test_literature_class_preferences_are_soft_with_unknown_fallback():
    method = recipe()
    assert chemistry_penalty(method, 'RP_Pos', '[M+H]+', 'PC') < chemistry_penalty(method, 'RP_Pos', '[M+NH4]+', 'PC')
    assert chemistry_penalty(method, 'RP_Pos', '[M+NH4]+', 'TG') < chemistry_penalty(method, 'RP_Pos', '[M+H]+', 'TG')
    assert chemistry_penalty(method, 'RP_Neg', '[M-H]-', 'PE') < chemistry_penalty(method, 'RP_Neg', '[M+CH3COO]-', 'PE')
    assert chemistry_penalty(method, 'RP_Pos', '[M+H]+', 'unknown-class') == chemistry_penalty(method, 'RP_Pos', '[M+H]+')
    assert ion_state(method, 'RP_Pos', '[M+NH4]+') != 'disabled'


@pytest.mark.parametrize('ion', ['[M+2H]2+', '[M+Na-2H]-'])
def test_explicitly_excluded_ions_cannot_be_reenabled(ion):
    method = recipe()
    method['methods']['RP']['adducts']['pos' if ion.endswith('+') else 'neg'][ion] = 'common'
    with pytest.raises(ValueError, match='Unsupported/excluded'):
        normalize_acquisition_method(method)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 0, True])
def test_invalid_additive_concentration_fails_without_fallback(value):
    method = recipe()
    method['methods']['NP']['solvents']['A']['additives'][0]['concentration'] = value
    with pytest.raises(ValueError):
        normalize_acquisition_method(method)


def test_malformed_method_file_and_nonchronological_gradient_fail(tmp_path):
    (tmp_path / 'metadata').mkdir()
    (tmp_path / 'metadata/acquisition_method.json').write_text('{')
    with pytest.raises(ValueError):
        load_acquisition_method(tmp_path)
    method = recipe()
    method['methods']['RP']['gradient'][2]['time_min'] = .5
    with pytest.raises(ValueError, match='Gradient times'):
        normalize_acquisition_method(method)


def test_disabled_ion_is_excluded_from_search_in_both_directions():
    method = recipe()
    method['methods']['RP']['adducts']['pos']['[M+Na]+'] = 'disabled'
    neutral = 700.0
    ions = {ion.name: ion for ion in ADDUCTS}
    source = pd.DataFrame({'feature_id': ['h', 'na'],
        'mz': [ions['[M+H]+'].mz(neutral), ions['[M+Na]+'].mz(neutral)],
        'rt': [5., 5.], 'already_annotated': [False, False]})
    hits = source.assign(raw_p_significant=True)
    matched = search_adduct_evidence({'RP_Pos': source}, {'RP_Pos': hits},
        ppm=2, rt_tolerance=.1, method=method)
    assert matched.empty
    assert ion_state(method, 'NP_Pos', '[M+Na]+') == 'possible'  # RP override stays local.


def test_formate_acetate_ch2_pair_retained_as_ambiguous_not_used_as_cluster():
    offsets = {ion.name: ion.offset for ion in ADDUCTS}
    assert offsets['[M+CH3COO]-'] - offsets['[M+HCOO]-'] == pytest.approx(14.015650064, abs=1e-9)
    raw = pd.DataFrame({'stream': ['NP_Neg'] * 2, 'feature_id': ['a', 'b'],
        'mz': [744.998202851, 759.013852915], 'rt': [5., 5.01],
        'p_value': [.001, .001], 'log2_fold_change_b_vs_a': [2., 2.], 'raw_p_significant': [True, True]})
    pairs = pd.DataFrame({'stream': ['NP_Neg'], 'source_feature_id': ['a'], 'target_feature_id': ['b'],
        'ch2_steps': [1], 'charge_hypothesis': [1], 'rt_delta_seconds': [.6]})
    analysis = {'features': raw, 'raw_p_features': raw, 'criteria': {}, 'homologous_matches': pairs}
    legacy = assign_compound_ids(analysis)
    assert legacy['compound_features'].cluster_id.ne('').all()
    aware = assign_compound_ids({**analysis, 'acquisition_method': recipe()})
    assert aware['compound_features'].cluster_id.eq('').all()
    assert aware['homologous_matches'].adduct_alias_ambiguous.all()  # Evidence remains inspectable.
    allowed = recipe()
    allowed['methods']['NP']['adducts']['neg']['[M+CH3COO]-'] = 'disabled'
    unambiguous = assign_compound_ids({**analysis, 'acquisition_method': allowed})
    assert unambiguous['compound_features'].cluster_id.ne('').all()
    pairs['rt_delta_seconds'] = 10.
    assert not guard_homologous_edges(pairs, recipe(), .05).adduct_alias_ambiguous.any()


def test_saved_result_keeps_its_own_method_when_project_metadata_changes(tmp_path):
    method = recipe()
    method['methods']['RP']['calibration']['pos']['mz'] = 'calibration-2026-01/mz-model-sha256'
    save_acquisition_method(tmp_path, method)
    result = StandardUntargetedTests()._result()
    result['acquisition_method'] = method
    analysis = analyse_standard_untargeted(result, UntargetedComparisonCriteria('Group_01_control', 'Group_02_disease'))
    save_untargeted_workflow(tmp_path, result, analysis)
    save_acquisition_method(tmp_path, default_acquisition_method())
    restored, _ = load_untargeted_workflow(tmp_path)
    assert restored['acquisition_method'] == method
    assert restored['untargeted_analysis']['acquisition_method'] == method
    assert (tmp_path / 'Untargeted/acquisition_method.json').is_file()
    for name in ('features', 'raw_p_features', 'compound_features', 'adduct_matches', 'homologous_matches', 'lipid_grid_matches'):
        table = restored['untargeted_analysis'][name]
        assert 'acquisition_method_id' in table
        if not table.empty:
            assert set(table.acquisition_method_id) == {method_id(method)}


def test_detached_job_uses_frozen_method_after_live_metadata_changes(tmp_path):
    method = recipe()
    method['methods']['RP']['adducts']['pos']['[M+Na]+'] = 'disabled'
    save_acquisition_method(tmp_path, method)
    result = StandardUntargetedTests()._result()
    result['acquisition_method'] = method
    job = start_untargeted_job(tmp_path,
        UntargetedComparisonCriteria('Group_01_control', 'Group_02_disease'), result=result)
    save_acquisition_method(tmp_path, default_acquisition_method())
    state = wait_job(job)
    assert state['status'] == 'complete', (job / 'untargeted.log').read_text()
    restored, _ = load_untargeted_workflow(tmp_path)
    assert restored['acquisition_method'] == method


def test_method_editor_unknown_and_populated_forms_save_only_project_settings(tmp_path):
    for method in (default_acquisition_method(), recipe()):
        # A tiny standalone page exercises the actual editor without a pipeline run.
        source = '''
import streamlit as st
from presentation.acquisition_settings import render_acquisition_settings
from standard_lipidomics.acquisition_method import load_acquisition_method
method = render_acquisition_settings({'acquisition_method':load_acquisition_method(PROJECT)}, PROJECT)
'''.replace('PROJECT', repr(str(tmp_path)))
        save_acquisition_method(tmp_path, method)
        app = AppTest.from_string(source, default_timeout=30).run()
        assert not app.exception
        next(b for b in app.button if b.label == 'Save method settings').click().run()
        assert not app.exception
        assert load_acquisition_method(tmp_path) == method
    assert not list(tmp_path.glob('*/LASER_PROJECT.json'))


def test_method_id_is_order_independent_and_normalization_is_a_copy():
    method = recipe()
    normalized = normalize_acquisition_method(method)
    assert method_id(method) == method_id(dict(reversed(list(method.items()))))
    normalized['methods']['RP']['solvents']['A']['composition'] = 'changed'
    assert normalized != method


def test_browser_dynamic_and_saved_links_respect_disabled_ions(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the browser model')
    from test_untargeted_mode_view import example
    from presentation.untargeted_mode import build_mode_payload
    result, analysis = example()
    method = recipe()
    method['methods']['RP']['adducts']['pos']['[M+Na]+'] = 'disabled'
    analysis['acquisition_method'] = method
    analysis = assign_compound_ids(analysis, result=result)
    payload = build_mode_payload(result, analysis)
    (tmp_path / 'payload.json').write_text(json.dumps(payload))
    shutil.copy2(ROOT / 'app/presentation/untargeted_mode/model.js', tmp_path / 'model.mjs')
    (tmp_path / 'check.mjs').write_text('''
import {readFileSync} from 'node:fs';
import {ModeViewerModel,featureKey} from './model.mjs';
const data=JSON.parse(readFileSync(new URL('./payload.json',import.meta.url)));
const model=new ModeViewerModel(data);
for(const key of [featureKey('RP_Pos','a'),featureKey('RP_Pos','c')]) {
  const bad=model.matches(key).filter(x=>(x.source.startsWith('RP_Pos')&&x.sourceAdduct==='[M+Na]+') ||
    (x.target.startsWith('RP_Pos')&&x.targetAdduct==='[M+Na]+'));
  if(bad.length) throw Error('Disabled sodium appeared in browser evidence');
}
''')
    subprocess.run([node, str(tmp_path / 'check.mjs')], check=True, capture_output=True, text=True)
