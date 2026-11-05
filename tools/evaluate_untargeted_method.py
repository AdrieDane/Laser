"""Compare legacy and method-aware grouping on one private frozen checkpoint.

Statistics/formula/isotope outputs are reused unchanged. No gold identities enter
the inference worker. Full feature-list discovery belongs to the next stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pandas as pd

from audit_untargeted_checkpoint import metric_rows
from freeze_untargeted_baseline import (
    ROOT, MODES, digest, fingerprint, project_files, score_representatives, write_json,
)


def scored_ledger(ledger, predictions):
    frame = ledger.copy()
    assert not predictions.duplicated(['stream', 'feature_id']).any()
    assigned = {(r.stream, str(r.feature_id)): r.adduct for r in predictions.itertuples() if r.adduct and getattr(r, 'isotope_index', 0) == 0}
    frame['assigned_adduct'] = [assigned.get((r.stream, str(r.feature_id)), '') for r in frame.itertuples()]
    frame['correct_adduct'] = frame.mapped & frame.assigned_adduct.ne('') & frame.assigned_adduct.eq(frame.adduct)
    return frame


def class_mode_metrics(frame):
    rows = []
    for cls, group in frame.groupby('lipid_class', sort=True):
        for row in metric_rows(group, pd.DataFrame(columns=['stream', 'feature_id'])):
            if row['scope'] == 'all_classes':
                rows.append({**row, 'scope': cls, 'unconfirmed_extra_ions': None,
                             'extras_reason': 'No lipid-class identity is inferred for unconfirmed ions'})
    return rows


def run(args):
    checkpoint, out = args.checkpoint.resolve(), args.output.resolve()
    if out.is_relative_to(ROOT) or out.is_relative_to(checkpoint) or checkpoint.is_relative_to(out):
        raise ValueError('Use a new private output directory outside the repo and checkpoint')
    if out.exists():
        raise ValueError('Never overwrite an evaluation attempt')
    node = Path(args.node or shutil.which('node') or '')
    if not node.is_file():
        raise ValueError('Node executable is required')
    inputs = json.loads((checkpoint / 'input_fingerprints.json').read_text())
    project = checkpoint / 'project'
    assert fingerprint(list(project_files(project)), project) == inputs
    old = json.loads((checkpoint / 'provenance.json').read_text())
    for name, expected in old['replay_hashes'][0].items():
        assert digest(checkpoint / 'replay_1' / name) == expected
    assert digest(checkpoint / 'reference/manual_mapping.csv') == old['manual_mapping_sha256']
    assert digest(checkpoint / 'reference/regression_queries.json') == old['regression_queries_sha256']
    revision = subprocess.check_output(['git', 'rev-parse', args.revision+'^{commit}'], cwd=ROOT, text=True).strip()
    out.mkdir(parents=True)
    archive = out / 'source.tar'
    with archive.open('wb') as handle:
        subprocess.run(['git', 'archive', revision], cwd=ROOT, stdout=handle, check=True)
    source = out / 'source'
    source.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter='data')
    method = json.loads(args.method.read_text())
    queries = json.loads((checkpoint / 'reference/regression_queries.json').read_text())
    ledger = pd.read_csv(checkpoint / 'reference/query_ledger.csv', dtype={'feature_id': str}).fillna('')
    keys = [{'stream': q['stream'], 'feature_id': str(q['feature_id'])} for q in queries]
    write_json(out / 'feature_keys.json', keys)
    # Freeze the protocol before the first prediction; do not choose arms after scoring.
    write_json(out / 'protocol.json', {'stage': 2, 'source_revision': revision,
        'arms': ['legacy', 'method_aware'], 'checkpoint': str(checkpoint),
        'reference_mapping_sha256': old['manual_mapping_sha256'],
        'regression_queries_sha256': old['regression_queries_sha256'],
        'source_archive_sha256': digest(archive), 'supplied_method_sha256': digest(args.method),
        'method_snapshot': method, 'scoring_weights': {'anchor_chemistry_penalty': .10, 'member_chemistry_penalty': .025},
        'statistics_formula_isotope_searches': 'Frozen, not rerun', 'default_promotion': False,
        'primary_metrics': ['all_manual_original_ion_adduct_recovery', 'conditional_assigned_adduct_agreement',
                            'assignment_coverage', 'unconfirmed_extras', 'exact_PC_counterparts'],
        'reference_scope': 'All frozen class labels, all four modes; development acquisition only'})
    baseline = pd.read_csv(checkpoint / 'replay_1/compound_features.csv', dtype={'feature_id': str}).fillna('')
    legacy_scored = scored_ledger(ledger, baseline)
    metrics, class_metrics, pc_metrics = {}, {}, {}
    arms = {'baseline': baseline}
    scores = {'baseline': legacy_scored}
    metrics['baseline'] = metric_rows(legacy_scored, baseline)
    class_metrics['baseline'] = class_mode_metrics(legacy_scored)
    for arm in ('legacy', 'method_aware'):
        output = out / arm
        output.mkdir()
        configured = {**method, 'adduct_scoring': arm}
        write_json(output / 'method.json', configured)
        print(f'Running frozen {arm} inference and browser selector', flush=True)
        subprocess.run([sys.executable, str(source / 'tools/freeze_untargeted_baseline.py'), '--worker',
                        str(source), str(project), str(out / 'feature_keys.json'), str(output),
                        str(node.resolve()), str(output / 'method.json')], cwd=source, check=True)
        predicted = pd.read_csv(output / 'compound_features.csv', dtype={'feature_id': str}).fillna('')
        arms[arm] = predicted
        scored = scored_ledger(ledger, predicted)
        scores[arm] = scored
        scored.to_csv(output / 'reference_predictions.csv', index=False)
        metrics[arm] = metric_rows(scored, predicted)
        class_metrics[arm] = class_mode_metrics(scored)
        rows, pc = score_representatives(queries, json.loads((output / 'representatives.json').read_text()), ledger)
        pc_metrics[arm] = pc
        pd.DataFrame(rows).to_csv(output / 'all_PC_mode_slots.csv', index=False)
    # Legacy implementation must preserve the complete scientific assignment table.
    columns = ['stream', 'feature_id', 'compound_id', 'adduct', 'neutral_mass', 'mode_representative', 'cluster_id']
    def scientific(frame):
        return frame[columns].sort_values(['stream', 'feature_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(scientific(baseline), scientific(arms['legacy']))
    comparisons = []
    for arm in ('legacy', 'method_aware'):
        after = { (r['scope'], r['mode']): r for r in metrics[arm] }
        for before in metrics['baseline']:
            row = after[(before['scope'], before['mode'])]
            comparisons.append({'arm': arm, 'scope': before['scope'], 'mode': before['mode'],
                **{key+'_before': value for key, value in before.items() if key not in ('scope', 'mode')},
                **{key+'_after': value for key, value in row.items() if key not in ('scope', 'mode')}})
    pd.DataFrame(comparisons).to_csv(out / 'all_mode_comparison.csv', index=False)
    class_before = {(r['scope'], r['mode']): r for r in class_metrics['baseline']}
    class_comparison = []
    for row in class_metrics['method_aware']:
        before = class_before[(row['scope'], row['mode'])]
        class_comparison.append({'class': row['scope'], 'mode': row['mode'],
            **{key+'_before': value for key, value in before.items() if key not in ('scope', 'mode')},
            **{key+'_after': value for key, value in row.items() if key not in ('scope', 'mode')}})
    pd.DataFrame(class_comparison).to_csv(out / 'all_class_mode_comparison.csv', index=False)
    transitions = ledger.copy()
    transitions['correct_before'] = scores['baseline'].correct_adduct
    transitions['correct_after'] = scores['method_aware'].correct_adduct
    transitions['adduct_before'] = scores['baseline'].assigned_adduct
    transitions['adduct_after'] = scores['method_aware'].assigned_adduct
    transitions['gained'] = ~transitions.correct_before & transitions.correct_after
    transitions['lost'] = transitions.correct_before & ~transitions.correct_after
    transitions.to_csv(out / 'all_reference_transitions.csv', index=False)
    gains = []
    for mode in MODES:
        frame = transitions[transitions.stream.eq(mode)]
        gains.append({'mode': mode, 'gained': int(frame.gained.sum()), 'lost': int(frame.lost.sum())})
    assert fingerprint(list(project_files(project)), project) == inputs
    summary = {'source_revision': revision, 'legacy_assignments_identical': True,
        'positive_PC_counterparts': pc_metrics, 'reference_transitions': gains,
        'all_four_modes_scored': True, 'all_class_mode_rows': len(class_comparison),
        'independent_validation': False, 'default_changed': False,
        'statistics_formula_isotope_searches_rerun': False, 'checkpoint_inputs_unchanged': True,
        'compound_ion_counts': {arm: {mode: int(frame.stream.eq(mode).sum()) for mode in MODES} for arm, frame in arms.items()}}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--method', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--node')
    run(parser.parse_args())
