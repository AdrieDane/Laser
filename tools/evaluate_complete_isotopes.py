"""Frozen Stage 4 comparison. Private study results must remain outside Git.

Runtime workers never receive reference formulas. The separate reference-formula
readout and raw QC check are conditional diagnostics, not a verified isotope gold
set or independent precision/recall estimate.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

from freeze_untargeted_baseline import ROOT, MODES, digest, fingerprint, project_files, write_json, ratio


def worker(source, checkpoint, method_file, out, model):
    sys.path.insert(0, str(source/'app'))
    from standard_lipidomics.project_store import load_untargeted_workflow
    from standard_lipidomics.untargeted_compounds import assign_compound_ids
    from standard_lipidomics.untargeted import UntargetedComparisonCriteria
    from standard_lipidomics.complete_isotopes import apply_complete_isotopes
    from presentation.untargeted_mode import build_mode_payload
    import time
    result, _ = load_untargeted_workflow(checkpoint/'project')
    analysis = result['untargeted_analysis']
    result['acquisition_method'] = analysis['acquisition_method'] = json.loads(method_file.read_text())
    criteria = UntargetedComparisonCriteria(**{**analysis['criteria'], 'isotope_model': model})
    analysis['criteria'] = vars(criteria)
    analysis['compound_version'] = None
    analysis = assign_compound_ids(analysis, result=result)
    original = analysis['features'].copy()
    compounds = analysis['compound_features'].copy()
    start = time.monotonic()
    analysis = apply_complete_isotopes(result, analysis, criteria,
        lambda f, msg: print(msg, flush=True))
    elapsed = time.monotonic()-start
    pd.testing.assert_frame_equal(original, analysis['features'][original.columns])
    pd.testing.assert_frame_equal(compounds, analysis['compound_features'][compounds.columns])
    for name in ('raw_p_features', 'compound_features', 'isotope_candidate_scores', 'isotope_peak_evidence'):
        analysis.get(name, pd.DataFrame()).to_csv(out/(name+'.csv'), index=False)
    write_json(out/'payload.json', build_mode_payload(result, analysis))
    write_json(out/'provenance.json', {'criteria': vars(criteria), 'model': analysis.get('isotope_model_provenance'),
        'model_seconds': elapsed, 'all_original_feature_values_unchanged': True,
        'all_original_compound_values_unchanged': True, 'tested_features': len(original)})


def raw_qc_check(result, raw_root, queries, out, ppm):
    """Centroid apex snapshots near existing feature RT; no trace model is fitted."""
    from standard_traces import _iter_mzxml_scans
    from standard_lipidomics.isotope_quality import IsotopeIndex
    lookup = {}
    for p in raw_root.rglob('*.mzXML'):
        if p.stem in lookup:
            raise ValueError('Ambiguous raw file stem; no implicit file selection allowed.')
        lookup[p.stem] = p
    output, files, missing = [], [], []
    for mode in MODES:
        frame = result['streams'][mode]['features']
        groups, types = frame.attrs['sample_group_map'], frame.attrs['sample_type_map']
        samples = [c for c in frame.attrs['sample_columns'] if IsotopeIndex.is_qc(groups.get(c, ''), types.get(c, ''))]
        rows = [q for q in queries if q['stream'] == mode]
        rt = np.array([q['rt'] for q in rows])
        for sample in samples:
            if sample not in lookup:
                missing.append({'mode': mode, 'sample': sample})
                continue
            path = lookup[sample]
            print('Raw QC check: '+mode+' '+path.name, flush=True)
            files.append({'mode': mode, 'sample': sample, 'sha256': digest(path)})
            best = {}
            for retention, mz, area in _iter_mzxml_scans(path):
                if np.any(np.diff(mz) < 0):
                    order = np.argsort(mz)
                    mz, area = mz[order], area[order]
                for i in np.flatnonzero(np.abs(rt-retention) <= .10):
                    q = rows[i]
                    left, right = np.searchsorted(mz, [q['mz']*(1-ppm*1e-6), q['mz']*(1+ppm*1e-6)])
                    if right-left != 1 or area[left] <= 0:
                        continue  # Ambiguous parent centroids are not verified observations.
                    if i in best and best[i][0] >= area[left]:
                        continue
                    observations = []
                    for peak in q['peaks']:
                        target = mz[left] + peak['expected_mz']-q['mz']
                        lo, hi = np.searchsorted(mz, [target*(1-ppm*1e-6), target*(1+ppm*1e-6)])
                        observations.append({**peak, 'raw_centroid_count': int(hi-lo),
                            'raw_ratio': float(area[lo]/area[left]) if hi-lo == 1 else None,
                            'raw_mz': float(mz[lo]) if hi-lo == 1 else None})
                    best[i] = (float(area[left]), retention, observations)
            for i, q in enumerate(rows):
                if i not in best:
                    output.append({'query_id': q['query_id'], 'stream': mode, 'sample': sample,
                                   'status': 'parent absent/ambiguous within fixed RT window'})
                    continue
                for peak in best[i][2]:
                    output.append({'query_id': q['query_id'], 'stream': mode, 'sample': sample,
                                   'status': 'centroid snapshot; not independently verified envelope',
                                   'parent_raw_intensity': best[i][0], 'raw_rt_min': best[i][1], **peak})
    pd.DataFrame(output).to_csv(out/'raw_QC_centroid_observations.csv', index=False)
    write_json(out/'raw_QC_provenance.json', {'files': files, 'missing_files': missing,
        'RT_half_window_min': .10, 'scope': 'apex centroid snapshots, no EIC fit; raw and XCMS alignment may differ',
        'verified_isotope_gold_set': False})
    return {'QC_files': len(files), 'missing_QC_files': len(missing), 'observation_rows': len(output)}


def reference_readout(source, cp, method, run, raw_root):
    sys.path.insert(0, str(source/'app'))
    from standard_lipidomics.project_store import load_untargeted_workflow
    from standard_lipidomics.untargeted import UntargetedComparisonCriteria
    from standard_lipidomics.isotope_quality import IsotopeIndex, score_envelope
    from standard_lipidomics.isotope_patterns import parse_formula, formula_text
    from standard_lipidomics.untargeted_formula_labels import _CANDIDATE
    result, _ = load_untargeted_workflow(cp/'project')
    criteria = replace(UntargetedComparisonCriteria(**result['untargeted_analysis']['criteria']), isotope_model='complete')
    indexes = {m: IsotopeIndex(m, v['features']) for m, v in result['streams'].items()}
    reference = pd.read_csv(cp/'reference/query_ledger.csv', dtype={'feature_id': str}).fillna('')
    before = pd.read_csv(run/'current/raw_p_features.csv', dtype={'feature_id': str}).fillna('')
    candidates = pd.read_csv(run/'complete/isotope_candidate_scores.csv', dtype={'feature_id': str}).fillna('')
    raw = {(r.stream, str(r.feature_id)): r for r in before.itertuples()}
    grouped = {k: f for k, f in candidates.groupby(['stream', 'feature_id'])}
    oracle, recall, raw_queries = [], [], []
    for r in reference.to_dict('records'):
        key = (r['stream'], str(r['feature_id']))
        base = {k: r[k] for k in ('query_id', 'stream', 'feature_id', 'lipid_class', 'formula', 'adduct')}
        base.update(status='not evaluable', reason='', mapped=bool(r['mapped']))
        if not r['mapped'] or not r['formula'] or '(IS)' in r['name']:
            base['reason'] = 'unmapped, formula missing or internal standard'
            oracle.append(base)
            continue
        try:
            neutral = formula_text(parse_formula(r['formula']))
            score, peaks = score_envelope(indexes[key[0]], key[1], r['formula'], r['adduct'], criteria,
                method['methods'][key[0][:2]]['instrument'])
        except ValueError as exc:
            base['reason'] = str(exc)
            oracle.append(base)
            continue
        base.update(score)
        base['query_id'], base['lipid_class'] = r['query_id'], r['lipid_class']
        oracle.append(base)
        unique = {p['component']: {k: p[k] for k in ('component', 'nominal', 'expected_mz', 'expected_ratio')} for p in peaks}
        if unique:
            index = indexes[key[0]]
            parent = index.by_id[key[1]]
            raw_queries.append({'query_id': r['query_id'], 'stream': key[0], 'mz': float(index.mz[parent]),
                                'rt': float(index.rt[parent]), 'peaks': list(unique.values())})
        if key not in raw or not r['ion_metric_eligible']:
            continue
        old = _CANDIDATE.findall(str(getattr(raw[key], 'elemental_formula_candidates', '')))
        new = grouped.get(key, pd.DataFrame())
        pairs = []
        if not new.empty and 'rank' in new:
            ranked = new[pd.to_numeric(new['rank'], errors='coerce').notna()].copy()
            ranked['rank'] = pd.to_numeric(ranked['rank'])
            pairs = [(str(v.adduct), str(v.formula)) for v in ranked.sort_values('rank').itertuples()]
        def canonical(items):
            return [(a, formula_text(parse_formula(f))) for a, f in items]
        old, pairs = canonical(old), canonical(pairs)
        recall.append({**{k: r[k] for k in ('query_id', 'stream', 'feature_id', 'lipid_class')},
            'baseline_formula_in_pool': any(f == neutral for _, f in old),
            'complete_formula_in_pool': any(f == neutral for _, f in pairs),
            'baseline_joint_in_pool': (r['adduct'], neutral) in old,
            'complete_joint_in_pool': (r['adduct'], neutral) in pairs,
            'baseline_joint_top1': bool(old and old[0] == (r['adduct'], neutral)),
            'complete_joint_top1': bool(pairs and pairs[0] == (r['adduct'], neutral)),
            'complete_joint_top5': (r['adduct'], neutral) in pairs[:5]})
    oracle, recall = pd.DataFrame(oracle), pd.DataFrame(recall)
    oracle.to_csv(run/'all_reference_formula_conditioned_readout.csv', index=False)
    recall.to_csv(run/'common_query_formula_recall.csv', index=False)
    metrics, cells = [], []
    for label, frame in [('all_classes', reference), *reference.groupby('lipid_class')]:
        for mode in MODES:
            all_rows = frame[frame.stream.eq(mode)]
            q = recall[recall.stream.eq(mode)]
            if label != 'all_classes':
                q = q[q.lipid_class.eq(label)]
            row = {'scope': label, 'mode': mode, 'all_manual_rows': len(all_rows), 'common_formula_queries': len(q)}
            for field in [c for c in recall if c.endswith(('_pool', '_top1', '_top5'))]:
                row[field] = ratio(int(q[field].sum()), len(q))
            (metrics if label == 'all_classes' else cells).append(row)
    write_json(run/'four_mode_formula_metrics.json', metrics)
    write_json(run/'all_252_class_mode_metrics.json', cells)
    raw_metrics = raw_qc_check(result, raw_root, raw_queries, run, criteria.cross_mode_ppm_tolerance)
    return {'all_reference_rows': len(reference), 'formula_conditioned_rows': len(oracle),
        'common_formula_queries': len(recall), 'raw_QC_check': raw_metrics,
        'common_query_metrics': metrics, 'independent_isotope_precision_recall': None,
        'limitation': 'No independently verified isotope-member gold set. Formula-conditioned fits and raw QC snapshots are diagnostics, not recognition accuracy.'}


def run(args):
    cp, out = args.checkpoint.resolve(), args.output.resolve()
    if out.exists() or out.is_relative_to(ROOT) or out.is_relative_to(cp) or cp.is_relative_to(out):
        raise ValueError('Use a new private output directory outside repository/checkpoint.')
    inputs = json.loads((cp/'input_fingerprints.json').read_text())
    assert fingerprint(list(project_files(cp/'project')), cp/'project') == inputs
    revision = subprocess.check_output(['git', 'rev-parse', args.revision+'^{commit}'], cwd=ROOT, text=True).strip()
    out.mkdir(parents=True)
    archive = out/'source.tar'
    with archive.open('wb') as handle:
        subprocess.run(['git', 'archive', revision], cwd=ROOT, stdout=handle, check=True)
    source = out/'source'
    source.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter='data')
    method = json.loads(args.method.read_text())
    method['adduct_scoring'] = 'legacy'
    write_json(out/'method.json', method)
    write_json(out/'protocol.json', {'source_revision': revision, 'source_archive_sha256': digest(archive),
        'arms': ['current', 'complete'], 'default_changed': False,
        'reference_scope': 'all 3668 rows retained; formula-conditioned diagnostics separated from runtime prediction',
        'candidate_recall_scope': 'same mapped, unambiguous, formula-bearing raw-p queries in both arms; all classes/four modes',
        'independent_isotope_gold_set': False, 'raw_QC_RT_half_window_min': .10,
        'QC_snapshots': 'apex centroid checks only; no full trace model, no inferred detection limit',
        'statistics_and_compound_assignments': 'exactly frozen; additive score fields only',
        'parameters': {'probability_coverage': .99999, 'relative_component_floor': .001,
                       'ratio_pairs_min': 3, 'log_ratio_MAD_max': .5, 'ratio_support_factor': 2,
                       'ratio_cost_scale': .35, 'missing_above_proxy_penalty': .25, 'centroid_group_span_FWHM': 2}})
    for arm in ['current', 'complete']:
        d = out/arm
        d.mkdir()
        print('RUNNING '+arm, flush=True)
        subprocess.run([sys.executable, str(source/'tools/evaluate_complete_isotopes.py'), '--worker',
                        str(source), str(cp), str(out/'method.json'), str(d), arm], check=True, cwd=source,
                       env=dict(os.environ))
    a = pd.read_csv(out/'current/compound_features.csv', dtype={'feature_id': str}).fillna('')
    b = pd.read_csv(out/'complete/compound_features.csv', dtype={'feature_id': str}).fillna('')
    pd.testing.assert_frame_equal(a, b[a.columns])
    original = pd.read_csv(cp/'replay_1/compound_features.csv', dtype={'feature_id': str}).fillna('')
    cols = ['stream', 'feature_id', 'compound_id', 'adduct', 'neutral_mass', 'mode_representative', 'cluster_id']
    pd.testing.assert_frame_equal(a[cols].sort_values(['stream', 'feature_id']).reset_index(drop=True),
                                  original[cols].sort_values(['stream', 'feature_id']).reset_index(drop=True))
    summary = reference_readout(source, cp, method, out, args.raw.resolve())
    assert fingerprint(list(project_files(cp/'project')), cp/'project') == inputs
    write_json(out/'summary.json', {**summary, 'source_revision': revision, 'checkpoint_unchanged': True,
                                  'current_assignments_equal_Stage1': True, 'default_changed': False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(*[Path(v) for v in sys.argv[2:6]], sys.argv[6])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for flag in ['checkpoint', 'method', 'raw', 'output']:
            parser.add_argument('--'+flag, type=Path, required=True)
        parser.add_argument('--revision', default='HEAD')
        run(parser.parse_args())
