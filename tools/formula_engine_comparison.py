"""Private, source-bound MS1 formula benchmark; study labels never enter workers.

Preparation explicitly supplies adducts in the conditional formula arm. Blind
LASER arms remove both adduct and charge. Feature-derived MS1 area spectra are
not claimed to be acquired, isolated raw MS1 spectra.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import pandas as pd

from freeze_untargeted_baseline import MODES, fingerprint, project_files, write_json, ratio


def prepare(cp, method_file, out):
    from standard_lipidomics.project_store import load_untargeted_workflow
    from standard_lipidomics.isotope_quality import IsotopeIndex
    from standard_lipidomics.isotope_patterns import parse_formula, formula_text
    from standard_lipidomics.complete_isotopes import formula_library
    if out.exists() or out.is_relative_to(Path(__file__).resolve().parents[1]) or out.is_relative_to(cp) or cp.is_relative_to(out):
        raise ValueError('Use a new private preparation directory.')
    out.mkdir(parents=True)
    assert fingerprint(list(project_files(cp/'project')), cp/'project') == json.loads((cp/'input_fingerprints.json').read_text())
    result, _ = load_untargeted_workflow(cp/'project')
    indexes = {m: IsotopeIndex(m, v['features']) for m, v in result['streams'].items()}
    mean_levels = {m: np.nanmean(i.frame[i.samples].to_numpy(float), axis=1) for m,i in indexes.items()}
    reference = pd.read_csv(cp/'reference/query_ledger.csv', dtype={'feature_id': str}).fillna('')
    known = {formula_text(parse_formula(f)) for f in formula_library()[1]}
    rows, inputs = [], []
    for number, row in enumerate(reference.to_dict('records')):
        qid = f'query_{number+1:05d}'
        row['benchmark_id'], row['reason'] = qid, ''
        if not row['formula']:
            row['reason'] = 'reference formula missing'
        elif '(IS)' in row['name']:
            row['reason'] = 'internal standard excluded'
        else:
            try:
                row['neutral_formula'] = formula_text(parse_formula(row['formula']))
            except ValueError:
                row['reason'] = 'unsupported/labelled reference formula'
        row['formula_query'] = not row['reason']
        row['library_known'] = row.get('neutral_formula', '') in known
        if row['formula_query'] and row['mapped']:
            index = indexes[row['stream']]
            parent = index.by_id[str(row['feature_id'])]
            mz, rt = float(index.mz[parent]), float(index.rt[parent])
            lo, hi = np.searchsorted(index.mz[index.order], [mz-.01, mz+6.01])
            selected = index.order[lo:hi]
            selected = selected[np.abs(index.rt[selected]-rt) <= .05]
            intensities = mean_levels[row['stream']][selected]
            spectrum = [[float(index.mz[i]), float(v)] for i, v in zip(selected, intensities) if np.isfinite(v) and v > 0]
            charge = -2 if row['adduct'] == '[M-2H]2-' else (-1 if row['stream'].endswith('Neg') else 1)
            inputs.append({'query_id': qid, 'stream': row['stream'], 'feature_id': str(row['feature_id']),
                           'mz': mz, 'rt': rt, 'adduct': row['adduct'], 'charge': charge, 'ms1': spectrum})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'reference_ledger.csv', index=False)
    write_json(out/'known_adduct_inputs.json', inputs)
    write_json(out/'blind_inputs.json', [{k:v for k,v in q.items() if k not in {'adduct', 'charge'}} for q in inputs])
    method = json.loads(method_file.read_text())
    method['adduct_scoring'] = 'legacy'
    write_json(out/'method.json', method)
    write_json(out/'preparation.json', {'reference_rows': len(rows), 'formula_queries': sum(r['formula_query'] for r in rows),
        'exported_queries': len(inputs), 'formula_missing_or_standard_rows': sum(not r['formula_query'] for r in rows),
        'mode_formula_queries': {m:sum(r['formula_query'] and r['stream']==m for r in rows) for m in MODES},
        'input_representation': 'feature-derived mean biological MS1 area spectrum, 6 Da above parent, RT ±0.05min; no formula-based isotope selection',
        'native_evidence_loss': 'external engines receive aggregate areas; LASER can use the original per-sample areas',
        'MSMS_available': False, 'independent_validation': False})


def legacy_candidates(q, criteria, method):
    from copy import deepcopy
    from standard_lipidomics.untargeted import _elemental_formula_candidates
    from standard_lipidomics.untargeted_formula_labels import _CANDIDATE
    from standard_lipidomics.untargeted_adducts import ADDUCTS
    restricted = deepcopy(method)
    polarity = 'neg' if q['charge'] < 0 else 'pos'
    for a in ADDUCTS:
        if a.polarity == polarity:
            restricted['methods'][q['stream'][:2]]['adducts'][polarity][a.name] = 'common' if a.name == q['adduct'] else 'disabled'
    def area(target):
        compatible = [(abs(m-target), v) for m,v in q['ms1'] if abs(m-target) <= target*criteria.cross_mode_ppm_tolerance*1e-6]
        return min(compatible)[1] if compatible else np.nan
    mono = area(q['mz'])
    one = area(q['mz']+1.00335483507/abs(q['charge']))
    two = area(q['mz']+2*1.00335483507/abs(q['charge']))
    carbon = one/mono/(.0107/.9893) if np.isfinite(mono) and mono > 0 else np.nan
    m2 = two/mono*100 if np.isfinite(mono) and mono > 0 else np.nan
    text, count = _elemental_formula_candidates(q['mz'], polarity, carbon, criteria.cross_mode_ppm_tolerance,
        maximum_candidates=1000000, charge=abs(q['charge']), m_plus_2_percent=m2, method=restricted, stream=q['stream'])
    pairs = _CANDIDATE.findall(text)
    return [{'formula': f, 'adduct': a, 'charge': q['charge'], 'rank': i+1, 'isotope_scored': True}
            for i,(a,f) in enumerate(pairs)], {'generated_candidates': count, 'native_display_limit': 5,
                                             'legacy_carbon_estimate': float(carbon) if np.isfinite(carbon) else None}


def worker(source, cp, preparation, out, engine):
    sys.path.insert(0, str(source/'app'))
    from standard_lipidomics.project_store import load_untargeted_workflow
    from standard_lipidomics.untargeted import UntargetedComparisonCriteria
    from standard_lipidomics.isotope_quality import IsotopeIndex, score_envelope
    from standard_lipidomics.complete_isotopes import candidate_hypotheses
    from standard_lipidomics.lipid_formulas import predict_formulas
    result, _ = load_untargeted_workflow(cp/'project')
    criteria = replace(UntargetedComparisonCriteria(**result['untargeted_analysis']['criteria']), isotope_model='complete')
    if engine in {'lipid_known', 'lipid_blind'}:
        criteria = replace(criteria, formula_model='lipid_aware')
    method = json.loads((preparation/'method.json').read_text())
    indexes = {m: IsotopeIndex(m, v['features']) for m,v in result['streams'].items()}
    inputs = json.loads((preparation/('blind_inputs.json' if engine=='lipid_blind' else 'known_adduct_inputs.json')).read_text())
    out.mkdir(parents=True, exist_ok=False)
    def timeout(_sig, _frame):
        raise TimeoutError('30-second query budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    start = time.monotonic()
    with (out/'predictions.jsonl').open('w') as handle:
        for number,q in enumerate(inputs):
            began = time.monotonic()
            signal.alarm(30)
            try:
                if engine == 'current':
                    candidates, audit = legacy_candidates(q, criteria, method)
                elif engine == 'stage4':
                    previous, _ = legacy_candidates(q, criteria, method)
                    row = {**q, 'elemental_formula_candidates': '; '.join(r['adduct']+' '+r['formula']+' (' for r in previous[:5])}
                    candidates = []
                    for formula, adduct, origin in candidate_hypotheses(row, q['stream'], criteria, method):
                        if adduct != q['adduct']:
                            continue
                        score, _ = score_envelope(indexes[q['stream']], q['feature_id'], formula, adduct, criteria,
                            method['methods'][q['stream'][:2]]['instrument'])
                        candidates.append({**score, 'charge': q['charge'], 'isotope_scored': True, 'origin': origin})
                    ranked = sorted([r for r in candidates if r['status'] != 'parent mass incompatible'],
                                    key=lambda r:(r['score'],r['formula']))
                    for rank,r in enumerate(ranked,1):
                        r['rank'] = rank
                    audit = {'generated_candidates':len(candidates)}
                else:
                    candidates, _, audit = predict_formulas(indexes[q['stream']], q, q['stream'],
                        replace(criteria, formula_model='lipid_aware'), method,
                        supplied_adduct=q.get('adduct'))
                # Nonfinite costs denote explicitly unscored candidates.
                for candidate in candidates:
                    if not np.isfinite(candidate.get('score', 0)):
                        candidate['score'] = None
                record = {'query_id':q['query_id'], 'status':'complete', 'candidates':candidates, 'audit':audit}
            except (TimeoutError, ValueError) as exc:
                record = {'query_id':q['query_id'],'status':type(exc).__name__+': '+str(exc),'candidates':[]}
            finally:
                signal.alarm(0)
            record['seconds'] = time.monotonic()-began
            handle.write(json.dumps(record,allow_nan=False)+'\n')
            handle.flush()
            if number % 100 == 0 or number+1 == len(inputs):
                print(engine,number+1,'/',len(inputs),'elapsed',round(time.monotonic()-start,1),flush=True)
    write_json(out/'receipt.json',{'engine':engine,'queries':len(inputs),'seconds':time.monotonic()-start,
        'known_adduct_supplied':engine!='lipid_blind','reference_formulas_supplied':False,'criteria':vars(criteria),
        'native_current_limits':'N0..5 O1..32 P0..2 S0..2, isotope-carbon window; unmodified',
        'lipid_formula_limits':'C1..180 H0..400 N0..6 O0..60 P0..3 S0..3; scoring budget32',
        'MSMS_used':False})


def evaluate(preparation, predictions, out):
    from standard_lipidomics.isotope_patterns import parse_formula, formula_text
    reference = pd.read_csv(preparation/'reference_ledger.csv', dtype={'feature_id':str}).fillna('')
    rows = []
    outputs = {}
    if predictions.is_file():
        outputs = {r['query_id']:r for r in (json.loads(line) for line in predictions.read_text().splitlines())}
    for r in reference[reference.formula_query].to_dict('records'):
        output = outputs.get(r['benchmark_id'], {'status':'unavailable/unsupported engine query', 'candidates':[]})
        candidates = []
        for c in output['candidates']:
            try:
                c = {**c, 'formula':formula_text(parse_formula(c['formula']))}
                candidates.append(c)
            except ValueError:
                continue
        ranked = sorted([c for c in candidates if c.get('rank')],key=lambda c:c['rank'])
        truth = r['neutral_formula']
        formula_ranks = [int(c['rank']) for c in ranked if c['formula']==truth]
        charge = -2 if r['adduct'] == '[M-2H]2-' else (-1 if r['stream'].endswith('Neg') else 1)
        joint_ranks = [int(c['rank']) for c in ranked if c['formula']==truth and c.get('adduct')==r['adduct']
                       and c.get('charge', charge)==charge]
        rows.append({'query_id':r['benchmark_id'],'stream':r['stream'],'class':r['lipid_class'],
            'feature_id':r['feature_id'],'library_known':r['library_known'],'status':output['status'],
            'candidate_count':len(candidates),'pool_recall':any(c['formula']==truth for c in candidates),
            'scored_recall':bool(formula_ranks),'top1':bool(formula_ranks and min(formula_ranks)==1),
            'top5':bool(formula_ranks and min(formula_ranks)<=5),
            'joint_top1':bool(joint_ranks and min(joint_ranks)==1),
            'joint_top5':bool(joint_ranks and min(joint_ranks)<=5),
            'seconds':output.get('seconds',0)})
    frame = pd.DataFrame(rows)
    out.mkdir(exist_ok=False, parents=True)
    frame.to_csv(out/'query_scores.csv',index=False)
    metrics=[]
    for scope, subset in [('all',frame),('library_known',frame[frame.library_known]),
                          ('library_absent',frame[~frame.library_known]),
                          *[(c,frame[frame['class'].eq(c)]) for c in sorted(reference.lipid_class.unique())]]:
        for mode in MODES:
            q=subset[subset.stream.eq(mode)]
            metrics.append({'scope':scope,'mode':mode,'N':len(q),
                **{field:ratio(int(q[field].sum()),len(q)) for field in ['pool_recall','scored_recall','top1','top5','joint_top1','joint_top5']},
                'unavailable_or_failed':int(q.status.ne('complete').sum()),'no_candidates':int(q.candidate_count.eq(0).sum())})
    write_json(out/'metrics.json',metrics)
    return frame


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','worker','evaluate'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--method',type=Path)
    parser.add_argument('--preparation',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--engine',choices=['current','stage4','lipid_known','lipid_blind'])
    args=parser.parse_args()
    if args.source:
        sys.path.insert(0,str(args.source.resolve()/'app'))
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
    if args.action=='prepare':
        prepare(args.checkpoint.resolve(),args.method,args.output.resolve())
    elif args.action=='worker':
        worker(args.source.resolve(),args.checkpoint.resolve(),args.preparation.resolve(),args.output.resolve(),args.engine)
    else:
        evaluate(args.preparation.resolve(),args.predictions.resolve(),args.output.resolve())
