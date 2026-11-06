"""Stage 6 source-bound blind worker. Detailed study outputs stay outside Git."""
import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from freeze_untargeted_baseline import write_json


def worker(source, checkpoint, preparation, runtime_file, output, arm, node=None):
    sys.path.insert(0, str(source/'app'))
    from standard_lipidomics.project_store import load_untargeted_workflow
    from standard_lipidomics.untargeted import UntargetedComparisonCriteria, _coordinate_sources
    from standard_lipidomics.joint_inference import apply_joint_inference
    from standard_lipidomics.untargeted_adducts import search_adduct_evidence
    from standard_lipidomics.untargeted_compounds import assign_compound_ids
    inputs = json.loads((preparation/'blind_inputs.json').read_text())
    assert all(not set(q) & {'formula', 'adduct', 'charge', 'name', 'lipid_class'} for q in inputs)
    result, _ = load_untargeted_workflow(checkpoint/'project')
    # Physical coordinates and sample metadata only. Existing annotation names
    # cannot provide an accidental class/formula prior in either replay arm.
    result['automatic_annotations'] = pd.DataFrame()
    analysis = result['untargeted_analysis']
    # The backwards-compatible loader supplies empty placeholders for new tables.
    # Those are outputs of this replay, rather than original statistical evidence.
    original = {name: value.copy(deep=True) for name, value in analysis.items()
                if isinstance(value, pd.DataFrame) and not name.startswith(('joint_', 'msbuddy_'))}
    queries = pd.DataFrame(inputs)[['stream', 'feature_id', 'mz', 'rt']].copy()
    queries['feature_id'] = queries.feature_id.astype(str)
    queries['raw_p_significant'], queries['significant'] = True, True
    queries['p_value'], queries['log2_fold_change_b_vs_a'] = np.nan, np.nan
    criteria = replace(UntargetedComparisonCriteria(**analysis['criteria']), isotope_model='complete',
        formula_model='lipid_aware', joint_model='joint_ms1' if arm != 'sequential' else 'current',
        use_msbuddy=arm == 'joint_msbuddy', formula_scoring_limit=32, ion_grouping_mode='current')
    analysis = {**analysis, 'criteria': asdict(criteria), 'raw_p_features': queries, 'significant_features': queries.copy(),
                'acquisition_method': json.loads((preparation/'method.json').read_text())}
    result['acquisition_method'] = analysis['acquisition_method']
    if arm == 'joint_msbuddy':
        result['_msbuddy_runtime'] = json.loads(runtime_file.read_text())
    output.mkdir(exist_ok=False, parents=True)
    began = time.monotonic()
    if arm == 'sequential':
        by_stream = {s: f.copy() for s, f in queries.groupby('stream')}
        analysis['adduct_matches'] = search_adduct_evidence(_coordinate_sources(result), by_stream,
            ppm=criteria.cross_mode_ppm_tolerance, rt_tolerance=criteria.opposite_polarity_rt_tolerance_min,
            same_mode_rt_tolerance=criteria.same_mode_adduct_rt_tolerance_min, method=analysis['acquisition_method'],
            progress=lambda n, total, msg: print(msg, flush=True))
        analysis = assign_compound_ids(analysis, result=result)
        predictions = analysis['compound_features']
        predictions.to_csv(output/'assignments.csv', index=False)
    else:
        analysis = apply_joint_inference(result, analysis, criteria, lambda f, m: print(m, flush=True))
        for name, frame in analysis.items():
            if isinstance(frame, pd.DataFrame) and name.startswith(('joint_', 'msbuddy_')):
                frame.to_csv(output/(name+'.csv'), index=False)
        hypotheses = {k: f for k, f in analysis['joint_hypotheses'].groupby(['stream', 'feature_id'], sort=False)}
        with (output/'predictions.jsonl').open('w') as handle:
            for q in inputs:
                frame = hypotheses.get((q['stream'], str(q['feature_id'])), pd.DataFrame())
                rows = []
                if not frame.empty:
                    for r in frame.itertuples():
                        rows.append({'formula': r.formula, 'adduct': r.adduct, 'charge': int(r.charge),
                                     'rank': int(r.joint_rank) if pd.notna(r.joint_rank) else None})
                handle.write(json.dumps({'query_id': q['query_id'], 'status': 'complete', 'candidates': rows})+'\n')
        # Every original column must be exactly preserved by additive inference.
        for name, before in original.items():
            if name not in {'raw_p_features', 'significant_features'}:
                pd.testing.assert_frame_equal(before, analysis[name])
        ledger = analysis['joint_feature_ledger']
        ledger[ledger.joint_adduct.ne('')].to_csv(output/'assignments.csv', index=False)
        write_json(output/'joint_provenance.json', analysis['joint_provenance'])
        write_json(output/'msbuddy_provenance.json', analysis['msbuddy_provenance'])
    inference_seconds = time.monotonic()-began
    from presentation.untargeted_mode import build_mode_payload
    payload = build_mode_payload(result, analysis, refresh_legacy=False)
    write_json(output/'payload.json', payload)
    write_json(output/'feature_keys.json', [{'stream': q['stream'], 'feature_id': q['feature_id'], 'query_id': q['query_id']} for q in inputs])
    assets = source/'app/presentation/untargeted_mode'
    (output/'viewer.mjs').write_text((assets/'model.js').read_text()+'\n'+(assets/'viewer.js').read_text())
    (output/'representatives.mjs').write_text('''import {readFileSync,writeFileSync} from 'node:fs';
import {ModeViewerModel,featureKey,sampleFigures} from './viewer.mjs';
const model=new ModeViewerModel(JSON.parse(readFileSync(new URL('./payload.json',import.meta.url))));
const queries=JSON.parse(readFileSync(new URL('./feature_keys.json',import.meta.url)));
const elapsed=[];
const rows=queries.map(q=>{const start=performance.now();
 const representatives=sampleFigures({model,selected:featureKey(q.stream,q.feature_id),canvas:'#fff',ink:'#111',grid:'#ddd'})
 .filter(x=>x.mode!=='empty').map(x=>({mode:x.mode,...model.point(x.key),...model.rows.get(x.key)}));
 elapsed.push(performance.now()-start);return {...q,representatives};});
elapsed.sort((a,b)=>a-b);
writeFileSync(new URL('./representatives.json',import.meta.url),JSON.stringify(rows));
writeFileSync(new URL('./browser_model_timing.json',import.meta.url),JSON.stringify({N:elapsed.length,median_ms:elapsed[Math.floor(elapsed.length/2)],max_ms:elapsed.at(-1),scope:'sampleFigures model only; not end-to-end Plotly/browser rendering'}));
''')
    node = node or shutil.which('node')
    if not node:
        raise ValueError('Node is required for the actual browser selector audit')
    subprocess.run([str(node), str(output/'representatives.mjs')], check=True, cwd=output)
    write_json(output/'receipt.json', {'arm': arm, 'seconds': inference_seconds, 'criteria': asdict(criteria),
        'input_query_rows': len(inputs), 'unique_original_query_keys': len(queries.drop_duplicates(['stream', 'feature_id'])),
        'reference_formulas_adducts_charges_names_supplied': False,
        'same_query_sequential_note': 'Current selector replayed on the same query keys, without reference charge or class hints',
        'old_statistics_unchanged': arm != 'sequential', 'original_statistical_rows': len(original['features'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['sequential', 'joint_laser', 'joint_msbuddy'], required=True)
    parser.add_argument('--node', type=Path)
    args = parser.parse_args()
    worker(args.source.resolve(), args.checkpoint.resolve(), args.preparation.resolve(),
           args.runtime.resolve() if args.runtime else None, args.output.resolve(), args.arm, args.node)
