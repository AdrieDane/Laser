"""Source-bound Stage 7 blind replay. Gold is read by a separate scorer only."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def isotope_trace_queries(queries, tested):
    columns = [c for c in tested if c in ('isotope_m_plus_1_feature_id','isotope_m_plus_2_feature_id')]
    physical = tested[['stream','feature_id',*columns]].copy()
    physical['feature_id'] = physical.feature_id.astype(str)
    return queries.merge(physical,on=['stream','feature_id'],how='left',validate='one_to_one')


def render(source, output, result, analysis, keys, node):
    from presentation.untargeted_mode import build_mode_payload
    from freeze_untargeted_baseline import write_json
    write_json(output/'payload.json', build_mode_payload(result, analysis, refresh_legacy=False))
    write_json(output/'feature_keys.json', keys)
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
writeFileSync(new URL('./browser_model_timing.json',import.meta.url),JSON.stringify({N:elapsed.length,median_ms:elapsed[Math.floor(elapsed.length/2)],max_ms:elapsed.at(-1),scope:'sampleFigures model only; not full browser rendering'}));
''')
    subprocess.run([str(node), str(output/'representatives.mjs')], cwd=output, check=True)


def worker(args):
    source, output = args.source.resolve(), args.output.resolve()
    sys.path[:0] = [str(source/'app'), str(source/'tools')]
    from freeze_untargeted_baseline import write_json
    from standard_lipidomics.project_store import load_untargeted_workflow, save_untargeted_workflow
    from standard_lipidomics.untargeted import UntargetedComparisonCriteria, _coordinate_sources
    from standard_lipidomics.untargeted_adducts import search_adduct_evidence
    from standard_lipidomics.untargeted_compounds import assign_compound_ids, LABEL_COLUMNS
    from standard_lipidomics.trace_evidence import collect_trace_evidence, candidate_pairs, summarize_measurements, digest, SETTINGS, VERSION
    from presentation.untargeted_mode import build_mode_payload
    output.mkdir(parents=True, exist_ok=False)
    result, _ = load_untargeted_workflow(args.checkpoint/'project')
    result['automatic_annotations'] = pd.DataFrame()
    original = result['untargeted_analysis']
    if args.scope == 'natural':
        inputs = json.loads((args.preparation/'blind_inputs.json').read_text())
        assert all(not set(q) & {'formula','adduct','charge','name','lipid_class'} for q in inputs)
        queries = pd.DataFrame(inputs)[['stream','feature_id','mz','rt']].copy()
        queries['feature_id'] = queries.feature_id.astype(str)
        queries['raw_p_significant'], queries['significant'] = True, True
        queries['p_value'], queries['log2_fold_change_b_vs_a'] = np.nan, np.nan
        keys = [{k:q[k] for k in ('stream','feature_id','query_id')} for q in inputs]
    else:
        queries = original['raw_p_features'].copy(deep=True)
        keys = queries[['stream','feature_id']].to_dict('records')
    criteria = replace(UntargetedComparisonCriteria(**original['criteria']), ion_grouping_mode='current',
        joint_model='current', use_msbuddy=False, trace_model={'current':'current','trace':'peak_shape','conservative':'peak_shape_conservative'}[args.arm])
    significant = queries.copy() if args.scope == 'natural' else original['significant_features'].copy(deep=True)
    analysis = {**original, 'criteria':asdict(criteria), 'raw_p_features':queries,
        'significant_features':significant, 'acquisition_method':json.loads((args.preparation/'method.json').read_text())}
    result['acquisition_method'] = analysis['acquisition_method']
    began = time.monotonic()
    # Natural-query replay must reproduce the Stage 6 matched sequential control.
    if args.scope == 'natural':
        analysis['adduct_matches'] = search_adduct_evidence(_coordinate_sources(result),
            {s:f.copy() for s,f in queries.groupby('stream')}, ppm=criteria.cross_mode_ppm_tolerance,
            rt_tolerance=criteria.opposite_polarity_rt_tolerance_min,
            same_mode_rt_tolerance=criteria.same_mode_adduct_rt_tolerance_min,
            method=analysis['acquisition_method'], progress=lambda n,t,m:print(m,flush=True))
    if args.arm != 'current':
        # Physical isotope endpoints are diagnostic only. Do not supply a source
        # charge, formula or adduct to the natural-query ion grouping control.
        if args.scope == 'natural':
            analysis['raw_p_features'] = isotope_trace_queries(queries,original['features'])
        if args.trace_cache:
            cached=args.trace_cache.resolve()
            provenance=json.loads((cached/'trace_provenance.json').read_text())
            assert provenance['settings']==SETTINGS
            assert all(digest(Path(p))==sha for p,sha in provenance['files_sha256'].items())
            pairs=candidate_pairs(analysis)
            dtype={'source_feature_id':str,'target_feature_id':str,'feature_id':str,'sample':str,'stream':str}
            tables={name:pd.read_csv(cached/(name+'.csv'),dtype=dtype,float_precision='round_trip')
                for name in ['trace_pair_evidence','trace_pair_samples','trace_sample_quality','trace_points']}
            old_pairs=tables['trace_pair_evidence'][pairs.columns].sort_values(list(pairs.columns)).reset_index(drop=True)
            pd.testing.assert_frame_equal(pairs.sort_values(list(pairs.columns)).reset_index(drop=True),old_pairs)
            tables['trace_pair_evidence']=summarize_measurements(pairs,tables['trace_pair_samples'],conservative=args.arm=='conservative')
            analysis={**analysis,**tables,'trace_provenance':{**provenance,'version':VERSION,
                'policy':'replicated joint discordance; veto only' if args.arm=='conservative' else 'legacy bonus and veto',
                'raw_files_processed':0,'measurements_reused':True,
                'cache_sha256':{p.name:digest(p) for p in cached.glob('trace_*.csv')}}}
        else:
            analysis = collect_trace_evidence(result, analysis, json.loads(args.context.read_text()),
                progress=lambda f,m:print(f'{f:.3f} {m}',flush=True))
        analysis['raw_p_features'] = queries
    analysis = assign_compound_ids(analysis, result=result)
    for name in ('features','raw_p_features','significant_features'):
        before = (original[name] if name == 'features' else significant if name == 'significant_features' else queries).copy()
        # Display joins use lossless textual original keys; the old checkpoint
        # contains integer keys. Verify their values while retaining exact tests
        # for every numerical statistical/evidence column.
        before['feature_id'] = before.feature_id.astype(str)
        columns = [c for c in before if c not in LABEL_COLUMNS]
        pd.testing.assert_frame_equal(before[columns],analysis[name][columns],check_exact=True)
    for name in ('homologous_matches','lipid_grid_matches'):
        pd.testing.assert_frame_equal(original[name],analysis[name][original[name].columns],check_exact=True)
    for name,frame in analysis.items():
        if isinstance(frame,pd.DataFrame) and (name.startswith('trace_') or name in ('compound_features','adduct_matches')):
            frame.to_csv(output/(name+'.csv'),index=False)
    write_json(output/'trace_provenance.json',analysis.get('trace_provenance',{}))
    seconds = time.monotonic()-began
    analysis['compound_features'].to_csv(output/'assignments.csv',index=False)
    render(source,output,result,analysis,keys,args.node)
    if args.arm != 'current' and args.scope == 'routine':
        save_untargeted_workflow(output/'saved_project',result,analysis)
        reopened,_ = load_untargeted_workflow(output/'saved_project')
        for name in ('features','raw_p_features','significant_features','trace_pair_evidence','trace_pair_samples','trace_sample_quality','trace_points'):
            pd.testing.assert_frame_equal(analysis[name],reopened['untargeted_analysis'][name],check_exact=True)
        assert build_mode_payload(result,analysis) == build_mode_payload(reopened,reopened['untargeted_analysis'])
    write_json(output/'receipt.json',{'arm':args.arm,'scope':args.scope,'query_N':len(queries),
        'seconds':seconds,'criteria':asdict(criteria),'original_statistics_exact':True,
        'labels_withheld':True,'saved_roundtrip_exact':args.arm!='current' and args.scope=='routine'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','preparation','output','context','node'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=['current','trace','conservative'],required=True)
    parser.add_argument('--trace-cache',type=Path)
    parser.add_argument('--scope',choices=['natural','routine'],required=True)
    worker(parser.parse_args())
