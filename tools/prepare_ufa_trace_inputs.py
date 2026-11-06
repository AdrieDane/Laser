"""Conditional, unranked shared-pool inputs for native IDSL.UFA; no gold labels."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def prepare(args):
    sys.path.insert(0,str(args.source/'app'))
    from standard_lipidomics.untargeted import UntargetedComparisonCriteria
    from standard_lipidomics.lipid_formulas import formula_candidates, formula_text
    from standard_lipidomics.isotope_patterns import ion_composition
    from standard_lipidomics.trace_evidence import digest, valid_native_bounds
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    context=json.loads(args.context.read_text())
    assert context['available']
    assert all(digest(Path(p))==sha for p,sha in context['files_sha256'].items())
    inputs=json.loads((args.preparation/'known_adduct_inputs.json').read_text())
    assert all(not set(q)&{'formula','name','lipid_class'} for q in inputs)
    method=json.loads((args.preparation/'method.json').read_text())
    criteria=UntargetedComparisonCriteria('conditional_A','conditional_B',
        formula_model='lipid_aware',isotope_model='complete')
    configs={}
    for mode,item in context['streams'].items():
        camera=json.loads((args.camera_configs/(mode+'_camera_config.json')).read_text())
        sample=camera['sample']
        frame=pd.read_csv(item['boundaries'],dtype={'feature_id':str,'sample':str})
        frame=frame[frame['sample'].eq(sample)]
        by_id={k:g for k,g in frame.groupby('feature_id',sort=False)}
        rows=[]
        for q in inputs:
            if q['stream']!=mode:
                continue
            candidates,audit=formula_candidates(q['mz'],mode,criteria,method,supplied_adduct=q['adduct'])
            pool=[{'formula':r['formula'],'ion_formula':formula_text(ion_composition(r['formula'],q['adduct'])[0])} for r in candidates]
            bounds=by_id.get(str(q['feature_id']),pd.DataFrame())
            status='ready' if len(bounds)==1 and not bool(bounds.iloc[0].filled) else 'native peak mapping missing, multiple or gap-filled'
            if status=='ready' and not valid_native_bounds(bounds).all():
                status='invalid native peak boundaries or apex'
            selected=bounds.iloc[0].to_dict() if status=='ready' else {}
            rows.append({**{k:q[k] for k in ('query_id','stream','feature_id','mz','rt','adduct','charge')},
                'status':status,'candidates':sorted(pool,key=lambda r:r['formula']),
                'boundaries':{k:float(selected[k]) for k in ('rt_start_corrected_min','rt_end_corrected_min')} if selected else {},
                'generated_N':len(pool),'candidate_budget_truncated':False})
        assert len(rows)==sum(q['stream']==mode for q in inputs)
        source_sample=next(s for s in item['samples'] if s['sample']==sample)
        configs[mode]={'r_library':str(args.r_library.resolve()),'stream':mode,'sample':source_sample,'queries':rows,
            'source_context_sha256':digest(args.context),'conditional_supplied_adduct':True,'reference_formulas_supplied':False,
            'shared_candidate_universe':True,'native_peaklist_adapter':'original XCMS boundaries plus actual observed raw apex height; not native IPA peak detection'}
        (output/(mode+'.json')).write_text(json.dumps(configs[mode],indent=2)+'\n')
        print(mode,len(rows),'queries',sum(len(q['candidates']) for q in rows),'unranked cases',flush=True)
    (output/'receipt.json').write_text(json.dumps({'queries':len(inputs),'formula_labels_supplied':False,
        'known_adduct_conditional':True,'input_hashes':{p.name:digest(p) for p in args.preparation.iterdir() if p.is_file()},
        'configs_sha256':{m:digest(output/(m+'.json')) for m in configs}},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','preparation','context','camera-configs','r-library','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    prepare(parser.parse_args())
