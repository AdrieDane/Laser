"""Native msbuddy 1.0.0 worker for an isolated supported Python environment.

No LASER imports, reference formulas or adduct prediction. The explicit supplied
adduct is part of the conditional formula benchmark, never an inferred success.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def run(inputs, data, output, ppm=4.):
    import numpy as np
    from msbuddy import Msbuddy, MsbuddyConfig, MetaFeature, Spectrum
    from msbuddy.utils import form_arr_to_str
    assert importlib.metadata.version('msbuddy') == '1.0.0'
    output.mkdir(exist_ok=False, parents=True)
    database_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.joblib')}
    if len(database_hashes) != 3:
        raise ValueError('All three pinned database/model files must be prepared before execution.')
    bounds = {'C':(1,180),'H':(0,400),'N':(0,6),'O':(0,60),'P':(0,3),'S':(0,3),
              'F':(0,0),'Cl':(0,0),'Br':(0,0),'I':(0,0),'Na':(0,0),'K':(0,0)}
    config = MsbuddyConfig(ms_instr=None,ppm=True,ms1_tol=ppm,ms2_tol=ppm,halogen=False,
                          parallel=False,timeout_secs=30,batch_size=128,element_ranges=bounds,
                          isotope_bin_mztol=.02,max_isotope_cnt=6,data_dir=data)
    assert config.ms1_tol == ppm
    receipt = {'engine':'msbuddy','version':'1.0.0','database_hashes':database_hashes,
               'input_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),
               'element_ranges':bounds,'ms1_tolerance_ppm':ppm,'instrument_preset':None,
               'MSMS_used':False,'adducts':'explicitly supplied, not inferred',
               'native_filter':'precursor 1<mz<1500; unsupported rows retained in predictions',
               'timeout_seconds':30,'dependency_versions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    (output/'frozen_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    engine = Msbuddy(config)
    queries = json.loads(inputs.read_text(encoding="utf-8"))
    started = time.monotonic()
    with (output/'predictions.jsonl').open('w') as handle:
        for start in range(0,len(queries),128):
            batch=queries[start:start+128]
            engine.clear_data()
            supported=[]
            for q in batch:
                if not 1<q['mz']<1500:
                    handle.write(json.dumps({'query_id':q['query_id'],'status':'unsupported native precursor range', 'candidates':[]})+'\n')
                    continue
                spectrum = np.array(q['ms1'],dtype=float)
                ms1=Spectrum(spectrum[:,0],spectrum[:,1]) if len(spectrum) else None
                try:
                    engine.add_data(MetaFeature(q['query_id'],q['mz'],q['charge'],rt=q['rt']*60,
                                                adduct=q['adduct'],ms1=ms1,ms2=None))
                    supported.append(q)
                except ValueError as exc:
                    handle.write(json.dumps({'query_id':q['query_id'],'status':'unsupported input: '+str(exc),'candidates':[]})+'\n')
            if not supported:
                continue
            try:
                engine.annotate_formula()
                native = {str(m.identifier):m for m in engine.data}
                for q in supported:
                    feature=native.get(q['query_id'])
                    candidates=[]
                    if feature is not None:
                        for rank,c in enumerate(feature.candidate_formula_list or [],1):
                            candidates.append({'formula':form_arr_to_str(c.formula.array),'adduct':q['adduct'],
                                               'charge':q['charge'],'rank':rank,'isotope_scored':True})
                    handle.write(json.dumps({'query_id':q['query_id'],'status':'complete' if feature else 'removed by native engine',
                                              'candidates':candidates})+'\n')
            except Exception as exc:
                for q in supported:
                    handle.write(json.dumps({'query_id':q['query_id'],'status':'native batch failure: '+type(exc).__name__+': '+str(exc),'candidates':[]})+'\n')
            handle.flush()
            print('msbuddy',min(start+128,len(queries)),'/',len(queries),'elapsed',round(time.monotonic()-started,1),flush=True)
    (output/'completion.json').write_text(json.dumps({'queries':len(queries),'seconds':time.monotonic()-started},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ppm',type=float,default=4.)
    args=parser.parse_args()
    run(args.inputs.resolve(),args.data.resolve(),args.output.resolve(),args.ppm)
