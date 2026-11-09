"""Optional original-feature EIC evidence, independent of curator identities.

Only traces from the same injection are shape-correlated. XCMS group RT spreads
are not treated as peak widths. Native per-injection boundaries and global RT
maps must be present; missing or ambiguous traces never supply positive evidence.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from standard_traces import _iter_ms1_scans
from standard_lipidomics.sample_groups import biological_group
from standard_lipidomics.isotope_quality import IsotopeIndex

VERSION = 2
SETTINGS = {'ppm': 4., 'padding_min': .05, 'min_points': 5, 'min_edge_points': 2,
    'relative_threshold': .1, 'snr_proxy_min': 5., 'max_gap_factor': 3.,
    'shape_r_min': .85, 'overlap_min': .5, 'min_samples': 2, 'agreement_fraction': .8,
    'discordant_r_max': .5, 'discordant_overlap_max': .25, 'ranking_bonus': .25}
TEXT = {'stream': str, 'feature_id': str, 'sample': str}


def valid_native_bounds(frame):
    """An original apex outside its own bounds cannot define a trusted window."""
    columns = ['target_mz','rt_start_corrected_min','rt_apex_corrected_min','rt_end_corrected_min']
    if not set(columns).issubset(frame.columns):
        return pd.Series(False,index=frame.index)
    values = frame[columns].apply(pd.to_numeric,errors='coerce')
    return (pd.Series(np.isfinite(values).all(axis=1),index=frame.index) & values.target_mz.gt(0)
        & values.rt_start_corrected_min.le(values.rt_apex_corrected_min)
        & values.rt_apex_corrected_min.le(values.rt_end_corrected_min))


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze_sources(project, manifest_path=None, streams=None):
    """Freeze a portable raw/RT bridge at launch; no native R needed in the job."""
    try:
        root = Path(project).expanduser().resolve()
        path = Path(manifest_path) if manifest_path else root/'metadata/trace_sources/manifest.json'
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get('schema_version') != 1 or manifest.get('peak_rt_basis') != 'corrected':
            raise ValueError('Unsupported trace bridge or unknown native peak RT basis.')
        bound_root = path.parent.resolve()
        files = {str(path.resolve()): digest(path)}
        selected_streams = set(streams) if streams is not None else set(manifest['streams'])
        frozen_streams = {}
        for mode, item in manifest['streams'].items():
            if mode not in selected_streams:
                continue
            boundaries = (bound_root/item['boundaries']).resolve()
            if not boundaries.is_relative_to(bound_root):
                raise ValueError('Trace boundaries must be inside the exported metadata bundle.')
            files[str(boundaries)] = digest(boundaries)
            table = (root/item['input_table']).resolve()
            if not table.is_relative_to(root) or digest(table, 'md5') != item['input_md5']:
                raise ValueError('Original XCMS input changed since the RT crosswalk was exported.')
            files[str(table)] = digest(table)
            samples, seen = [], set()
            for sample in item['samples']:
                name = str(sample['sample'])
                raw = (root/sample['raw_path']).resolve()
                if name in seen or not raw.is_relative_to(root):
                    raise ValueError('Ambiguous sample identity or raw file outside the project.')
                seen.add(name)
                files[str(raw)] = digest(raw)
                a, b = np.asarray(sample['raw_rt_min'], float), np.asarray(sample['corrected_rt_min'], float)
                if len(a) < 2 or len(a) != len(b) or not np.isfinite([a,b]).all() or (np.diff(a) <= 0).any() or (np.diff(b) <= 0).any():
                    raise ValueError('RT mapping must be finite, one-to-one and strictly increasing.')
                samples.append({**sample, 'raw_path': str(raw)})
            frozen_streams[mode] = {**item, 'boundaries': str(boundaries), 'samples': samples}
        if not frozen_streams:
            raise ValueError('No trace metadata matches the requested acquisition modes.')
        return {'available': True, 'version': VERSION, 'streams': frozen_streams, 'files_sha256': files,
                'manifest': str(path.resolve()), 'crosswalk': manifest.get('provenance', '')}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'available': False, 'version': VERSION, 'reason': str(error)}


def candidate_pairs(analysis):
    """Physical endpoints only; adduct labels do not affect shape measurements."""
    pairs = set()
    links = analysis.get('adduct_matches', pd.DataFrame())
    for row in links.itertuples():
        if row.source_stream == row.target_stream:
            a,b = sorted((str(row.source_feature_id), str(row.target_feature_id)))
            if a != b:
                pairs.add((str(row.source_stream), a, b, 'ion'))
    raw = analysis.get('raw_p_features', pd.DataFrame())
    for n in (1,2):
        column = f'isotope_m_plus_{n}_feature_id'
        if column not in raw:
            continue
        for row in raw.to_dict('records'):
            target = row.get(column)
            if pd.isna(target) or str(target) in {'', 'nan'}:
                continue
            target_id = str(int(target)) if isinstance(target,float) and target.is_integer() else str(target)
            a,b = sorted((str(row['feature_id']), target_id))
            if a != b:
                pairs.add((str(row['stream']), a, b, 'isotope'))
    return pd.DataFrame(sorted(pairs), columns=['stream','source_feature_id','target_feature_id','kind'])


def trace_quality(rt, y, centroid_counts, lower, upper, settings=SETTINGS):
    """Scan quality and noise proxies; no measured noise/LOD claim."""
    rt, y = np.asarray(rt,float), np.asarray(y,float)
    base = {'status': 'insufficient scan coverage', 'scan_count': len(y), 'signal_points': 0,
            'noise_proxy': None, 'snr_proxy': None, 'apex_corrected_min': None}
    if len(y) < settings['min_points'] or not np.isfinite(y).all():
        return base, y
    corrected = np.maximum(0., y-np.quantile(y,.2))
    inside = np.flatnonzero((rt >= lower) & (rt <= upper))
    if not len(inside) or corrected[inside].max() <= 0:
        return {**base,'status':'no peak signal inside native boundaries'}, corrected
    apex = int(inside[np.argmax(corrected[inside])])
    height = float(corrected[apex])
    noise = float(np.median(np.abs(np.diff(y)-np.median(np.diff(y)))) * 1.4826 / np.sqrt(2))
    signal = np.flatnonzero((corrected >= settings['relative_threshold']*height) & (rt >= lower) & (rt <= upper))
    base.update(noise_proxy=noise, snr_proxy=height/noise if noise else None,
                apex_corrected_min=float(rt[apex]), peak_height=height, signal_points=len(signal))
    if corrected.max() > height*(1+1e-10):
        return {**base,'status':'dominant neighbouring peak outside native boundaries'}, corrected
    if len(signal) < settings['min_points'] or (signal < apex).sum() < settings['min_edge_points'] or (signal > apex).sum() < settings['min_edge_points']:
        return {**base,'status':'insufficient peak points or apex edges'}, corrected
    delta = np.diff(rt)
    if not (delta > 0).all() or delta.max() > settings['max_gap_factor']*np.median(delta):
        return {**base,'status':'nonmonotone scans or acquisition gap'}, corrected
    if (np.asarray(centroid_counts)[signal] > 1).any():
        return {**base,'status':'multiple mass centroids in peak'}, corrected
    if noise and height/noise < settings['snr_proxy_min']:
        return {**base,'status':'low signal relative to noise proxy'}, corrected
    maxima,_ = find_peaks(corrected, height=.5*height, prominence=.3*height, distance=3)
    if len(maxima) > 1:
        return {**base,'status':'multiple peaks inside native boundaries'}, corrected
    return {**base,'status':'usable'}, corrected


def shape_pair(left, right, settings=SETTINGS):
    """Use common real scan indices, never lag shifts or densifying interpolation."""
    if left['quality']['status'] != 'usable' or right['quality']['status'] != 'usable':
        return {'status':'not evaluable; trace quality', 'shape_r':None, 'overlap':None, 'common_points':0}
    _, a,b = np.intersect1d(left['scan'], right['scan'], assume_unique=True, return_indices=True)
    x,y = left['corrected'][a], right['corrected'][b]
    if not len(x):
        return {'status':'not evaluable; no common scans','shape_r':None,'overlap':None,'common_points':0}
    if x.max() < left['quality']['peak_height']*(1-1e-10) or y.max() < right['quality']['peak_height']*(1-1e-10):
        return {'status':'not evaluable; apex outside common scans','shape_r':None,'overlap':None,'common_points':len(x)}
    chosen = (x >= settings['relative_threshold']*x.max()) | (y >= settings['relative_threshold']*y.max())
    x,y = x[chosen],y[chosen]
    result = {'status':'not evaluable; common peak coverage','shape_r':None,'overlap':None,'common_points':len(x)}
    if len(x) < settings['min_points'] or x.sum() <= 0 or y.sum() <= 0:
        return result
    for values in (x,y):
        apex = int(np.argmax(values))
        if apex < settings['min_edge_points'] or len(values)-apex-1 < settings['min_edge_points']:
            return result
    if np.std(x) <= 0 or np.std(y) <= 0:
        return result
    correlation = float(np.corrcoef(x,y)[0,1])
    overlap = float(np.minimum(x/x.sum(),y/y.sum()).sum())
    return {**result,'status':'measured','shape_r':correlation,'overlap':overlap}


def _sample_traces(sample, bounds, settings=SETTINGS):
    """Stream one raw file once for all requested original-feature windows."""
    ids = bounds.feature_id.astype(str).to_numpy()
    masses = bounds.target_mz.to_numpy(float)
    lower = bounds.rt_start_corrected_min.to_numpy(float)
    upper = bounds.rt_end_corrected_min.to_numpy(float)
    raw = np.asarray(sample['raw_rt_min'],float)
    aligned = np.asarray(sample['corrected_rt_min'],float)
    values = [[] for _ in ids]
    for scan,(time,mz,intensity) in enumerate(_iter_ms1_scans(Path(sample['raw_path']))):
        if time < raw[0] or time > raw[-1]:
            continue
        rt = float(np.interp(time,raw,aligned))
        active = np.flatnonzero((rt >= lower-settings['padding_min']) & (rt <= upper+settings['padding_min']))
        if not len(active):
            continue
        order = np.argsort(mz,kind='stable')
        mz,intensity = mz[order],np.maximum(0.,intensity[order])
        lo = np.searchsorted(mz,masses[active]*(1-settings['ppm']*1e-6))
        hi = np.searchsorted(mz,masses[active]*(1+settings['ppm']*1e-6),side='right')
        cumulative = np.r_[0.,np.cumsum(intensity,dtype=float)]
        # Directly sum windows to avoid cancellation of small peaks in a large TIC.
        area = cumulative[hi]-cumulative[lo]
        for i,a,b in zip(range(len(active)),lo,hi):
            if b>a:
                area[i] = np.sum(intensity[a:b])
        for index,z,n in zip(active,area,hi-lo):
            values[index].append((scan,time,rt,float(z),int(n)))
    traces = {}
    for index,rows in enumerate(values):
        array = np.asarray(rows,float).reshape((-1,5))
        quality,corrected = trace_quality(array[:,2],array[:,3],array[:,4],lower[index],upper[index],settings)
        if lower[index] < aligned[0] or upper[index] > aligned[-1]:
            quality['status'] = 'native boundaries outside recorded RT mapping'
        traces[ids[index]] = {'scan':array[:,0].astype(int),'raw_rt':array[:,1],'rt':array[:,2],
            'intensity':array[:,3],'centroids':array[:,4].astype(int),'corrected':corrected,'quality':quality}
    return traces


def collect_trace_evidence(result, analysis, context, progress=None):
    pairs = candidate_pairs(analysis)
    provenance = {'version':VERSION,'settings':dict(SETTINGS),'status':'unavailable',
        'same_injection_shapes_only':True,'pair_specific_alignment':False,'verified_isotope_gold':False}
    empty = {'trace_pair_evidence':pd.DataFrame(),'trace_pair_samples':pd.DataFrame(),
             'trace_sample_quality':pd.DataFrame(),'trace_points':pd.DataFrame(),'trace_provenance':provenance}
    if not context or not context.get('available'):
        provenance['reason'] = (context or {}).get('reason','Native per-injection trace metadata unavailable')
        return {**analysis,**empty}
    try:
        for path,sha in context['files_sha256'].items():
            if digest(path) != sha:
                provenance['reason'] = 'Frozen trace source changed: '+Path(path).name
                return {**analysis,**empty}
    except OSError as error:
        provenance['reason'] = str(error)
        return {**analysis,**empty}
    provenance.update(status='complete', files_sha256=context['files_sha256'], crosswalk=context['crosswalk'])
    sample_rows,quality_rows,points = [],[],[]
    completed,total = 0,sum(len(s['samples']) for s in context['streams'].values())
    for mode,item in context['streams'].items():
        requested = pairs[pairs.stream.eq(mode)]
        if requested.empty:
            continue
        keys = set(requested.source_feature_id)|set(requested.target_feature_id)
        frame = result['streams'].get(mode,{}).get('features',pd.DataFrame())
        groups,types = frame.attrs.get('sample_group_map',{}),frame.attrs.get('sample_type_map',{})
        selected = [s for s in item['samples'] if s['sample'] in groups and (
            biological_group(frame,groups[s['sample']]) or IsotopeIndex.is_qc(groups[s['sample']],types.get(s['sample'],'')))]
        representative = next((s['sample'] for s in selected if IsotopeIndex.is_qc(groups[s['sample']],types.get(s['sample'],''))),None)
        chunks = [chunk[chunk.feature_id.isin(keys)] for chunk in pd.read_csv(item['boundaries'],dtype=TEXT,chunksize=150000)]
        boundaries = pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
        for sample in selected:
            completed += 1
            if progress:
                progress(completed/max(1,total), f'Raw peak shapes: {mode}, file {completed}/{total}')
            subset = boundaries[boundaries['sample'].eq(sample['sample'])].copy()
            present = set(subset.feature_id)
            counts = subset.groupby('feature_id').size()
            single = subset[subset.feature_id.map(counts).eq(1) & ~subset.filled.astype(bool)]
            valid = valid_native_bounds(single)
            invalid_ids = set(single.loc[~valid,'feature_id'])
            single = single[valid]
            traces = _sample_traces(sample,single) if len(single) else {}
            for fid in keys:
                trace = traces.get(fid)
                why = 'gap-filled peak or multiple native peaks' if fid in present else 'native peak mapping absent'
                if fid in invalid_ids:
                    why = 'invalid native peak boundaries or apex'
                quality_rows.append({'stream':mode,'feature_id':fid,'sample':sample['sample'],
                    'raw_file':sample['raw_path'],**(trace['quality'] if trace else {'status':why})})
                if trace and sample['sample'] == representative:
                    points.extend({'stream':mode,'feature_id':fid,'sample':sample['sample'],
                        'scan_index':int(scan),'raw_rt_min':float(a),'corrected_rt_min':float(b),'intensity':float(z)}
                        for scan,a,b,z in zip(trace['scan'],trace['raw_rt'],trace['rt'],trace['intensity']))
            for row in requested.itertuples():
                a,b = traces.get(row.source_feature_id),traces.get(row.target_feature_id)
                measurement = shape_pair(a,b) if a and b else {'status':'not evaluable; native mapping', 'shape_r':None,'overlap':None,'common_points':0}
                sample_rows.append({'stream':mode,'source_feature_id':row.source_feature_id,
                    'target_feature_id':row.target_feature_id,'kind':row.kind,'sample':sample['sample'],**measurement})
    measurements = pd.DataFrame(sample_rows)
    conservative = analysis.get('criteria',{}).get('trace_model') == 'peak_shape_conservative'
    summaries_frame = summarize_measurements(pairs,measurements,conservative=conservative)
    summaries = summaries_frame.to_dict('records')
    provenance['policy'] = 'replicated joint discordance; veto only' if conservative else 'legacy bonus and veto'
    provenance.update(raw_files_processed=completed, pairs=len(pairs),
        supported_pairs=sum(r['trace_status']=='supported' for r in summaries),
        discordant_pairs=sum(r['trace_status']=='discordant' for r in summaries))
    if not any(r['usable_samples'] for r in summaries):
        provenance['status'] = 'no usable measurements'
    if progress:
        progress(1.,'Chromatographic trace evidence saved')
    return {**analysis,**empty,'trace_pair_evidence':pd.DataFrame(summaries),'trace_pair_samples':measurements,
        'trace_sample_quality':pd.DataFrame(quality_rows),'trace_points':pd.DataFrame(points)}


def annotate_trace_links(links, evidence):
    output = links.copy()
    lookup = {(r.stream,*sorted((str(r.source_feature_id),str(r.target_feature_id)))):r._asdict()
              for r in evidence.itertuples() if r.kind == 'ion'}
    values = [lookup.get((r.source_stream,*sorted((str(r.source_feature_id),str(r.target_feature_id)))),{})
              if r.source_stream == r.target_stream else {'trace_status':'different injections; no shape comparison'} for r in links.itertuples()]
    for field in ('trace_status','usable_samples','median_shape_r','median_overlap'):
        output[field] = [row.get(field,'not evaluable' if field=='trace_status' else None) for row in values]
    return output


def summarize_measurements(pairs, measurements, *, conservative=False):
    """Aggregate real same-injection observations; no reference identities.

    Conservative policy requires replicated low correlation AND low overlap.
    Strong overlap with dissimilar shapes remains ambiguous.
    """
    measured_groups = {k:g for k,g in measurements.groupby(['stream','source_feature_id','target_feature_id','kind'],sort=False)} if len(measurements) else {}
    summaries = []
    for row in pairs.itertuples():
        subset = measured_groups.get((row.stream,row.source_feature_id,row.target_feature_id,row.kind),pd.DataFrame())
        usable = subset[subset.status.eq('measured')] if len(subset) else subset
        n = len(usable)
        correlation = float(usable.shape_r.median()) if n else None
        overlap = float(usable.overlap.median()) if n else None
        fraction = float(((usable.shape_r >= SETTINGS['shape_r_min']) & (usable.overlap >= SETTINGS['overlap_min'])).mean()) if n else 0.
        discordance = float(((usable.shape_r < SETTINGS['discordant_r_max']) & (usable.overlap < SETTINGS['discordant_overlap_max'])).mean()) if n else 0.
        status = 'not evaluable'
        if n >= SETTINGS['min_samples']:
            status = 'supported' if fraction >= SETTINGS['agreement_fraction'] else (
                'discordant' if (discordance >= SETTINGS['agreement_fraction'] if conservative else
                    correlation < SETTINGS['discordant_r_max'] or overlap < SETTINGS['discordant_overlap_max']) else 'ambiguous')
        summaries.append({'stream':row.stream,'source_feature_id':row.source_feature_id,'target_feature_id':row.target_feature_id,
            'kind':row.kind,'trace_status':status,'usable_samples':n,'median_shape_r':correlation,'median_overlap':overlap,'support_fraction':fraction,**({'discordance_fraction':discordance} if conservative else {})})
    return pd.DataFrame(summaries)
