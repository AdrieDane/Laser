"""Browser-local mode plots, sample bars, feature table and evidence panels."""
from base64 import b64encode
from dataclasses import asdict
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
from plotly.colors import qualitative
import streamlit as st

from presentation.standard_grid.persistent_plot import _runtime
from presentation.viewer_theme import current_theme
from standard_lipidomics.untargeted_adducts import ADDUCTS
from standard_lipidomics.untargeted_compounds import assign_compound_ids, COMPOUND_VERSION
from standard_lipidomics.acquisition_method import normalize_acquisition_method, ion_state, method_id

HERE = Path(__file__).parent


def _packed(values, dtype='<f8'):
    return b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode('ascii')


def _records(frame):
    clean = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    return clean.where(pd.notna(clean), None).to_dict('records')



def _lipid_labels(result):
    """Only accepted identities; alternative candidates are not annotations."""
    frame = result.get('automatic_annotations', pd.DataFrame())
    labels = {}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return labels
    stream_col = next((c for c in ('stream', 'standard_stream') if c in frame), None)
    fid_col = next((c for c in ('feature_id', 'feat_feature_id') if c in frame), None)
    name_col = next((c for c in ('feature_identity', 'name', 'species') if c in frame), None)
    if not all((stream_col, fid_col, name_col)):
        return labels
    for row in frame.to_dict('records'):
        if pd.isna(row[name_col]):
            continue
        key = (str(row[stream_col]), str(row[fid_col]))
        label = str(row[name_col])
        if pd.notna(row.get('adduct')):
            label += ' ' + str(row['adduct'])
        labels.setdefault(key, [])
        if label not in labels[key]:
            labels[key].append(label)
    return {key: escape('; '.join(values)) for key, values in labels.items()}


def _joint_links(analysis):
    frames = []
    for table, status in [('joint_selected_links', 'joint_link_status'), ('joint_associations', 'association_status')]:
        frame = analysis.get(table, pd.DataFrame())
        if not frame.empty and status in frame:
            frames.append(frame[frame[status].isin(['selected', 'already_connected'])])
    rows = []
    ions = {a.name: a for a in ADDUCTS}
    for frame in frames:
        for r in frame.to_dict('records'):
            neutral = ions[r['source_adduct']].neutral(float(r['source_mz']))
            expected = ions[r['target_adduct']].mz(neutral)
            rows.append({**r, 'neutral_mass': neutral, 'mass_error_ppm': (r['target_mz']-expected)/expected*1e6,
                         'rt_delta_seconds': (r['target_rt']-r['source_rt'])*60})
    return pd.DataFrame(rows)


def build_mode_payload(result, analysis, *, token='preview', theme='Minimal', refresh_legacy=True):
    """One compact load; all subsequent selection work stays in the browser."""
    from standard_lipidomics.untargeted_grid import search_lipid_grids, GRID_VERSION
    method = normalize_acquisition_method(analysis.get('acquisition_method', result.get('acquisition_method')))
    joint_active = analysis.get('criteria', {}).get('joint_model', 'current') != 'current' and 'joint_compound_features' in analysis
    if refresh_legacy and not joint_active and analysis.get('lipid_grid_version') != GRID_VERSION:
        from standard_lipidomics.untargeted import _coordinate_sources, UntargetedComparisonCriteria
        analysis['lipid_grid_matches'] = search_lipid_grids(analysis.get('by_stream', {}), _coordinate_sources(result), UntargetedComparisonCriteria(**analysis['criteria']), method=method)
        analysis['lipid_grid_version'] = GRID_VERSION
        analysis['compound_version'] = None
    if refresh_legacy and not joint_active and analysis.get('compound_version') != COMPOUND_VERSION:
        analysis.update(assign_compound_ids(analysis, result=result))
    compounds = analysis.get('compound_features', pd.DataFrame())
    if joint_active:
        compounds = analysis['joint_compound_features'].copy()
        compounds['feature_id'] = compounds.feature_id.astype(str)
        clusters = analysis.get('cluster_members', pd.DataFrame())
        if not clusters.empty:
            names = clusters[['stream', 'feature_id', 'cluster_id']].copy()
            names['feature_id'] = names.feature_id.astype(str)
            names = names.drop_duplicates(['stream', 'feature_id'])
            compounds = compounds.drop(columns=['cluster_id'], errors='ignore').merge(names, how='left', on=['stream', 'feature_id'], validate='one_to_one')
    raw = analysis.get('raw_p_features', analysis.get('significant_features', pd.DataFrame()))
    detail_columns = ['isotope_charge_hypothesis', 'chemical_composition_hypothesis']
    for number in (1, 2):
        detail_columns.extend(f'isotope_m_plus_{number}_{suffix}' for suffix in
            ('feature_id', 'mz', 'rt_delta_seconds', 'intensity_percent'))
    detail_columns = ['stream', 'feature_id', *[name for name in detail_columns if name in raw]]
    known = raw[detail_columns].set_index(['stream', 'feature_id']).to_dict('index') if not raw.empty else {}
    statistics = []
    for row in _records(compounds):
        detail = known.get((row['stream'], row['feature_id']), {})
        statistics.append({**detail, **row})
    statistic_frame = pd.DataFrame(statistics)
    if not statistic_frame.empty:
        statistics = [{key: value for key, value in row.items() if value is not None}
                      for row in _records(statistic_frame.sort_values(['compound_id', 'stream', 'mz'], kind='stable'))]
    selected_ids = {(str(row['stream']), str(row['feature_id'])) for row in statistics}
    streams = {}
    lipid_labels = _lipid_labels(result)
    group_names = set()
    for name, item in result.get('streams', {}).items():
        features = item.get('features', pd.DataFrame())
        if features.empty or not {'feature_id', 'mz', 'rt'}.issubset(features):
            continue
        from standard_lipidomics.grid_browser import standard_sample_columns
        samples = standard_sample_columns(features)
        mapping = features.attrs.get('sample_group_map', {})
        groups = [str(mapping.get(sample, 'Unassigned')) for sample in samples]
        names = features.attrs.get('sample_group_name_map', {})
        labels = {group: str(names.get(sample, group)) for sample, group in zip(samples, groups)}
        group_names.update(groups)
        coords = features[['feature_id', 'mz', 'rt']].copy()
        coords['feature_id'] = coords.feature_id.astype(str)
        coords['mz'] = pd.to_numeric(coords.mz, errors='coerce')
        coords['rt'] = pd.to_numeric(coords.rt, errors='coerce')
        coords = coords[np.isfinite(coords.mz) & np.isfinite(coords.rt)].sort_values('mz', kind='stable')
        sample_rows = features[features.feature_id.astype(str).map(lambda value: (str(name), value) in selected_ids)]
        matrix = sample_rows[samples].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        unannotated = analysis.get('by_stream', {}).get(name, pd.DataFrame())
        unannotated_ids = set(unannotated.feature_id.astype(str)) if 'feature_id' in unannotated else set()
        raw_ids = set(raw.loc[raw.stream.eq(name), 'feature_id'].astype(str)) if not raw.empty else set()
        indices = np.flatnonzero(coords.feature_id.isin(unannotated_ids - raw_ids).to_numpy())
        if len(indices) > 5000:
            indices = np.random.default_rng(20261105).choice(indices, 5000, replace=False)
        streams[str(name)] = {'ids': coords.feature_id.tolist(), 'mz': _packed(coords.mz.to_numpy(float)),
            'rt': _packed(coords.rt.to_numpy(float)), 'background': sorted(indices.tolist()),
            'samples': list(samples), 'groups': groups, 'groupLabels': labels,
            'sampleIds': sample_rows.feature_id.astype(str).tolist(), 'levels': _packed(matrix),
            'lipidIds': {fid: label for (stream, fid), label in lipid_labels.items() if stream == str(name)}}
    palette = {group: qualitative.Safe[index % len(qualitative.Safe)] for index, group in enumerate(sorted(group_names))}
    links = _joint_links(analysis) if joint_active else analysis.get('adduct_matches', pd.DataFrame())
    link_rows = []
    if not links.empty:
        allowed = {ion.name for ion in ADDUCTS}
        links = links[links.source_adduct.isin(allowed) & links.target_adduct.isin(allowed)]
        for row in links.itertuples(index=False):
            link_rows.append([str(row.source_stream), str(row.source_feature_id), str(row.target_stream), str(row.target_feature_id),
                str(row.source_adduct), str(row.target_adduct), float(row.mass_error_ppm),
                float(row.rt_delta_seconds), bool(row.rt_compared), float(row.neutral_mass)])
    homologous = analysis.get('homologous_matches', pd.DataFrame())
    grid_links = analysis.get('lipid_grid_matches', pd.DataFrame())
    if method['adduct_scoring'] == 'method_aware':
        if 'adduct_alias_ambiguous' in homologous:
            homologous = homologous[~homologous.adduct_alias_ambiguous.fillna(False)]
        if 'adduct_alias_ambiguous' in grid_links:
            grid_links = grid_links[~grid_links.adduct_alias_ambiguous.fillna(False)]
    cluster_rows = [[str(row.stream), str(row.source_feature_id), str(row.target_feature_id),
                     int(row.ch2_steps), float(row.mass_error_ppm), float(row.rt_delta_seconds),
                     str(row.target_lipid_classes) if pd.notna(row.target_lipid_classes) else '']
                    for row in homologous.itertuples(index=False)]
    # Give otherwise ungrouped local neighbours their own tentative display ID.
    # Never infer that two unrelated neighbours are the same compound.
    cluster_members = _records(analysis.get('cluster_members', pd.DataFrame()))
    if joint_active:
        ids = {(r.stream, str(r.feature_id)): r.compound_id for r in analysis['joint_feature_ledger'].itertuples()}
        for row in cluster_members:
            row['compound_id'] = ids.get((str(row['stream']), str(row['feature_id'])), '')
    next_compound = max((int(row['compound_id'].split('_')[-1]) for row in statistics if row.get('compound_id')), default=0)
    for row in sorted(cluster_members, key=lambda value: (str(value['stream']), str(value['feature_id']))):
        if not row.get('compound_id'):
            next_compound += 1
            row['compound_id'] = f'compound_{next_compound:04d}'
    joint_hypotheses, joint_associations, joint_support = {}, {}, {}
    if joint_active:
        frame = analysis.get('joint_hypotheses', pd.DataFrame())
        if not frame.empty:
            columns = [c for c in ('stream', 'feature_id', 'formula', 'adduct', 'joint_rank', 'joint_cost',
                       'base_cost', 'native_rank', 'local_context_support', 'cross_phase_context_support', 'status') if c in frame]
            ranked = frame[frame.joint_rank.notna()].sort_values('joint_rank', kind='stable').groupby(['stream', 'feature_id'], sort=False).head(8)
            for row in _records(ranked[columns]):
                joint_hypotheses.setdefault(str(row['stream'])+'\0'+str(row['feature_id']), []).append(row)
        for row in _records(analysis.get('joint_associations', pd.DataFrame())):
            for a, b in [('source', 'target'), ('target', 'source')]:
                k = str(row[a+'_stream'])+'\0'+str(row[a+'_feature_id'])
                joint_associations.setdefault(k, []).append({'stream': row[b+'_stream'], 'feature_id': row[b+'_feature_id'],
                    'adduct': row[b+'_adduct'], 'formula': row['source_formula'], 'status': row['association_status']})
        supporting = analysis.get('joint_support', pd.DataFrame())
        if not supporting.empty:
            columns = [c for c in ('source_stream', 'source_feature_id', 'source_formula', 'source_adduct', 'target_stream',
                       'target_feature_id', 'target_adduct', 'formula_mass_error_ppm', 'paired_sample_count',
                       'abundance_spearman', 'within_group_correlation', 'target_spectral_cost',
                       'context_selected', 'target_budget_reached') if c in supporting]
            compact = supporting.sort_values('context_cost', kind='stable').groupby(['source_stream', 'source_feature_id'], sort=False).head(24)
            for row in _records(compact[columns]):
                k = str(row['source_stream'])+'\0'+str(row['source_feature_id'])
                joint_support.setdefault(k, []).append(row)
    return {'token': token, 'theme': theme, 'streams': streams, 'statistics': statistics,
            'links': link_rows, 'clusters': cluster_rows, 'palette': palette,
            'familyClusters': [[str(row.stream), str(row.source_feature_id), str(row.target_feature_id)] for row in grid_links.itertuples(index=False)],
            'clusterMembers': cluster_members,
            'adducts': [asdict(adduct) for adduct in ADDUCTS], 'criteria': dict(analysis.get('criteria', {})),
            'acquisitionMethodId': method_id(method),
            'fullFeatureGrouping': joint_active or analysis.get('criteria', {}).get('ion_grouping_mode', 'current') != 'current',
            'jointHypotheses': joint_hypotheses, 'jointAssociations': joint_associations,
            'jointSupport': joint_support,
            'jointModel': joint_active, 'msbuddyStatus': (analysis.get('msbuddy_provenance') or {}).get('status', 'not requested'),
            'adductPolicies': {stream: [ion.name for ion in ADDUCTS if ion_state(method, stream, ion.name) != 'disabled'] for stream in streams}}


def render_mode_view(result, analysis):
    theme = current_theme()
    signature = (id(result), id(analysis), theme, COMPOUND_VERSION)
    cached = st.session_state.get('standard_untargeted_mode_payload')
    if cached is None or cached[0] != signature:
        payload = build_mode_payload(result, analysis, token='::'.join(map(str, signature)), theme=theme)
        cached = signature, payload
        st.session_state.standard_untargeted_mode_payload = cached
    st.components.v2.component('standard_plotly_runtime', js=_runtime())(key='untargeted-mode-runtime', height=0)
    component = st.components.v2.component('untargeted_mode_workspace',
        js=(HERE/'model.js').read_text() + '\n' + (HERE/'viewer.js').read_text(),
        css=(HERE/'viewer.css').read_text())
    event = component(key='untargeted-mode-workspace', data=cached[1], width='stretch',
        on_review_change=lambda: None, default={})
    selected = event.get('review')
    if selected:
        st.session_state.standard_untargeted_review_focus = selected
    return st.session_state.get('standard_untargeted_review_focus')
