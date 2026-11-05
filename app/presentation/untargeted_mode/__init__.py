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


def build_mode_payload(result, analysis, *, token='preview', theme='Minimal'):
    """One compact load; all subsequent selection work stays in the browser."""
    if analysis.get('compound_version') != COMPOUND_VERSION:
        analysis.update(assign_compound_ids(analysis))
    compounds = analysis.get('compound_features', pd.DataFrame())
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
    links = analysis.get('adduct_matches', pd.DataFrame())
    link_rows = []
    if not links.empty:
        for row in links.itertuples(index=False):
            link_rows.append([str(row.source_stream), str(row.source_feature_id), str(row.target_stream), str(row.target_feature_id),
                str(row.source_adduct), str(row.target_adduct), float(row.mass_error_ppm),
                float(row.rt_delta_seconds), bool(row.rt_compared), float(row.neutral_mass)])
    homologous = analysis.get('homologous_matches', pd.DataFrame())
    cluster_rows = [[str(row.stream), str(row.source_feature_id), str(row.target_feature_id),
                     int(row.ch2_steps), float(row.mass_error_ppm), float(row.rt_delta_seconds),
                     str(row.target_lipid_classes) if pd.notna(row.target_lipid_classes) else '']
                    for row in homologous.itertuples(index=False)]
    # Give otherwise ungrouped local neighbours their own tentative display ID.
    # Never infer that two unrelated neighbours are the same compound.
    cluster_members = _records(analysis.get('cluster_members', pd.DataFrame()))
    next_compound = max((int(row['compound_id'].split('_')[-1]) for row in statistics if row.get('compound_id')), default=0)
    for row in sorted(cluster_members, key=lambda value: (str(value['stream']), str(value['feature_id']))):
        if not row.get('compound_id'):
            next_compound += 1
            row['compound_id'] = f'compound_{next_compound:04d}'
    return {'token': token, 'theme': theme, 'streams': streams, 'statistics': statistics,
            'links': link_rows, 'clusters': cluster_rows, 'palette': palette,
            'clusterMembers': cluster_members,
            'adducts': [asdict(adduct) for adduct in ADDUCTS], 'criteria': dict(analysis.get('criteria', {}))}


def render_mode_view(result, analysis):
    theme = current_theme()
    signature = (id(result), id(analysis), theme)
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
