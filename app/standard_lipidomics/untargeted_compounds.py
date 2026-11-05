"""Stable display names for tentative, neutral-mass-consistent ion groups.

These labels never replace lipid annotations or change statistical decisions.
CH2-series edges are deliberately excluded from compound grouping.
"""
from collections import defaultdict

import pandas as pd

from standard_lipidomics.untargeted_adducts import ADDUCTS

COMPOUND_VERSION = 3
LABEL_COLUMNS = ('compound_id', 'adduct', 'adduct_alternatives', 'detected_modes', 'association_status', 'cluster_id')


def _key(stream, feature):
    return str(stream), str(feature)


def assign_compound_ids(analysis):
    raw = analysis.get('raw_p_features', analysis.get('significant_features', pd.DataFrame()))
    if raw.empty:
        return {**analysis, 'compound_version': COMPOUND_VERSION, 'compound_features': pd.DataFrame()}
    hits = {_key(row.stream, row.feature_id): row._asdict() for row in raw.itertuples(index=False)}
    tested = analysis.get('features', raw)
    statistics = {_key(row.stream, row.feature_id): (row.p_value, row.log2_fold_change_b_vs_a)
                  for row in tested[['stream', 'feature_id', 'p_value', 'log2_fold_change_b_vs_a']].itertuples(index=False)}
    priorities = {adduct.name: index for index, adduct in enumerate(ADDUCTS)}
    candidates, alternatives = [], {}
    coelution = float(analysis.get('criteria', {}).get('opposite_polarity_rt_tolerance_min', .1))
    same_mode = float(analysis.get('criteria', {}).get('same_mode_adduct_rt_tolerance_min', .05))
    links = analysis.get('adduct_matches', pd.DataFrame())
    if not links.empty:
        allowed = {ion.name for ion in ADDUCTS}
        links = links[links.source_adduct.isin(allowed) & links.target_adduct.isin(allowed)]
        grouped = defaultdict(list)
        for row in links.itertuples(index=False):
            grouped[(str(row.source_stream), str(row.source_feature_id), str(row.source_adduct))].append(row)
        for (stream, feature, adduct), group in sorted(grouped.items()):
            anchor = _key(stream, feature)
            if anchor not in hits:
                continue
            first = group[0]
            members = {anchor: {'stream': anchor[0], 'feature_id': anchor[1], 'mz': float(first.source_mz),
                'rt': float(first.source_rt), 'adduct': adduct, 'neutral_mass': float(first.neutral_mass), 'already_annotated': False, '_error': 0.}}
            alternatives.setdefault(anchor, set()).add(str(adduct))
            # One neutral mass and one adduct per measured feature within a group.
            for row in sorted(group, key=lambda row: abs(row.mass_error_ppm)):
                target = _key(row.target_stream, row.target_feature_id)
                alternatives.setdefault(target, set()).add(str(row.target_adduct))
                members.setdefault(target, {'stream': target[0], 'feature_id': target[1], 'mz': float(row.target_mz),
                    'rt': float(row.target_rt), 'adduct': str(row.target_adduct), 'neutral_mass': float(first.neutral_mass),
                    'already_annotated': bool(row.target_is_annotated), '_error': abs(float(row.mass_error_ppm))})
            members = _coeluting_families(members, anchor, coelution, same_mode)
            score = (-len({key[0] for key in members}), -len(members), priorities.get(str(adduct), 99), anchor)
            candidates.append((score, anchor, members))
    claimed, groups = set(), []
    for _, anchor, members in sorted(candidates, key=lambda item: item[0]):
        if anchor in claimed:
            continue
        available = {key: value for key, value in members.items() if key not in claimed}
        groups.append((anchor, available))
        claimed.update(available)
    for key, row in sorted(hits.items()):
        if key not in claimed:
            groups.append((key, {key: {'stream': key[0], 'feature_id': key[1], 'mz': row['mz'], 'rt': row['rt'],
                'adduct': '', 'neutral_mass': None, 'already_annotated': False}}))
    records = []
    for number, (_, members) in enumerate(sorted(groups, key=lambda item: item[0]), 1):
        name = f'compound_{number:04d}'
        modes = '; '.join(sorted({key[0] for key in members}))
        for key, member in sorted(members.items()):
            p_value, fold = statistics.get(key, (None, None))
            records.append({**{key: value for key, value in member.items() if key != '_error'}, 'compound_id': name, 'detected_modes': modes,
                'raw_p_significant': key in hits, 'p_value': p_value, 'log2_fold_change_b_vs_a': fold,
                'adduct_alternatives': '; '.join(sorted(alternatives.get(key, ()), key=lambda value: priorities.get(value, 99))),
                'association_status': 'tentative mass/RT association' if len(members) > 1 else 'adduct unresolved'})
    compounds = pd.DataFrame(records)
    compounds, cluster_members = _assign_cluster_ids(compounds, analysis.get('homologous_matches', pd.DataFrame()))
    labels = compounds[['stream', 'feature_id', *LABEL_COLUMNS]]
    output = {**analysis, 'compound_version': COMPOUND_VERSION, 'compound_features': compounds, 'cluster_members': cluster_members}
    for name in ('features', 'raw_p_features', 'significant_features'):
        frame = analysis.get(name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame = frame.drop(columns=[column for column in LABEL_COLUMNS if column in frame]).copy()
            frame['feature_id'] = frame.feature_id.astype(str)
            output[name] = frame.merge(labels, on=['stream', 'feature_id'], how='left', validate='one_to_one', sort=False)
    if 'by_stream' in output and not output.get('features', pd.DataFrame()).empty:
        output['by_stream'] = {str(name): frame.copy() for name, frame in output['features'].groupby('stream', sort=False)}
    return output


def _coeluting_families(members, anchor, tolerance, same_mode=.05):
    """Do not collapse equal-mass RT isomers when linking RP to NP."""
    phases = defaultdict(list)
    for key, value in members.items():
        phases[key[0].split('_')[0]].append((key, value))
    result = {}
    for phase, items in phases.items():
        ordered = sorted(items, key=lambda item: (item[1]['rt'], item[0]))
        windows = []
        for index, (_, first) in enumerate(ordered):
            local = [item for item in ordered[index:] if item[1]['rt'] - first['rt'] <= tolerance + 1e-12]
            if phase == anchor[0].split('_')[0] and anchor not in {key for key, _ in local}:
                continue
            # A cross-chromatography link must not merge distant adduct RT
            # families within a single mode. Select one narrow family per mode.
            narrow = []
            for mode in sorted({key[0] for key, _ in local}):
                candidates = [item for item in local if item[0][0] == mode]
                families = []
                for offset, (_, first_mode) in enumerate(candidates):
                    family = [item for item in candidates[offset:] if item[1]['rt']-first_mode['rt'] <= same_mode+1e-12]
                    if mode == anchor[0] and anchor not in {key for key, _ in family}:
                        continue
                    score = (-len(family), sum(value.get('_error', 0.) for _, value in family), first_mode['rt'])
                    families.append((score, family))
                if families:
                    narrow.extend(min(families, key=lambda item: item[0])[1])
            local = narrow
            rank = (-len({key[0] for key, _ in local}), -len({value['adduct'] for _, value in local}),
                    -len(local), sum(value.get('_error', 0.) for _, value in local), first['rt'])
            windows.append((rank, local))
        if windows:
            result.update(min(windows, key=lambda item: item[0])[1])
    return result


def _assign_cluster_ids(compounds, links):
    """Cluster distinct compounds through saved homologous-series edges."""
    compound_of = {_key(row.stream, row.feature_id): row.compound_id for row in compounds.itertuples(index=False)}
    keys_by_node = defaultdict(set)
    for key, compound in compound_of.items():
        keys_by_node[compound].add(key)
    adjacency = defaultdict(set)
    for row in links.itertuples(index=False):
        a, b = _key(row.stream, row.source_feature_id), _key(row.stream, row.target_feature_id)
        left, right = compound_of.get(a, repr(a)), compound_of.get(b, repr(b))
        if left == right:
            continue  # Different adducts of one compound do not make a lipid series.
        keys_by_node[left].add(a)
        keys_by_node[right].add(b)
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited, components = set(), []
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack, members = [root], set()
        while stack:
            current = stack.pop()
            if current not in members:
                members.add(current)
                stack.extend(adjacency[current] - members)
        visited.update(members)
        keys = {key for member in members for key in keys_by_node[member]}
        components.append((min(keys), keys, len(members)))
    records = []
    for number, (_, keys, count) in enumerate(sorted(components), 1):
        for stream, feature in sorted(keys):
            records.append({'stream': stream, 'feature_id': feature, 'cluster_id': f'cluster_{number:04d}',
                'cluster_size': len(keys), 'cluster_compound_count': count, 'compound_id': compound_of.get((stream, feature), '')})
    members = pd.DataFrame(records, columns=['stream', 'feature_id', 'cluster_id', 'cluster_size', 'cluster_compound_count', 'compound_id'])
    labeled = compounds.merge(members[['stream', 'feature_id', 'cluster_id']], on=['stream', 'feature_id'], how='left', validate='one_to_one')
    labeled['cluster_id'] = labeled.cluster_id.fillna('')
    return labeled, members
