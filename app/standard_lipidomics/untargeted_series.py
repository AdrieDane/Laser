"""Local CH2 homologous-series evidence around raw-p-value hits."""
import numpy as np
import pandas as pd

CH2 = 14.015650064
LINK_COLUMNS = ('stream', 'source_feature_id', 'target_feature_id', 'ch2_steps',
                'composition_difference', 'charge_hypothesis', 'mass_error_da',
                'mass_error_ppm', 'rt_delta_seconds', 'target_is_annotated', 'target_lipid_classes')


def attach_lipid_series(by_stream, sources, annotations, criteria):
    """Link local ±1..4 CH2 steps, retaining annotated and non-hit neighbours.

    This assumes the same ion/adduct along a link. Without isotope evidence the
    charge is assumed to be one. A mass-spaced pair alone is reported as a pair;
    three connected features form a candidate series, never a lipid identity.
    """
    output, all_links = {}, []
    for stream, original in by_stream.items():
        table = original.copy()
        for column, value in {'lipid_series_id': '', 'lipid_series_size': 1,
            'lipid_series_neighbor_count': 0, 'lipid_series_class_context': '',
            'lipid_series_evidence': 'no local CH2-spaced neighbours'}.items():
            table[column] = value
        source = sources.get(stream, pd.DataFrame())
        if source.empty:
            output[stream] = table
            continue
        classes = annotations.get(stream, pd.DataFrame())
        class_map = (classes.groupby('feature_id').lipid_class.agg(lambda values: ', '.join(sorted(set(values.astype(str)))))
                     .to_dict() if not classes.empty else {})
        masses = source.mz.to_numpy(float)
        retention = source.rt.to_numpy(float)
        ids = source.feature_id.astype(str).to_numpy()
        annotated = source.already_annotated.to_numpy(bool)
        adjacency, links = {}, []
        hits = table[table.raw_p_significant.fillna(False)]
        known_charges = table.set_index('feature_id').get('isotope_charge_hypothesis', pd.Series(dtype=float)).to_dict()
        for _, hit in hits.iterrows():
            observed_charge = hit.get('isotope_charge_hypothesis', np.nan)
            charge = int(observed_charge) if pd.notna(observed_charge) else 1
            for step in (*range(-4, 0), *range(1, 5)):
                expected = hit.mz + step * CH2 / charge
                delta = min(criteria.ch2_mass_tolerance_da, expected * criteria.cross_mode_ppm_tolerance * 1e-6)
                left = np.searchsorted(masses, expected - delta, side='left')
                right = np.searchsorted(masses, expected + delta, side='right')
                for target_index in range(left, right):
                    target_id = ids[target_index]
                    target_mz = masses[target_index]
                    target_rt = retention[target_index]
                    rt_delta = (target_rt - hit.rt) * 60.
                    other_charge = known_charges.get(target_id, np.nan)
                    if abs(rt_delta) > criteria.ch2_rt_tolerance_min * 60. or (pd.notna(other_charge) and int(other_charge) != charge):
                        continue
                    a, b = str(hit.feature_id), target_id
                    if a == b:
                        continue
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
                    links.append(dict(zip(LINK_COLUMNS, (stream, a, b, step,
                        f'{"+" if step > 0 else "−"}C{abs(step)}H{2*abs(step)}', charge,
                        target_mz - expected, (target_mz - expected) / expected * 1e6,
                        rt_delta, bool(annotated[target_index]), class_map.get(b, '')))))
        components, visited = {}, set()
        number = 0
        for start in sorted(adjacency):
            if start in visited:
                continue
            number += 1
            stack, members = [start], set()
            while stack:
                current = stack.pop()
                if current not in members:
                    members.add(current)
                    stack.extend(adjacency.get(current, set()) - members)
            visited.update(members)
            for member in members:
                components[member] = (f'{stream}-CH2-{number:03d}', len(members))
        frame = pd.DataFrame(links, columns=LINK_COLUMNS)
        grouped = {key: group for key, group in frame.groupby('source_feature_id')}
        for index, hit in hits.iterrows():
            related = grouped.get(str(hit.feature_id))
            if related is None:
                continue
            series, size = components[str(hit.feature_id)]
            table.at[index, 'lipid_series_id'] = series
            table.at[index, 'lipid_series_size'] = size
            table.at[index, 'lipid_series_neighbor_count'] = related.target_feature_id.nunique()
            context = sorted(set(value for value in related.target_lipid_classes if value))
            table.at[index, 'lipid_series_class_context'] = '; '.join(context)
            label = 'candidate CH2 homologous series' if size >= 3 else 'CH2-spaced pair'
            table.at[index, 'lipid_series_evidence'] = f'{label}: {size} connected features; same-adduct hypothesis' + (
                '; annotated neighbours: ' + ', '.join(context) if context else '')
        all_links.extend(links)
        output[stream] = table
    return output, pd.DataFrame(all_links, columns=LINK_COLUMNS)
