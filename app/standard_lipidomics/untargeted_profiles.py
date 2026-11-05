"""Sample-profile support for tentative ion links, without lipid identities."""
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _sample_pairs(left, right, criteria):
    # Reuse the specimen-key and biological-group rules used by the evidence
    # evaluator. Never pair columns by position or include QC/blank injections.
    from standard_lipidomics.untargeted import (
        _eligible_abundance_samples, _unique_sample_key_map, _SPECIMEN_ID_COLUMNS,
    )
    samples = [_eligible_abundance_samples(frame, criteria) for frame in (left, right)]
    metadata = [frame.attrs.get('sample_metadata_records', {}) for frame in (left, right)]
    methods = [f'Excel metadata: {column}' for column in _SPECIMEN_ID_COLUMNS
               if all(any(column in row for row in records.values()) for records in metadata)]
    methods += ['exact XCMS sample name', 'canonical acquisition name']
    groups = [frame.attrs.get('sample_group_map', {}) for frame in (left, right)]
    for method in methods:
        a, b = [_unique_sample_key_map(frame, names, method) for frame, names in zip((left, right), samples)]
        keys = sorted(key for key in a.keys() & b.keys() if groups[0].get(a[key]) == groups[1].get(b[key]))
        if len(keys) >= 6:
            return [a[key] for key in keys], [b[key] for key in keys], [groups[0][a[key]] for key in keys], method
    return [], [], [], 'not evaluable; fewer than six uniquely paired biological samples'


def _correlation(a, b):
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    denominator = np.sqrt(np.nansum(a*a, axis=1) * np.nansum(b*b, axis=1))
    return np.divide(np.nansum(a*b, axis=1), denominator,
                     out=np.full(len(a), np.nan), where=denominator > 0)


def link_profile_evidence(result, analysis, links):
    """Evaluate all candidate links in batches; keep mass-only evidence intact."""
    output = links.copy()
    output['paired_sample_count'] = 0
    output['abundance_spearman'] = np.nan
    output['within_group_correlation'] = np.nan
    output['profile_status'] = 'not evaluable'
    output['pairing_method'] = 'not evaluable'
    criteria = SimpleNamespace(**analysis.get('criteria', {}))
    if not all(hasattr(criteria, name) for name in ('group_a', 'group_b')):
        return output
    frames = {name: item.get('features', pd.DataFrame()) for name, item in result.get('streams', {}).items()}
    for (source, target), group in links.groupby(['source_stream', 'target_stream'], sort=False):
        left, right = frames.get(source, pd.DataFrame()), frames.get(target, pd.DataFrame())
        if left.empty or right.empty:
            continue
        a, b, groups, method = _sample_pairs(left, right, criteria)
        output.loc[group.index, 'pairing_method'] = method
        if not a:
            continue
        matrices = []
        for frame, columns, ids in ((left, a, group.source_feature_id), (right, b, group.target_feature_id)):
            values = frame[['feature_id', *columns]].copy()
            values['feature_id'] = values.feature_id.astype(str)
            values = values.drop_duplicates('feature_id').set_index('feature_id')
            matrices.append(values.reindex(ids.astype(str)).apply(pd.to_numeric, errors='coerce').to_numpy(float))
        x, y = matrices
        valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        x = np.where(valid, np.log1p(np.maximum(x, 0)), np.nan)
        y = np.where(valid, np.log1p(np.maximum(y, 0)), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rho = _correlation(rankdata(x, axis=1, nan_policy='omit'), rankdata(y, axis=1, nan_policy='omit'))
            residual_x, residual_y = x.copy(), y.copy()
            for label in set(groups):
                mask = np.asarray(groups) == label
                residual_x[:, mask] -= np.nanmean(x[:, mask], axis=1, keepdims=True)
                residual_y[:, mask] -= np.nanmean(y[:, mask], axis=1, keepdims=True)
            within = _correlation(residual_x, residual_y)
        count = valid.sum(axis=1)
        evaluable = (count >= 6) & np.isfinite(rho) & np.isfinite(within)
        supported = evaluable & (rho >= .6) & (within >= .2)
        output.loc[group.index, 'paired_sample_count'] = count
        output.loc[group.index, 'abundance_spearman'] = rho
        output.loc[group.index, 'within_group_correlation'] = within
        output.loc[group.index, 'profile_status'] = np.where(supported, 'supported', np.where(evaluable, 'discordant', 'not evaluable'))
    return output
