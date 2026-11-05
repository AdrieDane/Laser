"""P-value-independent ion candidates from complete feature tables.

Only measured coordinates, acquisition settings and paired biological abundance
profiles enter the search. Raw chromatographic shapes are a separate evidence
route. Group-wide shifts are removed before within-group profile correlation.
"""
from itertools import combinations_with_replacement
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from standard_lipidomics.sample_groups import biological_group
from standard_lipidomics.untargeted_adducts import adduct_options
from standard_lipidomics.untargeted_profiles import _correlation

C13 = 1.00335483507
FULL_GROUP_VERSION = 1
CHUNK_SIZE = 10000


class FeatureIndex:
    def __init__(self, stream, frame, accepted=()):
        self.stream = stream
        self.frame = frame.copy()
        self.frame['feature_id'] = self.frame.feature_id.astype(str)
        if self.frame.feature_id.duplicated().any():
            raise ValueError(f'Duplicate original feature keys in {stream}.')
        self.frame = self.frame.sort_values('feature_id', kind='stable').reset_index(drop=True)
        self.frame.attrs.update(frame.attrs)
        self.ids = self.frame.feature_id.to_numpy(str)
        self.mz = pd.to_numeric(self.frame.mz, errors='coerce').to_numpy(float)
        self.rt = pd.to_numeric(self.frame.rt, errors='coerce').to_numpy(float)
        self.valid = np.isfinite(self.mz) & (self.mz > 0) & np.isfinite(self.rt)
        self.width = np.full(len(self.frame), np.nan)
        if {'rtmin', 'rtmax'} <= set(frame):
            lower = pd.to_numeric(self.frame.rtmin, errors='coerce').to_numpy(float)
            upper = pd.to_numeric(self.frame.rtmax, errors='coerce').to_numpy(float)
            good = np.isfinite(lower) & np.isfinite(upper) & (upper > lower) & (lower <= self.rt) & (self.rt <= upper)
            self.width[good] = (upper - lower)[good]
        self.accepted = np.isin(self.ids, list(map(str, accepted)))
        self.order = np.flatnonzero(self.valid)
        self.order = self.order[np.argsort(self.mz[self.order], kind='stable')]
        self.samples = [str(s) for s in frame.attrs.get('sample_columns', ()) if s in frame]
        groups = frame.attrs.get('sample_group_map', {})
        self.samples = [s for s in self.samples if s in groups and biological_group(frame, groups[s])]
        self.levels = self.frame[self.samples].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        self.groups = {s: str(groups[s]) for s in self.samples}


def paired_columns(left, right):
    from standard_lipidomics.untargeted import _SPECIMEN_ID_COLUMNS, _unique_sample_key_map
    methods = [f'Excel metadata: {c}' for c in _SPECIMEN_ID_COLUMNS]
    methods += ['exact XCMS sample name', 'canonical acquisition name']
    for method in methods:
        a = _unique_sample_key_map(left.frame, left.samples, method)
        b = _unique_sample_key_map(right.frame, right.samples, method)
        keys = sorted(k for k in a.keys() & b.keys() if left.groups[a[k]] == right.groups[b[k]])
        if len(keys) >= 6:
            return ([left.samples.index(a[k]) for k in keys], [right.samples.index(b[k]) for k in keys],
                    [left.groups[a[k]] for k in keys], method)
    return [], [], [], 'not evaluable; fewer than six uniquely paired biological specimens'


def rt_limits(left, right, a, b, cap):
    widths = (left.width[a] + right.width[b]) / 2
    return np.where(np.isfinite(widths), np.minimum(cap, np.maximum(.005, widths)), cap)


def profile_metrics(left, right, a, b, pairing):
    cols_a, cols_b, groups, method = pairing
    count = len(a)
    if not cols_a:
        return np.zeros(count, int), np.full(count, np.nan), np.full(count, np.nan), method
    x, y = left.levels[a][:, cols_a], right.levels[b][:, cols_b]
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = np.where(valid, np.log1p(np.maximum(x, 0)), np.nan)
    y = np.where(valid, np.log1p(np.maximum(y, 0)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        rho = _correlation(rankdata(x, axis=1, nan_policy='omit'), rankdata(y, axis=1, nan_policy='omit'))
        rx, ry = x.copy(), y.copy()
        for label in set(groups):
            mask = np.asarray(groups) == label
            rx[:, mask] -= np.nanmean(x[:, mask], axis=1, keepdims=True)
            ry[:, mask] -= np.nanmean(y[:, mask], axis=1, keepdims=True)
        within = _correlation(rx, ry)
    return valid.sum(axis=1), rho, within, method


def _mass_pairs(left, right, source_ion, target_ion, ppm, cap):
    """Yield compact arrays, pruning coelution before constructing table rows."""
    masses = right.mz[right.order]
    for start in range(0, len(left.order), CHUNK_SIZE):
        indices = left.order[start:start + CHUNK_SIZE]
        expected = target_ion.mz(source_ion.neutral(left.mz[indices]))
        lo = np.searchsorted(masses, expected * (1 - ppm * 1e-6))
        hi = np.searchsorted(masses, expected * (1 + ppm * 1e-6), side='right')
        lengths = hi - lo
        if not lengths.sum():
            continue
        ai = np.repeat(indices, lengths)
        offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
        bi = right.order[np.repeat(lo, lengths) + np.arange(lengths.sum()) - offsets]
        if left is right:
            keep = ai != bi
            ai, bi = ai[keep], bi[keep]
        if cap is not None:
            keep = np.abs(left.rt[ai] - right.rt[bi]) <= rt_limits(left, right, ai, bi, cap) + 1e-10
            ai, bi = ai[keep], bi[keep]
        if len(ai):
            yield ai, bi


def _edge_frame(left, right, a, b, source_ion, target_ion, ppm, cap, stage, pairing):
    expected = target_ion.mz(source_ion.neutral(left.mz[a]))
    error = (right.mz[b] - expected) / expected * 1e6
    delta = (right.rt[b] - left.rt[a]) * 60
    limits = rt_limits(left, right, a, b, cap) if cap is not None else np.full(len(a), np.nan)
    n, rho, within, paired_by = profile_metrics(left, right, a, b, pairing)
    evaluable = (n >= 6) & np.isfinite(rho) & np.isfinite(within)
    mass_score = np.maximum(0, 1 - np.abs(error) / ppm)
    rt_score = np.maximum(0, 1 - np.abs(delta) / (limits * 60)) if cap is not None else np.zeros(len(a))
    score = (.30 * np.clip(rho, 0, 1) + .40 * np.clip(within, 0, 1) + .15 * mass_score + .15 * rt_score)
    if cap is None:
        score /= .85  # An unavailable RP-to-NP RT comparison supplies no bonus.
    hard = evaluable & (rho >= .6) & (within >= .2)
    soft = evaluable & (rho >= .2) & (within >= .05) & (score >= .45)
    return pd.DataFrame({'source_stream': left.stream, 'source_feature_id': left.ids[a],
        'target_stream': right.stream, 'target_feature_id': right.ids[b],
        'source_index': a, 'target_index': b, 'source_adduct': source_ion.name,
        'target_adduct': target_ion.name, 'source_mz': left.mz[a], 'target_mz': right.mz[b],
        'source_rt': left.rt[a], 'target_rt': right.rt[b],
        'source_neutral_mass': source_ion.neutral(left.mz[a]),
        'target_neutral_mass': target_ion.neutral(right.mz[b]),
        'mass_error_ppm': error, 'rt_delta_seconds': delta, 'rt_compared': cap is not None,
        'rt_limit_seconds': limits * 60, 'paired_sample_count': n,
        'abundance_spearman': rho, 'within_group_correlation': within,
        'pairing_method': paired_by, 'profile_evaluable': evaluable,
        'hard_supported': hard, 'soft_supported': soft, 'evidence_score': score,
        'stage': stage, 'relationship': 'same mode' if stage == 0 else 'opposite polarity' if stage == 1 else 'other chromatography',
        'source_is_annotated': left.accepted[a], 'target_is_annotated': right.accepted[b]})


def build_adduct_candidates(indexes, method, criteria, progress=None):
    frames = []
    pairs = list(combinations_with_replacement(sorted(indexes), 2))
    pairs.sort(key=lambda p: (0 if p[0] == p[1] else 1 if p[0][:2] == p[1][:2] else 2, p))
    for number, (source, target) in enumerate(pairs):
        left, right = indexes[source], indexes[target]
        stage = 0 if source == target else 1 if source[:2] == target[:2] else 2
        cap = criteria.same_mode_adduct_rt_tolerance_min if stage == 0 else criteria.opposite_polarity_rt_tolerance_min if stage == 1 else None
        pairing = paired_columns(left, right)
        if progress:
            progress(number / max(1, len(pairs)), f'Full-feature ion search: {source} ↔ {target}')
        sources = adduct_options('neg' if source.endswith('Neg') else 'pos', method=method, stream=source)
        targets = adduct_options('neg' if target.endswith('Neg') else 'pos', method=method, stream=target)
        for si, source_ion in enumerate(sources):
            for ti, target_ion in enumerate(targets):
                if stage == 0 and si >= ti:
                    continue
                if source_ion.fragment and target_ion.fragment:
                    continue
                for a, b in _mass_pairs(left, right, source_ion, target_ion, criteria.cross_mode_ppm_tolerance, cap):
                    frames.append(_edge_frame(left, right, a, b, source_ion, target_ion,
                        criteria.cross_mode_ppm_tolerance, cap, stage, pairing))
    if not frames:
        return pd.DataFrame(columns=['source_stream', 'source_feature_id', 'target_stream', 'target_feature_id',
                                     'hard_supported', 'soft_supported', 'evidence_score', 'stage'])
    return pd.concat(frames, ignore_index=True)


def build_isotope_candidates(indexes, criteria):
    """Mass/profile candidates only; complete isotope-envelope validation is later."""
    from standard_lipidomics.untargeted_adducts import Adduct
    frames = []
    for stream, index in indexes.items():
        pairing = paired_columns(index, index)
        positive = np.where(index.levels > 0, index.levels, np.nan)
        for charge in (1, 2):
            for isotope in (1, 2):
                parent = Adduct('M', 'pos', 1, 0.)
                shifted = Adduct('isotope', 'pos', 1, isotope * C13 / charge)
                for a, b in _mass_pairs(index, index, parent, shifted, criteria.cross_mode_ppm_tolerance,
                                        criteria.isotope_rt_tolerance_min):
                    n, rho, within, method = profile_metrics(index, index, a, b, pairing)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        ratio = np.nanmedian(positive[b] / positive[a], axis=1)
                    frames.append(pd.DataFrame({'stream': stream, 'source_feature_id': index.ids[a],
                        'target_feature_id': index.ids[b], 'source_index': a, 'target_index': b,
                        'isotope_index': isotope, 'charge_hypothesis': charge,
                        'mass_error_ppm': (index.mz[b] - index.mz[a] - isotope*C13/charge) / index.mz[b] * 1e6,
                        'rt_delta_seconds': (index.rt[b] - index.rt[a])*60,
                        'paired_sample_count': n, 'abundance_spearman': rho,
                        'within_group_correlation': within, 'median_paired_ratio': ratio,
                        'pairing_method': method, 'envelope_verified': False}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['stream', 'source_feature_id', 'target_feature_id'])
