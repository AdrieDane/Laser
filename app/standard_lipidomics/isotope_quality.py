"""Per-sample isotope evidence with explicit censoring and quality limitations."""
import math
import re

import numpy as np
import pandas as pd

from standard_lipidomics.full_feature_evidence import FeatureIndex
from standard_lipidomics.isotope_patterns import theoretical_pattern, resolving_power
from standard_lipidomics.sample_groups import biological_group
from standard_lipidomics.untargeted_adducts import ADDUCTS


class IsotopeIndex(FeatureIndex):
    def __init__(self, stream, frame):
        super().__init__(stream, frame)
        self.by_id = {fid: i for i, fid in enumerate(self.ids)}
        columns = [str(c) for c in frame.attrs.get('sample_columns', ()) if c in frame]
        groups = frame.attrs.get('sample_group_map', {})
        types = frame.attrs.get('sample_type_map', {})
        self.ratio_samples = [c for c in columns if c in groups and
                              (biological_group(frame, groups[c]) or self.is_qc(groups[c], types.get(c, '')))]
        self.qc_mask = np.array([self.is_qc(groups[c], types.get(c, '')) for c in self.ratio_samples])
        self.areas = self.frame[self.ratio_samples].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        # Feature-area quantile is a censoring proxy, NOT a measured noise/LOD.
        positive = np.where(self.areas > 0, self.areas, np.nan)
        self.proxy = np.nanquantile(positive, .05, axis=0) if self.ratio_samples else np.array([])
        self.saturation_known = any(c in self.frame for c in ('saturated', 'is_saturated'))
        self.saturated = np.zeros(len(frame), bool)
        for c in ('saturated', 'is_saturated'):
            if c in self.frame:
                self.saturated |= self.frame[c].fillna(False).map(lambda v: str(v).lower() in {'true', '1', 'yes'}).to_numpy(bool)

    @staticmethod
    def is_qc(group, sample_type):
        text = str(group)+' '+str(sample_type)
        return bool(re.search(r'(^|[^a-z])(qc|quality control|pool(?:ed)?)([^a-z]|$)',
                              text, re.I)) and not re.search(r'blank|blanc|standard|\bis\b', text, re.I)

    def matches(self, parent, target, ppm, rt_cap):
        masses = self.mz[self.order]
        lo = np.searchsorted(masses, target*(1-ppm*1e-6))
        hi = np.searchsorted(masses, target*(1+ppm*1e-6), side='right')
        candidates = self.order[lo:hi]
        width = (self.width[parent]+self.width[candidates])/2
        limit = np.where(np.isfinite(width), np.minimum(rt_cap, np.maximum(.005, width)), rt_cap)
        return candidates[np.abs(self.rt[candidates]-self.rt[parent]) <= limit+1e-12]

    def ratios(self, parent, candidate, expected):
        mono, iso = self.areas[parent], self.areas[candidate]
        valid = np.isfinite(mono) & np.isfinite(iso) & (mono > 0) & (iso > 0)
        primary = valid & ~self.qc_mask
        origin = 'biological samples'
        if primary.sum() < 3:
            primary = valid & self.qc_mask
            origin = 'QC samples'
        n = int(primary.sum())
        qc = valid & self.qc_mask
        qc_n = int(qc.sum())
        qc_ratio = float(np.median(iso[qc]/mono[qc])) if qc_n else np.nan
        if not n:
            return {'paired_samples': 0, 'ratio_source': origin, 'median_ratio': np.nan,
                    'log_ratio_error': np.nan, 'log_ratio_mad': np.nan, 'qc_pairs': qc_n,
                    'qc_median_ratio': qc_ratio, 'ratio_evaluable': False}
        logs = np.log(iso[primary]/mono[primary])
        median = float(np.median(logs))
        mad = float(np.median(np.abs(logs-median)))
        return {'paired_samples': n, 'ratio_source': origin, 'median_ratio': math.exp(median),
                'log_ratio_error': abs(median-math.log(expected)), 'log_ratio_mad': mad,
                'qc_pairs': qc_n, 'qc_median_ratio': qc_ratio,
                'ratio_evaluable': n >= 3 and mad <= .5 and not (self.saturated[parent] or self.saturated[candidate])}


def score_envelope(index, feature_id, formula, adduct, criteria, instrument=None):
    """Score a hypothesis, retaining unavailable/ambiguous evidence as such.

    Scores are heuristic compatibility costs, not probabilities or identities.
    The same parent/feature cannot supply two different predicted components.
    """
    parent = index.by_id[str(feature_id)]
    pattern = theoretical_pattern(formula, adduct, instrument)
    mono = next(p for p in pattern if p.nominal == 0)
    power, basis = resolving_power(instrument or {}, mono.mz)
    mass_error = (index.mz[parent]-mono.mz)/mono.mz*1e6
    result = {'stream': index.stream, 'feature_id': str(feature_id), 'formula': formula,
              'adduct': adduct, 'parent_mass_error_ppm': mass_error, 'resolution_basis': basis,
              'resolution_verified': power is not None, 'observed_components': 0,
              'evaluable_components': 0, 'expected_components': 0, 'quality_flags': '',
              'score': abs(mass_error)/criteria.cross_mode_ppm_tolerance,
              'status': 'mass-only; isotope evidence not evaluable',
              'saturation_status': 'supplied flag' if index.saturation_known else 'not assessable from integrated areas'}
    records, errors, mass_errors, used, flags = [], [], [], {parent}, set()
    if power is None:
        flags.add('unverified resolution; nominal aggregation may mix resolved fine peaks')
    if abs(mass_error) > criteria.cross_mode_ppm_tolerance:
        result.update(status='parent mass incompatible', quality_flags='parent mass outside tolerance')
        return result, records
    if index.saturated[parent]:
        flags.add('parent saturation flagged; intensity fit disabled')
    for number, peak in enumerate(sorted(pattern, key=lambda p: (-p.probability, p.mz))):
        expected = peak.probability/mono.probability
        if peak.nominal == 0 or expected < .001:
            continue  # Explicit 0.1%-of-parent reporting floor; full simulation retained.
        result['expected_components'] += 1
        target = index.mz[parent] + peak.mz-mono.mz
        candidates = index.matches(parent, target, criteria.cross_mode_ppm_tolerance, criteria.isotope_rt_tolerance_min)
        alternatives = []
        for candidate in candidates:
            if candidate in used:
                flags.add('overlapping components share a feature')
                continue
            ratios = index.ratios(parent, candidate, expected)
            ppm = (index.mz[candidate]-target)/target*1e6
            rt = (index.rt[candidate]-index.rt[parent])*60
            cost = abs(ppm)/criteria.cross_mode_ppm_tolerance + abs(rt)/(criteria.isotope_rt_tolerance_min*60)
            if ratios['ratio_evaluable']:
                cost += ratios['log_ratio_error']/.35
            alternatives.append((cost, str(index.ids[candidate]), candidate, ppm, rt, ratios))
        alternatives.sort(key=lambda item: (item[0], item[1]))
        if not alternatives:
            mono_areas = index.areas[parent]
            valid = np.isfinite(mono_areas) & (mono_areas > 0)
            observable = int((valid & (mono_areas*expected >= index.proxy)).sum())
            status = 'missing above area proxy' if observable >= 3 else 'missing; possibly below area proxy'
            records.append({'stream': index.stream, 'feature_id': str(feature_id), 'formula': formula, 'adduct': adduct,
                            'component': number, 'nominal': peak.nominal, 'expected_mz': target, 'expected_ratio': expected,
                            'candidate_feature_id': '', 'selected': False, 'status': status, 'observable_samples_proxy': observable})
            flags.add(status)
            continue
        result['observed_components'] += 1
        ambiguous = len(alternatives) > 1 and alternatives[1][0]-alternatives[0][0] < .25
        best = alternatives[0]
        used.add(best[2])
        mass_errors.append(abs(best[3])/criteria.cross_mode_ppm_tolerance)
        supported = best[5]['ratio_evaluable'] and best[5]['log_ratio_error'] <= np.log(2.) and not ambiguous
        if ambiguous:
            flags.add('competing feature candidates')
        if not best[5]['ratio_evaluable']:
            flags.add('too few paired samples, variable ratio or saturation')
        elif best[5]['log_ratio_error'] > np.log(2.):
            flags.add('ratio discordance; possible interference or wrong hypothesis')
        if best[5]['ratio_evaluable'] and not ambiguous:
            errors.append(best[5]['log_ratio_error'])
        for rank, (_, fid, _, ppm, rt, ratios) in enumerate(alternatives):
            records.append({'stream': index.stream, 'feature_id': str(feature_id), 'formula': formula, 'adduct': adduct,
                            'component': number, 'nominal': peak.nominal, 'expected_mz': target, 'expected_ratio': expected,
                            'candidate_feature_id': fid, 'mass_error_ppm': ppm, 'rt_delta_seconds': rt,
                            'selected': rank == 0, 'status': 'supported' if rank == 0 and supported else 'ambiguous' if ambiguous else 'tentative', **ratios})
    result['evaluable_components'] = len(errors)
    if mass_errors:
        result['score'] += float(np.mean(mass_errors))
    if errors:
        result['score'] += float(np.mean(np.minimum(errors, 3.)))/.35
        result['status'] = 'compatible isotope evidence' if max(errors) <= np.log(2.) else 'isotope ratio discordance'
    if any('missing above' in f for f in flags):
        result['score'] += .25  # Missing evidence never deletes a formula candidate.
        if result['status'] == 'compatible isotope evidence':
            result['status'] = 'partial compatible isotope evidence'
    ion = next(a for a in ADDUCTS if a.name == adduct)
    lower = index.matches(parent, index.mz[parent]-1.00335483507/ion.charge,
                          criteria.cross_mode_ppm_tolerance, criteria.isotope_rt_tolerance_min)
    plausible = [str(index.ids[i]) for i in lower if index.ratios(i, parent, 1.)['ratio_evaluable']]
    result['possible_monoisotopic_parents'] = '; '.join(plausible)
    if plausible:
        flags.add('possible heavier-isotope selection; lower coeluting ion present')
    if not any(p['selected'] for p in records):
        flags.add('monoisotopic selection unverified')
    result['quality_flags'] = '; '.join(sorted(flags))
    return result, records
