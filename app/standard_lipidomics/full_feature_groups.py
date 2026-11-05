"""Bounded neutral-mass ion families, built before significance is considered.

Not connected-component clustering: every union must satisfy neutral-mass span,
complete coelution limits and one original monoisotopic ion per mode/adduct.
Competing hypotheses and all unmatched original features remain in the exports.
"""
from collections import defaultdict
import hashlib
import json

import numpy as np
import pandas as pd

from standard_lipidomics.acquisition_method import (
    normalize_acquisition_method, configure_adduct_matches, record_method_on_analysis,
)
from standard_lipidomics.full_feature_evidence import (
    FeatureIndex, FULL_GROUP_VERSION, build_adduct_candidates, build_isotope_candidates,
)
from standard_lipidomics.untargeted_adducts import ADDUCTS


def source_signature(result):
    value = hashlib.sha256()
    for mode, item in sorted(result.get('streams', {}).items()):
        frame = item['features']
        columns = [c for c in ['feature_id', 'mz', 'rt', 'rtmin', 'rtmax', *frame.attrs.get('sample_columns', ())] if c in frame]
        value.update(mode.encode())
        value.update(pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes())
        metadata = {name:frame.attrs.get(name) for name in ('sample_columns', 'sample_group_map', 'sample_type_map', 'sample_group_name_map', 'sample_metadata_records')}
        value.update(json.dumps(metadata,sort_keys=True,default=str).encode())
    accepted = result.get('automatic_annotations')
    if isinstance(accepted, pd.DataFrame) and not accepted.empty:
        columns = [c for c in ('stream', 'standard_stream', 'feature_id', 'feat_feature_id', 'lipid_class', 'class') if c in accepted]
        value.update(pd.util.hash_pandas_object(accepted[columns],index=False).to_numpy().tobytes())
    return value.hexdigest()


def group_settings(criteria, method):
    from standard_lipidomics.acquisition_method import method_id
    names = ('ion_grouping_mode', 'cross_mode_ppm_tolerance', 'same_mode_adduct_rt_tolerance_min', 'opposite_polarity_rt_tolerance_min', 'isotope_rt_tolerance_min')
    return {**{name:getattr(criteria,name) for name in names}, 'acquisition_method_id':method_id(method)}


class BoundedFamilies:
    def __init__(self, indexes, criteria):
        self.indexes, self.criteria = indexes, criteria
        self.offsets, frames = {}, []
        offset = 0
        for mode, index in sorted(indexes.items()):
            self.offsets[mode] = offset
            frames.append(pd.DataFrame({'stream': mode, 'feature_id': index.ids,
                'mz': index.mz, 'rt': index.rt, 'peak_width_min': index.width,
                'coordinate_valid': index.valid, 'already_annotated': index.accepted}))
            offset += len(index.ids)
        self.ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.parent = np.arange(offset)
        self.members = {}
        self.adduct = np.full(offset, '', dtype=object)
        self.neutral = np.full(offset, np.nan)
        self.quality = np.zeros(offset)
        self.isotope = np.zeros(offset, dtype=int)
        self.mode = self.ledger.stream.to_numpy(str) if offset else np.array([], str)
        self.rt = self.ledger.rt.to_numpy(float) if offset else np.array([], float)
        self.width = self.ledger.peak_width_min.to_numpy(float) if offset else np.array([], float)
        self.priorities = {ion.name: i for i, ion in enumerate(ADDUCTS)}
        self.ions = {ion.name: ion for ion in ADDUCTS}

    def root(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return int(node)

    def _rt_compatible(self, nodes):
        for phase in ('RP', 'NP'):
            same = [node for node in nodes if self.mode[node].startswith(phase)]
            for number, left in enumerate(same):
                for right in same[number + 1:]:
                    cap = (self.criteria.same_mode_adduct_rt_tolerance_min if self.mode[left] == self.mode[right]
                           else self.criteria.opposite_polarity_rt_tolerance_min)
                    width = (self.width[left] + self.width[right]) / 2
                    allowed = min(cap, max(.005, width)) if np.isfinite(width) else cap
                    if abs(self.rt[left] - self.rt[right]) > allowed + 1e-10:
                        return False
        return True

    def join(self, a, b, source_adduct, target_adduct, source_mass, target_mass, quality):
        ra, rb = self.root(a), self.root(b)
        if ((self.adduct[a] and self.adduct[a] != source_adduct)
                or (self.adduct[b] and self.adduct[b] != target_adduct)):
            return 'competing_adduct_hypothesis'
        if ra == rb:
            return 'already_connected'
        nodes = self.members.get(ra, {a}) | self.members.get(rb, {b})
        labels = {(self.mode[node], self.adduct[node]) for node in nodes if self.adduct[node]}
        proposed = [(self.mode[a], source_adduct), (self.mode[b], target_adduct)]
        if len(labels | set(proposed)) < sum(bool(self.adduct[n]) for n in nodes) + int(not self.adduct[a]) + int(not self.adduct[b]):
            return 'duplicate_mode_adduct_isomer'
        masses = [self.neutral[n] for n in nodes if np.isfinite(self.neutral[n])] + [source_mass, target_mass]
        if min(masses) <= 0 or max(masses)-min(masses) > max(masses)*self.criteria.cross_mode_ppm_tolerance*1e-6:
            return 'neutral_mass_span'
        if not self._rt_compatible(nodes):
            return 'complete_RT_span'
        root = min(ra, rb)
        for node in nodes:
            self.parent[node] = root
        self.members.pop(ra, None)
        self.members.pop(rb, None)
        self.members[root] = nodes
        self.adduct[a], self.adduct[b] = source_adduct, target_adduct
        self.neutral[a], self.neutral[b] = source_mass, target_mass
        self.quality[a], self.quality[b] = max(self.quality[a], quality), max(self.quality[b], quality)
        return 'selected'

    def assemble(self, candidates, mode, progress=None):
        candidates = candidates.copy()
        candidates['selection_status'] = 'profile_not_supported'
        if candidates.empty:
            return candidates
        eligible = candidates.hard_supported if mode == 'full_feature_hard' else candidates.soft_supported
        candidates['profile_status'] = np.where(~candidates.profile_evaluable, 'not evaluable',
                                                np.where(eligible, 'supported', 'discordant'))
        chosen = candidates[eligible].copy()
        # Fragment-only discoveries need same-mode intact-parent evidence first.
        chosen['rank_score'] = chosen.evidence_score - .05 * (chosen.source_chemistry_penalty + chosen.target_chemistry_penalty)
        chosen['rank_score'] -= .01 * (chosen.source_adduct.map(self.priorities) + chosen.target_adduct.map(self.priorities))
        chosen['absolute_ppm'] = chosen.mass_error_ppm.abs()
        chosen = chosen.sort_values(['stage', 'rank_score', 'absolute_ppm', 'source_stream', 'source_feature_id',
                                      'target_stream', 'target_feature_id', 'source_adduct', 'target_adduct'],
                                     ascending=[True, False, True, True, True, True, True, True, True], kind='stable')
        # Scalar writes to Arrow-backed strings copy the whole status column.
        # Accumulate by original row index, then materialize the same column once.
        statuses = dict.fromkeys(candidates.index, 'profile_not_supported')
        for number, row in enumerate(self._ordered_edges(chosen)):
            a = self.offsets[row.source_stream] + row.source_index
            b = self.offsets[row.target_stream] + row.target_index
            source, target = self.ions[row.source_adduct], self.ions[row.target_adduct]
            if source.fragment or target.fragment:
                intact_a = any(self.mode[n] == row.source_stream and self.adduct[n]
                               and not self.ions[self.adduct[n]].fragment for n in self.members.get(self.root(a), {a}))
                intact_b = any(self.mode[n] == row.target_stream and self.adduct[n]
                               and not self.ions[self.adduct[n]].fragment for n in self.members.get(self.root(b), {b}))
                if (source.fragment and not (intact_a or (row.source_stream == row.target_stream and not target.fragment))) or (
                        target.fragment and not (intact_b or (row.source_stream == row.target_stream and not source.fragment))):
                    statuses[row.Index] = 'fragment_without_same_mode_intact_parent'
                    continue
            status = self.join(a, b, row.source_adduct, row.target_adduct,
                               row.source_neutral_mass, row.target_neutral_mass, row.evidence_score)
            statuses[row.Index] = status
            if progress and number % 25000 == 0:
                progress(number/max(1, len(chosen)), f'Bounded ion-family assembly: {number:,}/{len(chosen):,} candidate links')
        candidates['selection_status'] = candidates.index.map(statuses)
        return candidates

    def _ordered_edges(self, chosen):
        """Prefer established coeluting families over isolated equal-mass isomers."""
        for stage in (0, 1, 2):
            frame = chosen[chosen.stage.eq(stage)].copy()
            if stage:
                support = []
                for row in frame.itertuples():
                    a = self.offsets[row.source_stream] + row.source_index
                    b = self.offsets[row.target_stream] + row.target_index
                    count = len(self.members.get(self.root(a), {a})) + len(self.members.get(self.root(b), {b})) - 2
                    support.append(.025 * min(3, count))
                frame['rank_score'] += support
            frame = frame.sort_values(['rank_score', 'absolute_ppm', 'source_stream', 'source_feature_id',
                                       'target_stream', 'target_feature_id', 'source_adduct', 'target_adduct'],
                                      ascending=[False, True, True, True, True, True, True, True], kind='stable')
            yield from frame.itertuples()

    def attach_isotopes(self, candidates):
        output = candidates.copy()
        output['selection_status'] = 'unverified_envelope_candidate'
        if output.empty:
            return output
        # Do not merge two pre-existing ion families using an unverified envelope.
        supported = output[(output.paired_sample_count >= 6) & (output.abundance_spearman >= .6)
                           & (output.within_group_correlation >= .2) & output.median_paired_ratio.between(.001, 1.8)]
        supported = supported.assign(error=supported.mass_error_ppm.abs()).sort_values(
            ['error', 'stream', 'source_feature_id', 'target_feature_id', 'isotope_index'], kind='stable')
        statuses = dict.fromkeys(output.index, 'unverified_envelope_candidate')
        for row in supported.itertuples():
            a = self.offsets[row.stream] + row.source_index
            b = self.offsets[row.stream] + row.target_index
            if not self.adduct[a] or self.isotope[a] or self.ions[self.adduct[a]].charge != row.charge_hypothesis:
                continue
            if self.adduct[b] or self.root(b) != b:
                statuses[row.Index] = 'already_in_supported_ion_family'
                continue
            root = self.root(a)
            nodes = self.members.get(root, {a})
            if any(self.mode[n] == row.stream and self.adduct[n] == self.adduct[a]
                   and self.isotope[n] == row.isotope_index for n in nodes):
                statuses[row.Index] = 'competing_isotope_peak'
                continue
            if not self._rt_compatible(nodes | {b}):
                continue
            self.parent[b] = root
            self.members[root] = nodes | {b}
            self.adduct[b], self.neutral[b], self.isotope[b] = self.adduct[a], self.neutral[a], row.isotope_index
            statuses[row.Index] = 'conditional_isotope_member_unverified_envelope'
        output['selection_status'] = output.index.map(statuses)
        return output

    def table(self):
        output = self.ledger.copy()
        roots = [self.root(i) for i in range(len(output))]
        names = {root: f'compound_{number:04d}' for number, root in enumerate(sorted(set(roots)), 1)}
        output['compound_id'] = [names[root] for root in roots]
        output['adduct'], output['neutral_mass'], output['isotope_index'] = self.adduct, self.neutral, self.isotope
        output['ion_evidence_score'] = self.quality
        output['association_status'] = np.where(output.adduct.ne(''), 'tentative full-feature ion family',
            np.where(output.coordinate_valid, 'unmatched or unresolved ion', 'invalid coordinates; original feature retained'))
        modes = output.groupby('compound_id').stream.agg(lambda x: '; '.join(sorted(set(x))))
        output['detected_modes'] = output.compound_id.map(modes)
        order = output.assign(rank=output.adduct.map(self.priorities).fillna(999)).sort_values(
            ['compound_id', 'stream', 'isotope_index', 'rank', 'ion_evidence_score', 'feature_id'],
            ascending=[True, True, True, True, False, True], kind='stable')
        representatives = order.drop_duplicates(['compound_id', 'stream']).index
        output['mode_representative'] = output.index.isin(representatives)
        return output


def build_full_feature_groups(result, criteria, method=None, progress=None):
    from standard_lipidomics.untargeted import _accepted_feature_ids
    method = normalize_acquisition_method(method or result.get('acquisition_method'))
    signature = source_signature(result)
    settings = group_settings(criteria, method)
    indexes = {name: FeatureIndex(name, item['features'], _accepted_feature_ids(result, name))
               for name, item in result.get('streams', {}).items() if not item['features'].empty}
    if not indexes:
        return {'ion_feature_ledger': pd.DataFrame(), 'ion_relationships': pd.DataFrame(),
                'ion_isotope_candidates': pd.DataFrame(), 'ion_group_version': FULL_GROUP_VERSION}
    cache = result.get('_ion_candidate_cache')
    if cache is not None:
        provenance = result.get('_ion_cache_provenance', {})
        physical = {k:v for k,v in settings.items() if k!='ion_grouping_mode'}
        previous = {k:v for k,v in provenance.get('ion_group_settings', {}).items() if k!='ion_grouping_mode'}
        if provenance.get('ion_group_source_signature') != signature or physical != previous:
            raise ValueError('Cached ion candidates came from different inputs or physical settings.')
    candidates = cache[0].copy() if cache is not None else build_adduct_candidates(indexes, method, criteria,
        (lambda fraction, message: progress(.65*fraction, message)) if progress else None)
    candidates = configure_adduct_matches(candidates, method, result)
    assembler = BoundedFamilies(indexes, criteria)
    candidates = assembler.assemble(candidates, criteria.ion_grouping_mode,
        (lambda fraction, message: progress(.7+.15*fraction, message)) if progress else None)
    if progress:
        progress(.9, 'Full-feature isotope-candidate search')
    isotopes = assembler.attach_isotopes(cache[1].copy() if cache is not None else build_isotope_candidates(indexes, criteria))
    ledger = assembler.table()
    alternatives = defaultdict(set)
    supported = candidates[candidates.hard_supported | candidates.soft_supported]
    for row in supported.itertuples(index=False):
        alternatives[(row.source_stream, row.source_feature_id)].add(row.source_adduct)
        alternatives[(row.target_stream, row.target_feature_id)].add(row.target_adduct)
    ledger['adduct_alternatives'] = ['; '.join(sorted(alternatives.get((r.stream, r.feature_id), ()))) for r in ledger.itertuples()]
    output = {'ion_group_source_signature': signature, 'ion_group_settings': settings, 'ion_feature_ledger': ledger, 'ion_relationships': candidates,
              'ion_isotope_candidates': isotopes, 'ion_group_version': FULL_GROUP_VERSION,
              'ion_grouping_mode': criteria.ion_grouping_mode, 'acquisition_method': method}
    return record_method_on_analysis(output, method)


def apply_full_feature_display(analysis, result):
    """Select significant families only after full grouping; keep the full ledger."""
    from standard_lipidomics.untargeted import UntargetedComparisonCriteria
    from standard_lipidomics.untargeted_compounds import COMPOUND_VERSION, LABEL_COLUMNS, _cluster_evidence, _assign_cluster_ids
    from standard_lipidomics.untargeted_formula_labels import add_formula_labels
    criteria = UntargetedComparisonCriteria(**analysis['criteria'])
    output = {**analysis}
    expected_settings = group_settings(criteria, normalize_acquisition_method(analysis.get('acquisition_method', result.get('acquisition_method'))))
    if (analysis.get('ion_group_version') != FULL_GROUP_VERSION or analysis.get('ion_group_settings') != expected_settings
            or analysis.get('ion_group_source_signature') != source_signature(result)):
        output.update(build_full_feature_groups(result, criteria, analysis.get('acquisition_method')))
    ledger = output['ion_feature_ledger']
    if ledger.empty:
        return {**output, 'compound_features': ledger, 'compound_version': COMPOUND_VERSION}
    raw = analysis.get('raw_p_features', pd.DataFrame())
    raw_keys = set(zip(raw.stream.astype(str), raw.feature_id.astype(str))) if not raw.empty else set()
    mask = [(r.stream, r.feature_id) in raw_keys for r in ledger.itertuples()]
    selected = set(ledger.loc[mask, 'compound_id'])
    # Resolve homologues through full compound IDs, including non-displayed families.
    cluster_links, evidence = _cluster_evidence(output, result, output['acquisition_method'], criteria.same_mode_adduct_rt_tolerance_min)
    touching = set(zip(cluster_links.stream.astype(str), cluster_links.source_feature_id.astype(str))) | set(zip(cluster_links.stream.astype(str), cluster_links.target_feature_id.astype(str))) if not cluster_links.empty else set()
    touched_ids = set(ledger.loc[[(r.stream, r.feature_id) in touching for r in ledger.itertuples()], 'compound_id'])
    cluster_scope = ledger[ledger.compound_id.isin(touched_ids | selected)].copy()
    clustered, members = _assign_cluster_ids(cluster_scope, cluster_links)
    cluster_of = clustered.set_index(['stream', 'feature_id']).cluster_id.to_dict()
    ledger = ledger.copy()
    ledger['cluster_id'] = [cluster_of.get((r.stream, r.feature_id), '') for r in ledger.itertuples()]
    compounds = ledger[ledger.compound_id.isin(selected)].copy()
    tested = analysis.get('features', pd.DataFrame()).copy()
    if not tested.empty:
        stats = tested[['stream', 'feature_id', 'p_value', 'log2_fold_change_b_vs_a']].copy()
        stats['feature_id'] = stats.feature_id.astype(str)
        compounds = compounds.merge(stats, on=['stream', 'feature_id'], how='left', validate='one_to_one')
    else:
        compounds['p_value'], compounds['log2_fold_change_b_vs_a'] = np.nan, np.nan
    compounds['raw_p_significant'] = [(r.stream, r.feature_id) in raw_keys for r in compounds.itertuples()]
    compounds = add_formula_labels(compounds, raw)
    # All original ions stay in ledger; biological tables still exclude accepted ions.
    output.update(evidence)
    output.update(ion_feature_ledger=ledger, compound_features=compounds, cluster_members=members, compound_version=COMPOUND_VERSION)
    for name in ('features', 'raw_p_features', 'significant_features'):
        frame = analysis.get(name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame = frame.drop(columns=[c for c in LABEL_COLUMNS if c in frame]).copy()
            frame['feature_id'] = frame.feature_id.astype(str)
            output[name] = frame.merge(compounds[['stream', 'feature_id', *LABEL_COLUMNS]], on=['stream', 'feature_id'], how='left', validate='one_to_one')
    output['by_stream'] = {str(mode): f.copy() for mode, f in output.get('features', pd.DataFrame()).groupby('stream')}
    # The viewer consumes only validated family membership; the complete search
    # remains in ion_relationships, never pushed wholesale into the browser.
    links = output['ion_relationships']
    if not links.empty:
        known = ledger.set_index(['stream', 'feature_id'])[['compound_id', 'adduct', 'isotope_index']].to_dict('index')
        allowed = []
        for row in links.itertuples(index=False):
            left = known[(row.source_stream, row.source_feature_id)]
            right = known[(row.target_stream, row.target_feature_id)]
            allowed.append(left['compound_id'] in selected and left['compound_id'] == right['compound_id']
                and left['adduct'] == row.source_adduct and right['adduct'] == row.target_adduct
                and left['isotope_index'] == right['isotope_index'] == 0
                and row.selection_status in {'selected', 'already_connected'})
        links = links[allowed].copy()
        links['neutral_mass'] = links.source_neutral_mass
        ions = {ion.name: ion for ion in ADDUCTS}
        links['expected_mz'] = [ions[r.target_adduct].mz(r.source_neutral_mass) for r in links.itertuples()]
        links['target_is_raw_p_hit'] = [(r.target_stream, r.target_feature_id) in raw_keys for r in links.itertuples()]
    output['adduct_matches'] = links
    return record_method_on_analysis(output, output['acquisition_method'])
