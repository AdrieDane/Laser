"""Neutral-mass hypotheses for untargeted evidence, never lipid assignments.

Common ion families follow https://www.lipidmaps.org/tools/ms/gp_ox_form.php.
Additive-dependent ions are hypotheses;
their presence is searched rather than assumed. Masses include electron charge.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Adduct:
    name: str
    polarity: str
    charge: int
    offset: float
    carbon: int = 0
    fragment: bool = False

    def neutral(self, mz):
        return mz * self.charge - self.offset

    def mz(self, neutral):
        return (neutral + self.offset) / self.charge


PROTON = 1.007276466621
ADDUCTS = (
    Adduct('[M+H]+', 'pos', 1, PROTON),
    Adduct('[M+NH4]+', 'pos', 1, 18.033825553),
    Adduct('[M+Na]+', 'pos', 1, 22.989220702),
    Adduct('[M+K]+', 'pos', 1, 38.963157907),
    Adduct('[M+H-H2O]+', 'pos', 1, PROTON - 18.010564684, fragment=True),
    Adduct('[M+2H]2+', 'pos', 2, 2 * PROTON),
    Adduct('[M-H]-', 'neg', 1, -PROTON),
    Adduct('[M+HCOO]-', 'neg', 1, 44.998202851, carbon=1),
    Adduct('[M+CH3COO]-', 'neg', 1, 59.013852915, carbon=2),
    Adduct('[M+Cl]-', 'neg', 1, 34.969401262),
    Adduct('[M+Na-2H]-', 'neg', 1, 22.989220702 - 2 * PROTON),
    Adduct('[M-H-H2O]-', 'neg', 1, -PROTON - 18.010564684, fragment=True),
    Adduct('[M-2H]2-', 'neg', 2, -2 * PROTON),
)
EVIDENCE_VERSION = 2
MATCH_COLUMNS = ('source_stream', 'source_feature_id', 'source_mz', 'source_rt',
    'source_adduct', 'neutral_mass', 'target_stream', 'target_feature_id',
    'target_mz', 'expected_mz', 'target_rt', 'target_adduct', 'mass_error_ppm',
    'rt_delta_seconds', 'rt_compared', 'relationship', 'water_loss_hypothesis',
    'target_is_annotated', 'target_is_raw_p_hit')


def adduct_options(polarity, charge=None):
    return tuple(a for a in ADDUCTS if a.polarity == polarity
                 and (charge is None or a.charge == charge))


def _polarity(stream):
    return 'neg' if stream.casefold().endswith('_neg') else 'pos'


def _hits(table):
    return table.get('raw_p_significant', table.get('significant', pd.Series(False, index=table.index))).fillna(False).astype(bool)


def search_adduct_evidence(sources, by_stream, *, ppm, rt_tolerance):
    """All mass-compatible links, requiring co-elution only within one phase.

    Searches include non-significant and already annotated counterparts. A
    feature may have several source/target adduct hypotheses; retain all of
    them instead of converting an ambiguous match into a unique identity.
    """
    raw_hits = {stream: set(table.loc[_hits(table), 'feature_id'].astype(str))
                for stream, table in by_stream.items()}
    target_charges = {(stream, str(row.feature_id)): row.isotope_charge_hypothesis
                      for stream, table in by_stream.items()
                      if 'isotope_charge_hypothesis' in table
                      for row in table.itertuples(index=False)}
    rows = []
    for stream, table in by_stream.items():
        for _, feature in table[_hits(table)].iterrows():
            observed_charge = feature.get('isotope_charge_hypothesis', np.nan)
            charge = int(observed_charge) if pd.notna(observed_charge) else None
            for source_adduct in adduct_options(_polarity(stream), charge):
                neutral = source_adduct.neutral(float(feature.mz))
                if neutral <= 0:
                    continue
                for target_stream, target in sources.items():
                    same_phase = stream.split('_')[0] == target_stream.split('_')[0]
                    for target_adduct in adduct_options(_polarity(target_stream)):
                        if target_stream == stream and target_adduct == source_adduct:
                            continue
                        expected = target_adduct.mz(neutral)
                        if expected <= 0:
                            continue
                        masses = target.mz.to_numpy(float)
                        delta = expected * ppm * 1e-6
                        left = np.searchsorted(masses, expected - delta, side='left')
                        right = np.searchsorted(masses, expected + delta, side='right')
                        for candidate in target.iloc[left:right].itertuples(index=False):
                            known_charge = target_charges.get((target_stream, str(candidate.feature_id)), np.nan)
                            if pd.notna(known_charge) and int(known_charge) != target_adduct.charge:
                                continue
                            if target_stream == stream and str(candidate.feature_id) == str(feature.feature_id):
                                continue
                            rt_delta = (candidate.rt - feature.rt) * 60.
                            if same_phase and abs(rt_delta) > rt_tolerance * 60.:
                                continue
                            relation = ('same mode' if target_stream == stream else
                                        'opposite polarity' if same_phase else
                                        'other chromatography, same polarity' if _polarity(stream) == _polarity(target_stream) else
                                        'other chromatography, opposite polarity')
                            rows.append(dict(zip(MATCH_COLUMNS, (
                                stream, str(feature.feature_id), feature.mz, feature.rt,
                                source_adduct.name, neutral, target_stream, str(candidate.feature_id),
                                candidate.mz, expected, candidate.rt, target_adduct.name,
                                (candidate.mz - expected) / expected * 1e6,
                                rt_delta, same_phase, relation,
                                source_adduct.fragment or target_adduct.fragment,
                                bool(candidate.already_annotated), str(candidate.feature_id) in raw_hits.get(target_stream, set()),
                            ))))
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def attach_adduct_summaries(by_stream, matches):
    groups = {key: group for key, group in matches.groupby(['source_stream', 'source_feature_id'])}
    output = {}
    for stream, original in by_stream.items():
        table = original.copy()
        for prefix in ('same_mode_adduct', 'all_mode_adduct'):
            table[f'{prefix}_candidate_count'] = 0
            table[f'{prefix}_evidence'] = ''
        for index, feature in table[_hits(table)].iterrows():
            candidates = groups.get((stream, str(feature.feature_id)))
            if candidates is None:
                continue
            for prefix, frame in (('same_mode_adduct', candidates[candidates.target_stream.eq(stream)]),
                                   ('all_mode_adduct', candidates)):
                table.at[index, f'{prefix}_candidate_count'] = len(frame)
                table.at[index, f'{prefix}_evidence'] = '; '.join(
                    f'{r.source_adduct} → {r.target_stream} {r.target_feature_id} {r.target_adduct} '
                    f'({r.mass_error_ppm:+.2f} ppm; ' +
                    (f'ΔRT {r.rt_delta_seconds:+.1f} s)' if r.rt_compared else 'RT not compared)')
                    for r in frame.itertuples(index=False))
        output[stream] = table
    return output
