"""Additive complete-isotope evidence; legacy tests and ion grouping are held fixed."""
from functools import lru_cache
import hashlib

import numpy as np
import pandas as pd

from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_lipidomics.isotope_patterns import (
    ISOTOPE_MODEL_VERSION, ISOSPEC_VERSION, PROBABILITY_COVERAGE, parse_formula,
)
from standard_lipidomics.isotope_quality import IsotopeIndex, score_envelope
from standard_lipidomics.untargeted_adducts import adduct_options
from standard_lipidomics.untargeted_formula_labels import _CANDIDATE


@lru_cache(maxsize=1)
def formula_library():
    path = REFERENCE_DATA_DIRECTORY/'4d_lipid_reference.csv'
    frame = pd.read_csv(path, usecols=['name', 'formula', 'neutral_mass'])
    frame = frame[~frame.name.str.contains(r'\(IS\)|\bD\d+|13C|15N', regex=True, na=False)]
    frame = frame.dropna(subset=['formula', 'neutral_mass']).drop_duplicates(['formula', 'neutral_mass'])
    valid = []
    for row in frame.itertuples():
        try:
            parse_formula(row.formula)
            valid.append(True)
        except ValueError:
            valid.append(False)
    frame = frame[valid].sort_values(['neutral_mass', 'formula'], kind='stable')
    return frame.neutral_mass.to_numpy(float), frame.formula.to_numpy(str), hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_hypotheses(row, stream, criteria, method):
    """Existing candidates plus a mass-only lookup of the existing public library.

    No study reference identities or formulas enter this function. Missing M+1
    cannot suppress library candidates. A broader de-novo generator is Stage 5.
    """
    polarity = 'neg' if stream.endswith('Neg') else 'pos'
    allowed = adduct_options(polarity, method=method, stream=stream)
    names = {a.name for a in allowed}
    candidates = {(formula, adduct): 'existing candidate' for adduct, formula in
                  _CANDIDATE.findall(str(row.get('elemental_formula_candidates', '')))
                  if adduct in names}
    masses, formulas, _ = formula_library()
    for adduct in allowed:
        neutral = adduct.neutral(float(row['mz']))
        lo, hi = np.searchsorted(masses, [neutral*(1-criteria.cross_mode_ppm_tolerance*1e-6),
                                         neutral*(1+criteria.cross_mode_ppm_tolerance*1e-6)])
        for formula in formulas[lo:hi]:
            candidates.setdefault((formula, adduct.name), 'public-library mass candidate')
    return [(formula, adduct, origin) for (formula, adduct), origin in sorted(candidates.items())]


def apply_complete_isotopes(result, analysis, criteria, progress=None):
    if criteria.isotope_model == 'current':
        return analysis
    method = analysis['acquisition_method']
    scores, peaks, summaries = [], [], []
    queries = analysis['raw_p_features']
    indexes = {m: IsotopeIndex(m, item['features']) for m, item in result['streams'].items()}
    labelled = set()
    annotations = result.get('automatic_annotations', pd.DataFrame())
    if {'stream', 'feature_id', 'name'} <= set(annotations):
        selected = annotations[annotations.name.astype(str).str.contains(r'\(IS\)|\bD\d+|13C|15N', regex=True)]
        labelled = set(zip(selected.stream.astype(str), selected.feature_id.astype(str)))
    for number, row in enumerate(queries.to_dict('records')):
        stream, fid = str(row['stream']), str(row['feature_id'])
        summary = {'stream': stream, 'feature_id': fid, 'full_isotope_formula': '', 'full_isotope_adduct': '',
                   'full_isotope_status': 'no candidate composition available', 'full_isotope_quality': '',
                   'full_isotope_score': np.nan, 'full_isotope_candidate_count': 0}
        if (stream, fid) in labelled:
            summary['full_isotope_status'] = 'labelled/internal standard excluded from natural-abundance inference'
            summaries.append(summary)
            continue
        for formula, adduct, origin in candidate_hypotheses(row, stream, criteria, method):
            try:
                score, observations = score_envelope(indexes[stream], fid, formula, adduct, criteria,
                    method['methods'][stream[:2]]['instrument'])
            except ValueError as exc:
                scores.append({'stream': stream, 'feature_id': fid, 'formula': formula, 'adduct': adduct,
                               'candidate_origin': origin, 'status': str(exc), 'score': np.nan})
                continue
            score['candidate_origin'] = origin
            scores.append(score)
            peaks.extend(observations)
        local = [r for r in scores if r['stream'] == stream and r['feature_id'] == fid and
                 np.isfinite(r.get('score', np.nan)) and r['status'] != 'parent mass incompatible']
        local.sort(key=lambda r: (r['score'], r['formula'], r['adduct']))
        for rank, candidate in enumerate(local, 1):
            candidate['rank'] = rank
        summary['full_isotope_candidate_count'] = len(local)
        if local:
            best = local[0]
            summary.update(full_isotope_formula=best['formula'], full_isotope_adduct=best['adduct'],
                           full_isotope_status=best['status'], full_isotope_quality=best['quality_flags'],
                           full_isotope_score=best['score'])
        summaries.append(summary)
        if progress and (number % 25 == 0 or number+1 == len(queries)):
            progress((number+1)/max(1, len(queries)), f'Complete isotope evidence: {number+1:,}/{len(queries):,} features')
    columns = ['stream', 'feature_id', 'full_isotope_formula', 'full_isotope_adduct', 'full_isotope_status',
               'full_isotope_quality', 'full_isotope_score', 'full_isotope_candidate_count']
    labels = pd.DataFrame(summaries, columns=columns)
    output = {**analysis, 'isotope_candidate_scores': pd.DataFrame(scores), 'isotope_peak_evidence': pd.DataFrame(peaks)}
    for name in ('features', 'raw_p_features', 'significant_features', 'compound_features'):
        frame = analysis.get(name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame = frame.drop(columns=[c for c in columns[2:] if c in frame]).copy()
            frame['feature_id'] = frame.feature_id.astype(str)
            output[name] = frame.merge(labels, on=['stream', 'feature_id'], how='left', validate='one_to_one')
    output['by_stream'] = {str(m): f.copy() for m, f in output['features'].groupby('stream')}
    output['isotope_model_provenance'] = {'version': ISOTOPE_MODEL_VERSION, 'engine': 'IsoSpecPy',
        'engine_version': ISOSPEC_VERSION, 'probability_coverage': PROBABILITY_COVERAGE,
        'component_floor_relative_to_parent': .001, 'minimum_ratio_pairs': 3, 'maximum_log_ratio_mad': .5,
        'formula_library_sha256': formula_library()[2], 'integration': 'additive evidence; current formulas/grouping unchanged',
        'resolution': 'bounded 2-FWHM centroid approximation; unknown laws/references explicitly flagged',
        'missingness': '5th-percentile positive area proxy; not measured noise/LOD',
        'saturation': 'only supplied flags; otherwise not assessable from integrated areas'}
    return output
