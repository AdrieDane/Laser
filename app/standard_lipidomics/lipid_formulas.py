"""Structured lipid-aware formula candidates with a broader CHNOPS fallback.

No study identities, retention-time matches or reference formulas enter this
search. Budgets limit isotope scoring, never silently truncate generation.
"""
import csv
from functools import lru_cache
import hashlib
import re

import numpy as np

from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_lipidomics.isotope_patterns import formula_text, parse_formula, ion_composition
from standard_lipidomics.untargeted_adducts import adduct_options

FORMULA_MODEL_VERSION = 1
ELEMENTS = ('C', 'H', 'N', 'O', 'P', 'S')
MAX_COUNTS = (180, 400, 6, 60, 3, 3)
MAX_CHAIN_CARBONS = 48
MAX_CHAIN_DOUBLE_BONDS = 36
MAX_EXTRA_OXYGEN = 4
RECORD = np.dtype([('mass', 'f8'), *[(e, 'i2') for e in ELEMENTS], ('template', 'i2'), ('oxidation', 'i1')])


@lru_cache(maxsize=1)
def atomic_masses():
    import IsoSpecPy
    return np.array([IsoSpecPy.Iso(formula=e+'1').getLightestPeakMass() for e in ELEMENTS])


def neutral_valid(atoms):
    """Broad neutral valence constraints, not peptide-style heteroatom ratios.

    P is odd-valent and S may be hypervalent. No S/P-specific DBE cap is used.
    """
    c, h, n, o, p, s = (atoms.get(e, 0) for e in ELEMENTS)
    if set(atoms)-set(ELEMENTS):
        return False
    if any(atoms.get(e, 0) < 0 or atoms.get(e, 0) > maximum for e, maximum in zip(ELEMENTS, MAX_COUNTS)) or c < 1:
        return False
    dbe_twice = 2*c+2+n+p-h
    return dbe_twice % 2 == 0 and dbe_twice >= -4*s and h <= 2*c+n+p+2+4*s


@lru_cache(maxsize=1)
def templates():
    path = REFERENCE_DATA_DIRECTORY/'matlab_pipeline/DataBase/lipids.tsv'
    groups, skipped = {}, []
    active, kind = '', ''
    with path.open(encoding='latin-1', newline='') as handle:
        rows = list(csv.reader(handle, delimiter='\t'))
    for row in rows[1:]:
        if not any(v.strip() for v in row):
            continue
        if row[0].strip():
            active, kind = row[0].strip(), row[2].strip()
        chem = row[10].strip()
        if not chem:
            skipped.append(active)
            continue
        key = (chem, kind)
        groups.setdefault(key, set()).add(active)
    definitions = []
    for (chem, kind), classes in sorted(groups.items()):
        token = re.search(r'R(\d*)', chem)
        nr = int(token[1] or 1) if token else 0
        text = re.sub(r'R\d*', '', chem)
        atoms = parse_formula(text) if text else {}
        if kind not in {'FA', 'FAfull', 'isoprene', ''}:
            raise ValueError('Unsupported public lipid R-group definition: '+kind)
        definitions.append({'chem': chem, 'kind': kind, 'nR': nr, 'atoms': atoms, 'classes': tuple(sorted(classes))})
    return definitions, {'template_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'unsupported_blank_classes': sorted(set(skipped))}


@lru_cache(maxsize=1)
def lipid_index():
    definitions, _ = templates()
    arrays = []
    for number, template in enumerate(definitions):
        kind, nr, base = template['kind'], template['nR'], template['atoms']
        if not kind:
            if nr:
                raise ValueError('R-group without a supported definition.')
            if neutral_valid(base):
                ox, loss = np.meshgrid(np.arange(MAX_EXTRA_OXYGEN+1), np.arange(5), indexing='ij')
                r = np.zeros(ox.size, dtype=RECORD)
                for e in ELEMENTS:
                    r[e] = base.get(e, 0)
                r['O'] += ox.ravel()
                r['H'] -= 2*loss.ravel()
                r['oxidation'] = ox.ravel()
                r['template'] = number
                arrays.append(r[r['H'] >= 0])
            continue
        if kind == 'isoprene':
            chain, db, ox = np.meshgrid(np.arange(1, 37), [0], np.arange(MAX_EXTRA_OXYGEN+1), indexing='ij')
            c, h = base.get('C', 0)+5*chain.ravel(), base.get('H', 0)+8*chain.ravel()
        else:
            chain, db, ox = np.meshgrid(np.arange(2*nr, min(180, MAX_CHAIN_CARBONS*nr)+1),
                                        np.arange(MAX_CHAIN_DOUBLE_BONDS+1), np.arange(MAX_EXTRA_OXYGEN+1), indexing='ij')
            c = base.get('C', 0)+chain.ravel()-nr
            h = base.get('H', 0)+2*chain.ravel()-2*db.ravel()-nr
        r = np.zeros(len(c), dtype=RECORD)
        r['C'], r['H'], r['O'] = c, h, base.get('O', 0)+ox.ravel()
        for e in ('N', 'P', 'S'):
            r[e] = base.get(e, 0)
        r['template'], r['oxidation'] = number, ox.ravel()
        valid = np.ones(len(r), bool)
        for e, maximum in zip(ELEMENTS, MAX_COUNTS):
            valid &= (r[e] >= (1 if e == 'C' else 0)) & (r[e] <= maximum)
        if kind != 'isoprene':
            valid &= db.ravel() <= chain.ravel()//2
        arrays.append(r[valid])
    rows = np.concatenate(arrays)
    rows['mass'] = sum(rows[e]*mass for e, mass in zip(ELEMENTS, atomic_masses()))
    return rows[np.argsort(rows['mass'], kind='stable')]


@lru_cache(maxsize=1)
def generic_index():
    """Meet-in-the-middle CH and NOPS indexes, not an isotope-carbon gate."""
    c, h = np.meshgrid(np.arange(1, MAX_COUNTS[0]+1), np.arange(MAX_COUNTS[1]+1), indexing='ij')
    c, h = c.ravel(), h.ravel()
    # The remaining N/P/S can increase valence; apply exact neutral constraints later.
    valid = h <= 2*c+MAX_COUNTS[2]+MAX_COUNTS[4]+2+4*MAX_COUNTS[5]
    c, h = c[valid], h[valid]
    mass = c*atomic_masses()[0]+h*atomic_masses()[1]
    order = np.argsort(mass, kind='stable')
    hetero = np.indices(tuple(v+1 for v in MAX_COUNTS[2:])).reshape(4, -1).T
    return mass[order], c[order], h[order], hetero, hetero@atomic_masses()[2:]


def formula_candidates(mz, stream, criteria, method, *, supplied_adduct=None):
    """All generated candidates and audit counts; no reference-class filtering."""
    from standard_lipidomics.complete_isotopes import formula_library
    rows, report = {}, {'generated_before_valence': 0, 'valence_rejected': 0, 'ion_composition_rejected': 0}
    definitions, metadata = templates()
    index = lipid_index()
    chmass, carbon, hydrogen, hetero, heteromass = generic_index()
    library_mass, library_formula, _ = formula_library()
    ions = adduct_options('neg' if stream.endswith('Neg') else 'pos', method=method, stream=stream)
    if supplied_adduct:
        ions = tuple(a for a in ions if a.name == supplied_adduct)
    def add(atoms, ion, mass, origin, classes=(), prior=0.):
        report['generated_before_valence'] += 1
        if not neutral_valid(atoms):
            report['valence_rejected'] += 1
            return
        formula = formula_text(atoms)
        try:
            ion_composition(formula, ion.name)
        except ValueError:
            report['ion_composition_rejected'] += 1
            return
        key = (formula, ion.name)
        if key not in rows:
            rows[key] = {'formula': formula, 'adduct': ion.name, 'charge': ion.charge*(1 if ion.polarity == 'pos' else -1),
                         'neutral_mass': float(mass), 'origins': set(), 'lipid_classes': set(), 'structure_prior': float(prior)}
        r = rows[key]
        r['origins'].add(origin)
        r['lipid_classes'].update(classes)
        r['structure_prior'] = min(r['structure_prior'], prior)
    for ion in ions:
        neutral = float(ion.neutral(mz))
        if neutral <= 0:
            continue
        delta = neutral*criteria.cross_mode_ppm_tolerance*1e-6
        lo, hi = np.searchsorted(index['mass'], [neutral-delta, neutral+delta])
        for r in index[lo:hi]:
            add({e: int(r[e]) for e in ELEMENTS if r[e]}, ion, r['mass'], 'lipid backbone',
                definitions[r['template']]['classes'], .15*int(r['oxidation']))
        lo, hi = np.searchsorted(library_mass, [neutral-delta, neutral+delta])
        for formula, mass in zip(library_formula[lo:hi], library_mass[lo:hi]):
            add(parse_formula(formula), ion, mass, 'public library')
        target = neutral-heteromass
        left = np.searchsorted(chmass, target-delta)
        right = np.searchsorted(chmass, target+delta, side='right')
        for i in np.flatnonzero(right > left):
            for j in range(left[i], right[i]):
                values = (int(carbon[j]), int(hydrogen[j]), *map(int, hetero[i]))
                add({e: n for e, n in zip(ELEMENTS, values) if n}, ion, chmass[j]+heteromass[i], 'de novo CHNOPS', prior=1.)
    output = []
    for r in rows.values():
        ion = next(a for a in ions if a.name == r['adduct'])
        neutral = float(ion.neutral(mz))
        r['mass_error_ppm'] = (r['neutral_mass']-neutral)/neutral*1e6
        r['origins'], r['lipid_classes'] = '; '.join(sorted(r['origins'])), '; '.join(sorted(r['lipid_classes']))
        output.append(r)
    output.sort(key=lambda r: (r['structure_prior']+abs(r['mass_error_ppm'])/criteria.cross_mode_ppm_tolerance,
                               r['formula'], r['adduct']))
    report.update(generated_candidates=len(output), template_sha256=metadata['template_sha256'],
                  formula_scoring_limit=criteria.formula_scoring_limit,
                  scoring_budget_exhausted=len(output) > criteria.formula_scoring_limit)
    return output, report


def predict_formulas(index, row, stream, criteria, method, *, supplied_adduct=None):
    from standard_lipidomics.isotope_quality import score_envelope
    candidates, audit = formula_candidates(float(row['mz']), stream, criteria, method, supplied_adduct=supplied_adduct)
    scores, peaks = [], []
    for number, candidate in enumerate(candidates):
        result = {'stream': stream, 'feature_id': str(row['feature_id']), **candidate,
                  'candidate_origin': candidate['origins'], 'score': np.nan, 'status': 'not isotope scored; explicit budget',
                  'isotope_scored': False}
        if number < criteria.formula_scoring_limit:
            measured, observations = score_envelope(index, row['feature_id'], candidate['formula'], candidate['adduct'],
                criteria, method['methods'][stream[:2]]['instrument'])
            result.update(measured)
            result['score'] += candidate['structure_prior']
            result['isotope_scored'] = True
            peaks.extend(observations)
        scores.append(result)
    ranked = [r for r in scores if r['isotope_scored'] and r['status'] != 'parent mass incompatible']
    ranked.sort(key=lambda r: (r['score'], r['formula'], r['adduct']))
    for rank, candidate in enumerate(ranked, 1):
        candidate['rank'] = rank
    return scores, peaks, {**audit, 'stream': stream, 'feature_id': str(row['feature_id']), 'scored_candidates': len(ranked)}
