"""Complete natural-abundance ion patterns; no identities or study labels.

IsoSpec supplies fine isotopologues. Instrument-width clustering is an explicit
centroid approximation, not a claim of experimentally resolved fine structure.
"""
from dataclasses import dataclass
from functools import lru_cache
import re

import numpy as np

from standard_lipidomics.untargeted_adducts import ADDUCTS

ISOTOPE_MODEL_VERSION = 1
ISOSPEC_VERSION = '2.6.1'
PROBABILITY_COVERAGE = .99999
ELECTRON_MASS = .000548579909065
TOKEN = re.compile(r'([A-Z][a-z]?)(\d*)')
ION_ATOMS = {
    '[M+H]+': {'H': 1}, '[M+NH4]+': {'N': 1, 'H': 4},
    '[M+Na]+': {'Na': 1}, '[M+K]+': {'K': 1},
    '[M+H-H2O]+': {'H': -1, 'O': -1}, '[M-H]-': {'H': -1},
    '[M+HCOO]-': {'C': 1, 'H': 1, 'O': 2},
    '[M+CH3COO]-': {'C': 2, 'H': 3, 'O': 2},
    '[M+Cl]-': {'Cl': 1}, '[M-H-H2O]-': {'H': -3, 'O': -1},
    '[M-2H]2-': {'H': -2},
}
QE_REFERENCE = 'https://documents.thermofisher.com/TFS-Assets/CMD/Specification-Sheets/PS-63912-LC-MS-Q-Exactive-Plus-Orbitrap-PS63912-EN.pdf'


def parse_formula(formula):
    tokens = list(TOKEN.finditer(str(formula)))
    if not tokens or ''.join(t.group() for t in tokens) != formula:
        raise ValueError('Unsupported or isotope-labelled formula syntax.')
    atoms = {}
    for t in tokens:
        if t[1] not in {'C', 'H', 'N', 'O', 'P', 'S', 'Cl', 'Na', 'K'}:
            raise ValueError('Unsupported or isotope-labelled element.')
        atoms[t[1]] = atoms.get(t[1], 0) + int(t[2] or 1)
    if any(n <= 0 or n > 1000 for n in atoms.values()) or sum(atoms.values()) > 2000:
        raise ValueError('Formula outside the documented lipid-size limits.')
    return atoms


def formula_text(atoms):
    return ''.join(e + (str(atoms[e]) if atoms[e] != 1 else '')
                   for e in ['C', 'H', *sorted(set(atoms) - {'C', 'H'})] if atoms.get(e, 0))


def ion_composition(formula, adduct):
    if adduct not in ION_ATOMS:
        raise ValueError('Unsupported or disabled ion.')
    atoms = parse_formula(formula)
    for element, count in ION_ATOMS[adduct].items():
        atoms[element] = atoms.get(element, 0) + count
    if any(n < 0 for n in atoms.values()):
        raise ValueError('Adduct removes atoms absent from the neutral formula.')
    ion = next(a for a in ADDUCTS if a.name == adduct)
    return {e: n for e, n in atoms.items() if n}, ion.charge * (1 if ion.polarity == 'pos' else -1)


@dataclass(frozen=True)
class PatternPeak:
    mz: float
    probability: float
    nominal: int
    components: int = 1


@lru_cache(maxsize=8192)
def fine_pattern(formula, adduct):
    import IsoSpecPy
    if IsoSpecPy.__version__ != ISOSPEC_VERSION:
        raise ValueError(f'Complete isotope mode requires IsoSpecPy {ISOSPEC_VERSION}.')
    atoms, charge = ion_composition(formula, adduct)
    text = formula_text(atoms)
    base = IsoSpecPy.Iso(formula=text).getLightestPeakMass()
    distribution = IsoSpecPy.IsoTotalProb(formula=text, prob_to_cover=PROBABILITY_COVERAGE)
    peaks = []
    for mass, probability in zip(distribution.np_masses(), distribution.np_probs()):
        peaks.append(PatternPeak(float((mass-charge*ELECTRON_MASS)/abs(charge)),
                                 float(probability), int(round(mass-base))))
    return tuple(sorted(peaks, key=lambda p: p.mz))


def resolving_power(instrument, mz):
    """Return local FWHM resolving power and the explicit reference basis."""
    resolution = instrument.get('resolution')
    reference = instrument.get('resolution_reference_mz')
    model = re.sub(r'[^a-z0-9]', '', str(instrument.get('model', '')).lower())
    basis = 'project method'
    if reference is None and 'qexactiveplus' in model:
        reference, basis = 200., 'Q Exactive Plus manufacturer specification; not measured per scan'
    if not resolution or not reference:
        return None, 'resolution/reference unavailable; nominal envelope only'
    if 'orbitrap' in model or 'qexactive' in model:
        return float(resolution) * np.sqrt(float(reference) / mz), basis
    return None, 'instrument resolution law unknown; nominal envelope only'


@lru_cache(maxsize=8192)
def _pattern(formula, adduct, model, resolution, reference):
    fine = fine_pattern(formula, adduct)
    instrument = {'model': model, 'resolution': resolution, 'resolution_reference_mz': reference}
    groups, current = [], []
    for peak in fine:
        power, _ = resolving_power(instrument, peak.mz)
        separation = 2 * peak.mz/power if power else np.inf
        # Bounded span avoids linking a chain of distinct fine peaks.
        if current and (peak.nominal != current[0].nominal or peak.mz-current[0].mz > separation):
            groups.append(current)
            current = []
        current.append(peak)
    if current:
        groups.append(current)
    merged = []
    for group in groups:
        probability = sum(p.probability for p in group)
        merged.append(PatternPeak(sum(p.mz*p.probability for p in group)/probability,
                                  probability, group[0].nominal, len(group)))
    return tuple(merged)


def theoretical_pattern(formula, adduct, instrument=None):
    instrument = instrument or {}
    return _pattern(formula, adduct, instrument.get('model', 'unknown'),
                    instrument.get('resolution'), instrument.get('resolution_reference_mz'))
