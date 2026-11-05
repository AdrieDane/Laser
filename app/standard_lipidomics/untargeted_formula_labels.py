"""Display existing isotope-supported neutral formulas without re-predicting."""
from collections import defaultdict
import re

_CANDIDATE = re.compile(r'(\[[^\]]+\]\d*[+-])\s+(C\d+H\d+(?:[NOPS]\d+)*)\s+\(')


def add_formula_labels(compounds, raw):
    output = compounds.copy()
    predictions = {}
    if 'chemical_composition_hypothesis' in raw:
        for row in raw.itertuples(index=False):
            predictions[(str(row.stream), str(row.feature_id))] = _CANDIDATE.findall(str(row.chemical_composition_hypothesis))
    direct, by_compound = {}, defaultdict(set)
    for row in output.itertuples(index=False):
        candidates = [formula for adduct, formula in predictions.get((str(row.stream), str(row.feature_id)), ()) if adduct == row.adduct]
        if candidates:
            direct[(str(row.stream), str(row.feature_id))] = candidates[0]
            by_compound[row.compound_id].add(candidates[0])
    formulas, statuses = [], []
    for row in output.itertuples(index=False):
        formula = direct.get((str(row.stream), str(row.feature_id)), '')
        status = 'isotope-supported candidate' if formula else 'unresolved'
        family = by_compound[row.compound_id]
        if not formula and row.adduct and len(family) == 1:
            formula = next(iter(family))
            status = 'tentative linked-compound candidate'
        if len(family) > 1:
            status = 'conflicting compound candidates' if formula else 'unresolved; conflicting compound candidates'
        formulas.append(formula)
        statuses.append(status)
    output['predicted_chemical_formula'] = formulas
    output['formula_status'] = statuses
    return output
