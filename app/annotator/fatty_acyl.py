"""Composition-aware fatty-acyl evidence from lipid MS/MS spectra."""

from __future__ import annotations

import itertools
import json
import re
from bisect import bisect_left, bisect_right
from typing import Any, Iterable

import numpy as np
import pandas as pd


MASS_H = 1.00782503223
MASS_O = 15.99491461957
MASS_N = 14.00307400443
PROTON_MASS = 1.007276466621
AMMONIA_MASS = MASS_N + 3 * MASS_H
WATER_MASS = 2 * MASS_H + MASS_O


def fatty_acid_neutral_mass(carbon: int, double_bonds: int, oxygens: int = 0) -> float:
    """Monoisotopic mass of RCOOH, optionally with additional oxygens."""
    return 12.0 * carbon + MASS_H * (2 * carbon - 2 * double_bonds) + MASS_O * (2 + oxygens)


def fatty_acid_anion_mz(carbon: int, double_bonds: int, oxygens: int = 0) -> float:
    """Monoisotopic m/z of singly charged [FA-H]- including electron mass."""
    return fatty_acid_neutral_mass(carbon, double_bonds, oxygens) - PROTON_MASS


def _row_classes(row: pd.Series) -> set[str]:
    lipid_class = str(row.get("class", ""))
    base = row.get("base", "")
    result = {lipid_class}
    if isinstance(base, str) and base:
        result.add(f"{lipid_class}-{base}")
    return result


def applicable_fatty_acyl_rule(
    row: pd.Series, polarity: str, rules: Iterable[dict]
) -> dict | None:
    classes = _row_classes(row)
    for rule in rules:
        if polarity != rule.get("polarity"):
            continue
        if not classes.intersection(map(str, rule.get("lipid_classes", ()))):
            continue
        if classes.intersection(map(str, rule.get("excluded_lipid_classes", ()))):
            continue
        adducts = rule.get("adducts")
        if adducts and str(row.get("adduct", "")) not in set(map(str, adducts)):
            continue
        return rule
    return None


def _possible_fatty_acyls(row: pd.Series, rule: dict) -> list[tuple[int, int, int]]:
    try:
        total_c, total_db = int(row["carbon"]), int(row["db"])
    except (KeyError, TypeError, ValueError):
        return []
    chain_count = int(rule.get("chain_count", 2))
    carbon_min = int(rule.get("carbon_min", 4))
    carbon_max = int(rule.get("carbon_max", 40))
    db_max = int(rule.get("double_bond_max", 12))
    remainder_min = int(rule.get("remainder_carbon_min", carbon_min))
    remainder_max = int(rule.get("remainder_carbon_max", carbon_max))
    oxygenations = [int(value) for value in rule.get("additional_oxygens", [0])]

    result = []
    for carbon in range(carbon_min, min(carbon_max, total_c) + 1):
        remaining_c = total_c - carbon
        remaining_chains = chain_count - 1
        if remaining_chains:
            if not (remainder_min * remaining_chains <= remaining_c
                    <= remainder_max * remaining_chains):
                continue
        elif remaining_c != 0:
            continue
        for double_bonds in range(0, min(db_max, total_db, carbon - 1) + 1):
            remaining_db = total_db - double_bonds
            if remaining_db < 0 or remaining_db > db_max * max(1, remaining_chains):
                continue
            for oxygens in oxygenations:
                result.append((carbon, double_bonds, oxygens))
    return result


def _fa_label(composition: tuple[int, int, int]) -> str:
    carbon, double_bonds, oxygens = composition
    suffix = f";O{oxygens}" if oxygens > 1 else ";O" if oxygens == 1 else ""
    return f"FA({carbon}:{double_bonds}{suffix})"


def _chain_label(composition: tuple[int, int, int]) -> str:
    return _fa_label(composition)[3:-1]


def _valid_chain(composition: tuple[int, int, int], carbon_min=4, carbon_max=40) -> bool:
    carbon, double_bonds, _ = composition
    return carbon_min <= carbon <= carbon_max and 0 <= double_bonds <= min(12, carbon - 1)


def _molecular_species_candidates(
    row: pd.Series, rule: dict, detected: list[tuple[int, int, int]]
) -> list[str]:
    if not detected:
        return []
    total_c, total_db = int(row["carbon"]), int(row["db"])
    lipid_class = str(row.get("class", ""))
    base = row.get("base", "") if isinstance(row.get("base", ""), str) else ""
    family = str(rule.get("family", "glycerophospholipid"))
    unique = list(dict.fromkeys(detected))
    candidates: set[str] = set()

    if family == "sphingolipid":
        for fatty_acyl in unique:
            lcb = (total_c - fatty_acyl[0], total_db - fatty_acyl[1], 0)
            if not _valid_chain(
                lcb,
                int(rule.get("remainder_carbon_min", 12)),
                int(rule.get("remainder_carbon_max", 24)),
            ):
                continue
            lcb_prefix = base or "d"
            candidates.add(
                f"{lipid_class}({lcb_prefix}{lcb[0]}:{lcb[1]}/{_chain_label(fatty_acyl)})"
            )
        return sorted(candidates)

    chain_count = int(rule.get("chain_count", 2))
    class_label = f"{lipid_class}-{base}" if base in {"O", "P"} else lipid_class
    if chain_count == 1:
        for fatty_acyl in unique:
            if fatty_acyl[:2] == (total_c, total_db):
                candidates.add(f"{class_label}({_chain_label(fatty_acyl)})")
        return sorted(candidates)

    if chain_count == 2:
        for fatty_acyl in unique:
            complement = (total_c - fatty_acyl[0], total_db - fatty_acyl[1], 0)
            if not _valid_chain(complement):
                continue
            if base in {"O", "P"}:
                chains = [f"{base}-{_chain_label(complement)}", _chain_label(fatty_acyl)]
            else:
                chains = sorted([_chain_label(fatty_acyl), _chain_label(complement)])
            candidates.add(f"{lipid_class}({'_'.join(chains)})")
        return sorted(candidates)

    if chain_count == 3:
        fixed_pairs = list(itertools.combinations_with_replacement(unique, 2))
        if len(unique) == 1:
            first = unique[0]
            fixed_pairs = []
            for carbon in range(4, 41):
                for double_bonds in range(0, min(12, total_db, carbon - 1) + 1):
                    fixed_pairs.append((first, (carbon, double_bonds, 0)))
        for first, second in fixed_pairs:
            third = (
                total_c - first[0] - second[0],
                total_db - first[1] - second[1],
                0,
            )
            if not _valid_chain(third):
                continue
            chains = sorted([
                _chain_label(first), _chain_label(second), _chain_label(third)
            ])
            candidates.add(f"{lipid_class}({'_'.join(chains)})")
        return sorted(candidates)

    if chain_count > 3:
        unresolved = chain_count - 1
        for fatty_acyl in unique:
            remaining_c = total_c - fatty_acyl[0]
            remaining_db = total_db - fatty_acyl[1]
            if remaining_c > 0 and remaining_db >= 0:
                candidates.add(
                    f"{lipid_class}({_chain_label(fatty_acyl)}_"
                    f"{remaining_c}:{remaining_db}[{unresolved} chains unresolved])"
                )
        return sorted(candidates)

    return []


def _rank_molecular_species(
    molecular_species: list[str],
    detections: list[dict],
    species_mapping: dict[str, list[str]],
) -> tuple[list[dict], str, float | None, bool]:
    """Rank molecular species by the summed intensity of supporting FA ions.

    A fragment is counted once per candidate, even when that acyl composition
    occurs more than once in the candidate. Ordinary MS/MS does not establish
    chain multiplicity or sn-position from fragment intensity alone.
    """
    ranking: list[dict[str, Any]] = []
    for species in molecular_species:
        support = [
            item for item in detections
            if species in species_mapping.get(str(item["fatty_acyl"]), [])
        ]
        relative_score = sum(
            float(item.get("relative_intensity_percent") or 0.0)
            for item in support
        )
        absolute_score = sum(float(item.get("intensity") or 0.0) for item in support)
        ranking.append({
            "molecular_species": species,
            "supporting_fatty_acyls": [str(item["fatty_acyl"]) for item in support],
            "supporting_fragment_count": len(support),
            "summed_fragment_intensity": absolute_score,
            "summed_relative_intensity_percent": relative_score,
        })
    ranking.sort(key=lambda item: (
        -float(item["summed_relative_intensity_percent"]),
        -int(item["supporting_fragment_count"]),
        str(item["molecular_species"]),
    ))
    if not ranking:
        return ranking, "", None, False
    top_score = float(ranking[0]["summed_relative_intensity_percent"])
    top_count = int(ranking[0]["supporting_fragment_count"])
    tied = [
        item for item in ranking
        if np.isclose(
            float(item["summed_relative_intensity_percent"]), top_score,
            rtol=1e-9, atol=1e-12,
        ) and int(item["supporting_fragment_count"]) == top_count
    ]
    # Do not manufacture a unique structural call when the evidence is tied.
    most_likely = str(tied[0]["molecular_species"]) if len(tied) == 1 else ""
    return ranking, most_likely, top_score, len(tied) > 1


def detect_fatty_acyl_evidence(
    row: pd.Series,
    spectrum: dict | None,
    polarity: str,
    rules: Iterable[dict],
    tolerance_da: float,
    min_relative_intensity_percent: float = 0.0,
) -> dict:
    """Detect sufficiently intense compatible FA ions and infer molecular species."""
    min_relative_intensity_percent = max(
        0.0, float(min_relative_intensity_percent)
    )
    defaults = {
        "fatty_acyl_rule_id": "",
        "fatty_acyl_status": "not_applicable",
        "fatty_acyl_min_relative_intensity_percent": (
            min_relative_intensity_percent
        ),
        "fatty_acyl_detected": "",
        "fatty_acyl_detected_count": 0,
        "fatty_acyl_evidence_json": "[]",
        "fatty_acyl_species_mapping_json": "{}",
        "fatty_acyl_fragment_intensities": "",
        "molecular_species_candidates": "",
        "molecular_species_candidate_count": 0,
        "molecular_species_ambiguous": False,
        "molecular_species_ranking_json": "[]",
        "most_likely_molecular_species": "",
        "most_likely_molecular_species_score": np.nan,
        "most_likely_molecular_species_ambiguous": False,
    }
    rule = applicable_fatty_acyl_rule(row, polarity, rules)
    if rule is None:
        return defaults
    result = {**defaults, "fatty_acyl_rule_id": str(rule["id"])}
    if spectrum is None:
        return {**result, "fatty_acyl_status": "no_msms"}

    possible = _possible_fatty_acyls(row, rule)
    if not possible:
        return {**result, "fatty_acyl_status": "parent_composition_unavailable"}
    peaks = spectrum.get("peaks") or []
    precursor = spectrum.get("precursor_mz") or row.get("feat_mz")
    try:
        precursor = float(precursor)
    except (TypeError, ValueError):
        precursor = np.nan
    base_intensity = max((float(peak[1]) for peak in peaks), default=0.0)
    evidence_type = str(rule["evidence_type"])
    expected_entries = []
    for composition in possible:
        if evidence_type == "fatty_acid_anion":
            expected_entries.append((
                fatty_acid_anion_mz(*composition), composition, evidence_type
            ))
        elif evidence_type == "fatty_acid_plus_ammonia_loss":
            if not np.isfinite(precursor):
                continue
            expected_entries.append((
                fatty_acid_neutral_mass(*composition) + AMMONIA_MASS,
                composition,
                evidence_type,
            ))
        elif evidence_type == "fatty_acid_or_ketene_loss":
            if not np.isfinite(precursor):
                continue
            neutral_mass = fatty_acid_neutral_mass(*composition)
            expected_entries.extend((
                (neutral_mass, composition, "fatty_acid_neutral_loss"),
                (
                    neutral_mass - WATER_MASS,
                    composition,
                    "fatty_acyl_ketene_loss",
                ),
            ))
        else:
            continue
    expected_entries.sort(key=lambda value: value[0])
    expected_values = [value[0] for value in expected_entries]
    best_matches: dict[
        tuple[tuple[int, int, int], str], tuple[float, float, float, float]
    ] = {}
    for fragment_mz, intensity in peaks:
        fragment_mz, intensity = float(fragment_mz), float(intensity)
        relative_intensity = (
            100.0 * intensity / base_intensity if base_intensity > 0 else 0.0
        )
        if relative_intensity < min_relative_intensity_percent:
            continue
        observed = fragment_mz if evidence_type == "fatty_acid_anion" else (
            precursor - fragment_mz
        )
        start = bisect_left(expected_values, observed - tolerance_da)
        stop = bisect_right(expected_values, observed + tolerance_da)
        for expected, composition, matched_evidence_type in expected_entries[start:stop]:
            match_key = (composition, matched_evidence_type)
            previous = best_matches.get(match_key)
            if previous is None or intensity > previous[1]:
                best_matches[match_key] = (
                    fragment_mz, intensity, observed, expected
                )

    detections: list[dict[str, Any]] = []
    for (composition, matched_evidence_type), match in best_matches.items():
        fragment_mz, intensity, observed, expected = match
        detections.append({
            "fatty_acyl": _fa_label(composition),
            "carbon": composition[0],
            "double_bonds": composition[1],
            "additional_oxygens": composition[2],
            "evidence_type": matched_evidence_type,
            "expected_mz_or_loss": expected,
            "observed_fragment_mz": fragment_mz,
            "observed_mz_or_loss": observed,
            "error_da": observed - expected,
            "intensity": intensity,
            "relative_intensity_percent": (
                100.0 * intensity / base_intensity if base_intensity > 0 else None
            ),
        })
    detections.sort(
        key=lambda value: value["relative_intensity_percent"] or 0.0, reverse=True
    )
    detected_compositions: list[tuple[int, int, int]] = [
        (int(value["carbon"]), int(value["double_bonds"]), int(value["additional_oxygens"]))
        for value in detections
    ]
    molecular_species = _molecular_species_candidates(row, rule, detected_compositions)
    species_tokens = {
        species: set(re.findall(r"\d+:\d+(?:;O\d*)?", species))
        for species in molecular_species
    }
    species_mapping: dict[str, list[str]] = {}
    for composition in dict.fromkeys(detected_compositions):
        label = _fa_label(composition)
        chain_label = _chain_label(composition)
        species_mapping[label] = [
            species for species in molecular_species
            if chain_label in species_tokens[species]
        ]
    ranking, most_likely, most_likely_score, most_likely_ambiguous = (
        _rank_molecular_species(molecular_species, detections, species_mapping)
    )
    return {
        **result,
        "fatty_acyl_status": "supported" if detections else "not_detected",
        "fatty_acyl_detected": "; ".join(
            dict.fromkeys(value["fatty_acyl"] for value in detections)
        ),
        "fatty_acyl_detected_count": len(set(detected_compositions)),
        "fatty_acyl_evidence_json": json.dumps(detections, separators=(",", ":")),
        "fatty_acyl_species_mapping_json": json.dumps(
            species_mapping, separators=(",", ":")
        ),
        "fatty_acyl_fragment_intensities": "; ".join(
            f"{value['fatty_acyl']}"
            + (
                " [RCOOH loss]"
                if value["evidence_type"] == "fatty_acid_neutral_loss"
                else " [ketene loss]"
                if value["evidence_type"] == "fatty_acyl_ketene_loss"
                else ""
            )
            + f": {float(value['intensity']):.6g} "
            f"({float(value['relative_intensity_percent']):.2f}%)"
            for value in detections
        ),
        "molecular_species_candidates": "; ".join(molecular_species),
        "molecular_species_candidate_count": len(molecular_species),
        "molecular_species_ambiguous": len(molecular_species) > 1,
        "molecular_species_ranking_json": json.dumps(
            ranking, separators=(",", ":")
        ),
        "most_likely_molecular_species": most_likely,
        "most_likely_molecular_species_score": most_likely_score,
        "most_likely_molecular_species_ambiguous": most_likely_ambiguous,
    }
