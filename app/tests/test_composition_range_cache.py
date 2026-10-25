from dataclasses import replace
import re

from annotator.composition_ranges import (
    canonical_lipid_class,
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
)


def original(value, *, species=""):
    """Uncached implementation preserved verbatim in logic for parity."""
    label = " ".join(str(value).strip().split())
    compact = label.replace(" ", "")
    positional = re.fullmatch(r"([12]-acyl)(LPC|LPE)", compact, re.I)
    if positional:
        return f"{positional.group(1).lower()} {positional.group(2).upper()}"
    marker = re.fullmatch(r"(.+)-(O\+P|O|P|d|t|m|q)", label)
    if marker:
        label = f"{marker.group(1)}[{marker.group(2)}]"
    hydroxy = re.search(r"\)-(\d*OH)$", str(species), re.I)
    if hydroxy and label in {"AC", "FA"}:
        return f"{label}-{hydroxy.group(1).upper()}"
    return label


def test_cache_matches_original_and_preserves_unhashable_current_string_values():
    labels = [r.lipid_class for r in DEFAULT_COMPOSITION_RANGE_SETTINGS.class_ranges]
    labels += [
        None,
        [],
        {},
        " 1-acyl LPC ",
        "2-acylLPE",
        "PE-P",
        "PE-O",
        "PC-O+P",
        "FA",
        "AC",
        "ß",
    ]
    species = ["", "FA(18:1)-OH", "AC(20:2)-2OH", "FA(18:1)-oh", None, []]
    for label in labels:
        for name in species:
            assert canonical_lipid_class(label, species=name) == original(
                label, species=name
            )
    changing = []
    assert canonical_lipid_class(changing) == "[]"
    changing.append("PE")
    assert canonical_lipid_class(changing) == "['PE']"

    class UnusedSpecies:
        def __str__(self):
            raise AssertionError("Positional classes must still ignore species")

    assert canonical_lipid_class("1-acyl LPC", species=UnusedSpecies()) == "1-acyl LPC"


def test_range_cache_is_immutable_input_sensitive_and_returns_fresh_dictionaries():
    base = DEFAULT_COMPOSITION_RANGE_SETTINGS
    expected = {original(r.lipid_class): r for r in base.class_ranges}
    assert base.class_lookup() == expected
    base.class_lookup().clear()
    assert base.class_lookup() == expected
    rule = replace(base.class_ranges[0], enabled=False)
    changed = replace(base, class_ranges=(*base.class_ranges, rule))
    assert changed.class_lookup()[original(rule.lipid_class)] == rule
    assert base.class_lookup() == expected
    assert (
        replace(base, class_ranges=list(base.class_ranges)).class_lookup() == expected
    )
