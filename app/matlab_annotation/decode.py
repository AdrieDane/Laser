"""Structured decoding of MATLAB annotation codes.

The vendored ``naming.resolved_compound_name`` renders one display string per
annotation code. LASER needs the same information as separate fields — species,
adduct, isotopologue, carbon and double-bond count — so that annotations land on
the standard-mode grid the same way the native engine's do.

This module decodes the mixed-radix code itself and is checked against
``resolved_compound_name`` in the tests: the joined form must reproduce MATLAB's
display string exactly, so any drift in the vendored naming path is caught
rather than silently reinterpreted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

# ``_values`` is imported rather than reimplemented: it maps MATLAB's empty
# cells to "" (the monoisotopic decoration), and a second copy of that rule
# would be exactly the kind of drift this module exists to avoid.
from .vendor.naming import _values, compound_display_template

# MATLAB's composition axes, in code order. Axis 1 is unused by every shipped
# lipid definition but is kept so the radix arithmetic stays literal.
_ADDUCT_AXIS = 0
_ISOTOPE_AXIS = 2
_CARBON_AXIS = 3
_DOUBLE_BOND_AXIS = 4


@dataclass(frozen=True)
class DecodedAnnotation:
    """One MATLAB annotation code resolved into LASER's annotation fields."""

    code: int
    display_name: str
    species: str
    lipid_class: str
    adduct: str
    isotopologue: str
    carbon: int | None
    double_bonds: int | None
    internal_standard: bool

    @property
    def monoisotopic(self) -> bool:
        return not self.isotopologue


def composition_dimensions(
    composition_attr: dict, phase: str
) -> list[list[str]]:
    """Return the phase-resolved text axes MATLAB indexes a code against.

    The adduct axis is filtered by the definition's ``column`` field, because a
    lipid class may offer different adducts on NP than on RP.
    """
    dimensions = [_values(value) for value in composition_attr["text_comp"]]
    columns = _values(composition_attr.get("column", []))
    if columns and len(dimensions[_ADDUCT_AXIS]) == len(columns):
        dimensions[_ADDUCT_AXIS] = [
            value for value, column in zip(dimensions[_ADDUCT_AXIS], columns)
            if column == phase
        ]
    return [values or [""] for values in dimensions]


def _digits(code: int, dimensions: list[list[str]]) -> list[int]:
    bases = [len(values) for values in dimensions]
    remainder = abs(int(code)) - 1
    out = []
    for index in range(len(bases)):
        weight = int(np.prod(bases[index + 1:], dtype=np.int64))
        out.append(remainder // weight)
        remainder %= weight
    return out


def _species_name(template: str, chosen: list[str]) -> str:
    """Substitute the composition into MATLAB's ``TG(x:x)``-style template."""
    if "x:x" in template and len(chosen) > _DOUBLE_BOND_AXIS:
        return template.replace(
            "x:x", f"{chosen[_CARBON_AXIS]}:{chosen[_DOUBLE_BOND_AXIS]}"
        )
    if re.search(r"x\)$", template) and len(chosen) > _CARBON_AXIS:
        return re.sub(r"x\)$", f"{chosen[_CARBON_AXIS]})", template)
    return template


def _integer(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def lipid_class_from_template(template: str) -> str:
    """``TG(x:x)`` -> ``TG``; ``PC(O-x:x)`` -> ``PC-O``.

    LASER groups grids by class including the ether/plasmalogen/sphingoid
    subclass, so the ``O-``/``P-``/``d``/``t`` marker moves out of the brackets
    and onto the class the same way ``grid_class_series`` does for the 4D
    reference.
    """
    match = re.match(r"^([^(]*)\((?P<marker>[OPdtm]-?)?", template)
    if not match:
        return template
    head = match.group(1).strip()
    marker = (match.group("marker") or "").rstrip("-")
    return f"{head}-{marker}" if marker else head


class CodeDecoder:
    """Decode many codes for one ``(compound, phase)`` without redoing setup.

    Resolving the text axes and the display template costs far more than the
    radix arithmetic itself, so a definition with hundreds of thousands of
    theoretical targets must not repeat that work per code.
    """

    def __init__(self, project, compound: int, phase: str):
        self.compound = int(compound)
        self.phase = phase
        self.template = compound_display_template(project, compound - 1)
        self.lipid_class = lipid_class_from_template(self.template)
        self.dimensions = composition_dimensions(
            project.compounds.composition_attr[compound - 1], phase
        )
        self.bases = [len(values) for values in self.dimensions]
        self.weights = [
            int(np.prod(self.bases[index + 1:], dtype=np.int64))
            for index in range(len(self.bases))
        ]
        self._cache: dict[int, DecodedAnnotation] = {}

    @property
    def total_codes(self) -> int:
        return int(np.prod(self.bases, dtype=np.int64))

    def axis(self, index: int) -> list[str]:
        return self.dimensions[index]

    def code_for(self, indices: dict[int, int]) -> int:
        """Compose a code from zero-based indices on each axis."""
        return 1 + sum(
            indices.get(axis, 0) * weight
            for axis, weight in enumerate(self.weights)
        )

    def decode(self, code: int) -> DecodedAnnotation:
        code = int(code)
        cached = self._cache.get(code)
        if cached is not None:
            return cached
        if not code:
            decoded = DecodedAnnotation(
                code=0, display_name=self.template, species=self.template,
                lipid_class=self.lipid_class, adduct="", isotopologue="",
                carbon=None, double_bonds=None, internal_standard=False,
            )
            self._cache[code] = decoded
            return decoded
        remainder = abs(code) - 1
        chosen = []
        for values, weight in zip(self.dimensions, self.weights):
            chosen.append(values[remainder // weight])
            remainder %= weight
        species = _species_name(self.template, chosen)
        suffix = " ".join(value for value in chosen[:3] if value).strip()
        display_name = f"{species} {suffix}".strip()
        internal_standard = code < 0
        if internal_standard:
            display_name = f"IS: {display_name}"
        decoded = DecodedAnnotation(
            code=code,
            display_name=display_name,
            species=species,
            lipid_class=self.lipid_class,
            adduct=chosen[_ADDUCT_AXIS] if chosen else "",
            isotopologue=(
                chosen[_ISOTOPE_AXIS] if len(chosen) > _ISOTOPE_AXIS else ""
            ),
            carbon=(
                _integer(chosen[_CARBON_AXIS])
                if len(chosen) > _CARBON_AXIS else None
            ),
            double_bonds=(
                _integer(chosen[_DOUBLE_BOND_AXIS])
                if len(chosen) > _DOUBLE_BOND_AXIS else None
            ),
            internal_standard=internal_standard,
        )
        self._cache[code] = decoded
        return decoded


def decode_annotation_code(
    project, compound: int, phase: str, code: int
) -> DecodedAnnotation:
    """Resolve one ``(compound, code)`` pair into LASER's annotation fields."""
    composition_attr = project.compounds.composition_attr[compound - 1]
    template = compound_display_template(project, compound - 1)
    if not code:
        return DecodedAnnotation(
            code=0, display_name=template, species=template,
            lipid_class=lipid_class_from_template(template), adduct="",
            isotopologue="", carbon=None, double_bonds=None,
            internal_standard=False,
        )
    dimensions = composition_dimensions(composition_attr, phase)
    chosen = [
        values[index]
        for values, index in zip(dimensions, _digits(code, dimensions))
    ]
    species = _species_name(template, chosen)
    # MATLAB joins the first three axes as the display suffix. The monoisotopic
    # decoration is an empty MATLAB cell, so it drops out of the join and the
    # isotopologue field below is left blank for the unlabelled peak.
    suffix = " ".join(value for value in chosen[:3] if value).strip()
    display_name = f"{species} {suffix}".strip()
    internal_standard = int(code) < 0
    if internal_standard:
        display_name = f"IS: {display_name}"
    isotopologue = chosen[_ISOTOPE_AXIS] if len(chosen) > _ISOTOPE_AXIS else ""
    return DecodedAnnotation(
        code=int(code),
        display_name=display_name,
        species=species,
        lipid_class=lipid_class_from_template(template),
        adduct=chosen[_ADDUCT_AXIS] if chosen else "",
        isotopologue=isotopologue,
        carbon=_integer(chosen[_CARBON_AXIS]) if len(chosen) > _CARBON_AXIS else None,
        double_bonds=(
            _integer(chosen[_DOUBLE_BOND_AXIS])
            if len(chosen) > _DOUBLE_BOND_AXIS else None
        ),
        internal_standard=internal_standard,
    )
