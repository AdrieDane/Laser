"""MATLAB-equivalent resolved feature names from dense annotation codes."""

from __future__ import annotations

import re
import numpy as np

from .project import LegacyProject


def _values(value) -> list[str]:
    array = np.asarray(value, dtype=object).reshape(-1)
    return [str(item) if np.asarray(item).size else "" for item in array]


def resolved_compound_name(
    template: str, composition_attr: dict, phase: str, code: int
) -> str:
    """Port the relevant `write_compound_name([compound code])` path."""
    if not code:
        return template
    dimensions = [_values(value) for value in composition_attr["text_comp"]]
    columns = _values(composition_attr.get("column", []))
    if columns and len(dimensions[0]) == len(columns):
        dimensions[0] = [
            value for value, column in zip(dimensions[0], columns, strict=True)
            if column == phase
        ]
    dimensions = [values or [""] for values in dimensions]
    bases = [len(values) for values in dimensions]
    remainder = abs(int(code)) - 1
    digits = []
    for index in range(len(bases)):
        weight = int(np.prod(bases[index + 1 :], dtype=np.int64))
        digits.append(remainder // weight)
        remainder %= weight
    chosen = [values[index] for values, index in zip(dimensions, digits, strict=True)]
    name = template
    if "x:x" in name and len(chosen) >= 5:
        name = name.replace("x:x", f"{chosen[3]}:{chosen[4]}")
    elif re.search(r"x\)$", name) and len(chosen) >= 4:
        name = re.sub(r"x\)$", f"{chosen[3]})", name)
    suffix = " ".join(value for value in chosen[:3] if value).strip()
    resolved = f"{name} {suffix}".strip()
    return f"IS: {resolved}" if code < 0 else resolved


def compound_display_template(project: LegacyProject, index: int) -> str:
    """Port ``write_compound_name([compound, 0])`` for one catalog row."""

    fields = project.compounds.legacy_fields
    decorations = np.asarray(fields.get("composition", []), dtype=object)
    r_groups = np.asarray(fields.get("R_group", []), dtype=object).reshape(-1)

    def text(value) -> str:
        array = np.asarray(value, dtype=object).reshape(-1)
        return str(array[0]) if array.size else ""

    parts = ["", "", ""]
    if decorations.ndim == 2 and index < decorations.shape[0]:
        parts = [text(value) for value in decorations[index, :3]]
    r_group = text(r_groups[index]) if index < r_groups.size else ""
    base = project.compounds.classes[index]
    if r_group in {"FA", "FAfull", "ST"}:
        return f"{parts[0]}{base}{parts[1]}({parts[2]}x:x)".strip()
    if r_group == "isoprene":
        return f"{parts[0]}{base}{parts[1]}({parts[2]}x)".strip()
    return f"{parts[0]}{base}{parts[1]}{parts[2]}".strip()
