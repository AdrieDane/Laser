"""Pure state transitions for the runtime-validated annotation subset."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class AnnotationState:
    annotation: list[list[float]]
    rtp_grouped: list[float | None]
    accepted: list[int]
    inliers: list[dict[str, Any]]
    outliers: list[dict[str, Any]]
    interpolation: list[int]

    def clone(self) -> "AnnotationState":
        return copy.deepcopy(self)


def resolve_annotation_siblings(
    selected_compound: int,
    comb: Iterable[Any],
    priority: Iterable[float],
) -> list[int]:
    """Return legacy-ordered, non-negative-priority MATLAB compound indices."""
    groups = list(comb)
    priorities = list(priority)
    if len(groups) != len(priorities):
        raise ValueError("comb and priority lengths differ")
    if selected_compound < 1 or selected_compound > len(groups):
        raise IndexError("selected compound is outside MATLAB 1-based range")
    group = groups[selected_compound - 1]
    if group is None or (isinstance(group, str) and not group.strip()):
        return [selected_compound] if priorities[selected_compound - 1] >= 0 else []
    ordered = [selected_compound]
    ordered.extend(i for i, value in enumerate(groups, start=1)
                   if i != selected_compound and value == group)
    return [compound for compound in ordered if priorities[compound - 1] >= 0]


def _radix_lengths(composition_attr: dict[str, Any]) -> list[int]:
    text_comp = composition_attr["text_comp"]
    return [len(item) if isinstance(item, list) else 1 for item in text_comp]


def _num_base(value: int, bases: list[int]) -> list[int]:
    if value < 0 or any(base <= 0 for base in bases):
        raise ValueError("invalid mixed-radix input")
    result = []
    remaining = value
    divisors = []
    for index in range(len(bases)):
        divisor = math.prod(bases[index + 1 :])
        divisors.append(divisor)
    for divisor in divisors:
        digit = remaining // divisor
        result.append(digit)
        remaining -= digit * divisor
    return result


def _membership_keys(entry: dict[str, Any], phase: str) -> set[tuple[int, ...]]:
    phases = entry["phase"] if isinstance(entry["phase"], list) else [entry["phase"]]
    codes = entry["code"] if isinstance(entry["code"], list) else [entry["code"]]
    coordinates = entry["coordinates"]
    if coordinates and not isinstance(coordinates[0], list):
        coordinates = [coordinates]
    return {
        (abs(math.floor(codes[i])), *(abs(math.floor(1000 * x)) for x in coordinates[i]))
        for i, item_phase in enumerate(phases)
        if item_phase == phase
    }


def _append_inliers(
    state: AnnotationState,
    phase: str,
    command: dict[str, Any],
    composition_attrs: tuple[dict[str, Any], ...],
) -> None:
    working = command["selected_working_annotation"]
    if working and not isinstance(working[0], list):
        working = [working]
    coordinates = command["viewer_X_selected"]
    if coordinates and not isinstance(coordinates[0], list):
        coordinates = [coordinates]
    for sibling_position, entry in enumerate(state.inliers, start=1):
        existing = _membership_keys(entry, phase)
        bases = _radix_lengths(composition_attrs[0])
        phases = entry["phase"] if isinstance(entry["phase"], list) else [entry["phase"]]
        codes = entry["code"] if isinstance(entry["code"], list) else [entry["code"]]
        stored_coordinates = entry["coordinates"]
        if stored_coordinates and not isinstance(stored_coordinates[0], list):
            stored_coordinates = [stored_coordinates]
        for row, coordinate in zip(working, coordinates, strict=True):
            if int(row[1]) != sibling_position:
                continue
            code = int(row[2])
            digits = _num_base(abs(code) - 1, bases)
            if digits[0] != 0 or digits[2] != 0:
                continue
            key = (abs(math.floor(code)), *(abs(math.floor(1000 * x)) for x in coordinate))
            if key in existing:
                continue
            phases.append(phase)
            codes.append(code)
            stored_coordinates.append(list(coordinate))
            existing.add(key)
        entry["phase"] = phases
        entry["code"] = codes
        entry["coordinates"] = stored_coordinates


def validate_state(state: AnnotationState, compound_count: int) -> None:
    if len(state.annotation) != len(state.rtp_grouped):
        raise ValueError("annotation and RTp row counts differ")
    if not (len(state.accepted) == len(state.inliers) == len(state.outliers)
            == len(state.interpolation) == compound_count):
        raise ValueError("compound-state vector lengths differ")
    if any(len(row) != 2 for row in state.annotation):
        raise ValueError("dense annotation rows must have two columns")


def edit_membership(
    state: AnnotationState,
    *,
    compound_position: int,
    action: str,
    rows_to_add: list[dict[str, Any]],
    inlier_rows_to_remove_matlab: Iterable[int] = (),
    outlier_rows_to_remove_matlab: Iterable[int] = (),
) -> AnnotationState:
    """Apply already-resolved membership mutations using MATLAB 1-based row indices."""
    validate_state(state, len(state.accepted))
    if compound_position < 1 or compound_position > len(state.inliers):
        raise IndexError("compound position is outside the resolved set")
    if action not in {"inlier", "outlier", "remove"}:
        raise ValueError(f"unknown membership action: {action}")
    output = state.clone()
    position = compound_position - 1
    _remove_membership_rows(output.inliers[position], inlier_rows_to_remove_matlab)
    _remove_membership_rows(output.outliers[position], outlier_rows_to_remove_matlab)
    if action != "remove":
        target = output.inliers[position] if action == "inlier" else output.outliers[position]
        for row in rows_to_add:
            phase = row["phase"]
            code = int(row["code"])
            coordinates = list(row["coordinates"])
            target["phase"] = _as_list(target["phase"])
            target["code"] = _as_list(target["code"])
            existing_coordinates = target["coordinates"]
            if existing_coordinates and not isinstance(existing_coordinates[0], list):
                existing_coordinates = [existing_coordinates]
            target["phase"].append(phase)
            target["code"].append(code)
            existing_coordinates.append(coordinates)
            target["coordinates"] = existing_coordinates
    return output


def _remove_membership_rows(entry: dict[str, Any], rows_matlab: Iterable[int]) -> None:
    indices = sorted({int(row) - 1 for row in rows_matlab}, reverse=True)
    phases = _as_list(entry["phase"])
    codes = _as_list(entry["code"])
    coordinates = entry["coordinates"]
    if coordinates and not isinstance(coordinates[0], list):
        coordinates = [coordinates]
    for index in indices:
        if index < 0 or index >= len(codes):
            raise IndexError("membership row is outside MATLAB 1-based range")
        del phases[index]
        del codes[index]
        del coordinates[index]
    entry["phase"] = phases
    entry["code"] = codes
    entry["coordinates"] = coordinates


def apply_acceptance(
    state: AnnotationState,
    *,
    phase: str,
    compound_set: tuple[int, ...],
    target_rows_matlab: tuple[int, ...],
    accept: bool,
    command: dict[str, Any] | None = None,
    composition_attrs: tuple[dict[str, Any], ...] = (),
) -> AnnotationState:
    """Apply acceptance to a compact fixture state and return a new state."""
    validate_state(state, len(compound_set))
    output = state.clone()
    output.accepted = [int(accept)] * len(compound_set)
    output.annotation = [[0, 0] for _ in output.annotation]
    output.rtp_grouped = [None for _ in output.rtp_grouped]
    if not accept:
        return output
    if command is None or tuple(int(item) for item in _as_list(command["compound_set"])) != compound_set:
        raise ValueError("accept ON requires a matching captured command")
    if command["phase"] != phase:
        raise ValueError("command phase mismatch")
    row_lookup = {row: index for index, row in enumerate(target_rows_matlab)}
    working = command["selected_working_annotation"]
    if working and not isinstance(working[0], list):
        working = [working]
    data_rows = _as_list(command["data_rows_matlab"])
    predicted = _as_list(command["predicted_RTp_selected"])
    if not (len(working) == len(data_rows) == len(predicted)):
        raise ValueError("captured command arrays differ in length")
    for row, data_row, rtp in zip(working, data_rows, predicted, strict=True):
        sibling_position = int(row[1])
        if sibling_position < 1 or sibling_position > len(compound_set):
            raise ValueError("working row has invalid sibling position")
        try:
            local_row = row_lookup[int(data_row)]
        except KeyError as exc:
            raise ValueError("command writes outside compact target rows") from exc
        output.annotation[local_row] = [compound_set[sibling_position - 1], int(row[2])]
        output.rtp_grouped[local_row] = rtp
    output.interpolation = [0] * len(compound_set)
    if composition_attrs:
        _append_inliers(output, phase, command, composition_attrs)
    return output


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]
