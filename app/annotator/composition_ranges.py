"""Shared configurable carbon/double-bond limits for LASER annotation.

The initial defaults were the inclusive extrema observed in the accepted
``MS_data.xlsx`` tables of four example projects. Ordinary PE now permits
28--50 carbons after broader example-data validation; other defaults retain
their original bounds. A
major-group range and a class range are both applied by default; their
intersection is the effective range.  Users may disable all limits, disable
the major-group layer, opt one class out of its group, or disable filtering for
one class entirely.

The policy constrains the standard LASER matcher, automatic MATLAB engine, and
4D annotators at the theoretical target boundary, before RT-grid fitting and
feature competition.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, cast
from resource_paths import REFERENCE_DATA_DIRECTORY


SINGLE_ACYL = "Single acyl chain"
TWO_CHAIN = "Two acyl/alkyl chains"
SPHINGOLIPID = "Sphingolipids"
THREE_CHAIN = "Three acyl/alkyl chains"
FOUR_CHAIN = "Four acyl chains"
MAJOR_GROUP_ORDER = (
    SINGLE_ACYL,
    TWO_CHAIN,
    SPHINGOLIPID,
    THREE_CHAIN,
    FOUR_CHAIN,
)

DEFAULT_COMPOSITION_RANGE_PATH = (
    REFERENCE_DATA_DIRECTORY / "standard_lipid_composition_ranges.tsv"
)


@dataclass(frozen=True)
class CompositionRange:
    """One inclusive total-composition rectangle."""

    carbon_min: int
    carbon_max: int
    db_min: int
    db_max: int

    def __post_init__(self) -> None:
        values = (
            self.carbon_min, self.carbon_max, self.db_min, self.db_max,
        )
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise TypeError("composition-range limits must be integers")
        if self.carbon_min < 0 or self.db_min < 0:
            raise ValueError("composition-range minima cannot be negative")
        if self.carbon_min > self.carbon_max:
            raise ValueError("carbon minimum cannot exceed carbon maximum")
        if self.db_min > self.db_max:
            raise ValueError("double-bond minimum cannot exceed maximum")

    def contains(self, carbon: int, double_bonds: int) -> bool:
        return bool(
            self.carbon_min <= int(carbon) <= self.carbon_max
            and self.db_min <= int(double_bonds) <= self.db_max
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "carbon_min": self.carbon_min,
            "carbon_max": self.carbon_max,
            "db_min": self.db_min,
            "db_max": self.db_max,
        }


@dataclass(frozen=True)
class ClassCompositionRange:
    """Class range plus its relationship to the major-group layer."""

    lipid_class: str
    major_group: str
    limits: CompositionRange
    enabled: bool = True
    use_group_range: bool = True


@dataclass(frozen=True)
class CompositionRangeDecision:
    allowed: bool
    lipid_class: str
    major_group: str
    reason: str
    class_limits: CompositionRange | None
    group_limits: CompositionRange | None


@dataclass(frozen=True)
class CompositionRangeSettings:
    """Immutable, cache-safe standard-lipidomics composition settings."""

    enabled: bool
    use_group_ranges: bool
    group_ranges: tuple[tuple[str, CompositionRange], ...]
    class_ranges: tuple[ClassCompositionRange, ...]
    source: str = "bundled example-data extrema"

    def group_lookup(self) -> dict[str, CompositionRange]:
        return dict(self.group_ranges)

    def class_lookup(self) -> dict[str, ClassCompositionRange]:
        return {
            canonical_lipid_class(rule.lipid_class): rule
            for rule in self.class_ranges
        }

    def decide(
        self,
        lipid_class: object,
        carbon: int | float | None,
        double_bonds: int | float | None,
        *,
        species: object = "",
    ) -> CompositionRangeDecision:
        canonical = canonical_lipid_class(lipid_class, species=species)
        rule = self.class_lookup().get(canonical)
        major_group = (
            rule.major_group if rule is not None
            else major_group_for_class(canonical)
        )
        if not self.enabled:
            return CompositionRangeDecision(
                True, canonical, major_group, "all_ranges_disabled",
                rule.limits if rule else None, None,
            )
        if rule is not None and not rule.enabled:
            return CompositionRangeDecision(
                True, canonical, major_group, "class_filter_disabled",
                rule.limits, None,
            )
        configured_group_limits = (
            self.group_lookup().get(major_group)
            if self.use_group_ranges
            and (rule is None or rule.use_group_range)
            else None
        )
        if rule is None and configured_group_limits is None:
            return CompositionRangeDecision(
                True, canonical, major_group, "no_configured_range", None, None,
            )
        try:
            carbon_value = int(carbon)  # type: ignore[arg-type]
            db_value = int(double_bonds)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return CompositionRangeDecision(
                False, canonical, major_group, "composition_missing",
                rule.limits if rule else None, None,
            )

        group_limits = configured_group_limits
        if group_limits is not None and not group_limits.contains(
            carbon_value, db_value
        ):
            return CompositionRangeDecision(
                False, canonical, major_group, "outside_major_group_range",
                rule.limits if rule else None, group_limits,
            )
        if rule is not None and not rule.limits.contains(carbon_value, db_value):
            return CompositionRangeDecision(
                False, canonical, major_group, "outside_class_range",
                rule.limits, group_limits,
            )
        return CompositionRangeDecision(
            True, canonical, major_group, "within_composition_range",
            rule.limits if rule else None, group_limits,
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "use_group_ranges": self.use_group_ranges,
            "source": self.source,
            "groups": {
                name: limits.to_dict() for name, limits in self.group_ranges
            },
            "classes": {
                rule.lipid_class: {
                    "major_group": rule.major_group,
                    "enabled": rule.enabled,
                    "use_group_range": rule.use_group_range,
                    **rule.limits.to_dict(),
                }
                for rule in self.class_ranges
            },
        }


def _range_from_mapping(
    values: Mapping[str, object], fallback: CompositionRange,
) -> CompositionRange:
    return CompositionRange(
        int(cast(Any, values.get("carbon_min", fallback.carbon_min))),
        int(cast(Any, values.get("carbon_max", fallback.carbon_max))),
        int(cast(Any, values.get("db_min", fallback.db_min))),
        int(cast(Any, values.get("db_max", fallback.db_max))),
    )


def composition_range_settings_from_dict(
    values: Mapping[str, object] | None,
    *,
    defaults: CompositionRangeSettings | None = None,
) -> CompositionRangeSettings:
    """Overlay a compact headless/UI mapping onto bundled defaults."""
    base = defaults or DEFAULT_COMPOSITION_RANGE_SETTINGS
    if not values:
        return base
    raw_groups = values.get("groups", {})
    raw_classes = values.get("classes", {})
    group_overrides = raw_groups if isinstance(raw_groups, Mapping) else {}
    class_overrides = raw_classes if isinstance(raw_classes, Mapping) else {}

    groups = []
    for name, limits in base.group_ranges:
        override = group_overrides.get(name, {})
        groups.append((
            name,
            _range_from_mapping(
                override if isinstance(override, Mapping) else {}, limits
            ),
        ))
    classes = []
    for rule in base.class_ranges:
        override = class_overrides.get(rule.lipid_class, {})
        override = override if isinstance(override, Mapping) else {}
        classes.append(ClassCompositionRange(
            lipid_class=rule.lipid_class,
            major_group=str(override.get("major_group", rule.major_group)),
            limits=_range_from_mapping(override, rule.limits),
            enabled=bool(override.get("enabled", rule.enabled)),
            use_group_range=bool(
                override.get("use_group_range", rule.use_group_range)
            ),
        ))
    return CompositionRangeSettings(
        enabled=bool(values.get("enabled", base.enabled)),
        use_group_ranges=bool(
            values.get("use_group_ranges", base.use_group_ranges)
        ),
        group_ranges=tuple(groups),
        class_ranges=tuple(classes),
        source=str(values.get("source", base.source)),
    )


def canonical_lipid_class(value: object, *, species: object = "") -> str:
    """Map LASER/MATLAB spellings onto the example-data class labels."""
    label = " ".join(str(value).strip().split())
    compact = label.replace(" ", "")
    positional = re.fullmatch(r"([12]-acyl)(LPC|LPE)", compact, re.I)
    if positional:
        return f"{positional.group(1).lower()} {positional.group(2).upper()}"
    marker = re.fullmatch(r"(.+)-(O\+P|O|P|d|t|m|q)", label)
    if marker:
        label = f"{marker.group(1)}[{marker.group(2)}]"
    species_text = str(species)
    hydroxy = re.search(r"\)-(\d*OH)$", species_text, re.I)
    if hydroxy and label in {"AC", "FA"}:
        suffix = hydroxy.group(1).upper()
        return f"{label}-{suffix}"
    return label


_SPHINGO_ROOTS = {
    "C1P", "Cer", "GA1", "GA2", "GD1", "GD2", "GD3", "GM1",
    "GM2", "GM3", "GM4", "GQ1", "GT1", "HexCer", "Hex2Cer",
    "Hex3Cer", "Lyso HexCer", "Lyso Hex2Cer", "Lyso Hex3Cer",
    "Lyso SM", "PE-Cer", "PI-Cer", "S1P", "SM", "SM4", "SPH",
}
_SINGLE_ROOTS = {
    "AC", "CE", "FA", "FOH", "LPA", "LPC", "LPE", "LPG", "LPI",
    "LPS", "MG", "retinyl ester",
}
_TWO_ROOTS = {
    "BMP", "CDP-DG", "DG", "DGDG", "DLCL", "FAHFA", "MGDG", "PA",
    "PC", "PE", "PG", "PI", "PS", "WE",
}
_THREE_ROOTS = {"LUB", "MLCL", "SLBPA", "TG", "X1"}
_FOUR_ROOTS = {"BPA", "CL", "NAPS"}


def _root_class(value: str) -> str:
    label = re.sub(r"\[[^]]+\]$", "", value)
    label = re.sub(r"-(?:\d*)OH$", "", label, flags=re.I)
    label = re.sub(r"^[12]-acyl\s*", "", label, flags=re.I)
    return label


def major_group_for_class(value: object) -> str:
    canonical = canonical_lipid_class(value)
    root = _root_class(canonical)
    if root in _SPHINGO_ROOTS or re.match(r"^(?:GA|GD|GM|GQ|GT)\d", root):
        return SPHINGOLIPID
    if root in _THREE_ROOTS:
        return THREE_CHAIN
    if root in _FOUR_ROOTS:
        return FOUR_CHAIN
    if root in _TWO_ROOTS:
        return TWO_CHAIN
    if root in _SINGLE_ROOTS:
        return SINGLE_ACYL
    return "Other / unclassified"


def _load_bundled_defaults(path: Path) -> CompositionRangeSettings:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    class_rules = tuple(ClassCompositionRange(
        lipid_class=row["lipid_class"],
        major_group=row["major_group"],
        limits=CompositionRange(
            int(row["carbon_min"]), int(row["carbon_max"]),
            int(row["db_min"]), int(row["db_max"]),
        ),
    ) for row in rows)
    group_ranges: list[tuple[str, CompositionRange]] = []
    for group in MAJOR_GROUP_ORDER:
        members = [rule.limits for rule in class_rules if rule.major_group == group]
        if not members:
            continue
        group_ranges.append((group, CompositionRange(
            min(item.carbon_min for item in members),
            max(item.carbon_max for item in members),
            min(item.db_min for item in members),
            max(item.db_max for item in members),
        )))
    return CompositionRangeSettings(
        enabled=True,
        use_group_ranges=True,
        group_ranges=tuple(group_ranges),
        class_ranges=class_rules,
        source=str(path),
    )


DEFAULT_COMPOSITION_RANGE_SETTINGS = _load_bundled_defaults(
    DEFAULT_COMPOSITION_RANGE_PATH
)
