"""MATLAB-compatible class, composition, stream, and adduct policies.

The historical MATLAB pipeline separates the chemical database from the
project data. ``lipids.tsv`` describes which structural class variants are
active in NP or RP, which measurements/adducts may represent them, and their
phase-specific priority. The analysis script separately limits every fatty
acyl R group by carbon and double-bond count.

LASER's standard mode uses these definitions as an allow-list before mass
matching. This module is intentionally not imported by the 4D workflow.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import cast
import numpy as np
import pandas as pd

from annotator.composition_ranges import CompositionRangeSettings
from resource_paths import REFERENCE_DATA_DIRECTORY


MATLAB_POLICY_COLUMNS = (
    "matlab_class",
    "class_name",
    "r_group",
    "composition_prefix",
    "composition_suffix",
    "composition_base",
    "priority_np",
    "priority_rp",
    "combined_class",
    "measurement",
    "chemistry",
)

DEFAULT_FA_CARBON_RANGE = (8, 42)
DEFAULT_FA_DB_RANGE = (0, 8)
DEFAULT_MATLAB_POLICY_PATH = REFERENCE_DATA_DIRECTORY / "matlab_lipid_policy.tsv"

_FA_RANGE_RE = re.compile(
    r"R_group\.FA\.range'\s*,\s*\[\s*"
    r"(-?\d+)\s*,\s*(-?\d+)\s*;\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)

_MEASUREMENT_ADDUCTS = {
    "pos_0": "[M+H]+",
    "pos2_0": "[M+2H]2+",
    "pos3_0": "[M+3H]3+",
    "pos4_0": "[M+4H]4+",
    "pos_NH3": "[M+NH4]+",
    "pos2_NH3": "[M+2NH4]2+",
    "pos_Na": "[M+Na]+",
    "pos_K": "[M+K]+",
    "pos_Li": "[M+Li]+",
    "pos2_Ca": "[M+Ca]2+",
    "pos2_Mg": "[M+Mg]2+",
    "pos_0-H2O": "[M+H-H2O]+",
    "neg_0": "[M-H]-",
    "neg2_0": "[M-2H]2-",
    "neg3_0": "[M-3H]3-",
    "neg4_0": "[M-4H]4-",
    "neg_HCl": "[M+Cl]-",
    "neg_CH2O2": "[M+HCOO]-",
    "neg_Na": "[M-2H+Na]-",
    "neg_0-H2O": "[M-H-H2O]-",
}

_HYDROXY_SUFFIX_RE = re.compile(r"-(?:\d+)?OH$", re.IGNORECASE)


def discover_matlab_policy_file(directory: str | Path) -> Path | None:
    """Find ``lipids.tsv`` in a MATLAB project or one of its ancestors."""
    start = Path(directory).expanduser()
    if start.is_file():
        start = start.parent
    for root in (start, *start.parents):
        candidates = (
            root / "lipids.tsv",
            root / "MS_database" / "lipids.tsv",
            root / "Meta" / "MS_database" / "lipids.tsv",
            root / "Data" / "Meta" / "MS_database" / "lipids.tsv",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def discover_matlab_fa_ranges(
    directory: str | Path,
) -> tuple[tuple[int, int], tuple[int, int], Path | None]:
    """Read ``R_group.FA.range`` from the nearest MATLAB analysis script."""
    start = Path(directory).expanduser()
    if start.is_file():
        start = start.parent
    for root in (start, *start.parents):
        direct = root / "analyse_MS_data.m"
        candidates = [direct] if direct.is_file() else []
        annotation = root / "Processing" / "Annotation" / "analyse_MS_data.m"
        if annotation.is_file():
            candidates.append(annotation)
        for candidate in candidates:
            text = candidate.read_text(encoding="latin1", errors="replace")
            match = _FA_RANGE_RE.search(text)
            if match:
                values = tuple(int(value) for value in match.groups())
                return (
                    (values[0], values[1]),
                    (values[2], values[3]),
                    candidate,
                )
    return DEFAULT_FA_CARBON_RANGE, DEFAULT_FA_DB_RANGE, None


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _is_internal_standard(row: pd.Series) -> bool:
    value = row.get("is_is", False)
    if pd.isna(value):
        value = False
    if isinstance(value, str):
        value = value.strip().casefold() in {"1", "true", "yes"}
    return bool(value) or str(row.get("name", "")).endswith(" (IS)")


def _normalise_suffix(value) -> str:
    suffix = _clean(value)
    if suffix == "*":
        return ""
    return suffix.replace("_", "-")


def _normalise_base(value) -> str:
    return _clean(value).rstrip("-")


def _matlab_policy_class(
    matlab_class: str,
    prefix: str,
    suffix: str,
    base: str,
) -> str:
    if prefix in {"ML", "DL"} and matlab_class == "CL":
        return f"{prefix}CL"
    if prefix:
        return f"{prefix.strip()} {matlab_class}"
    if suffix:
        return f"{matlab_class}{suffix}"
    if base:
        return f"{matlab_class}[{base}]"
    return matlab_class


def _r_group_count(chemistry: str) -> int:
    match = re.search(r"R(\d*)$", chemistry)
    if not match:
        return 0
    return int(match.group(1) or 1)


def _parse_measurement_token(token: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(NP|RP)_(.+)", token.strip())
    if not match:
        return None
    phase, measurement = match.groups()
    adduct = _MEASUREMENT_ADDUCTS.get(measurement)
    if adduct is None:
        return None
    polarity = "Pos" if "_pos" in f"_{measurement}" else "Neg"
    return f"{phase}_{polarity}", adduct


def load_matlab_lipid_policy(
    path: str | Path,
    *,
    fa_carbon_range: tuple[int, int] = DEFAULT_FA_CARBON_RANGE,
    fa_db_range: tuple[int, int] = DEFAULT_FA_DB_RANGE,
    range_source: str | Path | None = None,
) -> pd.DataFrame:
    """Normalise a MATLAB ``lipids.tsv`` into one row per allowed measurement."""
    source = Path(path).expanduser()
    raw = pd.read_csv(source, sep="\t", encoding="latin1", dtype=object)
    if raw.shape[1] < len(MATLAB_POLICY_COLUMNS):
        raise ValueError(
            f"MATLAB lipid policy {source} has {raw.shape[1]} columns; "
            f"expected at least {len(MATLAB_POLICY_COLUMNS)}."
        )
    raw = raw.iloc[:, : len(MATLAB_POLICY_COLUMNS)].copy()
    raw.columns = MATLAB_POLICY_COLUMNS

    rules: list[dict] = []
    current_class = ""
    current_r_group = ""
    for rule_index, row in raw.iterrows():
        explicit_class = _clean(row["matlab_class"])
        if explicit_class:
            current_class = explicit_class
            current_r_group = _clean(row["r_group"])
        elif _clean(row["r_group"]):
            current_r_group = _clean(row["r_group"])
        if not current_class:
            continue

        prefix = _clean(row["composition_prefix"])
        suffix = _normalise_suffix(row["composition_suffix"])
        base = _normalise_base(row["composition_base"])
        policy_class = _matlab_policy_class(current_class, prefix, suffix, base)
        n_r_groups = _r_group_count(_clean(row["chemistry"]))
        for token in _clean(row["measurement"]).split(";"):
            parsed = _parse_measurement_token(token)
            if parsed is None:
                continue
            stream, adduct = parsed
            phase = stream.split("_", 1)[0]
            priority = pd.to_numeric(
                row["priority_rp" if phase == "RP" else "priority_np"],
                errors="coerce",
            )
            if current_r_group == "FA" and n_r_groups:
                carbon_min = cast(float, fa_carbon_range[0] * n_r_groups)
                carbon_max = cast(float, fa_carbon_range[1] * n_r_groups)
                db_min = cast(float, fa_db_range[0] * n_r_groups)
                db_max = cast(float, fa_db_range[1] * n_r_groups)
            else:
                carbon_min = carbon_max = db_min = db_max = np.nan
            rules.append(
                {
                    "policy_class": policy_class,
                    "matlab_class": current_class,
                    "standard_stream": stream,
                    "adduct": adduct,
                    "phase_priority": priority,
                    "r_group": current_r_group,
                    "n_r_groups": n_r_groups,
                    "carbon_min": carbon_min,
                    "carbon_max": carbon_max,
                    "db_min": db_min,
                    "db_max": db_max,
                    "measurement_token": token.strip(),
                    "combined_class": _clean(row["combined_class"]),
                    "rule_index": int(rule_index),
                    "policy_source": str(source),
                    "composition_range_source": (
                        str(range_source) if range_source else "MATLAB default"
                    ),
                }
            )
    policy = pd.DataFrame(rules)
    if policy.empty:
        raise ValueError(
            f"MATLAB lipid policy {source} contains no recognised measurements."
        )
    policy.attrs["policy_source"] = str(source)
    policy.attrs["fa_carbon_range"] = tuple(fa_carbon_range)
    policy.attrs["fa_db_range"] = tuple(fa_db_range)
    policy.attrs["range_source"] = (
        str(range_source) if range_source else "MATLAB default"
    )
    return policy


def _reference_policy_labels(row: pd.Series) -> list[str]:
    lipid_class = str(row["class"])
    name = str(row["name"])
    base = _clean(row.get("base"))
    policy_root = (
        f"Lyso {lipid_class}"
        if name.casefold().startswith(f"lyso {lipid_class.casefold()}")
        else lipid_class
    )

    if lipid_class in {"MLCL", "DLCL"}:
        exact = lipid_class
    elif name.casefold().startswith("1-acyl "):
        exact = f"1-acyl {lipid_class}"
    elif name.casefold().startswith("2-acyl "):
        exact = f"2-acyl {lipid_class}"
    elif _HYDROXY_SUFFIX_RE.search(lipid_class):
        exact = lipid_class
    elif base:
        exact = f"{policy_root}[{base}]"
    else:
        exact = policy_root

    labels = [exact]
    # MATLAB uses a combined ether/plasmalogen policy in phases where the two
    # compositions cannot be distinguished reliably.
    if base in {"O", "P"}:
        labels.append(f"{policy_root}[O+P]")
    # Positional lyso isomers are likewise combined in the lower-priority
    # phase while remaining distinct in the preferred phase.
    if exact.startswith(("1-acyl ", "2-acyl ")):
        labels.append(lipid_class)
    return labels


def apply_composition_range_filter(
    reference: pd.DataFrame,
    settings: CompositionRangeSettings,
    *,
    stream: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply composition limits independently of the MATLAB class policy."""
    allowed_indices: list[int] = []
    audit_rows: list[dict] = []
    for index, row in reference.iterrows():
        carbon = pd.to_numeric(row.get("carbon"), errors="coerce")
        db = pd.to_numeric(row.get("db"), errors="coerce")
        lipid_class = _reference_policy_labels(row)[0]
        if _is_internal_standard(row):
            decision = None
            allowed = True
            reason = "internal_standard_protected"
        else:
            decision = settings.decide(
                lipid_class, carbon, db, species=row.get("name", "")
            )
            allowed = decision.allowed
            reason = decision.reason
        if allowed:
            allowed_indices.append(index)
        audit_rows.append(
            {
                "stream": stream,
                "name": row.get("name", ""),
                "class": row.get("class", ""),
                "adduct": row.get("adduct", ""),
                "carbon": row.get("carbon"),
                "db": row.get("db"),
                "policy_result": reason,
                "composition_range_class": (
                    decision.lipid_class if decision is not None else lipid_class
                ),
                "composition_range_group": (
                    decision.major_group if decision is not None else ""
                ),
                "composition_range_result": reason,
            }
        )
    return (
        reference.loc[allowed_indices].reset_index(drop=True),
        pd.DataFrame(audit_rows),
    )


def apply_matlab_lipid_policy(
    reference: pd.DataFrame,
    stream: str,
    policy: pd.DataFrame,
    *,
    composition_ranges: CompositionRangeSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a reference to positive-priority MATLAB rules for one stream.

    Returns the allowed reference and a row-level audit of every reference
    entry considered for that stream.
    """
    stream_rules = policy[policy["standard_stream"].eq(stream)].copy()
    lookup: dict[tuple[str, str], list[pd.Series]] = {}
    for _, rule in stream_rules.iterrows():
        lookup.setdefault(
            (str(rule["policy_class"]).casefold(), str(rule["adduct"])), []
        ).append(rule)

    audit_rows: list[dict] = []
    allowed_indices: list[int] = []
    selected_metadata: dict[int, dict] = {}
    for index, row in reference.iterrows():
        labels = _reference_policy_labels(row)
        candidates: list[tuple[int, pd.Series]] = []
        for label_rank, label in enumerate(labels):
            candidates.extend(
                (label_rank, rule)
                for rule in lookup.get((label.casefold(), str(row["adduct"])), ())
            )

        positive = [
            (label_rank, rule)
            for label_rank, rule in candidates
            if pd.notna(rule["phase_priority"]) and float(rule["phase_priority"]) > 0
        ]
        if positive:
            label_rank, selected = sorted(
                positive,
                key=lambda item: (
                    -float(item[1]["phase_priority"]),
                    item[0],
                    int(item[1]["rule_index"]),
                ),
            )[0]
            carbon = pd.to_numeric(row.get("carbon"), errors="coerce")
            db = pd.to_numeric(row.get("db"), errors="coerce")
            has_range = pd.notna(selected["carbon_min"])
            composition_pass = bool(
                not has_range
                or (
                    pd.notna(carbon)
                    and pd.notna(db)
                    and float(selected["carbon_min"])
                    <= float(carbon)
                    <= float(selected["carbon_max"])
                    and float(selected["db_min"])
                    <= float(db)
                    <= float(selected["db_max"])
                )
            )
            reason = "allowed" if composition_pass else "composition_out_of_range"
            range_decision = None
            is_internal_standard = _is_internal_standard(row)
            if (
                reason == "allowed"
                and composition_ranges is not None
                and not is_internal_standard
            ):
                range_decision = composition_ranges.decide(
                    selected["policy_class"],
                    carbon,
                    db,
                    species=row.get("name", ""),
                )
                if not range_decision.allowed:
                    reason = range_decision.reason
        else:
            selected = None
            composition_pass = False
            range_decision = None
            folded_labels = {label.casefold() for label in labels}
            all_labels = (
                policy["policy_class"].astype(str).str.casefold().isin(folded_labels)
            )
            if not all_labels.any():
                reason = "class_variant_not_defined"
            elif not (all_labels & policy["standard_stream"].eq(stream)).any():
                reason = "stream_not_enabled"
            elif not (
                all_labels
                & policy["standard_stream"].eq(stream)
                & policy["adduct"].astype(str).eq(str(row["adduct"]))
            ).any():
                reason = "adduct_not_enabled"
            else:
                reason = "nonpositive_phase_priority"

        if reason == "allowed" and selected is not None:
            allowed_indices.append(index)
            selected_metadata[index] = {
                "matlab_policy_class": selected["policy_class"],
                "matlab_policy_priority": float(selected["phase_priority"]),
                "matlab_policy_measurement": selected["measurement_token"],
                "matlab_policy_n_r_groups": int(selected["n_r_groups"]),
                "matlab_policy_carbon_range": (
                    f"{selected['carbon_min']:g}-{selected['carbon_max']:g}"
                    if pd.notna(selected["carbon_min"])
                    else ""
                ),
                "matlab_policy_db_range": (
                    f"{selected['db_min']:g}-{selected['db_max']:g}"
                    if pd.notna(selected["db_min"])
                    else ""
                ),
                "matlab_policy_rule_index": int(selected["rule_index"]),
            }
        audit_rows.append(
            {
                "stream": stream,
                "name": row["name"],
                "class": row["class"],
                "adduct": row["adduct"],
                "carbon": row.get("carbon"),
                "db": row.get("db"),
                "candidate_policy_labels": "; ".join(labels),
                "policy_result": reason,
                "selected_policy_class": (
                    selected["policy_class"] if selected is not None else ""
                ),
                "selected_priority": (
                    selected["phase_priority"] if selected is not None else np.nan
                ),
                "composition_range_class": (
                    range_decision.lipid_class if range_decision is not None else ""
                ),
                "composition_range_group": (
                    range_decision.major_group if range_decision is not None else ""
                ),
                "composition_range_result": (
                    range_decision.reason if range_decision is not None else ""
                ),
            }
        )

    filtered = reference.loc[allowed_indices].copy()
    for column in (
        "matlab_policy_class",
        "matlab_policy_priority",
        "matlab_policy_measurement",
        "matlab_policy_n_r_groups",
        "matlab_policy_carbon_range",
        "matlab_policy_db_range",
        "matlab_policy_rule_index",
    ):
        filtered[column] = [
            selected_metadata[index][column] for index in allowed_indices
        ]
    filtered["matlab_policy_applied"] = True
    filtered["matlab_policy_source"] = policy.attrs.get(
        "policy_source",
        policy["policy_source"].iloc[0] if "policy_source" in policy else "",
    )
    filtered["matlab_policy_range_source"] = policy.attrs.get(
        "range_source",
        policy["composition_range_source"].iloc[0]
        if "composition_range_source" in policy
        else "",
    )
    return filtered.reset_index(drop=True), pd.DataFrame(audit_rows)
