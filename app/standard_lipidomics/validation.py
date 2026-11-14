"""Evaluation-only comparison with independently mapped manual annotations.

Never imported by annotation selection. Candidate-table observations are not
reconstructed pipeline history: absent rows and failed checks are not causal
explanations of an earlier decision. No thresholds or labels are fed back.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable

import pandas as pd


STREAMS = frozenset({"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"})
_IDENTITY = ["stream", "species", "adduct", "internal_standard"]
_EXACT = [*_IDENTITY, "feature_id"]
_MANUAL_REQUIRED = {
    "stream",
    "matlab_species",
    "matlab_class",
    "matlab_adduct",
    "xcms_feature_id",
    "crosswalk_status",
}
_LASER_REQUIRED = {"stream", "name", "class", "adduct", "feat_feature_id"}
MAPPED_CROSSWALK_STATUSES = frozenset({"profile_match", "exact_feature_id"})
_CHECKS = (
    "grid_coherence_assessed",
    "grid_coherence_pass",
    "post_regression_eligible",
    "positional_isomer_selected",
    "automatic_accepted",
    "pe_completion_score",
    "pe_support_count",
    "pe_decision",
    "pe_final_selected",
    "auto_annotation_status",
)
_SUMMARY_FIELDS = (
    "manual_rows",
    "mapped_manual_rows",
    "unresolved_manual_rows",
    "duplicate_mapped_manual_rows",
    "unique_mapped_manual_annotations",
    "manual_same_peak",
    "manual_same_peak_rows",
    "manual_same_lipid_different_peak",
    "manual_peak_other_lipid",
    "manual_mapped_not_matched",
    "conflicting_manual_rows",
    "mapped_manual_pairs_observed_in_candidate_tables",
    "mapped_manual_pairs_absent_from_available_candidate_tables",
    "mapped_manual_pairs_without_diagnostics",
    "laser_rows",
    "unique_laser_annotations",
    "duplicate_laser_rows",
    "laser_same_manual_peak",
    "laser_same_manual_lipid_different_peak",
    "laser_manual_identity_peak_unresolved",
    "laser_not_in_manual_list",
    "percent_mapped_manual_same_peak",
    "percent_all_manual_rows_same_peak",
)


@dataclass(frozen=True)
class AnnotationComparison:
    manual: pd.DataFrame
    laser: pd.DataFrame


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def canonical_feature_id(value) -> str:
    """Undo numeric CSV .0 suffixes, but never collapse text IDs with leading zeros."""
    value = _text(value)
    return re.sub(r"^((?:0|[1-9]\d*))\.0+$", r"\1", value)


def _species(value) -> str:
    value = " ".join(_text(value).split()).removeprefix("IS: ").removesuffix(" (IS)")
    return re.sub(r"\b([12]-acyl)\s+(LPC|LPE)\b", r"\1\2", value)


def _lipid_class(value) -> str:
    return re.sub(r"\[([^\]]+)\]", r"-\1", _text(value)).replace(" ", "")


def _boolean(value) -> bool | None:
    if pd.isna(value) or _text(value) == "":
        return None
    text = _text(value).casefold()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"Invalid boolean flag: {value!r}")


def _standard_flags(names: pd.Series, flags: pd.Series) -> tuple[list, list]:
    standard, conflicts = [], []
    for name, flag in zip(names, flags):
        label = _text(name).startswith("IS: ") or _text(name).endswith(" (IS)")
        observed = _boolean(flag)
        standard.append(label or observed is True)
        conflicts.append(label and observed is False)
    return standard, conflicts


def _require(frame: pd.DataFrame, columns: set[str], kind: str) -> None:
    if not columns.issubset(frame):
        raise ValueError(f"{kind} missing columns: {sorted(columns - set(frame))}")


def _normalise(frame: pd.DataFrame, *, manual: bool) -> pd.DataFrame:
    _require(
        frame,
        _MANUAL_REQUIRED if manual else _LASER_REQUIRED,
        "Manual" if manual else "LASER",
    )
    name, cls, adduct, fid, flag = (
        (
            "matlab_species",
            "matlab_class",
            "matlab_adduct",
            "xcms_feature_id",
            "matlab_is_internal_standard",
        )
        if manual
        else ("name", "class", "adduct", "feat_feature_id", "is_is")
    )
    x = pd.DataFrame(index=frame.index)
    x["source_row"] = range(1, len(frame) + 1)
    x["stream"] = frame.stream.map(_text)
    x["original_name"] = frame[name].map(_text)
    x["species"] = frame[name].map(_species)
    x["lipid_class"] = frame[cls].map(_lipid_class)
    x["adduct"] = frame[adduct].map(_text)
    x["feature_id"] = frame[fid].map(canonical_feature_id)
    flags = (
        frame[flag]
        if flag in frame
        else pd.Series(None, index=frame.index, dtype=object)
    )
    x["internal_standard"], x["standard_flag_conflict"] = _standard_flags(
        frame[name], flags
    )
    unknown = set(x.stream) - STREAMS
    if unknown:
        raise ValueError(f"Unknown measurement streams: {sorted(unknown)}")
    if x[["species", "lipid_class", "adduct"]].eq("").any().any():
        raise ValueError("Annotation names, classes and adducts must not be blank")
    if manual:
        x["crosswalk_status"] = frame.crosswalk_status.map(_text)
        # profile_match: mapped by intensity profile; exact_feature_id: the
        # truth already names the original feature (a Moon .moon document).
        x["mapped"] = x.crosswalk_status.isin(MAPPED_CROSSWALK_STATUSES)
        if (x.mapped & x.feature_id.eq("")).any():
            raise ValueError(
                "A profile-matched manual entry has no original feature ID"
            )
        for c in (
            "matlab_peak_row",
            "profile_correlation",
            "profile_n",
            "crosswalk_reason",
        ):
            if c in frame:
                x[c] = frame[c]
    else:
        if x.feature_id.eq("").any():
            raise ValueError("A LASER annotation has no original feature ID")
        for c in _CHECKS:
            if c in frame:
                x[c] = frame[c]
    return x.reset_index(drop=True)


def _tuples(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    return list(frame[columns].itertuples(index=False, name=None))


def _json(values: Iterable) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def _lookup(frame: pd.DataFrame, keys: list[str], value: str) -> dict:
    result: dict[tuple, set] = {}
    for key, val in zip(_tuples(frame, keys), frame[value]):
        result.setdefault(key, set()).add(val)
    return result


def _candidate_groups(frame: pd.DataFrame | None, needed: set[tuple]) -> dict:
    if frame is None:
        return {}
    x = _normalise(frame, manual=False)
    x = x.loc[[key in needed for key in _tuples(x, _EXACT)]]
    return {k: v for k, v in x.groupby(_EXACT, sort=False, dropna=False)}


def _observed_checks(group: pd.DataFrame | None) -> str:
    counts: dict[str, int] = {}
    if group is not None:
        for row in group.to_dict("records"):
            if (
                _boolean(row.get("grid_coherence_assessed")) is True
                and _boolean(row.get("grid_coherence_pass")) is False
            ):
                counts["grid_coherence_not_passed"] = (
                    counts.get("grid_coherence_not_passed", 0) + 1
                )
            if _boolean(row.get("post_regression_eligible")) is False:
                counts["post_regression_not_eligible"] = (
                    counts.get("post_regression_not_eligible", 0) + 1
                )
            if (
                row["lipid_class"]
                in {"1-acylLPC", "2-acylLPC", "1-acylLPE", "2-acylLPE"}
                and _boolean(row.get("positional_isomer_selected")) is False
            ):
                counts["positional_isomer_not_selected"] = (
                    counts.get("positional_isomer_not_selected", 0) + 1
                )
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def _candidate_evidence(key, mapped, final, pe, supplied) -> dict:
    f, p = final.get(key), pe.get(key)
    if not mapped:
        state = "manual_peak_unresolved"
    elif not supplied:
        state = "diagnostics_not_supplied"
    elif f is not None and p is not None:
        state = "present_in_both_candidate_tables"
    elif f is not None:
        state = "present_in_final_candidate_table"
    elif p is not None:
        state = "present_in_PE_discovery_table"
    else:
        state = "absent_from_available_candidate_tables"
    support = (
        pd.to_numeric(
            p.get("pe_support_count", pd.Series(dtype=float)), errors="coerce"
        )
        if p is not None
        else pd.Series(dtype=float)
    )
    scores = (
        pd.to_numeric(
            p.get("pe_completion_score", pd.Series(dtype=float)), errors="coerce"
        )
        if p is not None
        else pd.Series(dtype=float)
    )
    decisions = (
        p.pe_decision.dropna().map(_text)
        if p is not None and "pe_decision" in p
        else []
    )
    return dict(
        candidate_evidence=state,
        final_candidate_table_supplied=supplied[0] if supplied else False,
        PE_discovery_table_supplied=supplied[1] if supplied else False,
        final_candidate_rows=len(f) if f is not None else 0,
        PE_discovery_rows=len(p) if p is not None else 0,
        recorded_final_check_failures=_observed_checks(f),
        recorded_PE_decisions=_json(v for v in decisions if v),
        best_recorded_PE_score=float(scores.max()) if scores.notna().any() else None,
        max_recorded_PE_local_support=float(support.max())
        if support.notna().any()
        else None,
        all_recorded_PE_support_below_five=(
            bool(support.max() < 5) if len(support) and support.notna().all() else None
        ),
    )


def compare_manual_annotations(
    manual: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    candidates: pd.DataFrame | None = None,
    pe_decisions: pd.DataFrame | None = None,
    dataset: str = "dataset",
    include_in_pooled: bool = True,
) -> AnnotationComparison:
    """Compare exact original features; never alter annotations or source frames.

    Standards are distinct identities. Only harmless numeric-ID and positional-
    lyso spacing aliases are normalized. Unresolved truth never establishes a
    wrong peak. Duplicate rows and conflicting labels remain explicitly visible.
    """
    m, a = _normalise(manual, manual=True), _normalise(annotations, manual=False)
    needed = set(_tuples(m[m.mapped], _EXACT))
    final, pe = (
        _candidate_groups(candidates, needed),
        _candidate_groups(pe_decisions, needed),
    )
    supplied = (candidates is not None, pe_decisions is not None)
    selected = set(_tuples(a, _EXACT))
    manual_exact = set(_tuples(m[m.mapped], _EXACT))
    manual_ids = set(_tuples(m, _IDENTITY))
    mapped_ids = set(_tuples(m[m.mapped], _IDENTITY))
    unresolved_ids = set(_tuples(m[~m.mapped], _IDENTITY))
    chosen_features = _lookup(a, _IDENTITY, "feature_id")
    chosen_labels = a.assign(
        _label=[
            f"{n}{' (IS)' if s else ''}|{d}"
            for n, d, s in a[["species", "adduct", "internal_standard"]].itertuples(
                index=False, name=None
            )
        ]
    )
    labels_on_feature = _lookup(chosen_labels, ["stream", "feature_id"], "_label")
    manual_feature_counts = (
        m[m.mapped].drop_duplicates(_EXACT).groupby(["stream", "feature_id"]).size()
    )
    m["duplicate_manual_entry"] = m.mapped & m.duplicated(_EXACT)
    a["duplicate_laser_annotation"] = a.duplicated(_EXACT)
    identities = a.assign(_identity=_tuples(a, _IDENTITY))
    a["multiple_labels_on_feature"] = (
        identities.groupby(["stream", "feature_id"])
        ._identity.transform("nunique")
        .gt(1)
    )
    records = []
    for row in m.to_dict("records"):
        key = tuple(row[c] for c in _EXACT)
        identity = key[:-1]
        mapped = row["mapped"]
        same = mapped and key in selected
        other_peak = mapped and not same and identity in chosen_features
        other_label = (
            mapped
            and not same
            and (row["stream"], row["feature_id"]) in labels_on_feature
        )
        if not mapped:
            outcome = "manual_peak_unresolved"
        elif same:
            outcome = "same_manual_peak"
        elif other_peak and other_label:
            outcome = "different_peak_and_other_lipid_on_manual_peak"
        elif other_peak:
            outcome = "same_lipid_different_peak"
        elif other_label:
            outcome = "other_lipid_on_manual_peak"
        else:
            outcome = "manual_lipid_not_selected"
        records.append(
            dict(
                dataset=dataset,
                include_in_pooled=include_in_pooled,
                same_manual_peak=bool(same),
                same_lipid_different_peak=bool(other_peak),
                other_lipid_on_manual_peak=bool(other_label),
                outcome=outcome,
                conflicting_manual_feature=bool(
                    mapped
                    and manual_feature_counts.get((row["stream"], row["feature_id"]), 0)
                    > 1
                ),
                laser_features_for_lipid=_json(chosen_features.get(identity, [])),
                laser_labels_on_manual_peak=_json(
                    labels_on_feature.get((row["stream"], row["feature_id"]), [])
                )
                if mapped
                else "[]",
                **_candidate_evidence(
                    key, mapped, final, pe, supplied if any(supplied) else ()
                ),
            )
        )
    # Explicit column schemas preserve well-formed exports for empty inputs.
    empty_manual_columns = [
        "dataset",
        "include_in_pooled",
        "same_manual_peak",
        "same_lipid_different_peak",
        "other_lipid_on_manual_peak",
        "outcome",
        "conflicting_manual_feature",
        "laser_features_for_lipid",
        "laser_labels_on_manual_peak",
        *_candidate_evidence(None, False, {}, {}, ()).keys(),
    ]
    m = pd.concat([m, pd.DataFrame(records, columns=empty_manual_columns)], axis=1)
    laser_records = []
    for key in _tuples(a, _EXACT):
        identity = key[:-1]
        comparison = (
            "same_manual_peak"
            if key in manual_exact
            else "same_manual_lipid_different_peak"
            if identity in mapped_ids
            else "manual_identity_peak_unresolved"
            if identity in manual_ids
            else "not_in_manual_list"
        )
        laser_records.append(
            dict(
                dataset=dataset,
                include_in_pooled=include_in_pooled,
                comparison=comparison,
                has_unresolved_manual_entry=identity in unresolved_ids,
            )
        )
    a = pd.concat(
        [
            a,
            pd.DataFrame(
                laser_records,
                columns=[
                    "dataset",
                    "include_in_pooled",
                    "comparison",
                    "has_unresolved_manual_entry",
                ],
            ),
        ],
        axis=1,
    )
    return AnnotationComparison(m, a)


def summarize_comparison(
    comparison: AnnotationComparison,
    *,
    group_by: tuple[str, ...] = ("dataset", "stream"),
    pooled_only: bool = False,
) -> pd.DataFrame:
    """Summarize unique matched pairs while retaining raw row/duplicate counts."""
    m, a = comparison.manual, comparison.laser
    if not group_by or not set(group_by).issubset(set(m) & set(a)):
        raise ValueError(
            "Summary groups must be nonempty and present in both comparison tables"
        )
    if pooled_only:
        m, a = m[m.include_in_pooled.eq(True)], a[a.include_in_pooled.eq(True)]
    grouped_manual = {
        k: v for k, v in m.groupby(list(group_by), sort=True, dropna=False)
    }
    grouped_laser = {
        k: v for k, v in a.groupby(list(group_by), sort=True, dropna=False)
    }
    rows = []
    for key in sorted(set(grouped_manual) | set(grouped_laser)):
        x, y = grouped_manual.get(key, m.iloc[:0]), grouped_laser.get(key, a.iloc[:0])
        unique = x[x.mapped.eq(True) & x.duplicate_manual_entry.eq(False)]
        calls = y[y.duplicate_laser_annotation.eq(False)]
        matched = int(unique.same_manual_peak.sum())
        available = unique.candidate_evidence.isin(
            {
                "present_in_both_candidate_tables",
                "present_in_final_candidate_table",
                "present_in_PE_discovery_table",
            }
        )
        row = dict(zip(group_by, key if isinstance(key, tuple) else (key,)))
        row.update(
            manual_rows=len(x),
            mapped_manual_rows=int(x.mapped.sum()),
            unresolved_manual_rows=int(x.mapped.eq(False).sum()),
            duplicate_mapped_manual_rows=int(x.duplicate_manual_entry.sum()),
            unique_mapped_manual_annotations=len(unique),
            manual_same_peak=matched,
            manual_same_peak_rows=int(x.same_manual_peak.sum()),
            manual_same_lipid_different_peak=int(
                unique.same_lipid_different_peak.sum()
            ),
            manual_peak_other_lipid=int(unique.other_lipid_on_manual_peak.sum()),
            manual_mapped_not_matched=len(unique) - matched,
            conflicting_manual_rows=int(x.conflicting_manual_feature.sum()),
            mapped_manual_pairs_observed_in_candidate_tables=int(available.sum()),
            mapped_manual_pairs_absent_from_available_candidate_tables=int(
                unique.candidate_evidence.eq(
                    "absent_from_available_candidate_tables"
                ).sum()
            ),
            mapped_manual_pairs_without_diagnostics=int(
                unique.candidate_evidence.eq("diagnostics_not_supplied").sum()
            ),
            laser_rows=len(y),
            unique_laser_annotations=len(calls),
            duplicate_laser_rows=int(y.duplicate_laser_annotation.sum()),
            laser_same_manual_peak=int(calls.comparison.eq("same_manual_peak").sum()),
            laser_same_manual_lipid_different_peak=int(
                calls.comparison.eq("same_manual_lipid_different_peak").sum()
            ),
            laser_manual_identity_peak_unresolved=int(
                calls.comparison.eq("manual_identity_peak_unresolved").sum()
            ),
            laser_not_in_manual_list=int(
                calls.comparison.eq("not_in_manual_list").sum()
            ),
            percent_mapped_manual_same_peak=100 * matched / len(unique)
            if len(unique)
            else None,
            percent_all_manual_rows_same_peak=100
            * int(x.same_manual_peak.sum())
            / len(x)
            if len(x)
            else None,
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*group_by, *_SUMMARY_FIELDS])
