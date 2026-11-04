"""Batch annotation for conventional XCMS lipidomics data.

This module is deliberately separate from LASER's 4D workflow. It accepts the
four peak-group streams used by the MATLAB pipeline (NP/RP, positive/negative),
normalises them through the existing feature loader, and reuses the same
reference and matching engine without requiring mobility or MS/MS.
"""

from __future__ import annotations
from dataclasses import dataclass
import io
from pathlib import Path
import re
from typing import BinaryIO, Mapping

import numpy as np
import pandas as pd

from scientific_defaults import SCIENTIFIC_DEFAULTS

from annotator.export import build_annotated_features_export
from annotator.features import load_feature_list, read_raw
from annotator.grid_coherence import GRID_COHERENCE_COLUMNS, score_grid_coherence
from annotator.matching import annotate
from annotator.composition_ranges import CompositionRangeSettings
from annotator.np_rt import (
    attach_np_rt_reference,
    combine_np_ether_plasmalogen_labels,
    fit_np_rt_anchor_alignment,
)
from standard_lipidomics.policy import (
    apply_composition_range_filter,
    apply_matlab_lipid_policy,
)
from standard_lipidomics.mass_calibration import apply_standard_mass_calibration, freeze_mass_calibration, restore_calibrated_xcms_provenance


STREAM_ORDER = ("NP_Neg", "NP_Pos", "RP_Neg", "RP_Pos")
_STREAM_RE = re.compile(r"(NP|RP)[_-]?(Neg|Pos)", re.IGNORECASE)

DEFAULT_STANDARD_PPM_TOLERANCE, MAX_STANDARD_PPM_TOLERANCE = SCIENTIFIC_DEFAULTS.standard_ppm_tolerance, 20.0
DEFAULT_GRID_COHERENCE_THRESHOLD = SCIENTIFIC_DEFAULTS.grid_coherence_threshold
MIN_CLASS_RT_ANCHORS = 5
MIN_REFERENCE_RT_OBSERVATIONS = 5
MAX_CLASS_RT_OFFSET_MIN = 0.75
MIN_CLASS_RT_IMPROVEMENT_MIN = 0.02
MIN_CLASS_RT_RELATIVE_IMPROVEMENT = 0.10
AUTOMATIC_SCORE_LIMIT = 1.0
HIGH_CONFIDENCE_SCORE_LIMIT = 0.75
HIGH_CONFIDENCE_REFERENCE_OBSERVATIONS = 10

POLICY_EXPORT_COLUMNS = (
    "matlab_policy_applied",
    "matlab_policy_class",
    "matlab_policy_priority",
    "matlab_policy_measurement",
    "matlab_policy_n_r_groups",
    "matlab_policy_carbon_range",
    "matlab_policy_db_range",
    "matlab_policy_rule_index",
    "matlab_policy_source",
    "matlab_policy_range_source",
)


def _validate_standard_ppm_tolerance(value: float) -> float:
    tolerance = float(value)
    if not 0.0 < tolerance <= MAX_STANDARD_PPM_TOLERANCE:
        raise ValueError(
            "Standard-lipidomics m/z tolerance must be greater than zero "
            f"and no larger than {MAX_STANDARD_PPM_TOLERANCE:g} ppm."
        )
    return tolerance


def _validate_grid_coherence_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Grid-coherence threshold must be between 0 and 1.")
    return threshold


@dataclass(frozen=True)
class StandardStream:
    """Chromatography and ionisation metadata for one XCMS feature stream."""

    name: str
    chromatography: str
    polarity: str

    @property
    def reference_polarity(self) -> str:
        return "pos" if self.polarity == "Pos" else "neg"

    @property
    def uses_reference_rt(self) -> bool:
        # The bundled LASER RT database is explicitly reverse-phase.
        return self.chromatography == "RP"


def parse_standard_stream(value: str | Path) -> StandardStream:
    """Infer ``NP_Neg``/``NP_Pos``/``RP_Neg``/``RP_Pos`` from a filename."""
    name = Path(value).name
    match = _STREAM_RE.search(name)
    if not match:
        raise ValueError(
            f"Could not infer an NP/RP and Neg/Pos stream from {name!r}. "
            "Use a filename containing NP_Neg, NP_Pos, RP_Neg, or RP_Pos."
        )
    chromatography = match.group(1).upper()
    polarity = match.group(2).title()
    stream_name = f"{chromatography}_{polarity}"
    return StandardStream(stream_name, chromatography, polarity)


def discover_standard_streams(directory: str | Path) -> dict[str, Path]:
    """Find MATLAB-style peak-group lists from a project or Annotation folder."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"Standard-lipidomics input folder not found: {root}")

    metadata_root = root / "metadata"
    search_directories = [metadata_root, root] if metadata_root.is_dir() else [root]
    for candidate_root in (root, *root.parents):
        metadata = candidate_root / "metadata"
        if metadata.is_dir() and metadata not in search_directories:
            search_directories.append(metadata)
        candidate = candidate_root / "Processing" / "Annotation"
        if candidate.is_dir() and candidate not in search_directories:
            search_directories.append(candidate)

    for search_directory in search_directories:
        found: dict[str, Path] = {}
        paths = sorted(search_directory.glob("*.txt")) + sorted(
            search_directory.glob("*.tsv")
        )
        for path in paths:
            try:
                stream = parse_standard_stream(path)
            except ValueError:
                continue
            if "peakgrouplist" in path.name.casefold():
                found.setdefault(stream.name, path)
        if found:
            return {name: found[name] for name in STREAM_ORDER if name in found}
    return {}


def _source_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, bytes):
        return source
    data = source.read()
    return data.encode("utf-8") if isinstance(data, str) else bytes(data)


def load_standard_feature_list(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    """Load a MATLAB-pipeline XCMS list without treating group counts as samples."""
    data = _source_bytes(source)
    raw = read_raw(io.BytesIO(data))
    group_columns = [
        str(column)
        for column in raw.columns
        if str(column).casefold().startswith("group_")
    ]
    features = load_feature_list(
        io.BytesIO(data),
        rt_unit="minutes",
        exclude_intensity_columns=[str(column) for column in raw.columns if str(column).casefold().startswith(("group_", "laser_"))],
    )
    features.attrs["xcms_group_columns"] = group_columns
    features.attrs["sample_columns"] = [
        column
        for column in features.columns
        if column
        not in {
            "feature_id",
            "mz",
            "rt",
            "mobility",
            "mobility_unit",
            "npeaks",
            "intensity",
            "quantification_measure",
            "mzmin",
            "mzmax",
            "rtmin",
            "rtmax",
            "into",
            "intb",
            "maxo",
            "sn",
        }
        and not str(column).endswith("::detected")
    ]
    # XCMS writes one detected-feature count per sample group followed by the
    # individual sample intensity columns in the same group order. Preserve a
    # compact, validated map for post-annotation statistics. The group-count
    # columns themselves remain excluded from intensities.
    group_sizes = []
    for column in group_columns:
        values = pd.to_numeric(raw[column], errors="coerce")
        group_sizes.append(int(values.max()) if values.notna().any() else 0)
    sample_columns = list(features.attrs["sample_columns"])
    if group_sizes and sum(group_sizes) == len(sample_columns):
        sample_group_map: dict[str, str] = {}
        offset = 0
        for group, size in zip(group_columns, group_sizes, strict=True):
            for sample in sample_columns[offset : offset + size]:
                sample_group_map[str(sample)] = str(group)
            offset += size
        features.attrs["xcms_group_sizes"] = dict(
            zip(group_columns, group_sizes, strict=True)
        )
        features.attrs["sample_group_map"] = sample_group_map
        features.attrs["sample_group_mapping_status"] = "validated"
    else:
        features.attrs["xcms_group_sizes"] = dict(
            zip(group_columns, group_sizes, strict=True)
        )
        features.attrs["sample_group_map"] = {}
        features.attrs["sample_group_mapping_status"] = (
            "unavailable" if not group_columns else "group_size_mismatch"
        )
    return restore_calibrated_xcms_provenance(features, raw)


def _median_absolute_deviation(values: pd.Series | np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan")
    centre = float(np.median(data))
    return float(np.median(np.abs(data - centre)))


def _leave_one_out_offset_error(residuals: np.ndarray) -> float:
    """Median absolute error after fitting an offset without the held-out node."""
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < 3:
        return float("nan")
    errors = [
        abs(value - np.median(np.delete(residuals, index)))
        for index, value in enumerate(residuals)
    ]
    return float(np.median(errors))


def fit_automatic_class_rt_models(
    mass_pairs: pd.DataFrame,
    rt_tolerance_min: float,
) -> pd.DataFrame:
    """Fit guarded class/adduct RT offsets from unambiguous grid anchors.

    The original historical RT remains the baseline. A class correction is
    activated only when at least five distinct grid nodes span carbon or
    double-bond space and leave-one-out prediction improves materially. This
    prevents a few chance mass matches from moving an entire lipid class.
    """
    columns = [
        "class",
        "adduct",
        "model_type",
        "offset_min",
        "anchor_nodes",
        "carbon_span",
        "db_span",
        "raw_median_abs_error_min",
        "adjusted_loo_median_abs_error_min",
        "improvement_min",
        "relative_improvement",
        "anchor_mad_min",
        "validated",
    ]
    if mass_pairs.empty:
        return pd.DataFrame(columns=columns)

    pairs = mass_pairs.copy()
    pairs["raw_rt_error"] = pd.to_numeric(
        pairs["feat_rt"], errors="coerce"
    ) - pd.to_numeric(pairs["ref_rt"], errors="coerce")
    identity_count = pairs.groupby("feat_feature_id")["name"].transform("nunique")
    seed_window = max(0.75, 1.5 * float(rt_tolerance_min))
    seeds = pairs[
        identity_count.eq(1)
        & pairs["raw_rt_error"].abs().le(seed_window)
        & pd.to_numeric(pairs["n_RT"], errors="coerce").ge(
            MIN_REFERENCE_RT_OBSERVATIONS
        )
        & pairs["carbon"].notna()
        & pairs["db"].notna()
    ].copy()
    # Split/repeated features at one reference node must not inflate support.
    seeds = seeds.sort_values(
        "raw_rt_error", key=lambda values: values.abs()
    ).drop_duplicates(["class", "adduct", "name"], keep="first")

    models = []
    for (lipid_class, adduct), group in seeds.groupby(["class", "adduct"], sort=False):
        residuals = group["raw_rt_error"].to_numpy(float)
        offset = float(np.median(residuals))
        raw_error = float(np.median(np.abs(residuals)))
        adjusted_error = _leave_one_out_offset_error(residuals)
        improvement = raw_error - adjusted_error
        relative_improvement = improvement / raw_error if raw_error > 0 else 0.0
        carbon_span = float(group["carbon"].max() - group["carbon"].min())
        db_span = float(group["db"].max() - group["db"].min())
        validated = bool(
            len(group) >= MIN_CLASS_RT_ANCHORS
            and (carbon_span >= 4 or db_span >= 2)
            and abs(offset) <= MAX_CLASS_RT_OFFSET_MIN
            and improvement >= MIN_CLASS_RT_IMPROVEMENT_MIN
            and relative_improvement >= MIN_CLASS_RT_RELATIVE_IMPROVEMENT
        )
        models.append(
            {
                "class": lipid_class,
                "adduct": adduct,
                "model_type": (
                    "validated_class_offset" if validated else "historical_reference"
                ),
                "offset_min": offset if validated else 0.0,
                "anchor_nodes": int(len(group)),
                "carbon_span": carbon_span,
                "db_span": db_span,
                "raw_median_abs_error_min": raw_error,
                "adjusted_loo_median_abs_error_min": adjusted_error,
                "improvement_min": improvement,
                "relative_improvement": relative_improvement,
                "anchor_mad_min": _median_absolute_deviation(residuals),
                "validated": validated,
            }
        )
    return pd.DataFrame(models, columns=columns)


def _apply_standard_rt_models(
    mass_pairs: pd.DataFrame,
    models: pd.DataFrame,
    *,
    ppm_tolerance: float,
    rt_tolerance_min: float,
    use_rt: bool,
    identity_column: str = "name",
    filter_passing: bool = True,
) -> pd.DataFrame:
    """Apply validated class offsets and optionally rebuild passing ranks.

    ``filter_passing=False`` is used to score the complete hard-window
    candidate neighbourhood for grid coherence before the final annotation
    ranking is rebuilt.  The default preserves the original matcher contract.
    """
    pairs = mass_pairs.copy()
    if pairs.empty:
        return pairs.reset_index(drop=True)
    if use_rt and not models.empty:
        lookup = models.set_index(["class", "adduct"])
        keys = pd.MultiIndex.from_arrays([pairs["class"], pairs["adduct"]])
        aligned = lookup.reindex(keys)
        pairs["rt_model_type"] = (
            aligned["model_type"].fillna("historical_reference").to_numpy()
        )
        pairs["rt_model_offset_min"] = aligned["offset_min"].fillna(0.0).to_numpy()
        pairs["rt_model_anchor_nodes"] = (
            aligned["anchor_nodes"].fillna(0).astype(int).to_numpy()
        )
        pairs["rt_model_validated"] = (
            aligned["validated"].fillna(False).astype(bool).to_numpy()
        )
    else:
        pairs["rt_model_type"] = (
            "historical_reference" if use_rt else "mass_only_no_reference_rt"
        )
        pairs["rt_model_offset_min"] = 0.0
        pairs["rt_model_anchor_nodes"] = 0
        pairs["rt_model_validated"] = False

    if use_rt:
        pairs["rt_expected"] = pairs["ref_rt"] + pairs["rt_model_offset_min"]
        pairs["rt_error"] = pairs["feat_rt"] - pairs["rt_expected"]
        rt_component = (pairs["rt_error"] / rt_tolerance_min) ** 2
        rt_pass = pairs["rt_error"].abs().le(rt_tolerance_min)
    else:
        pairs["rt_expected"] = np.nan
        pairs["rt_error"] = np.nan
        rt_component = 0.0
        rt_pass = pd.Series(True, index=pairs.index)
    pairs["has_rt_evidence"] = bool(use_rt)
    pairs["score_2d"] = np.sqrt(
        (pairs["ppm_error"] / ppm_tolerance) ** 2 + rt_component
    )
    pairs["score"] = pairs["score_2d"]
    if not filter_passing:
        return pairs.reset_index(drop=True)
    passing = pairs[pairs["ppm_error"].abs().le(ppm_tolerance) & rt_pass].copy()
    if passing.empty:
        return passing

    if identity_column not in passing:
        identity_column = "name"
    # NP O/P and positional-lyso aliases may produce multiple theoretical
    # reference rows for the same unresolved chromatographic identity.
    passing = passing.sort_values(
        ["score", "n_RT"], ascending=[True, False], kind="stable"
    ).drop_duplicates(["feat_feature_id", identity_column, "adduct"], keep="first")
    passing["n_alternatives"] = passing.groupby("feat_feature_id")[
        identity_column
    ].transform("nunique")
    passing["rank_for_feature"] = passing.groupby("feat_feature_id").cumcount()

    # Peak support only breaks ties between repeated features for one identity;
    # it never changes whether a feature passes the scientific windows.
    feature_quality = (
        pd.to_numeric(passing.get("feat_npeaks"), errors="coerce")
        .rank(pct=True)
        .fillna(0.5)
    )
    passing["feature_selection_score"] = passing["score"] + 0.10 * (
        1.0 - feature_quality
    )
    passing["rank_for_lipid"] = (
        passing.sort_values(
            ["feature_selection_score", "n_RT"],
            ascending=[True, False],
            kind="stable",
        )
        .groupby([identity_column, "adduct"])
        .cumcount()
        .reindex(passing.index)
    )

    next_score = passing[passing["rank_for_feature"].eq(1)].set_index(
        "feat_feature_id"
    )["score"]
    first_score = passing[passing["rank_for_feature"].eq(0)].set_index(
        "feat_feature_id"
    )["score"]
    margin = next_score.subtract(first_score)
    passing["score_margin_to_next"] = passing["feat_feature_id"].map(margin)
    return passing.reset_index(drop=True)


def _calculate_standard_grid_coherence(
    candidates: pd.DataFrame,
    *,
    ppm_tolerance: float,
    rt_tolerance_min: float,
    threshold: float,
    used_for_annotation: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the complete standard-mode candidate neighbourhood.

    Standard mode has no CCS evidence, so the shared scorer uses only the
    normalized m/z and RT residual dimensions.  Unassessed edge/sparse nodes
    retain ``grid_coherence_pass=True`` and are never removed by the optional
    filter.
    """
    threshold = _validate_grid_coherence_threshold(threshold)
    scored = score_grid_coherence(
        candidates,
        ppm_tolerance=float(ppm_tolerance),
        rt_tolerance=float(rt_tolerance_min),
        ccs_tolerance_percent=1.0,
        use_ccs=False,
        threshold=threshold,
    )
    scored["grid_coherence_threshold"] = threshold
    assessed = (
        scored.get("grid_coherence_assessed", pd.Series(False, index=scored.index))
        .fillna(False)
        .astype(bool)
    )
    outliers = (
        scored.get("grid_coherence_outlier", pd.Series(False, index=scored.index))
        .fillna(False)
        .astype(bool)
    )
    values = (
        pd.to_numeric(
            scored.loc[assessed, "grid_coherence_score"], errors="coerce"
        ).dropna()
        if "grid_coherence_score" in scored
        else pd.Series(dtype=float)
    )
    audit = pd.DataFrame(
        [
            {
                "status": "calculated" if not candidates.empty else "no_candidates",
                "threshold": threshold,
                "ppm_scale": float(ppm_tolerance),
                "rt_scale_min": float(rt_tolerance_min),
                "candidates": int(len(scored)),
                "assessed": int(assessed.sum()),
                "unassessed": int((~assessed).sum()),
                "diagnostic_outliers": int(outliers.sum()),
                "median_score": float(values.median()) if len(values) else np.nan,
                "used_for_annotation": bool(used_for_annotation),
            }
        ]
    )
    return scored, audit


def _copy_standard_grid_coherence(
    target: pd.DataFrame,
    scored_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Copy full-pool coherence values onto the selected annotation rows."""
    if target.empty or scored_candidates.empty:
        return target
    key_columns = [
        column
        for column in ("feat_feature_id", "name", "adduct")
        if column in target.columns and column in scored_candidates.columns
    ]
    if not key_columns:
        return target
    available = [
        column
        for column in [*GRID_COHERENCE_COLUMNS, "grid_coherence_threshold"]
        if column in scored_candidates.columns
    ]
    if not available:
        return target
    lookup = scored_candidates.drop_duplicates(key_columns, keep="first")
    left_keys = pd.MultiIndex.from_frame(target[key_columns].astype(str))
    right_keys = pd.MultiIndex.from_frame(lookup[key_columns].astype(str))
    result = target.copy()
    for column in available:
        result[column] = (
            pd.Series(lookup[column].to_numpy(), index=right_keys)
            .reindex(left_keys)
            .to_numpy()
        )
    return result


def _assign_automatic_status(
    annotations: pd.DataFrame,
    stream: StandardStream,
    *,
    rt_evidence: str | None = None,
    np_alignment_validated: bool = False,
) -> pd.DataFrame:
    """Assign auditable confidence and abstention states to candidates."""
    data = annotations.copy()
    if data.empty:
        return data
    top_feature = data["rank_for_feature"].eq(0)
    top_identity = data["rank_for_lipid"].eq(0)
    ambiguous = data["n_alternatives"].gt(1)
    enough_reference = pd.to_numeric(data["n_RT"], errors="coerce").ge(
        MIN_REFERENCE_RT_OBSERVATIONS
    )
    score_supported = data["score"].le(AUTOMATIC_SCORE_LIMIT)

    data["auto_annotation_status"] = "alternative_identity"
    data.loc[top_feature & ~top_identity, "auto_annotation_status"] = (
        "duplicate_feature"
    )
    data.loc[top_feature & top_identity, "auto_annotation_status"] = "low_confidence"
    rt_evidence = rt_evidence or (
        "rp_reference" if stream.uses_reference_rt else "none"
    )
    if rt_evidence == "none":
        data.loc[top_feature & top_identity, "auto_annotation_status"] = "mass_only"
    elif rt_evidence == "np_grid":
        data.loc[top_feature & top_identity, "auto_annotation_status"] = (
            "np_grid_unvalidated"
        )
        if np_alignment_validated:
            data.loc[top_feature & top_identity, "auto_annotation_status"] = (
                "np_grid_low_confidence"
            )
        data.loc[
            top_feature & top_identity & score_supported & np_alignment_validated,
            "auto_annotation_status",
        ] = "np_grid_supported"
        data.loc[
            top_feature
            & top_identity
            & data["score"].le(HIGH_CONFIDENCE_SCORE_LIMIT)
            & np_alignment_validated,
            "auto_annotation_status",
        ] = "high_confidence_np_grid"
        data.loc[
            top_feature & top_identity & ambiguous,
            "auto_annotation_status",
        ] = "ambiguous_isobar"
    else:
        data.loc[
            top_feature & top_identity & ~enough_reference,
            "auto_annotation_status",
        ] = "low_reference_support"
        data.loc[
            top_feature & top_identity & enough_reference & score_supported,
            "auto_annotation_status",
        ] = "mass_rt_supported"
        data.loc[
            top_feature
            & top_identity
            & enough_reference
            & score_supported
            & data["rt_model_validated"],
            "auto_annotation_status",
        ] = "class_grid_supported"
        data.loc[
            top_feature
            & top_identity
            & enough_reference
            & data["score"].le(HIGH_CONFIDENCE_SCORE_LIMIT)
            & pd.to_numeric(data["n_RT"], errors="coerce").ge(
                HIGH_CONFIDENCE_REFERENCE_OBSERVATIONS
            ),
            "auto_annotation_status",
        ] = "high_confidence_auto"
        data.loc[
            top_feature & top_identity & ambiguous,
            "auto_annotation_status",
        ] = "ambiguous_isobar"

    accepted_statuses = {
        "high_confidence_auto",
        "class_grid_supported",
        "mass_rt_supported",
        "high_confidence_np_grid",
        "np_grid_supported",
    }
    data["automatic_accepted"] = data["auto_annotation_status"].isin(accepted_statuses)
    data["automatic_candidate"] = data["automatic_accepted"]
    data["status"] = data["auto_annotation_status"]
    return data


def _attach_policy_metadata(
    annotations: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    policy_applied: bool,
) -> pd.DataFrame:
    """Attach the selected MATLAB allow-list rule to each candidate."""
    data = annotations.copy()
    if data.empty:
        for column in POLICY_EXPORT_COLUMNS:
            data[column] = pd.Series(dtype=object)
        return data
    if not policy_applied:
        data["matlab_policy_applied"] = False
        for column in POLICY_EXPORT_COLUMNS[1:]:
            data[column] = ""
        return data

    metadata = (
        reference[["name", "adduct", *POLICY_EXPORT_COLUMNS]]
        .drop_duplicates(["name", "adduct"], keep="first")
        .set_index(["name", "adduct"])
    )
    keys = pd.MultiIndex.from_frame(data[["name", "adduct"]].astype(str))
    aligned = metadata.reindex(keys)
    for column in POLICY_EXPORT_COLUMNS:
        data[column] = aligned[column].to_numpy()
    return data


def annotate_standard_stream(
    source: str | Path | bytes | BinaryIO,
    stream: StandardStream | str,
    reference: pd.DataFrame,
    *,
    ppm_tolerance: float = DEFAULT_STANDARD_PPM_TOLERANCE,
    rp_rt_tolerance_min: float = SCIENTIFIC_DEFAULTS.rt_tolerance_min,
    align_rp_rt: bool = False,
    adapt_class_rt: bool = True,
    matlab_policy: pd.DataFrame | None = None,
    composition_ranges: CompositionRangeSettings | None = None,
    np_rt_reference: pd.DataFrame | None = None,
    np_rt_tolerance_min: float = 0.30,
    use_grid_coherence: bool = True,
    grid_coherence_threshold: float = DEFAULT_GRID_COHERENCE_THRESHOLD,
    mass_calibration: object = None,
) -> dict:
    """Annotate one conventional stream without CCS or MS/MS evidence.

    RP uses the bundled RP RT database. NP uses a supplied Lipid-Wizard
    reference plus a guarded dataset anchor offset; without that reference it
    remains mass-only. This distinction is exposed in every result row.
    """
    ppm_tolerance = _validate_standard_ppm_tolerance(ppm_tolerance)
    grid_coherence_threshold = _validate_grid_coherence_threshold(
        grid_coherence_threshold
    )
    stream = parse_standard_stream(stream) if isinstance(stream, str) else stream
    features = apply_standard_mass_calibration(load_standard_feature_list(source), mass_calibration)
    ref = reference[
        reference["polarity"].astype(str).eq(stream.reference_polarity)
    ].copy()
    if matlab_policy is not None:
        ref, policy_audit = apply_matlab_lipid_policy(
            ref,
            stream.name,
            matlab_policy,
            composition_ranges=composition_ranges,
        )
    elif composition_ranges is not None:
        ref, policy_audit = apply_composition_range_filter(
            ref, composition_ranges, stream=stream.name
        )
    else:
        policy_audit = pd.DataFrame()
    use_np_rt = bool(stream.chromatography == "NP" and np_rt_reference is not None)
    np_alignment_validated = False
    np_rt_audit = pd.DataFrame()
    np_rt_alignment = pd.DataFrame()
    if use_np_rt:
        ref, np_rt_audit = attach_np_rt_reference(ref, np_rt_reference, stream.name)
    rt_tolerance_min = (
        float(np_rt_tolerance_min) if use_np_rt else float(rp_rt_tolerance_min)
    )
    baseline = annotate(
        features,
        ref,
        ppm_tol=ppm_tolerance,
        rt_tol=rt_tolerance_min,
        align_rt=align_rp_rt and stream.uses_reference_rt,
        use_rt=stream.uses_reference_rt,
        use_mobility=False,
    )
    if use_np_rt and not baseline["all_pairs"].empty:
        np_columns = [
            "name",
            "adduct",
            "np_rt_identity",
            "np_rt_reference_class",
            "np_rt_reference_seconds",
            "np_rt_reference_source",
        ]
        metadata = ref[np_columns].drop_duplicates(["name", "adduct"], keep="first")
        baseline["all_pairs"] = baseline["all_pairs"].merge(
            metadata, on=["name", "adduct"], how="left", validate="many_to_one"
        )
    rt_models = (
        fit_automatic_class_rt_models(baseline["all_pairs"], rp_rt_tolerance_min)
        if stream.uses_reference_rt and adapt_class_rt and not align_rp_rt
        else pd.DataFrame()
    )
    if use_np_rt:
        np_rt_alignment = fit_np_rt_anchor_alignment(
            baseline["all_pairs"],
            stream=stream.name,
            rt_tolerance_min=rt_tolerance_min,
        )
        alignment = np_rt_alignment.iloc[0]
        rt_models = (
            baseline["all_pairs"][["class", "adduct"]].drop_duplicates().copy()
            if not baseline["all_pairs"].empty
            else pd.DataFrame(columns=["class", "adduct"])
        )
        rt_models["model_type"] = alignment["model_type"]
        rt_models["offset_min"] = float(alignment["offset_min"])
        rt_models["anchor_nodes"] = int(alignment["anchor_nodes"])
        rt_models["validated"] = bool(alignment["validated"])
        np_alignment_validated = bool(alignment["validated"])
        annotations = _apply_standard_rt_models(
            baseline["all_pairs"],
            rt_models,
            ppm_tolerance=ppm_tolerance,
            rt_tolerance_min=rt_tolerance_min,
            # A rejected calibration must not let an unshifted or unreliable
            # grid hide mass candidates. Preserve the previous mass-only
            # diagnostic result and explicitly withhold it below.
            use_rt=np_alignment_validated,
            identity_column="np_rt_identity",
        )
        coherence_input = _apply_standard_rt_models(
            baseline["all_pairs"],
            rt_models,
            ppm_tolerance=ppm_tolerance,
            rt_tolerance_min=rt_tolerance_min,
            use_rt=np_alignment_validated,
            identity_column="np_rt_identity",
            filter_passing=False,
        )
        coherence_identity_column = "np_rt_identity"
        for column in (
            "anchor_classes",
            "reference_rt_span_min",
            "anchor_mad_min",
            "anchor_p90_abs_error_min",
            "validation_reason",
        ):
            annotations[f"np_rt_alignment_{column}"] = alignment[column]
    elif align_rp_rt and stream.uses_reference_rt:
        # Backwards-compatible explicit legacy option. Standard-mode UI and CLI
        # intentionally do not enable this because a global line can hide
        # opposing class shifts.
        annotations = baseline["annotations"].copy()
        annotations["rt_model_type"] = "legacy_global_alignment"
        annotations["rt_model_offset_min"] = (
            annotations["rt_expected"] - annotations["ref_rt"]
        )
        annotations["rt_model_anchor_nodes"] = int(baseline["alignment"][2])
        annotations["rt_model_validated"] = False
        next_score = annotations[annotations["rank_for_feature"].eq(1)].set_index(
            "feat_feature_id"
        )["score"]
        first_score = annotations[annotations["rank_for_feature"].eq(0)].set_index(
            "feat_feature_id"
        )["score"]
        annotations["score_margin_to_next"] = annotations["feat_feature_id"].map(
            next_score.subtract(first_score)
        )
        # The legacy path is retained for callers that explicitly request the
        # old global alignment.  Standard GUI/CLI runs use the class-model path.
        coherence_input = annotations.copy()
        coherence_identity_column = "name"
    else:
        annotations = _apply_standard_rt_models(
            baseline["all_pairs"],
            rt_models,
            ppm_tolerance=ppm_tolerance,
            rt_tolerance_min=rp_rt_tolerance_min,
            use_rt=stream.uses_reference_rt,
        )
        coherence_input = _apply_standard_rt_models(
            baseline["all_pairs"],
            rt_models,
            ppm_tolerance=ppm_tolerance,
            rt_tolerance_min=rp_rt_tolerance_min,
            use_rt=stream.uses_reference_rt,
            filter_passing=False,
        )
        coherence_identity_column = "name"
    scored_coherence, grid_coherence_audit = _calculate_standard_grid_coherence(
        coherence_input,
        ppm_tolerance=ppm_tolerance,
        rt_tolerance_min=rt_tolerance_min,
        threshold=grid_coherence_threshold,
        used_for_annotation=bool(use_grid_coherence),
    )
    # Keep the complete scored pool available to the grid browser.  It is
    # deliberately broader than the final selected annotation table so the
    # diagnostics can explain an excluded/outlier candidate.
    baseline["all_pairs"] = scored_coherence
    annotations = _copy_standard_grid_coherence(annotations, scored_coherence)
    if use_grid_coherence and not annotations.empty:
        outlier_mask = annotations["grid_coherence_outlier"].fillna(False).astype(bool)
        filtered_count = int(outlier_mask.sum())
        if align_rp_rt and stream.uses_reference_rt:
            annotations = annotations.loc[~outlier_mask].copy()
        else:
            kept_pool = scored_coherence[
                scored_coherence["grid_coherence_pass"].fillna(True).astype(bool)
            ].copy()
            annotations = _apply_standard_rt_models(
                kept_pool,
                rt_models,
                ppm_tolerance=ppm_tolerance,
                rt_tolerance_min=rt_tolerance_min,
                use_rt=(
                    np_alignment_validated if use_np_rt else stream.uses_reference_rt
                ),
                identity_column=coherence_identity_column,
            )
            if use_np_rt and not annotations.empty:
                for column in (
                    "anchor_classes",
                    "reference_rt_span_min",
                    "anchor_mad_min",
                    "anchor_p90_abs_error_min",
                    "validation_reason",
                ):
                    annotations[f"np_rt_alignment_{column}"] = alignment[column]
        grid_coherence_audit.loc[0, "filtered_annotations"] = filtered_count
    else:
        grid_coherence_audit.loc[0, "filtered_annotations"] = 0
    annotations = _assign_automatic_status(
        annotations,
        stream,
        rt_evidence=(
            "np_grid"
            if use_np_rt
            else "rp_reference"
            if stream.uses_reference_rt
            else "none"
        ),
        np_alignment_validated=bool(use_np_rt and np_alignment_validated),
    )
    annotations = _attach_policy_metadata(
        annotations,
        ref,
        policy_applied=matlab_policy is not None,
    )
    if use_np_rt and not annotations.empty:
        annotations["reference_identity"] = annotations["name"]
        annotations["name"] = annotations["np_rt_identity"]
    if stream.chromatography == "NP":
        annotations = combine_np_ether_plasmalogen_labels(annotations)
        baseline["all_pairs"] = combine_np_ether_plasmalogen_labels(
            baseline["all_pairs"]
        )
        for table_name, table in (
            ("annotations", annotations),
            ("all_pairs", baseline["all_pairs"]),
        ):
            keys = [
                column
                for column in ("feat_feature_id", "name", "adduct", "isotopologue")
                if column in table
            ]
            collapsed = (
                table.drop_duplicates(keys, keep="first").reset_index(drop=True)
                if not table.empty and keys
                else table
            )
            if table_name == "annotations":
                annotations = collapsed
            else:
                baseline["all_pairs"] = collapsed
        ref = combine_np_ether_plasmalogen_labels(ref)
        if not ref.empty:
            ref = ref.drop_duplicates(["name", "adduct"], keep="first")
    annotations["standard_stream"] = stream.name
    annotations["chromatography"] = stream.chromatography
    annotations["ion_mode"] = stream.reference_polarity
    annotations["rt_evidence_mode"] = (
        "global_aligned_RP_reference"
        if stream.uses_reference_rt and align_rp_rt
        else "automatic_class_RP_reference"
        if stream.uses_reference_rt and adapt_class_rt
        else "RP_reference"
        if stream.uses_reference_rt
        else "anchored_NP_Lipid_Wizard_reference"
        if use_np_rt and np_alignment_validated
        else "unvalidated_NP_reference_mass_only"
        if use_np_rt
        else "mass_only_no_NP_reference"
    )
    return {
        **baseline,
        "annotations": annotations,
        "features": features,
        "reference": ref,
        "stream": stream,
        "rt_models": rt_models,
        "np_rt_alignment": np_rt_alignment,
        "np_rt_audit": np_rt_audit,
        "grid_coherence_audit": grid_coherence_audit,
        "policy_audit": policy_audit,
        # The grid browser reviews candidates against the tolerances this run
        # used. A rejected NP calibration leaves the stream mass-only, so this records
        # what was applied rather than what was available.
        "criteria": {
            "ppm_tolerance": float(ppm_tolerance),
            "rt_tolerance_min": float(rt_tolerance_min),
            "uses_rt_grid": bool(
                stream.uses_reference_rt or (use_np_rt and np_alignment_validated)
            ),
            "mass_calibration": freeze_mass_calibration(mass_calibration),
            "use_grid_coherence": bool(use_grid_coherence),
            "grid_coherence_threshold": float(grid_coherence_threshold),
        },
    }


def _standard_export(result: dict) -> pd.DataFrame:
    annotations = result["annotations"]
    features = result["features"]
    sample_columns = tuple(features.attrs.get("sample_columns", ()))
    exported = build_annotated_features_export(annotations, sample_columns)
    if exported.empty:
        return exported
    metadata = annotations.set_index(["feat_feature_id", "name", "adduct"])
    keys = pd.MultiIndex.from_frame(
        exported[["feature_id", "feature_identity", "adduct"]].astype(str)
    )
    aligned = metadata.reindex(keys)
    leading = pd.DataFrame(
        {
            "stream": aligned["standard_stream"].to_numpy(),
            "chromatography": aligned["chromatography"].to_numpy(),
            "polarity": aligned["ion_mode"].to_numpy(),
            "rt_evidence_mode": aligned["rt_evidence_mode"].to_numpy(),
            "candidate_rank_for_feature": aligned["rank_for_feature"].to_numpy(),
            "candidate_rank_for_lipid": aligned["rank_for_lipid"].to_numpy(),
            "auto_annotation_status": aligned["auto_annotation_status"].to_numpy(),
            "automatic_accepted": aligned["automatic_accepted"].to_numpy(),
            "score_margin_to_next": aligned["score_margin_to_next"].to_numpy(),
            "reference_rt_observations": aligned["n_RT"].to_numpy(),
            "rt_model_type": aligned["rt_model_type"].to_numpy(),
            "rt_model_offset_min": aligned["rt_model_offset_min"].to_numpy(),
            "rt_model_anchor_nodes": aligned["rt_model_anchor_nodes"].to_numpy(),
            "rt_model_validated": aligned["rt_model_validated"].to_numpy(),
            "np_rt_reference_identity": (
                aligned["np_rt_identity"].to_numpy()
                if "np_rt_identity" in aligned
                else ""
            ),
            "np_rt_reference_seconds": (
                aligned["np_rt_reference_seconds"].to_numpy()
                if "np_rt_reference_seconds" in aligned
                else np.nan
            ),
            "np_rt_reference_source": (
                aligned["np_rt_reference_source"].to_numpy()
                if "np_rt_reference_source" in aligned
                else ""
            ),
            "np_rt_alignment_anchor_classes": (
                aligned["np_rt_alignment_anchor_classes"].to_numpy()
                if "np_rt_alignment_anchor_classes" in aligned
                else np.nan
            ),
            "np_rt_alignment_reference_rt_span_min": (
                aligned["np_rt_alignment_reference_rt_span_min"].to_numpy()
                if "np_rt_alignment_reference_rt_span_min" in aligned
                else np.nan
            ),
            "np_rt_alignment_anchor_mad_min": (
                aligned["np_rt_alignment_anchor_mad_min"].to_numpy()
                if "np_rt_alignment_anchor_mad_min" in aligned
                else np.nan
            ),
            "np_rt_alignment_anchor_p90_abs_error_min": (
                aligned["np_rt_alignment_anchor_p90_abs_error_min"].to_numpy()
                if "np_rt_alignment_anchor_p90_abs_error_min" in aligned
                else np.nan
            ),
            "np_rt_alignment_validation_reason": (
                aligned["np_rt_alignment_validation_reason"].to_numpy()
                if "np_rt_alignment_validation_reason" in aligned
                else ""
            ),
            "matlab_policy_applied": aligned["matlab_policy_applied"].to_numpy(),
            "matlab_policy_class": aligned["matlab_policy_class"].to_numpy(),
            "matlab_policy_priority": aligned["matlab_policy_priority"].to_numpy(),
            "matlab_policy_measurement": aligned[
                "matlab_policy_measurement"
            ].to_numpy(),
            "matlab_policy_n_r_groups": aligned["matlab_policy_n_r_groups"].to_numpy(),
            "matlab_policy_carbon_range": aligned[
                "matlab_policy_carbon_range"
            ].to_numpy(),
            "matlab_policy_db_range": aligned["matlab_policy_db_range"].to_numpy(),
            "matlab_policy_rule_index": aligned["matlab_policy_rule_index"].to_numpy(),
            "matlab_policy_source": aligned["matlab_policy_source"].to_numpy(),
            "matlab_policy_range_source": aligned[
                "matlab_policy_range_source"
            ].to_numpy(),
        }
    )
    return pd.concat([leading, exported.reset_index(drop=True)], axis=1)


def annotate_standard_batch(
    sources: Mapping[str, str | Path | bytes | BinaryIO],
    reference: pd.DataFrame,
    *,
    ppm_tolerance: float = DEFAULT_STANDARD_PPM_TOLERANCE,
    rp_rt_tolerance_min: float = SCIENTIFIC_DEFAULTS.rt_tolerance_min,
    align_rp_rt: bool = False,
    adapt_class_rt: bool = True,
    matlab_policy: pd.DataFrame | None = None,
    composition_ranges: CompositionRangeSettings | None = None,
    np_rt_reference: pd.DataFrame | None = None,
    np_rt_tolerance_min: float = 0.30,
    use_grid_coherence: bool = True,
    grid_coherence_threshold: float = DEFAULT_GRID_COHERENCE_THRESHOLD,
    mass_calibrations: Mapping[str, object] | None = None,
) -> dict:
    """Annotate all supplied conventional streams and combine stable exports."""
    if not sources:
        raise ValueError("Supply at least one standard-lipidomics XCMS stream.")
    results: dict[str, dict] = {}
    exports: list[pd.DataFrame] = []
    policy_audits: list[pd.DataFrame] = []
    np_rt_audits: list[pd.DataFrame] = []
    for name in STREAM_ORDER:
        if name not in sources:
            continue
        stream = parse_standard_stream(name)
        result = annotate_standard_stream(
            sources[name],
            stream,
            reference,
            ppm_tolerance=ppm_tolerance,
            rp_rt_tolerance_min=rp_rt_tolerance_min,
            align_rp_rt=align_rp_rt,
            adapt_class_rt=adapt_class_rt,
            matlab_policy=matlab_policy,
            composition_ranges=composition_ranges,
            np_rt_reference=np_rt_reference,
            np_rt_tolerance_min=np_rt_tolerance_min,
            use_grid_coherence=use_grid_coherence,
            grid_coherence_threshold=grid_coherence_threshold,
            mass_calibration=(mass_calibrations or {}).get(name),
        )
        results[name] = result
        exports.append(_standard_export(result))
        if not result["policy_audit"].empty:
            policy_audits.append(result["policy_audit"])
        if not result["np_rt_audit"].empty:
            np_rt_audits.append(result["np_rt_audit"])
    all_candidates = (
        pd.concat(exports, ignore_index=True, sort=False) if exports else pd.DataFrame()
    )
    if all_candidates.empty:
        automatic = best_identifications = equivalent_identifications = all_candidates.copy()
    else:
        automatic = all_candidates[
            all_candidates["automatic_accepted"].fillna(False).astype(bool)
        ].copy()
        evidence_rank = {
            "high_confidence_auto": 3,
            "high_confidence_np_grid": 3,
            "class_grid_supported": 2,
            "np_grid_supported": 2,
            "mass_rt_supported": 1,
        }
        automatic["_evidence_rank"] = (
            automatic["auto_annotation_status"].map(evidence_rank).fillna(0)
        )
        automatic = automatic.sort_values(
            [
                "feature_identity",
                "_evidence_rank",
                "annotation_score",
                "reference_rt_observations",
            ],
            ascending=[True, False, True, False],
            kind="stable",
        )
        automatic["quantitative_role"] = np.where(
            automatic.groupby("feature_identity").cumcount().eq(0),
            "primary",
            "equivalent",
        )
        automatic = automatic.drop(columns="_evidence_rank")
        best_identifications = automatic[
            automatic["quantitative_role"].eq("primary")
        ].reset_index(drop=True)
        equivalent_identifications = automatic[
            automatic["quantitative_role"].eq("equivalent")
        ].reset_index(drop=True)
    return {
        "streams": results,
        "all_candidates": all_candidates,
        "automatic_annotations": automatic.reset_index(drop=True),
        "identified_lipids": best_identifications,
        "equivalent_lipids": equivalent_identifications,
        "policy_audit": (
            pd.concat(policy_audits, ignore_index=True, sort=False)
            if policy_audits
            else pd.DataFrame()
        ),
        "np_rt_audit": (
            pd.concat(np_rt_audits, ignore_index=True, sort=False)
            if np_rt_audits
            else pd.DataFrame()
        ),
        "matlab_policy_applied": matlab_policy is not None,
        "np_rt_reference_applied": np_rt_reference is not None,
        "grid_coherence_enabled": bool(use_grid_coherence),
        "grid_coherence_threshold": float(grid_coherence_threshold),
        "composition_range_settings": (
            composition_ranges.to_dict() if composition_ranges is not None else None
        ),
        "grid_coherence_audit": (
            pd.concat(
                [
                    item["grid_coherence_audit"]
                    for item in results.values()
                    if not item["grid_coherence_audit"].empty
                ],
                ignore_index=True,
                sort=False,
            )
            if any(not item["grid_coherence_audit"].empty for item in results.values())
            else pd.DataFrame()
        ),
    }
