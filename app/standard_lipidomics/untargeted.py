"""Post-annotation differential analysis for standard XCMS lipidomics.

This module deliberately starts *after* standard lipid annotation. Accepted
feature/identity pairs are removed before any group comparison is calculated.
Local lipid-class neighbourhoods are interpretation hints only; they never
create an annotation or alter the accepted annotation tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, ttest_ind

from scientific_defaults import SCIENTIFIC_DEFAULTS
from standard_lipidomics.untargeted_adducts import (
    EVIDENCE_VERSION, adduct_options, search_adduct_evidence, attach_adduct_summaries, SearchProgress,
)
from standard_lipidomics.untargeted_series import attach_lipid_series
from standard_lipidomics.untargeted_compounds import assign_compound_ids


_NON_BIOLOGICAL_GROUP_TOKENS = {
    "blank",
    "blanco",
    "qc",
    "pool",
    "is",
    "standard",
    "calibrant",
    "calibration",
    "reference",
}
_CH2_MASS = 14.015650064
_C13_C12_MASS_DIFFERENCE = 1.00335483507
_PROTON_MASS = 1.007276466621
_POSITIVE_ADDUCT_OFFSETS = (
    ("[M+H]+", _PROTON_MASS),
    ("[M+NH4]+", 18.033825553),
    ("[M+Na]+", 22.989218),
)
_NEGATIVE_ADDUCT = ("[M-H]-", -_PROTON_MASS)
_ATOMIC_MASS = {
    "H": 1.00782503223,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
}
_SPECIMEN_ID_COLUMNS = (
    "Biological Sample ID",
    "Biological Sample",
    "Sample ID",
    "Specimen ID",
    "Subject ID",
    "Patient ID",
    "Animal ID",
    "Individual ID",
    "Pair ID",
)


@dataclass(frozen=True)
class UntargetedComparisonCriteria:
    """Reproducible gates for one two-group untargeted comparison."""

    group_a: str
    group_b: str
    p_value_threshold: float = SCIENTIFIC_DEFAULTS.untargeted_p_value_threshold
    min_abs_log2_fold_change: float = (
        SCIENTIFIC_DEFAULTS.untargeted_min_abs_log2_fold_change
    )
    min_detection_percent: float = SCIENTIFIC_DEFAULTS.untargeted_min_detection_percent
    cluster_min_neighbors: int = SCIENTIFIC_DEFAULTS.untargeted_cluster_min_neighbors
    cluster_min_dominance: float = SCIENTIFIC_DEFAULTS.untargeted_cluster_min_dominance
    ch2_mass_tolerance_da: float = SCIENTIFIC_DEFAULTS.untargeted_ch2_mass_tolerance_da
    ch2_rt_tolerance_min: float = SCIENTIFIC_DEFAULTS.untargeted_ch2_rt_tolerance_min
    cross_mode_ppm_tolerance: float = SCIENTIFIC_DEFAULTS.standard_ppm_tolerance
    opposite_polarity_rt_tolerance_min: float = (
        SCIENTIFIC_DEFAULTS.untargeted_opposite_polarity_rt_tolerance_min
    )
    isotope_rt_tolerance_min: float = (
        SCIENTIFIC_DEFAULTS.untargeted_isotope_rt_tolerance_min
    )

    def __post_init__(self) -> None:
        if not self.group_a or not self.group_b or self.group_a == self.group_b:
            raise ValueError("Choose two different sample groups.")
        if not 0.0 < float(self.p_value_threshold) <= 1.0:
            raise ValueError(
                "The raw p-value threshold must be greater than zero and at most one."
            )
        if float(self.min_abs_log2_fold_change) < 0.0:
            raise ValueError(
                "The minimum absolute log2 fold change cannot be negative."
            )
        if not 0.0 < float(self.min_detection_percent) <= 100.0:
            raise ValueError("The minimum detection percentage must be in (0, 100].")
        if int(self.cluster_min_neighbors) < 1:
            raise ValueError("At least one cluster neighbour is required.")
        if not 0.0 <= float(self.cluster_min_dominance) <= 1.0:
            raise ValueError("Cluster dominance must be between zero and one.")
        if float(self.cross_mode_ppm_tolerance) <= 0.0:
            raise ValueError("Cross-mode ppm tolerance must be positive.")
        if float(self.ch2_mass_tolerance_da) <= 0.0:
            raise ValueError("CH2 mass tolerance must be positive.")
        if float(self.ch2_rt_tolerance_min) <= 0.0:
            raise ValueError("CH2 RT tolerance must be positive.")
        if float(self.opposite_polarity_rt_tolerance_min) <= 0.0:
            raise ValueError("Opposite-polarity RT tolerance must be positive.")
        if float(self.isotope_rt_tolerance_min) <= 0.0:
            raise ValueError("Isotope RT tolerance must be positive.")


def _group_label(group: str) -> str:
    label = re.sub(r"^group[_ -]*\d+[_ -]*", "", str(group), flags=re.I)
    return re.sub(r"[_-]+", " ", label).strip() or str(group)


def is_biological_group(group: str) -> bool:
    """Return False for QC, blank, pool, IS and calibration groups."""
    tokens = set(re.findall(r"[a-z0-9]+", _group_label(group).casefold()))
    return not bool(tokens & _NON_BIOLOGICAL_GROUP_TOKENS)


def _metadata_workbooks(project: str | Path) -> list[Path]:
    root = Path(project).expanduser().resolve()
    metadata_root = root / "metadata"
    directories = [metadata_root, root] if metadata_root.is_dir() else [root]
    if root.is_dir():
        directories.append(root / "Processing" / "Annotation")
    for parent in root.parents:
        metadata = parent / "metadata"
        if metadata not in directories:
            directories.append(metadata)
        candidate = parent / "Processing" / "Annotation"
        if candidate not in directories:
            directories.append(candidate)
    if root.is_dir():
        from standard_lipidomics.workflow import discover_standard_streams
        directories.extend(path.parent for path in discover_standard_streams(root).values())
    paths: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.xlsx")):
            if path.name.casefold() == "ms_data.xlsx" or path.name.startswith("~$"):
                continue
            if path not in paths:
                paths.append(path)
    return paths


def attach_excel_sample_metadata(
    result: dict, project: str | Path | None
) -> pd.DataFrame:
    """Attach sample groups/types from the phase Excel metadata workbooks.

    ``File Name`` is joined exactly to the XCMS intensity-column header and
    ``Mode`` selects NP/RP and polarity. No filename-derived group assignment
    is performed. The returned audit records coverage for each stream.
    """
    columns = [
        "stream",
        "metadata_workbook",
        "sample_columns",
        "matched_samples",
        "mapping_status",
    ]
    if not project:
        return pd.DataFrame(columns=columns)
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    for path in _metadata_workbooks(project):
        try:
            frame = pd.read_excel(path)
        except (OSError, ValueError, ImportError):
            continue
        frame.columns = [str(column).strip() for column in frame.columns]
        required = {"File Name", "Mode", "Group Label", "Sample Type"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["_metadata_workbook"] = str(path)
        frames.append(frame)
        sources.append(str(path))
    if not frames:
        return pd.DataFrame(columns=columns)
    metadata = pd.concat(frames, ignore_index=True, sort=False)
    audit_rows: list[dict] = []
    for stream, stream_result in result.get("streams", {}).items():
        features = stream_result.get("features", pd.DataFrame())
        if not isinstance(features, pd.DataFrame):
            continue
        samples = [
            str(column)
            for column in features.attrs.get("sample_columns", ())
            if column in features
        ]
        selected = metadata[
            metadata["Mode"].astype(str).str.strip().eq(str(stream))
        ].drop_duplicates("File Name", keep="first")
        lookup = selected.set_index(selected["File Name"].astype(str))
        matched = [sample for sample in samples if sample in lookup.index]
        mapping = {
            sample: str(lookup.loc[sample, "Group Label"]).strip() for sample in matched
        }
        type_map = {
            sample: str(lookup.loc[sample, "Sample Type"]).strip() for sample in matched
        }
        name_map = {
            sample: str(lookup.loc[sample, "Group Name"]).strip()
            if "Group Name" in lookup.columns
            else mapping[sample]
            for sample in matched
        }
        if matched:
            features.attrs["sample_group_map"] = mapping
            features.attrs["sample_type_map"] = type_map
            features.attrs["sample_group_name_map"] = name_map
            features.attrs["sample_metadata_records"] = {
                sample: {
                    str(column): lookup.loc[sample, column]
                    for column in selected.columns
                    if not str(column).startswith("_")
                    and pd.notna(lookup.loc[sample, column])
                }
                for sample in matched
            }
            features.attrs["sample_metadata_sources"] = sorted(
                set(str(lookup.loc[sample, "_metadata_workbook"]) for sample in matched)
            )
            features.attrs["sample_group_mapping_status"] = (
                "excel_metadata_validated"
                if len(matched) == len(samples)
                else "excel_metadata_partial"
            )
        audit_rows.append(
            {
                "stream": str(stream),
                "metadata_workbook": "; ".join(
                    features.attrs.get("sample_metadata_sources", sources)
                ),
                "sample_columns": len(samples),
                "matched_samples": len(matched),
                "mapping_status": features.attrs.get(
                    "sample_group_mapping_status", "not_matched"
                ),
            }
        )
    return pd.DataFrame(audit_rows, columns=columns)


def group_catalog(result: dict) -> pd.DataFrame:
    """List validated XCMS sample groups available across result streams."""
    rows: list[dict] = []
    for stream, stream_result in result.get("streams", {}).items():
        features = stream_result.get("features", pd.DataFrame())
        mapping = (
            features.attrs.get("sample_group_map", {})
            if isinstance(features, pd.DataFrame)
            else {}
        )
        type_map = (
            features.attrs.get("sample_type_map", {})
            if isinstance(features, pd.DataFrame)
            else {}
        )
        name_map = (
            features.attrs.get("sample_group_name_map", {})
            if isinstance(features, pd.DataFrame)
            else {}
        )
        for group in dict.fromkeys(mapping.values()):
            group_samples = [
                sample for sample, value in mapping.items() if str(value) == str(group)
            ]
            sample_types = {
                str(type_map.get(sample, "")).strip().casefold()
                for sample in group_samples
                if str(type_map.get(sample, "")).strip()
            }
            group_names = [
                str(name_map.get(sample, "")).strip()
                for sample in group_samples
                if str(name_map.get(sample, "")).strip()
            ]
            biological = (
                (
                    bool(sample_types)
                    and sample_types.issubset({"study", "sample", "biological"})
                )
                if sample_types
                else is_biological_group(str(group))
            )
            rows.append(
                {
                    "stream": str(stream),
                    "group": str(group),
                    "label": group_names[0]
                    if group_names
                    else _group_label(str(group)),
                    "biological": biological,
                    "samples": sum(
                        str(value) == str(group) for value in mapping.values()
                    ),
                    "metadata_source": "; ".join(
                        features.attrs.get("sample_metadata_sources", [])
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "stream",
                "group",
                "label",
                "biological",
                "samples",
                "metadata_source",
            ]
        )
    return (
        pd.DataFrame(rows).drop_duplicates(["stream", "group"]).reset_index(drop=True)
    )


def default_biological_groups(result: dict) -> tuple[str, str] | None:
    """Return the first two biological groups shared by the most streams."""
    catalog = group_catalog(result)
    if catalog.empty:
        return None
    biological = catalog[catalog["biological"]]
    counts = (
        biological.groupby("group")["stream"].nunique().sort_values(ascending=False)
    )
    ordered = list(counts.index)
    return (str(ordered[0]), str(ordered[1])) if len(ordered) >= 2 else None


def _annotation_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _accepted_feature_ids(result: dict, stream: str) -> set[str]:
    accepted = result.get("automatic_annotations", pd.DataFrame())
    if not isinstance(accepted, pd.DataFrame) or accepted.empty:
        return set()
    stream_col = _annotation_column(accepted, ("stream", "standard_stream"))
    feature_col = _annotation_column(accepted, ("feature_id", "feat_feature_id"))
    if feature_col is None:
        return set()
    selected = accepted
    if stream_col is not None:
        selected = selected[selected[stream_col].astype(str).eq(str(stream))]
    return set(selected[feature_col].astype(str))


def _accepted_neighbour_frame(result: dict, stream: str) -> pd.DataFrame:
    accepted = result.get("automatic_annotations", pd.DataFrame())
    if not isinstance(accepted, pd.DataFrame) or accepted.empty:
        return pd.DataFrame(columns=["feature_id", "mz", "rt", "lipid_class"])
    stream_col = _annotation_column(accepted, ("stream", "standard_stream"))
    feature_col = _annotation_column(accepted, ("feature_id", "feat_feature_id"))
    mz_col = _annotation_column(accepted, ("mz", "feat_mz"))
    rt_col = _annotation_column(accepted, ("rt", "feat_rt"))
    class_col = _annotation_column(
        accepted, ("lipid_class", "grid_class", "class", "ref_class")
    )
    identity_col = _annotation_column(accepted, ("feature_identity", "name"))
    if feature_col is None or mz_col is None or rt_col is None:
        return pd.DataFrame(columns=["feature_id", "mz", "rt", "lipid_class"])
    selected = accepted
    if stream_col is not None:
        selected = selected[selected[stream_col].astype(str).eq(str(stream))]
    classes = (
        selected[class_col].astype(str)
        if class_col is not None
        else selected[identity_col].astype(str).str.extract(r"^([^\(]+)", expand=False)
        if identity_col is not None
        else pd.Series("", index=selected.index)
    )
    neighbours = pd.DataFrame(
        {
            "feature_id": selected[feature_col].astype(str),
            "mz": pd.to_numeric(selected[mz_col], errors="coerce"),
            "rt": pd.to_numeric(selected[rt_col], errors="coerce"),
            "lipid_class": classes.fillna("").astype(str),
        }
    ).dropna(subset=["mz", "rt"])
    return neighbours.drop_duplicates(["feature_id", "lipid_class"], keep="first")


def _differential_table(
    features: pd.DataFrame,
    stream: str,
    criteria: UntargetedComparisonCriteria,
    identified_feature_ids: set[str],
) -> pd.DataFrame:
    mapping = {
        str(key): str(value)
        for key, value in features.attrs.get("sample_group_map", {}).items()
    }
    group_a_columns = [
        column
        for column, group in mapping.items()
        if group == criteria.group_a and column in features
    ]
    group_b_columns = [
        column
        for column, group in mapping.items()
        if group == criteria.group_b and column in features
    ]
    if len(group_a_columns) < 2 or len(group_b_columns) < 2:
        return pd.DataFrame()
    feature_ids = features["feature_id"].astype(str)
    unannotated = ~feature_ids.isin(identified_feature_ids)
    data = features.loc[unannotated].copy()
    if data.empty:
        return pd.DataFrame()
    a = (
        data[group_a_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(float)
    )
    b = (
        data[group_b_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(float)
    )
    combined = np.concatenate([a, b], axis=1)
    positive = np.where(combined > 0.0, combined, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        minimum_positive = np.nanmin(positive, axis=1)
    pseudocount = np.where(np.isfinite(minimum_positive), minimum_positive * 0.5, 1.0)
    log_a = np.log2(a + pseudocount[:, None])
    log_b = np.log2(b + pseudocount[:, None])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # Equal or nearly equal replicate vectors legitimately produce scipy's
        # precision-loss warning. They are converted to non-significant rows
        # below, so the warning is not useful to an end user.
        warnings.simplefilter("ignore", RuntimeWarning)
        test = ttest_ind(log_b, log_a, axis=1, equal_var=False, nan_policy="omit")
    p_value = np.asarray(test.pvalue, dtype=float)
    p_value = np.where(np.isfinite(p_value), p_value, 1.0)
    detected_a_columns = [
        f"{column}::detected"
        for column in group_a_columns
        if f"{column}::detected" in data
    ]
    detected_b_columns = [
        f"{column}::detected"
        for column in group_b_columns
        if f"{column}::detected" in data
    ]
    if len(detected_a_columns) == len(group_a_columns) and len(
        detected_b_columns
    ) == len(group_b_columns):
        detection_a = (
            data[detected_a_columns]
            .fillna(False)
            .astype(bool)
            .mean(axis=1)
            .to_numpy(float)
            * 100.0
        )
        detection_b = (
            data[detected_b_columns]
            .fillna(False)
            .astype(bool)
            .mean(axis=1)
            .to_numpy(float)
            * 100.0
        )
        detection_evidence = "MZmine DETECTED state"
    else:
        detection_a = (a > 0.0).mean(axis=1) * 100.0
        detection_b = (b > 0.0).mean(axis=1) * 100.0
        detection_evidence = "non-zero intensity fallback"
    log2_fc = log_b.mean(axis=1) - log_a.mean(axis=1)
    significant = (
        (p_value < float(criteria.p_value_threshold))
        & (np.abs(log2_fc) >= float(criteria.min_abs_log2_fold_change))
        & (
            np.maximum(detection_a, detection_b)
            >= float(criteria.min_detection_percent)
        )
    )
    output = pd.DataFrame(
        {
            "stream": str(stream),
            "feature_id": data["feature_id"].astype(str).to_numpy(),
            "mz": pd.to_numeric(data["mz"], errors="coerce").to_numpy(),
            "rt": pd.to_numeric(data["rt"], errors="coerce").to_numpy(),
            "group_a": _group_label(criteria.group_a),
            "group_b": _group_label(criteria.group_b),
            "n_a": len(group_a_columns),
            "n_b": len(group_b_columns),
            "mean_intensity_a": a.mean(axis=1),
            "mean_intensity_b": b.mean(axis=1),
            "detection_a_percent": detection_a,
            "detection_b_percent": detection_b,
            "log2_fold_change_b_vs_a": log2_fc,
            "p_value": p_value,
            "raw_p_significant": p_value < float(criteria.p_value_threshold),
            "significant": significant,
            "already_identified": False,
            "detection_evidence": detection_evidence,
        }
    )
    for column in ("mobility", "mobility_unit", "ccs"):
        if column in data:
            output[column] = data[column].to_numpy()
    return output


def _attach_cluster_hypotheses(
    table: pd.DataFrame,
    neighbours: pd.DataFrame,
    criteria: UntargetedComparisonCriteria,
) -> pd.DataFrame:
    output = table.copy()
    output["cluster_class"] = ""
    output["cluster_neighbor_count"] = 0
    output["cluster_dominance"] = np.nan
    output["cluster_median_distance"] = np.nan
    output["cluster_hypothesis"] = "no coherent local lipid-class cluster"
    if output.empty or neighbours.empty:
        return output
    source = neighbours.dropna(subset=["mz", "rt"]).copy()
    source = source[source["lipid_class"].astype(str).ne("")]
    if source.empty:
        return output
    coordinates = np.column_stack(
        [
            source["mz"].to_numpy(float) / _CH2_MASS,
            source["rt"].to_numpy(float) / 0.25,
        ]
    )
    query = np.column_stack(
        [
            output["mz"].to_numpy(float) / _CH2_MASS,
            output["rt"].to_numpy(float) / 0.25,
        ]
    )
    k = min(8, len(source))
    distances, indices = cKDTree(coordinates).query(query, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    class_values = source["lipid_class"].astype(str).to_numpy()
    for row_index, (row_distances, row_indices) in enumerate(
        zip(distances, indices, strict=True)
    ):
        local = np.isfinite(row_distances) & (row_distances <= 3.0)
        if not local.any():
            continue
        local_classes = class_values[np.asarray(row_indices)[local]]
        counts = pd.Series(local_classes).value_counts()
        dominant = str(counts.index[0])
        dominance = float(counts.iloc[0] / len(local_classes))
        count = int(counts.iloc[0])
        output.iloc[row_index, output.columns.get_loc("cluster_neighbor_count")] = len(
            local_classes
        )
        output.iloc[row_index, output.columns.get_loc("cluster_dominance")] = dominance
        output.iloc[row_index, output.columns.get_loc("cluster_median_distance")] = (
            float(np.median(np.asarray(row_distances)[local]))
        )
        if count >= int(criteria.cluster_min_neighbors) and dominance >= float(
            criteria.cluster_min_dominance
        ):
            output.iloc[row_index, output.columns.get_loc("cluster_class")] = dominant
            output.iloc[row_index, output.columns.get_loc("cluster_hypothesis")] = (
                f"local {dominant} region ({count}/{len(local_classes)} neighbours)"
            )
    return output


def _attach_homologous_series_evidence(
    significant: pd.DataFrame,
    all_unannotated: pd.DataFrame,
    criteria: UntargetedComparisonCriteria,
) -> pd.DataFrame:
    """Add suggestive CH2-series evidence without assigning an identity.

    The source includes all tested unannotated features, not only significant
    ones, because a biological series member need not itself pass the
    differential gate. Mass and RT tolerances are recorded run criteria.
    """
    output = significant.copy()
    output["homologous_series_neighbor_count"] = 0
    output["homologous_series_mass_steps"] = ""
    output["homologous_series_hypothesis"] = "no local CH2-series evidence"
    output["significant_ch2_neighbor_count"] = 0
    output["significant_ch2_neighbor_feature_ids"] = ""
    output["significant_ch2_mass_steps"] = ""
    output["significant_ch2_rt_deltas_min"] = ""
    output["significant_ch2_series_id"] = ""
    output["significant_ch2_series_size"] = 1
    output["significant_ch2_cluster_hypothesis"] = (
        "no CH2-spaced relationship to another raw-p hit"
    )
    if output.empty or all_unannotated.empty:
        return output
    source = all_unannotated[["feature_id", "mz", "rt"]].copy()
    source["mz"] = pd.to_numeric(source["mz"], errors="coerce")
    source["rt"] = pd.to_numeric(source["rt"], errors="coerce")
    source = source.dropna(subset=["mz", "rt"]).sort_values("mz")
    if source.empty:
        return output
    source_mz = source["mz"].to_numpy(float)
    source_rt = source["rt"].to_numpy(float)
    source_ids = source["feature_id"].astype(str).to_numpy()
    mass_tolerance = float(criteria.ch2_mass_tolerance_da)
    rt_tolerance = float(criteria.ch2_rt_tolerance_min)
    significant_ids = set(output["feature_id"].astype(str))
    adjacency: dict[str, set[str]] = {
        feature_id: set() for feature_id in significant_ids
    }
    significant_evidence: dict[str, list[tuple[str, int, float]]] = {
        feature_id: [] for feature_id in significant_ids
    }
    for row_index, row in output.iterrows():
        feature_id = str(row["feature_id"])
        mz = float(row["mz"])
        rt = float(row["rt"])
        neighbour_ids: set[str] = set()
        observed_steps: set[int] = set()
        for step in range(1, 5):
            for direction in (-1.0, 1.0):
                target = mz + direction * step * _CH2_MASS
                left = int(
                    np.searchsorted(source_mz, target - mass_tolerance, side="left")
                )
                right = int(
                    np.searchsorted(source_mz, target + mass_tolerance, side="right")
                )
                if right <= left:
                    continue
                local = np.abs(source_rt[left:right] - rt) <= rt_tolerance
                for candidate_id in source_ids[left:right][local]:
                    if candidate_id != feature_id:
                        neighbour_ids.add(str(candidate_id))
                        observed_steps.add(step)
                local_offsets = np.flatnonzero(local)
                for local_offset in local_offsets:
                    candidate_id = str(source_ids[left + int(local_offset)])
                    if (
                        candidate_id == feature_id
                        or candidate_id not in significant_ids
                    ):
                        continue
                    candidate_rt = float(source_rt[left + int(local_offset)])
                    adjacency[feature_id].add(candidate_id)
                    adjacency[candidate_id].add(feature_id)
                    significant_evidence[feature_id].append(
                        (
                            candidate_id,
                            step,
                            candidate_rt - rt,
                        )
                    )
        count = len(neighbour_ids)
        steps = ",".join(map(str, sorted(observed_steps)))
        output.at[row_index, "homologous_series_neighbor_count"] = count
        output.at[row_index, "homologous_series_mass_steps"] = steps
        if count:
            output.at[row_index, "homologous_series_hypothesis"] = (
                f"{count} nearby unannotated feature(s) at CH2 step(s) {steps}"
            )
        raw_p_neighbours = significant_evidence[feature_id]
        if raw_p_neighbours:
            neighbour_ids_text = ";".join(
                sorted({item[0] for item in raw_p_neighbours})
            )
            raw_p_steps = ",".join(
                map(str, sorted({item[1] for item in raw_p_neighbours}))
            )
            rt_deltas = ";".join(
                f"{item[0]}:{item[2]:+.3f}"
                for item in sorted(raw_p_neighbours, key=lambda item: item[0])
            )
            output.at[row_index, "significant_ch2_neighbor_count"] = len(
                set(item[0] for item in raw_p_neighbours)
            )
            output.at[row_index, "significant_ch2_neighbor_feature_ids"] = (
                neighbour_ids_text
            )
            output.at[row_index, "significant_ch2_mass_steps"] = raw_p_steps
            output.at[row_index, "significant_ch2_rt_deltas_min"] = rt_deltas

    # Connected components distinguish an isolated CH2-spaced pair from a
    # more convincing repeated homologous series among the raw-p hits.
    visited: set[str] = set()
    component_number = 0
    feature_to_component: dict[str, tuple[str, int]] = {}
    for feature_id in sorted(significant_ids):
        if feature_id in visited:
            continue
        component_number += 1
        stack = [feature_id]
        members: set[str] = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(adjacency.get(current, set()) - members)
        visited.update(members)
        series_id = f"CH2-{component_number:03d}" if len(members) > 1 else ""
        for member in members:
            feature_to_component[member] = (series_id, len(members))
    for row_index, row in output.iterrows():
        feature_id = str(row["feature_id"])
        series_id, series_size = feature_to_component.get(feature_id, ("", 1))
        output.at[row_index, "significant_ch2_series_id"] = series_id
        output.at[row_index, "significant_ch2_series_size"] = series_size
        if series_size >= 3:
            output.at[row_index, "significant_ch2_cluster_hypothesis"] = (
                f"{series_id}: {series_size} raw-p hits form a local CH2-spaced series"
            )
        elif series_size == 2:
            output.at[row_index, "significant_ch2_cluster_hypothesis"] = (
                f"{series_id}: CH2-spaced pair of raw-p hits"
            )
    return output


def _stream_phase_polarity(stream: str) -> tuple[str, str] | None:
    match = re.match(r"^(NP|RP)[_\- ]*(Pos|Neg)$", str(stream), flags=re.I)
    if not match:
        return None
    return match.group(1).upper(), match.group(2).casefold()


def _coordinate_sources(result: dict) -> dict[str, pd.DataFrame]:
    """Return searchable feature coordinates, including accepted features."""
    sources: dict[str, pd.DataFrame] = {}
    for stream, stream_result in result.get("streams", {}).items():
        features = stream_result.get("features", pd.DataFrame())
        if not isinstance(features, pd.DataFrame) or features.empty:
            continue
        required = {"feature_id", "mz", "rt"}
        if not required.issubset(features.columns):
            continue
        source = features[["feature_id", "mz", "rt"]].copy()
        source["feature_id"] = source["feature_id"].astype(str)
        source["mz"] = pd.to_numeric(source["mz"], errors="coerce")
        source["rt"] = pd.to_numeric(source["rt"], errors="coerce")
        source = source.dropna(subset=["mz", "rt"]).sort_values("mz")
        source["already_annotated"] = source["feature_id"].isin(
            _accepted_feature_ids(result, str(stream))
        )
        sources[str(stream)] = source.reset_index(drop=True)
    return sources


def _canonical_acquisition_sample(sample: str) -> str:
    """Return a conservative cross-mode specimen key from an acquisition name."""
    value = Path(str(sample)).name.casefold().strip()
    value = re.sub(r"\.(d|mzml|mzxml|cdf|raw)$", "", value, flags=re.I)
    value = re.sub(r"(?:[_\- ](?:pos|positive|neg|negative))+$", "", value)
    value = re.sub(r"[_\- ]\d{3,}$", "", value)
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _eligible_abundance_samples(
    features: pd.DataFrame,
    criteria: UntargetedComparisonCriteria,
) -> list[str]:
    group_map = {
        str(key): str(value)
        for key, value in features.attrs.get("sample_group_map", {}).items()
    }
    configured = [
        str(sample)
        for sample in features.attrs.get("sample_columns", ())
        if str(sample) in features.columns
    ]
    samples = configured or [
        sample for sample in group_map if sample in features.columns
    ]
    return [
        sample
        for sample in samples
        if group_map.get(sample) in {criteria.group_a, criteria.group_b}
        and is_biological_group(group_map.get(sample, ""))
    ]


def _unique_sample_key_map(
    features: pd.DataFrame,
    samples: list[str],
    method: str,
) -> dict[str, str]:
    metadata = features.attrs.get("sample_metadata_records", {})
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        if method == "exact XCMS sample name":
            key = str(sample).casefold()
        elif method.startswith("Excel metadata: "):
            column = method.split(": ", 1)[1]
            value = metadata.get(sample, {}).get(column, "")
            key = str(value).strip().casefold() if pd.notna(value) else ""
        else:
            key = _canonical_acquisition_sample(sample)
        if key:
            grouped.setdefault(key, []).append(sample)
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def paired_abundance_table(
    result: dict,
    stream_a: str,
    feature_a: str,
    stream_b: str,
    feature_b: str,
    criteria: UntargetedComparisonCriteria,
) -> tuple[pd.DataFrame, str]:
    """Return biological abundance pairs using the strongest safe sample key."""
    columns = [
        "pair_key",
        "group",
        "sample_a",
        "sample_b",
        "intensity_a",
        "intensity_b",
    ]
    features_a = (
        result.get("streams", {}).get(stream_a, {}).get("features", pd.DataFrame())
    )
    features_b = (
        result.get("streams", {}).get(stream_b, {}).get("features", pd.DataFrame())
    )
    if (
        not isinstance(features_a, pd.DataFrame)
        or features_a.empty
        or not isinstance(features_b, pd.DataFrame)
        or features_b.empty
    ):
        return pd.DataFrame(columns=columns), "not evaluable"
    row_a = features_a[features_a["feature_id"].astype(str).eq(str(feature_a))]
    row_b = features_b[features_b["feature_id"].astype(str).eq(str(feature_b))]
    if row_a.empty or row_b.empty:
        return pd.DataFrame(columns=columns), "not evaluable"
    samples_a = _eligible_abundance_samples(features_a, criteria)
    samples_b = _eligible_abundance_samples(features_b, criteria)
    metadata_a = features_a.attrs.get("sample_metadata_records", {})
    metadata_b = features_b.attrs.get("sample_metadata_records", {})
    common_metadata_columns = [
        column
        for column in _SPECIMEN_ID_COLUMNS
        if any(column in record for record in metadata_a.values())
        and any(column in record for record in metadata_b.values())
    ]
    methods = [f"Excel metadata: {column}" for column in common_metadata_columns] + [
        "exact XCMS sample name",
        "canonical acquisition name",
    ]
    groups_a = {
        str(key): str(value)
        for key, value in features_a.attrs.get("sample_group_map", {}).items()
    }
    groups_b = {
        str(key): str(value)
        for key, value in features_b.attrs.get("sample_group_map", {}).items()
    }
    values_a = row_a.iloc[0]
    values_b = row_b.iloc[0]
    for method in methods:
        keys_a = _unique_sample_key_map(features_a, samples_a, method)
        keys_b = _unique_sample_key_map(features_b, samples_b, method)
        shared = [
            key
            for key in keys_a.keys() & keys_b.keys()
            if groups_a.get(keys_a[key]) == groups_b.get(keys_b[key])
        ]
        pairs: list[dict[str, object]] = []
        for key in sorted(shared):
            value_a = pd.to_numeric(values_a.get(keys_a[key]), errors="coerce")
            value_b = pd.to_numeric(values_b.get(keys_b[key]), errors="coerce")
            if pd.notna(value_a) and pd.notna(value_b):
                pairs.append(
                    {
                        "pair_key": key,
                        "group": groups_a.get(keys_a[key], ""),
                        "sample_a": keys_a[key],
                        "sample_b": keys_b[key],
                        "intensity_a": float(value_a),
                        "intensity_b": float(value_b),
                    }
                )
        if len(pairs) < 3:
            continue
        return pd.DataFrame.from_records(pairs, columns=columns), method
    return (
        pd.DataFrame(columns=columns),
        "not evaluable; no unique paired specimen key",
    )


def _paired_abundance_correlation(
    result: dict,
    stream_a: str,
    feature_a: str,
    stream_b: str,
    feature_b: str,
    criteria: UntargetedComparisonCriteria,
) -> tuple[float, int, str]:
    """Spearman correlation across authoritatively paired biological samples."""
    pairs, method = paired_abundance_table(
        result, stream_a, feature_a, stream_b, feature_b, criteria
    )
    if pairs.empty:
        return np.nan, 0, method
    paired = pairs[["intensity_a", "intensity_b"]].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation = spearmanr(
            np.log2(np.clip(paired[:, 0], a_min=0.0, a_max=None) + 1.0),
            np.log2(np.clip(paired[:, 1], a_min=0.0, a_max=None) + 1.0),
        ).statistic
    return (
        float(correlation) if np.isfinite(correlation) else np.nan,
        len(pairs),
        method,
    )


def _mass_candidates(
    source: pd.DataFrame,
    target_mz: float,
    ppm_tolerance: float,
) -> pd.DataFrame:
    if source.empty or not np.isfinite(target_mz) or target_mz <= 0.0:
        return source.iloc[0:0].copy()
    masses = source["mz"].to_numpy(float)
    delta = target_mz * float(ppm_tolerance) * 1e-6
    left = int(np.searchsorted(masses, target_mz - delta, side="left"))
    right = int(np.searchsorted(masses, target_mz + delta, side="right"))
    candidates = source.iloc[left:right].copy()
    if candidates.empty:
        return candidates
    candidates["mass_error_ppm"] = (candidates["mz"] - target_mz) / target_mz * 1e6
    return candidates


def _neutral_mass_options(mz: float, polarity: str) -> list[tuple[str, float]]:
    return [(adduct.name, adduct.neutral(mz)) for adduct in adduct_options(polarity)
            if adduct.neutral(mz) > 0]


def _neutral_mass_hypotheses(mz: float, polarity: str) -> str:
    return "; ".join(
        f"{adduct}: neutral {neutral_mass:.5f} Da"
        for adduct, neutral_mass in _neutral_mass_options(mz, polarity)
    )


def _elemental_formula_candidates(
    mz: float,
    polarity: str,
    carbon_estimate: float,
    ppm_tolerance: float,
    maximum_candidates: int = 5,
    charge: int = 1,
    m_plus_2_percent: float = np.nan,
) -> tuple[str, int]:
    """Enumerate coarse CHNOPS formulas constrained by isotope-derived C.

    The search is intentionally narrow and returns hypotheses, not an identity.
    Neutral CHNOPS candidates use the isotope charge and ion carbon correction.
    M+2, when measured, ranks candidates without forcing an exact isotope fit.
    """
    if not np.isfinite(carbon_estimate) or carbon_estimate < 4.0:
        return "", 0
    carbon_centre = int(round(carbon_estimate))
    carbon_values = range(max(4, carbon_centre - 3), carbon_centre + 4)
    candidates: list[tuple[float, str]] = []
    for ion in adduct_options(polarity, charge):
        adduct, neutral_mass = ion.name, ion.neutral(mz)
        if neutral_mass <= 0:
            continue
        allowed_error = neutral_mass * float(ppm_tolerance) * 1e-6
        for ion_carbon in carbon_values:
            carbon = ion_carbon - ion.carbon
            if carbon < 1:
                continue
            carbon_mass = float(carbon) * 12.0
            for nitrogen in range(0, 4):
                for oxygen in range(1, 15):
                    for phosphorus in range(0, 3):
                        for sulphur in range(0, 3):
                            heavy_mass = (
                                carbon_mass
                                + nitrogen * _ATOMIC_MASS["N"]
                                + oxygen * _ATOMIC_MASS["O"]
                                + phosphorus * _ATOMIC_MASS["P"]
                                + sulphur * _ATOMIC_MASS["S"]
                            )
                            hydrogen = int(
                                round((neutral_mass - heavy_mass) / _ATOMIC_MASS["H"])
                            )
                            if hydrogen < 0 or hydrogen > 2 * carbon + nitrogen + phosphorus + 2:
                                continue
                            double_bond_equivalents = (
                                carbon - hydrogen / 2.0 + (nitrogen + phosphorus) / 2.0 + 1.0
                            )
                            if (
                                double_bond_equivalents < 0.0
                                or abs(
                                    double_bond_equivalents
                                    - round(double_bond_equivalents)
                                )
                                > 1e-6
                            ):
                                continue
                            calculated = heavy_mass + hydrogen * _ATOMIC_MASS["H"]
                            error = calculated - neutral_mass
                            if abs(error) > allowed_error:
                                continue
                            formula = f"C{carbon}H{hydrogen}"
                            for element, count in (
                                ("N", nitrogen),
                                ("O", oxygen),
                                ("P", phosphorus),
                                ("S", sulphur),
                            ):
                                if count:
                                    formula += f"{element}{count}"
                            error_ppm = error / neutral_mass * 1e6
                            label = (
                                f"{adduct} {formula} "
                                f"({error_ppm:+.2f} ppm; "
                                f"DBE {double_bond_equivalents:.0f})"
                            )
                            # Search targets specifically use C13 spacings, not
                            # an integrated nominal envelope. Do not add Cl37,
                            # S34 or O18 intensities that may be resolved apart.
                            ion_carbon_count = carbon + ion.carbon
                            r1 = ion_carbon_count * .0107
                            r2 = ion_carbon_count * (ion_carbon_count - 1) / 2. * .0107**2
                            isotope_error = abs((r1 * 100.) - carbon_estimate * 1.07) / max(3., carbon_estimate * .15)
                            if np.isfinite(m_plus_2_percent):
                                isotope_error += abs(r2 * 100. - m_plus_2_percent) / max(3., m_plus_2_percent * .25)
                            label += f"; predicted C13 M+1 {r1*100.:.1f}%, M+2 {r2*100.:.1f}%"
                            candidates.append((isotope_error + abs(error_ppm) / ppm_tolerance, label))
    candidates.sort(key=lambda item: (item[0], item[1]))
    unique: list[str] = []
    for _, label in candidates:
        if label not in unique:
            unique.append(label)
    return "; ".join(unique[:maximum_candidates]), len(unique)


def _attach_isotope_evidence(
    result: dict,
    by_stream: dict[str, pd.DataFrame],
    criteria: UntargetedComparisonCriteria,
) -> dict[str, pd.DataFrame]:
    """Find co-eluting C13 partners across the complete feature table."""
    output: dict[str, pd.DataFrame] = {}
    defaults: dict[str, object] = {
        "isotope_m_plus_1_feature_id": "",
        "isotope_m_plus_1_mz": np.nan,
        "isotope_m_plus_1_rt_delta_seconds": np.nan,
        "isotope_m_plus_1_intensity_percent": np.nan,
        "isotope_m_plus_1_is_annotated": False,
        "isotope_m_plus_2_feature_id": "",
        "isotope_m_plus_2_mz": np.nan,
        "isotope_m_plus_2_rt_delta_seconds": np.nan,
        "isotope_m_plus_2_intensity_percent": np.nan,
        "isotope_m_plus_2_is_annotated": False,
        "isotope_charge_hypothesis": np.nan,
        "possible_isotope_of_feature_id": "",
        "estimated_carbon_count_from_m_plus_1": np.nan,
        "isotope_pattern": "no co-eluting isotope partner found",
        "isotope_pattern_assessment": "not evaluable",
        "neutral_mass_hypotheses": "",
        "kendrick_mass_defect": np.nan,
        "elemental_formula_candidates": "",
        "elemental_formula_candidate_count": 0,
        "chemical_composition_hypothesis": (
            "exact formula unresolved from standard feature data"
        ),
    }
    for stream, original in by_stream.items():
        table = original.copy()
        for column, default in defaults.items():
            table[column] = default
        parsed = _stream_phase_polarity(stream)
        polarity = (
            parsed[1] if parsed else ("neg" if "neg" in stream.casefold() else "pos")
        )
        stream_features = (
            result.get("streams", {}).get(stream, {}).get("features", pd.DataFrame())
        )
        if not isinstance(stream_features, pd.DataFrame) or stream_features.empty:
            output[stream] = table
            continue
        mapping = {
            str(key): str(value)
            for key, value in stream_features.attrs.get("sample_group_map", {}).items()
        }
        abundance_columns = [
            column
            for column, group in mapping.items()
            if group in {criteria.group_a, criteria.group_b}
            and column in stream_features.columns
        ]
        source = stream_features[["feature_id", "mz", "rt"]].copy()
        source["feature_id"] = source["feature_id"].astype(str)
        if abundance_columns:
            source["representative_intensity"] = (
                stream_features[abundance_columns]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1)
            )
        else:
            source["representative_intensity"] = np.nan
        source["already_annotated"] = source["feature_id"].isin(
            _accepted_feature_ids(result, stream)
        )
        source["mz"] = pd.to_numeric(source["mz"], errors="coerce")
        source["rt"] = pd.to_numeric(source["rt"], errors="coerce")
        source = source.dropna(subset=["mz", "rt"])
        source = source.sort_values("mz").reset_index(drop=True)
        hit_mask = table.get("raw_p_significant", table["significant"]).fillna(False).astype(bool)
        for row_index, row in table[hit_mask].iterrows():
            mz = float(row["mz"])
            rt = float(row["rt"])
            mono_intensity = float(
                np.nanmean(
                    [
                        row.get("mean_intensity_a", np.nan),
                        row.get("mean_intensity_b", np.nan),
                    ]
                )
            )
            neutral_hypotheses = _neutral_mass_hypotheses(mz, polarity)
            table.at[row_index, "neutral_mass_hypotheses"] = neutral_hypotheses
            first_neutral = (
                mz - _POSITIVE_ADDUCT_OFFSETS[0][1]
                if polarity == "pos"
                else mz - _NEGATIVE_ADDUCT[1]
            )
            kendrick_mass = first_neutral * 14.0 / _CH2_MASS
            table.at[row_index, "kendrick_mass_defect"] = (
                round(kendrick_mass) - kendrick_mass
            )

            # First determine whether the hit may itself be an isotope.
            lower_matches: list[tuple[float, pd.Series]] = []
            for charge in (1, 2):
                target = mz - _C13_C12_MASS_DIFFERENCE / charge
                candidates = _mass_candidates(
                    source, target, criteria.cross_mode_ppm_tolerance
                )
                if candidates.empty:
                    continue
                candidates["rt_delta_seconds"] = (candidates["rt"] - rt) * 60.0
                candidates = candidates[
                    candidates["rt_delta_seconds"]
                    .abs()
                    .le(criteria.isotope_rt_tolerance_min * 60.0)
                ]
                for _, candidate in candidates.iterrows():
                    score = abs(
                        float(candidate["mass_error_ppm"])
                    ) / criteria.cross_mode_ppm_tolerance + abs(
                        float(candidate["rt_delta_seconds"])
                    ) / (criteria.isotope_rt_tolerance_min * 60.0)
                    lower_matches.append((score, candidate))
            if lower_matches:
                _, lower = min(lower_matches, key=lambda item: item[0])
                table.at[row_index, "possible_isotope_of_feature_id"] = str(
                    lower["feature_id"]
                )

            best_pattern: tuple[float, int, pd.Series, pd.Series | None] | None = None
            for charge in (1, 2):
                target_m1 = mz + _C13_C12_MASS_DIFFERENCE / charge
                m1_candidates = _mass_candidates(
                    source, target_m1, criteria.cross_mode_ppm_tolerance
                )
                if m1_candidates.empty:
                    continue
                m1_candidates["rt_delta_seconds"] = (m1_candidates["rt"] - rt) * 60.0
                m1_candidates = m1_candidates[
                    m1_candidates["rt_delta_seconds"]
                    .abs()
                    .le(criteria.isotope_rt_tolerance_min * 60.0)
                ].copy()
                if m1_candidates.empty:
                    continue
                m1_candidates["match_score"] = m1_candidates[
                    "mass_error_ppm"
                ].abs() / criteria.cross_mode_ppm_tolerance + m1_candidates[
                    "rt_delta_seconds"
                ].abs() / (criteria.isotope_rt_tolerance_min * 60.0)
                m1 = m1_candidates.iloc[m1_candidates["match_score"].argmin()]
                target_m2 = mz + 2.0 * _C13_C12_MASS_DIFFERENCE / charge
                m2_candidates = _mass_candidates(
                    source, target_m2, criteria.cross_mode_ppm_tolerance
                )
                m2: pd.Series | None = None
                if not m2_candidates.empty:
                    m2_candidates["rt_delta_seconds"] = (
                        m2_candidates["rt"] - rt
                    ) * 60.0
                    m2_candidates = m2_candidates[
                        m2_candidates["rt_delta_seconds"]
                        .abs()
                        .le(criteria.isotope_rt_tolerance_min * 60.0)
                    ].copy()
                    if not m2_candidates.empty:
                        m2_candidates["match_score"] = m2_candidates[
                            "mass_error_ppm"
                        ].abs() / criteria.cross_mode_ppm_tolerance + m2_candidates[
                            "rt_delta_seconds"
                        ].abs() / (criteria.isotope_rt_tolerance_min * 60.0)
                        m2 = m2_candidates.iloc[m2_candidates["match_score"].argmin()]
                candidate_pattern = (float(m1["match_score"]), charge, m1, m2)
                if best_pattern is None or candidate_pattern[0] < best_pattern[0]:
                    best_pattern = candidate_pattern
            if best_pattern is None:
                table.at[row_index, "chemical_composition_hypothesis"] = (
                    f"{neutral_hypotheses}; no isotope-derived carbon estimate; "
                    "exact formula unresolved"
                )
                continue
            _, charge, m1, m2 = best_pattern
            m1_percent = (
                float(m1["representative_intensity"]) / mono_intensity * 100.0
                if np.isfinite(mono_intensity) and mono_intensity > 0.0
                else np.nan
            )
            m2_percent = (
                float(m2["representative_intensity"]) / mono_intensity * 100.0
                if m2 is not None
                and np.isfinite(mono_intensity)
                and mono_intensity > 0.0
                else np.nan
            )
            carbon_estimate = m1_percent / 1.07 if np.isfinite(m1_percent) else np.nan
            table.at[row_index, "isotope_m_plus_1_feature_id"] = str(m1["feature_id"])
            table.at[row_index, "isotope_m_plus_1_mz"] = float(m1["mz"])
            table.at[row_index, "isotope_m_plus_1_rt_delta_seconds"] = float(
                m1["rt_delta_seconds"]
            )
            table.at[row_index, "isotope_m_plus_1_intensity_percent"] = m1_percent
            table.at[row_index, "isotope_m_plus_1_is_annotated"] = bool(
                m1["already_annotated"]
            )
            table.at[row_index, "isotope_charge_hypothesis"] = charge
            table.at[row_index, "estimated_carbon_count_from_m_plus_1"] = (
                carbon_estimate
            )
            if m2 is not None:
                table.at[row_index, "isotope_m_plus_2_feature_id"] = str(
                    m2["feature_id"]
                )
                table.at[row_index, "isotope_m_plus_2_mz"] = float(m2["mz"])
                table.at[row_index, "isotope_m_plus_2_rt_delta_seconds"] = float(
                    m2["rt_delta_seconds"]
                )
                table.at[row_index, "isotope_m_plus_2_intensity_percent"] = m2_percent
                table.at[row_index, "isotope_m_plus_2_is_annotated"] = bool(
                    m2["already_annotated"]
                )
            pattern = f"M 100%; M+1 {m1_percent:.1f}%"
            if np.isfinite(m2_percent):
                pattern += f"; M+2 {m2_percent:.1f}%"
            table.at[row_index, "isotope_pattern"] = pattern
            assessment = "M+1 partner supports a carbon-containing ion"
            if np.isfinite(m2_percent) and np.isfinite(carbon_estimate):
                carbon_integer = max(1, int(round(carbon_estimate)))
                expected_m2_c13 = (
                    carbon_integer * (carbon_integer - 1) / 2.0 * (0.0107**2) * 100.0
                )
                assessment = (
                    "M+1 and M+2 partners support an isotope envelope; "
                    f"carbon-13-only M+2 expectation is approximately "
                    f"{expected_m2_c13:.1f}% (heteroatom isotopes not modelled)"
                )
            table.at[row_index, "isotope_pattern_assessment"] = assessment
            carbon_text = (
                f"approximately {carbon_estimate:.0f} carbon atoms from M+1/M"
                if np.isfinite(carbon_estimate)
                else "carbon estimate unavailable"
            )
            formula_candidates, formula_count = _elemental_formula_candidates(
                mz,
                polarity,
                carbon_estimate,
                criteria.cross_mode_ppm_tolerance,
                charge=charge, m_plus_2_percent=m2_percent,
            )
            table.at[row_index, "elemental_formula_candidates"] = formula_candidates
            table.at[row_index, "elemental_formula_candidate_count"] = formula_count
            formula_text = (
                f"candidate formulas: {formula_candidates}"
                if formula_candidates
                else "no CHNOPS formula found within the constrained search"
            )
            table.at[row_index, "chemical_composition_hypothesis"] = (
                f"charge {charge}; {carbon_text}; {neutral_hypotheses}; "
                f"{formula_text}; exact formula unresolved"
            )
        output[stream] = table
    return output


def _attach_cross_acquisition_evidence(
    result: dict,
    by_stream: dict[str, pd.DataFrame],
    criteria: UntargetedComparisonCriteria,
    progress=None,
) -> dict[str, pd.DataFrame]:
    """Attach cross-phase and opposite-polarity hypotheses to raw-p hits."""
    sources = _coordinate_sources(result)
    reporter = SearchProgress(progress, sum(int(table["significant"].fillna(False).sum())
        for table in by_stream.values()), 'Cross-mode abundance search')
    stream_lookup = {stream.casefold(): stream for stream in sources}
    hit_ids = {
        stream: set(
            table.loc[
                table["significant"].fillna(False).astype(bool), "feature_id"
            ].astype(str)
        )
        for stream, table in by_stream.items()
        if isinstance(table, pd.DataFrame) and not table.empty
    }
    output: dict[str, pd.DataFrame] = {}
    default_columns: dict[str, object] = {
        "cross_phase_match": False,
        "cross_phase_stream": "",
        "cross_phase_feature_id": "",
        "cross_phase_mz": np.nan,
        "cross_phase_mass_error_ppm": np.nan,
        "cross_phase_candidate_count": 0,
        "cross_phase_abundance_spearman": np.nan,
        "cross_phase_abundance_pairs": 0,
        "cross_phase_abundance_pairing": "not evaluable",
        "cross_phase_counterpart_is_annotated": False,
        "cross_phase_counterpart_is_raw_p_hit": False,
        "opposite_polarity_match": False,
        "opposite_polarity_stream": "",
        "opposite_polarity_feature_id": "",
        "opposite_polarity_mz": np.nan,
        "opposite_polarity_rt": np.nan,
        "opposite_polarity_rule": "",
        "opposite_polarity_mass_error_ppm": np.nan,
        "opposite_polarity_rt_delta_seconds": np.nan,
        "opposite_polarity_candidate_count": 0,
        "opposite_polarity_abundance_spearman": np.nan,
        "opposite_polarity_abundance_pairs": 0,
        "opposite_polarity_abundance_pairing": "not evaluable",
        "opposite_polarity_counterpart_is_annotated": False,
        "opposite_polarity_counterpart_is_raw_p_hit": False,
        "cross_acquisition_evidence": "not evaluated; feature did not pass screening gates",
    }
    for stream, original in by_stream.items():
        table = original.copy()
        for column, default in default_columns.items():
            table[column] = default
        parsed = _stream_phase_polarity(stream)
        if parsed is None:
            output[stream] = table
            continue
        phase, polarity = parsed
        other_phase = "NP" if phase == "RP" else "RP"
        cross_phase_stream = stream_lookup.get(
            f"{other_phase}_{'pos' if polarity == 'pos' else 'neg'}".casefold()
        )
        opposite_stream = stream_lookup.get(
            f"{phase}_{'neg' if polarity == 'pos' else 'pos'}".casefold()
        )
        hit_mask = table["significant"].fillna(False).astype(bool)
        for row_index, row in table[hit_mask].iterrows():
            reporter.step(stream)
            evidence: list[str] = []
            mz = float(row["mz"])
            rt = float(row["rt"])
            if cross_phase_stream:
                candidates = _mass_candidates(
                    sources[cross_phase_stream],
                    mz,
                    criteria.cross_mode_ppm_tolerance,
                )
                table.at[row_index, "cross_phase_candidate_count"] = len(candidates)
                if not candidates.empty:
                    correlations = [
                        _paired_abundance_correlation(
                            result,
                            stream,
                            str(row["feature_id"]),
                            cross_phase_stream,
                            str(candidate["feature_id"]),
                            criteria,
                        )
                        for _, candidate in candidates.iterrows()
                    ]
                    candidates = candidates.copy()
                    candidates["abundance_spearman"] = [
                        item[0] for item in correlations
                    ]
                    candidates["abundance_pairs"] = [item[1] for item in correlations]
                    candidates["abundance_pairing"] = [item[2] for item in correlations]
                    candidates["correlation_rank"] = candidates[
                        "abundance_spearman"
                    ].fillna(-2.0)
                    candidates["absolute_mass_error"] = candidates[
                        "mass_error_ppm"
                    ].abs()
                    best = candidates.sort_values(
                        ["correlation_rank", "absolute_mass_error"],
                        ascending=[False, True],
                    ).iloc[0]
                    counterpart_id = str(best["feature_id"])
                    table.at[row_index, "cross_phase_match"] = True
                    table.at[row_index, "cross_phase_stream"] = cross_phase_stream
                    table.at[row_index, "cross_phase_feature_id"] = counterpart_id
                    table.at[row_index, "cross_phase_mz"] = float(best["mz"])
                    table.at[row_index, "cross_phase_mass_error_ppm"] = float(
                        best["mass_error_ppm"]
                    )
                    table.at[row_index, "cross_phase_abundance_spearman"] = (
                        float(best["abundance_spearman"])
                        if pd.notna(best["abundance_spearman"])
                        else np.nan
                    )
                    table.at[row_index, "cross_phase_abundance_pairs"] = int(
                        best["abundance_pairs"]
                    )
                    table.at[row_index, "cross_phase_abundance_pairing"] = str(
                        best["abundance_pairing"]
                    )
                    table.at[row_index, "cross_phase_counterpart_is_annotated"] = bool(
                        best["already_annotated"]
                    )
                    table.at[row_index, "cross_phase_counterpart_is_raw_p_hit"] = (
                        counterpart_id in hit_ids.get(cross_phase_stream, set())
                    )
                    correlation_text = (
                        f"; abundance rho {best['abundance_spearman']:+.2f}, "
                        f"n={int(best['abundance_pairs'])} via {best['abundance_pairing']}"
                        if pd.notna(best["abundance_spearman"])
                        else "; abundance correlation not evaluable"
                    )
                    evidence.append(
                        f"{cross_phase_stream} mass candidate ({len(candidates)} "
                        f"within tolerance; RT not compared{correlation_text})"
                    )
            if opposite_stream:
                paired: list[pd.DataFrame] = []
                negative_name, negative_offset = _NEGATIVE_ADDUCT
                for positive_name, positive_offset in _POSITIVE_ADDUCT_OFFSETS:
                    if polarity == "pos":
                        target = mz - positive_offset + negative_offset
                    else:
                        target = mz - negative_offset + positive_offset
                    candidates = _mass_candidates(
                        sources[opposite_stream],
                        target,
                        criteria.cross_mode_ppm_tolerance,
                    )
                    if candidates.empty:
                        continue
                    candidates["rt_delta_seconds"] = (candidates["rt"] - rt) * 60.0
                    candidates = candidates[
                        candidates["rt_delta_seconds"]
                        .abs()
                        .le(criteria.opposite_polarity_rt_tolerance_min * 60.0)
                    ].copy()
                    if candidates.empty:
                        continue
                    candidates["rule"] = f"{positive_name} ↔ {negative_name}"
                    candidates["match_score"] = candidates[
                        "mass_error_ppm"
                    ].abs() / criteria.cross_mode_ppm_tolerance + candidates[
                        "rt_delta_seconds"
                    ].abs() / (criteria.opposite_polarity_rt_tolerance_min * 60.0)
                    paired.append(candidates)
                if paired:
                    candidates = pd.concat(paired, ignore_index=True, sort=False)
                    table.at[row_index, "opposite_polarity_candidate_count"] = len(
                        candidates
                    )
                    correlations = [
                        _paired_abundance_correlation(
                            result,
                            stream,
                            str(row["feature_id"]),
                            opposite_stream,
                            str(candidate["feature_id"]),
                            criteria,
                        )
                        for _, candidate in candidates.iterrows()
                    ]
                    candidates["abundance_spearman"] = [
                        item[0] for item in correlations
                    ]
                    candidates["abundance_pairs"] = [item[1] for item in correlations]
                    candidates["abundance_pairing"] = [item[2] for item in correlations]
                    candidates["correlation_rank"] = candidates[
                        "abundance_spearman"
                    ].fillna(-2.0)
                    best = candidates.sort_values(
                        ["correlation_rank", "match_score"], ascending=[False, True]
                    ).iloc[0]
                    counterpart_id = str(best["feature_id"])
                    table.at[row_index, "opposite_polarity_match"] = True
                    table.at[row_index, "opposite_polarity_stream"] = opposite_stream
                    table.at[row_index, "opposite_polarity_feature_id"] = counterpart_id
                    table.at[row_index, "opposite_polarity_mz"] = float(best["mz"])
                    table.at[row_index, "opposite_polarity_rt"] = float(best["rt"])
                    table.at[row_index, "opposite_polarity_rule"] = str(best["rule"])
                    table.at[row_index, "opposite_polarity_mass_error_ppm"] = float(
                        best["mass_error_ppm"]
                    )
                    table.at[row_index, "opposite_polarity_rt_delta_seconds"] = float(
                        best["rt_delta_seconds"]
                    )
                    table.at[row_index, "opposite_polarity_abundance_spearman"] = (
                        float(best["abundance_spearman"])
                        if pd.notna(best["abundance_spearman"])
                        else np.nan
                    )
                    table.at[row_index, "opposite_polarity_abundance_pairs"] = int(
                        best["abundance_pairs"]
                    )
                    table.at[row_index, "opposite_polarity_abundance_pairing"] = str(
                        best["abundance_pairing"]
                    )
                    table.at[
                        row_index, "opposite_polarity_counterpart_is_annotated"
                    ] = bool(best["already_annotated"])
                    table.at[
                        row_index, "opposite_polarity_counterpart_is_raw_p_hit"
                    ] = counterpart_id in hit_ids.get(opposite_stream, set())
                    correlation_text = (
                        f"; abundance rho {best['abundance_spearman']:+.2f}, "
                        f"n={int(best['abundance_pairs'])} via {best['abundance_pairing']}"
                        if pd.notna(best["abundance_spearman"])
                        else "; abundance correlation not evaluable"
                    )
                    evidence.append(
                        f"{opposite_stream} {best['rule']} candidate "
                        f"(ΔRT {best['rt_delta_seconds']:+.1f} s{correlation_text})"
                    )
            table.at[row_index, "cross_acquisition_evidence"] = (
                "; ".join(evidence)
                if evidence
                else "no cross-acquisition candidate within tolerances"
            )
        output[stream] = table
    return output


def analyse_standard_untargeted(
    result: dict,
    criteria: UntargetedComparisonCriteria,
    progress=None,
) -> dict:
    """Compare two groups for every unannotated feature in every stream."""
    by_stream: dict[str, pd.DataFrame] = {}
    skipped: list[dict] = []
    if progress:
        progress(.05, "Comparing sample groups")
    for stream, stream_result in result.get("streams", {}).items():
        features = stream_result.get("features", pd.DataFrame())
        mapping = (
            features.attrs.get("sample_group_map", {})
            if isinstance(features, pd.DataFrame)
            else {}
        )
        available = set(map(str, mapping.values()))
        if criteria.group_a not in available or criteria.group_b not in available:
            skipped.append(
                {"stream": stream, "reason": "comparison groups unavailable"}
            )
            continue
        table = _differential_table(
            features,
            str(stream),
            criteria,
            _accepted_feature_ids(result, str(stream)),
        )
        if table.empty:
            skipped.append({"stream": stream, "reason": "no unannotated features available"})
            continue
        # Neighbour interpretation is needed only for points that can light up
        # in the viewer. Keeping the full test vectorized and clustering this
        # small subset keeps large XCMS projects responsive.
        significant = (
            table[table["raw_p_significant"]].copy() if not table.empty else table.copy()
        )
        enriched = _attach_cluster_hypotheses(
            significant,
            _accepted_neighbour_frame(result, str(stream)),
            criteria,
        )
        enriched = _attach_homologous_series_evidence(enriched, table, criteria)
        cluster_columns = [
            "feature_id",
            "cluster_class",
            "cluster_neighbor_count",
            "cluster_dominance",
            "cluster_median_distance",
            "cluster_hypothesis",
            "homologous_series_neighbor_count",
            "homologous_series_mass_steps",
            "homologous_series_hypothesis",
            "significant_ch2_neighbor_count",
            "significant_ch2_neighbor_feature_ids",
            "significant_ch2_mass_steps",
            "significant_ch2_rt_deltas_min",
            "significant_ch2_series_id",
            "significant_ch2_series_size",
            "significant_ch2_cluster_hypothesis",
        ]
        table = table.merge(
            enriched[cluster_columns]
            if not enriched.empty
            else pd.DataFrame(columns=cluster_columns),
            on="feature_id",
            how="left",
            validate="one_to_one",
        )
        table["cluster_class"] = table["cluster_class"].fillna("")
        table["cluster_neighbor_count"] = (
            pd.to_numeric(table["cluster_neighbor_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        table["cluster_hypothesis"] = table["cluster_hypothesis"].fillna(
            "not evaluated because feature did not pass significance gates"
        )
        table["homologous_series_neighbor_count"] = (
            pd.to_numeric(table["homologous_series_neighbor_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        table["homologous_series_mass_steps"] = table[
            "homologous_series_mass_steps"
        ].fillna("")
        table["homologous_series_hypothesis"] = table[
            "homologous_series_hypothesis"
        ].fillna("not evaluated because feature did not pass significance gates")
        table["significant_ch2_neighbor_count"] = (
            pd.to_numeric(table["significant_ch2_neighbor_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        table["significant_ch2_series_size"] = (
            pd.to_numeric(table["significant_ch2_series_size"], errors="coerce")
            .fillna(1)
            .astype(int)
        )
        for column in (
            "significant_ch2_neighbor_feature_ids",
            "significant_ch2_mass_steps",
            "significant_ch2_rt_deltas_min",
            "significant_ch2_series_id",
            "significant_ch2_cluster_hypothesis",
        ):
            table[column] = table[column].fillna(
                "not evaluated because feature did not pass significance gates"
                if column == "significant_ch2_cluster_hypothesis"
                else ""
            )
        by_stream[str(stream)] = table
    if progress:
        progress(.45, "Searching isotope envelopes and candidate formulas")
    by_stream = _attach_isotope_evidence(result, by_stream, criteria)
    if progress:
        progress(.65, "Searching adducts across acquisition modes")
    by_stream = _attach_cross_acquisition_evidence(result, by_stream, criteria,
        progress=(lambda count, total, message: progress(.65 + .08 * (count - 1) / max(total, 1), message)) if progress else None)
    adduct_matches = search_adduct_evidence(_coordinate_sources(result), by_stream,
        ppm=criteria.cross_mode_ppm_tolerance,
        rt_tolerance=criteria.opposite_polarity_rt_tolerance_min,
        progress=(lambda count, total, message: progress(.73 + .12 * (count - 1) / max(total, 1), message)) if progress else None)
    by_stream = attach_adduct_summaries(by_stream, adduct_matches)
    if progress:
        progress(.85, "Finding local CH2 homologous series")
    by_stream, homologous_matches = attach_lipid_series(by_stream, _coordinate_sources(result),
        {stream: _accepted_neighbour_frame(result, stream) for stream in by_stream}, criteria)
    combined = (
        pd.concat(by_stream.values(), ignore_index=True, sort=False)
        if by_stream
        else pd.DataFrame()
    )
    output = {
        "criteria": asdict(criteria),
        "evidence_version": EVIDENCE_VERSION,
        "adduct_matches": adduct_matches,
        "homologous_matches": homologous_matches,
        "by_stream": by_stream,
        "features": combined,
        "significant_features": (
            combined[combined["significant"]].copy()
            if not combined.empty
            else combined.copy()
        ),
        "raw_p_features": combined[combined["raw_p_significant"]].copy() if not combined.empty else combined.copy(),
        "group_catalog": group_catalog(result),
        "skipped_streams": pd.DataFrame(skipped),
        "method": (
            "Welch t-test on log2 intensities with half-minimum pseudocount; "
            "significance uses the unadjusted two-sided p-value"
        ),
        "interpretation_boundary": (
            "Local lipid-class, significant-feature CH2-series and isotope "
            "signals are interpretation hypotheses; they are not lipid "
            "identifications or exact molecular formulas."
        ),
    }
    if progress:
        progress(.94, "Assigning tentative compound IDs")
    return assign_compound_ids(output)


__all__ = [
    "UntargetedComparisonCriteria",
    "analyse_standard_untargeted",
    "attach_excel_sample_metadata",
    "default_biological_groups",
    "group_catalog",
    "is_biological_group",
    "paired_abundance_table",
]
