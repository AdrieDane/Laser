"""Normal-phase retention-time reference and guarded anchor calibration.

The Lipid-Wizard table stores NP retention times in seconds at the identity
level used by the MATLAB workflow. Standard-mode LASER maps its more explicit
ether/plasmalogen and positional-lyso reference names onto those combined
identities, then fits one robust dataset offset from unambiguous mass matches.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from resource_paths import REFERENCE_DATA_DIRECTORY


DEFAULT_NP_RT_REFERENCE_PATH = (
    REFERENCE_DATA_DIRECTORY / "lipid_wizard_retention_times.csv"
)

MIN_NP_RT_ALIGNMENT_ANCHORS = 12
MIN_NP_RT_ALIGNMENT_CLASSES = 3
MIN_NP_RT_ALIGNMENT_SPAN_MIN = 2.0
MAX_NP_RT_ALIGNMENT_OFFSET_MIN = 1.0
MAX_NP_RT_ALIGNMENT_MAD_MIN = 0.15
MAX_NP_RT_ALIGNMENT_P90_MIN = 0.30
NP_RT_ALIGNMENT_HISTOGRAM_BIN_MIN = 0.05
NP_RT_ALIGNMENT_CLUSTER_HALF_WIDTH_MIN = 0.20

_POSITIONAL_LYSO_RE = re.compile(r"^(?:1-acyl|2-acyl)\s+")
_ETHER_RE = re.compile(r"\(O-(\d+):(\d+)\)")
_PLASMALOGEN_RE = re.compile(r"\(P-(\d+):(\d+)\)")
_NP_ETHER_PLASMALOGEN_RE = re.compile(
    r"^(?P<head>[^()]+)\((?P<form>O\+P|O|P)-(?P<carbon>\d+):(?P<db>\d+)\)$"
)


def combine_np_ether_plasmalogen_labels(
    frame: pd.DataFrame,
    *,
    species_column: str = "name",
) -> pd.DataFrame:
    """Label every coeluting NP ether/plasmalogen pair as shared O+P.

    ``O-n:d`` and ``P-n:(d-1)`` are the same unresolved NP sum identity.
    Original species/class labels are retained in audit columns.
    """
    output = frame.copy()
    if output.empty or species_column not in output:
        return output
    source_species_column = f"np_unresolved_source_{species_column}"
    output[source_species_column] = output[species_column].astype(str)
    if "class" in output:
        output["np_unresolved_source_class"] = output["class"].astype(str)
    replacements: dict[int, tuple[str, str, int]] = {}
    for index, species in output[species_column].astype(str).items():
        match = _NP_ETHER_PLASMALOGEN_RE.match(species)
        if not match:
            continue
        db = int(match.group("db")) + (1 if match.group("form") == "P" else 0)
        source_class = (
            str(output.at[index, "class"])
            if "class" in output else match.group("head")
        )
        if source_class.endswith("-O+P"):
            combined_class = source_class
        else:
            combined_class = re.sub(r"-(?:O|P)$", "-O+P", source_class)
        if combined_class == source_class and not source_class.endswith("-O+P"):
            combined_class = f"{source_class}-O+P"
        replacements[index] = (
            f"{match.group('head')}(O+P-{match.group('carbon')}:{db})",
            combined_class,
            db,
        )
    if not replacements:
        return output.drop(columns=[
            column for column in (
                source_species_column, "np_unresolved_source_class"
            ) if column in output
        ])
    for index, (combined_species, combined_class, db) in replacements.items():
        source_species = str(output.at[index, species_column])
        output.at[index, species_column] = combined_species
        if "class" in output:
            output.at[index, "class"] = combined_class
        if "base" in output:
            output.at[index, "base"] = ""
        for db_column in ("db", "double_bonds"):
            if db_column in output:
                output.at[index, db_column] = db
        if "display_name" in output:
            output.at[index, "display_name"] = str(
                output.at[index, "display_name"]
            ).replace(source_species, combined_species, 1)
    return output


def load_np_rt_reference(path: str | Path) -> pd.DataFrame:
    """Load Lipid-Wizard NP retention times and convert seconds to minutes."""
    source = Path(path).expanduser()
    data = pd.read_csv(source, encoding="utf-8-sig")
    required = {"name", "class", "NP_RT"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"NP RT reference {source} is missing columns: "
            f"{', '.join(sorted(missing))}"
        )
    data = data.copy()
    data["np_rt_seconds"] = pd.to_numeric(data["NP_RT"], errors="coerce")
    data = data[data["np_rt_seconds"].notna()].copy()
    data["np_rt_min"] = data["np_rt_seconds"] / 60.0
    if data["name"].astype(str).duplicated().any():
        duplicates = sorted(
            data.loc[
                data["name"].astype(str).duplicated(keep=False), "name"
            ].astype(str).unique()
        )
        raise ValueError(
            "NP RT reference contains duplicate lipid names: "
            + ", ".join(duplicates[:5])
        )
    data["np_rt_source"] = str(source)
    return data[
        [
            "name", "class", "np_rt_seconds", "np_rt_min",
            "np_rt_source",
        ]
    ].reset_index(drop=True)


def np_rt_identity(name: str, policy_class: str = "") -> str:
    """Map a LASER reference name to the MATLAB/Lipid-Wizard NP identity."""
    identity = str(name).strip()
    policy_class = str(policy_class).strip()
    if policy_class in {"LPC", "LPE"}:
        identity = _POSITIONAL_LYSO_RE.sub("", identity)
    if "[O+P]" not in policy_class:
        return identity

    ether = _ETHER_RE.search(identity)
    if ether:
        replacement = f"(O+P-{ether.group(1)}:{ether.group(2)})"
        return identity[:ether.start()] + replacement + identity[ether.end():]
    plasmalogen = _PLASMALOGEN_RE.search(identity)
    if plasmalogen:
        # O-n:d and P-n:(d-1) are the same unresolved NP composition.
        replacement = (
            f"(O+P-{plasmalogen.group(1)}:"
            f"{int(plasmalogen.group(2)) + 1})"
        )
        return (
            identity[:plasmalogen.start()]
            + replacement
            + identity[plasmalogen.end():]
        )
    return identity


def attach_np_rt_reference(
    reference: pd.DataFrame,
    np_reference: pd.DataFrame,
    stream: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach direct NP RT values and return only reference-supported rows."""
    data = reference.copy()
    policy_class = data.get(
        "matlab_policy_class", pd.Series("", index=data.index)
    ).fillna("").astype(str)
    data["np_rt_identity"] = [
        np_rt_identity(name, selected_class)
        for name, selected_class in zip(data["name"], policy_class)
    ]
    lookup = np_reference.set_index("name")
    aligned = lookup.reindex(data["np_rt_identity"].astype(str))
    data["np_rt_reference_class"] = aligned["class"].to_numpy()
    data["np_rt_reference_seconds"] = aligned[
        "np_rt_seconds"
    ].to_numpy()
    data["np_rt_reference_source"] = aligned["np_rt_source"].to_numpy()
    data["np_rt_reference_available"] = data[
        "np_rt_reference_seconds"
    ].notna()

    audit = data[
        [
            "name", "class", "adduct", "np_rt_identity",
            "np_rt_reference_class", "np_rt_reference_seconds",
            "np_rt_reference_available",
        ]
    ].copy()
    audit.insert(0, "stream", stream)

    supported = data[data["np_rt_reference_available"]].copy()
    supported["ref_rt"] = supported["np_rt_reference_seconds"] / 60.0
    # The source contains one RT value per identity. Dataset-level anchor
    # support is exported separately and must not be misreported here.
    supported["n_RT"] = 1
    return supported.reset_index(drop=True), audit.reset_index(drop=True)


def fit_np_rt_anchor_alignment(
    mass_pairs: pd.DataFrame,
    *,
    stream: str,
    rt_tolerance_min: float,
) -> pd.DataFrame:
    """Fit and validate one robust NP reference-to-dataset RT offset."""
    columns = [
        "stream", "model_type", "offset_min", "anchor_nodes",
        "anchor_classes", "reference_rt_span_min", "anchor_mad_min",
        "anchor_p90_abs_error_min", "validated", "validation_reason",
    ]
    if mass_pairs.empty:
        return pd.DataFrame([{
            "stream": stream,
            "model_type": "unvalidated_np_reference",
            "offset_min": 0.0,
            "anchor_nodes": 0,
            "anchor_classes": 0,
            "reference_rt_span_min": 0.0,
            "anchor_mad_min": np.nan,
            "anchor_p90_abs_error_min": np.nan,
            "validated": False,
            "validation_reason": "no_mass_pairs",
        }], columns=columns)

    pairs = mass_pairs.copy()
    pairs["raw_rt_error"] = (
        pd.to_numeric(pairs["feat_rt"], errors="coerce")
        - pd.to_numeric(pairs["ref_rt"], errors="coerce")
    )
    identity_column = (
        "np_rt_identity" if "np_rt_identity" in pairs else "name"
    )
    identity_count = pairs.groupby("feat_feature_id")[
        identity_column
    ].transform("nunique")
    seed_window = max(1.0, 3.0 * float(rt_tolerance_min))
    seeds = pairs[
        identity_count.eq(1)
        & pairs["raw_rt_error"].abs().le(seed_window)
        & pairs["ref_rt"].notna()
    ].copy()
    # A mass-only list contains many unrelated features. Find the dominant
    # residual cluster before selecting one feature per identity; choosing the
    # smallest raw residual first would incorrectly force a genuinely shifted
    # batch back toward zero.
    residual_pool = seeds["raw_rt_error"].dropna().to_numpy(float)
    if len(residual_pool):
        bin_width = NP_RT_ALIGNMENT_HISTOGRAM_BIN_MIN
        edges = np.arange(
            -seed_window,
            seed_window + 2 * bin_width,
            bin_width,
        )
        counts, edges = np.histogram(residual_pool, bins=edges)
        mode_index = int(np.argmax(counts))
        mode_centre = float(
            (edges[mode_index] + edges[mode_index + 1]) / 2.0
        )
        local = residual_pool[
            np.abs(residual_pool - mode_centre) <= bin_width
        ]
        if len(local):
            mode_centre = float(np.median(local))
        seeds["distance_to_shift_cluster"] = (
            seeds["raw_rt_error"] - mode_centre
        ).abs()
        seeds = seeds[
            seeds["distance_to_shift_cluster"].le(
                NP_RT_ALIGNMENT_CLUSTER_HALF_WIDTH_MIN
            )
        ]
    else:
        seeds["distance_to_shift_cluster"] = np.nan
    seeds = (
        seeds.sort_values(
            "distance_to_shift_cluster"
        )
        .drop_duplicates([identity_column, "adduct"], keep="first")
    )

    residuals = seeds["raw_rt_error"].to_numpy(float)
    offset = float(np.median(residuals)) if len(residuals) else 0.0
    centred = np.abs(residuals - offset)
    mad = float(np.median(centred)) if len(centred) else np.nan
    p90 = float(np.quantile(centred, 0.90)) if len(centred) else np.nan
    class_count = int(seeds["class"].astype(str).nunique())
    rt_span = (
        float(seeds["ref_rt"].max() - seeds["ref_rt"].min())
        if len(seeds) else 0.0
    )

    checks = {
        "too_few_anchors": len(seeds) < MIN_NP_RT_ALIGNMENT_ANCHORS,
        "too_few_classes": class_count < MIN_NP_RT_ALIGNMENT_CLASSES,
        "insufficient_rt_span": rt_span < MIN_NP_RT_ALIGNMENT_SPAN_MIN,
        "offset_too_large": abs(offset) > MAX_NP_RT_ALIGNMENT_OFFSET_MIN,
        "anchor_mad_too_large": not np.isfinite(mad)
        or mad > MAX_NP_RT_ALIGNMENT_MAD_MIN,
        "anchor_p90_too_large": not np.isfinite(p90)
        or p90 > MAX_NP_RT_ALIGNMENT_P90_MIN,
    }
    failures = [reason for reason, failed in checks.items() if failed]
    validated = not failures
    return pd.DataFrame([{
        "stream": stream,
        "model_type": (
            "validated_np_anchor_offset"
            if validated else "unvalidated_np_reference"
        ),
        "offset_min": offset if validated else 0.0,
        "anchor_nodes": int(len(seeds)),
        "anchor_classes": class_count,
        "reference_rt_span_min": rt_span,
        "anchor_mad_min": mad,
        "anchor_p90_abs_error_min": p90,
        "validated": validated,
        "validation_reason": "validated" if validated else ";".join(failures),
    }], columns=columns)
