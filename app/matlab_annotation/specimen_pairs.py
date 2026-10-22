"""Explicit cross-mode specimen links from non-annotation acquisition metadata.

No pairing inferred from filename offsets/order or lipid intensities. Caller
may load ONLY MS_metadata; curated MS/MS_data structures are not inputs.
"""

from __future__ import annotations

import re

import numpy as np


def _text(value):
    a = np.asarray(value, dtype=object).reshape(-1)
    return str(a[0]).strip() if len(a) == 1 else ""


def _filename(value):
    name = re.split(r"[/\\]", _text(value))[-1].strip().casefold()
    return re.sub(r"\.(?:mzml|mzxml|raw|cdf)$", "", name)


def _group(value):
    value = re.sub(r"^group[_ -]*\d+[_ -]*", "", _text(value), flags=re.I)
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def explicit_specimen_pairs(metadata, features_by_stream):
    """Validate exact filenames, stream and group; return sample-column->ID.

    Physical specimen keys come ONLY from shared metadata.sample_id rows. Both
    mode sign and phase are explicit per cell; channel/row order is irrelevant.
    Conflicting file/ID/group links fail explicitly, not first-match heuristics.
    Technical replicate columns may share a specimen ID only in the same group;
    downstream profiles must aggregate those rather than multiply sample count.
    Unmatched original sample columns remain unmapped/unknown and are reported.
    """
    required = {"filename", "sample_id", "column", "mode", "group", "group_label"}
    if not required.issubset(metadata):
        raise ValueError("Explicit specimen metadata fields are missing")
    ids = np.atleast_1d(metadata["sample_id"]).reshape(-1)
    groups = np.atleast_1d(metadata["group"]).reshape(-1)
    labels = np.atleast_1d(metadata["group_label"]).reshape(-1)
    n = len(ids)
    if not n or len(groups) != n:
        raise ValueError("Specimen IDs/group rows must align")

    def matrix(key):
        a = np.asarray(metadata[key], dtype=object)
        if a.ndim == 1 and n == 1:
            a = a.reshape(1, -1)
        if a.ndim != 2 or a.shape[0] != n:
            raise ValueError(
                "Metadata filename/phase/mode matrices must align with specimen rows"
            )
        return a

    files, phases, modes = matrix("filename"), matrix("column"), matrix("mode")
    if files.shape != phases.shape or files.shape != modes.shape:
        raise ValueError("Metadata measurement channel shapes disagree")
    specimen_groups = {}
    links = {}
    expected_group = {}
    for i in range(n):
        specimen = _text(ids[i])
        if not specimen or specimen.casefold() in {"nan", "none"}:
            raise ValueError("Physical specimen IDs must be explicit")
        try:
            gid = int(groups[i])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Metadata group indices must be integers") from exc
        if float(groups[i]) != gid or not 1 <= gid <= len(labels):
            raise ValueError("Metadata group index outside label table")
        if specimen in specimen_groups and specimen_groups[specimen] != gid:
            raise ValueError("One specimen ID belongs to conflicting biological groups")
        specimen_groups[specimen] = gid
        for j in range(files.shape[1]):
            filename = _filename(files[i, j])
            if not filename or filename in {"nan", "none"}:
                continue
            phase = _text(phases[i, j]).upper()
            try:
                mode = float(modes[i, j])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Measurement mode must explicitly identify positive/negative ions"
                ) from exc
            if phase not in {"NP", "RP"} or mode not in {-1, 1}:
                raise ValueError("Unknown explicit acquisition phase/mode")
            stream = phase + ("_Pos" if mode == 1 else "_Neg")
            key = (stream, filename)
            if key in links and links[key] != specimen:
                raise ValueError(
                    "One acquisition filename maps to multiple specimen IDs"
                )
            links[key] = specimen
            expected_group[key] = _group(labels[gid - 1])
    mapping = {}
    audit = {}
    for stream, features in features_by_stream.items():
        if stream not in {"NP_Pos", "NP_Neg", "RP_Pos", "RP_Neg"}:
            raise ValueError("Unknown feature stream")
        if features.attrs.get("sample_group_mapping_status") != "validated":
            raise ValueError("Raw feature group mapping must be validated first")
        columns = [
            str(c)
            for c in features.attrs.get("sample_columns", ())
            if str(c) in features
        ]
        names = [_filename(c) for c in columns]
        if len(set(names)) != len(names):
            raise ValueError("Original sample-column filenames must be unambiguous")
        group_map = features.attrs.get("sample_group_map", {})
        result = {}
        for col, name in zip(columns, names):
            key = (stream, name)
            if key not in links:
                continue
            if _group(group_map.get(col, "")) != expected_group[key]:
                raise ValueError(
                    "Explicit specimen group disagrees with original peak-list sample group"
                )
            result[col] = links[key]
        mapping[stream] = result
        audit[stream] = dict(
            raw_sample_columns=len(columns),
            mapped_columns=len(result),
            unique_specimens=len(set(result.values())),
            unmapped_columns=len(columns) - len(result),
            mapping_source="explicit_MS_metadata_sample_id_filename_phase_mode_group",
        )
    return mapping, audit
