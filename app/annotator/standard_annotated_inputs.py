"""Append final LASER identities to conventional XCMS input text files."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Mapping

import pandas as pd

from output_io import atomic_write_bytes


ANNOTATION_COLUMN = "LASER_annotation"
ADDUCT_COLUMN = "LASER_adduct"


def _source_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    data = source.read()
    return data.encode("utf-8") if isinstance(data, str) else bytes(data)


def _annotation_lookup(
    annotations: pd.DataFrame, stream: str,
) -> dict[str, tuple[str, str]]:
    if annotations is None or annotations.empty:
        return {}
    frame = annotations.copy()
    stream_column = next(
        (name for name in ("stream", "standard_stream") if name in frame),
        None,
    )
    if stream_column is not None:
        frame = frame[frame[stream_column].astype(str).eq(str(stream))]
    feature_column = next(
        (name for name in ("feature_id", "feat_feature_id") if name in frame),
        None,
    )
    identity_column = next(
        (name for name in ("feature_identity", "name", "species") if name in frame),
        None,
    )
    if feature_column is None or identity_column is None or "adduct" not in frame:
        return {}
    lookup: dict[str, tuple[str, str]] = {}
    for feature_id, group in frame.groupby(
        frame[feature_column].fillna("").astype(str), sort=False
    ):
        pairs: list[tuple[str, str]] = []
        for _, row in group.iterrows():
            identity_value = row.get(identity_column, "")
            adduct_value = row.get("adduct", "")
            pair = (
                "" if pd.isna(identity_value) else str(identity_value),
                "" if pd.isna(adduct_value) else str(adduct_value),
            )
            if pair[0] and pair not in pairs:
                pairs.append(pair)
        lookup[str(feature_id)] = (
            "; ".join(identity for identity, _adduct in pairs),
            "; ".join(adduct for _identity, adduct in pairs),
        )
    return lookup


def append_standard_annotations(
    source: str | Path | bytes | BinaryIO,
    annotations: pd.DataFrame,
    *,
    stream: str,
) -> bytes:
    """Return the input text with two columns appended and no row reordering.

    The original text before the two appended fields is copied verbatim. This
    avoids round-trip changes to numeric formatting, column order, or row order.
    The input is expected to be the tab-delimited XCMS peak-group format used by
    standard lipidomics.
    """
    raw = _source_bytes(source)
    lines = raw.splitlines(keepends=True)
    if not lines:
        return f"{ANNOTATION_COLUMN}\t{ADDUCT_COLUMN}\n".encode("utf-8")

    def content_and_ending(line: bytes) -> tuple[bytes, bytes]:
        if line.endswith(b"\r\n"):
            return line[:-2], b"\r\n"
        if line.endswith((b"\n", b"\r")):
            return line[:-1], line[-1:]
        return line, b""

    header, header_ending = content_and_ending(lines[0])
    header_text = header.decode("utf-8-sig", errors="replace")
    header_fields = header_text.split("\t")
    id_index = next(
        (
            index for index, value in enumerate(header_fields)
            if str(value).strip().casefold() in {"name", "feature", "feature_id", "id"}
        ),
        None,
    )
    if id_index is None:
        raise ValueError(
            f"Could not find the XCMS feature ID column in {stream}."
        )
    lookup = _annotation_lookup(annotations, stream)
    output = [
        header + f"\t{ANNOTATION_COLUMN}\t{ADDUCT_COLUMN}".encode("utf-8")
        + header_ending
    ]
    for line in lines[1:]:
        content, ending = content_and_ending(line)
        fields = content.split(b"\t")
        feature_id = (
            fields[id_index].decode("utf-8", errors="replace")
            if id_index < len(fields) else ""
        )
        identity, adduct = lookup.get(str(feature_id), ("", ""))
        output.append(
            content + b"\t" + identity.encode("utf-8") + b"\t"
            + adduct.encode("utf-8") + ending
        )
    return b"".join(output)


def write_standard_annotated_inputs(
    sources: Mapping[str, str | Path | bytes | BinaryIO],
    annotations: pd.DataFrame,
    output_directory: str | Path,
) -> dict[str, Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for stream, source in sources.items():
        source_name = (
            Path(source).name if isinstance(source, (str, Path))
            else f"peakgrouplist_{stream}.txt"
        )
        stem = Path(source_name).stem
        path = destination / f"{stem}_LASER_annotated.txt"
        atomic_write_bytes(
            path,
            append_standard_annotations(source, annotations, stream=stream)
        )
        paths[str(stream)] = path
    return paths


__all__ = [
    "ADDUCT_COLUMN", "ANNOTATION_COLUMN", "append_standard_annotations",
    "write_standard_annotated_inputs",
]
