"""Compact, indexed access to MZmine chromatogram and mobilogram traces.

MZmine's ``Export feature data`` XML preserves the complete feature data but is
too large to parse interactively (the bundled source XML files are several GB
each). This module stores only the arrays required by the review GUI in a small
SQLite database:

* chromatogram retention times and intensities;
* the summed mobilogram's mobility and intensity arrays;
* feature ID, raw-file/sample name and peak height.

The original XML remains the scientific source. No traces are reconstructed or
smoothed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4
import zlib

import numpy as np


TRACE_DB_FORMAT_VERSION = "3"
SUMMED_CHROMATOGRAM_BIN_WIDTH_MIN = 0.01

_FEATURE_START = re.compile(r'<feature rawdatafile="([^"]+)">')
_ROW_START = re.compile(r'<row id="([^"]+)">')
_HEIGHT = re.compile(r'<datatype type="height">([^<]+)</datatype>')
_ARRAY = re.compile(r"<(intensities|rts|mobilities)\b[^>]*>(.*?)</\1>")
_ROOT_ROWS = re.compile(r'numberofrows="(\d+)"')


@dataclass
class FeatureTrace:
    """One feature trace from one MZmine raw-data file."""

    feature_id: str
    sample_name: str
    peak_height: float
    rt_values: np.ndarray
    rt_intensities: np.ndarray
    mobility_values: np.ndarray
    mobility_intensities: np.ndarray
    is_representative: bool = False


@dataclass
class SummedFeatureChromatogram:
    """Feature-derived TIC for one injection.

    This is the sum of all exported MZmine feature chromatograms, not a raw
    scan-level TIC.  Keeping the distinction in the type and report prevents a
    method-QC diagnostic from being overstated as raw-data evidence.
    """

    sample_name: str
    rt_values: np.ndarray
    intensities: np.ndarray


def _parse_array(text: str) -> np.ndarray:
    if not text.strip():
        return np.empty(0, dtype=np.float32)
    return np.fromstring(text, sep=";", dtype=np.float32)


def _pack_array(values: np.ndarray) -> bytes | None:
    values = np.asarray(values, dtype="<f4")
    if not len(values):
        return None
    return zlib.compress(values.tobytes(), level=6)


def _unpack_array(data: bytes | None) -> np.ndarray:
    if data is None:
        return np.empty(0, dtype=float)
    return np.frombuffer(zlib.decompress(data), dtype="<f4").astype(float)


def _dataset_key(name: str | Path) -> str:
    path = Path(name)
    if path.suffix.casefold() == ".gz":
        path = path.with_suffix("")
    stem = path.stem.upper()
    for suffix in (
        "_LIPID_TRACES",
        "_FEATURE_TRACES",
        "_TRACE_BUNDLE",
        "_FEATURE_DATA",
        "_SCAN_EXPORT",
        "_EXPORT",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_associated_feature_trace_db(
    feature_name: str,
    candidates: list[str] | tuple[str, ...],
) -> str | None:
    """Return the best companion trace database for a feature-table name."""
    feature_key = _dataset_key(feature_name)
    feature_parent = Path(feature_name).parent
    matches = [
        candidate for candidate in candidates if _dataset_key(candidate) == feature_key
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda candidate: (
            Path(candidate).parent != feature_parent,
            len(Path(candidate).name),
            candidate,
        ),
    )[0]


def _connect_for_write(
    path: Path,
    mode: str,
    source_xml: Path,
    selection_metadata: dict[str, str] | None = None,
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE summed_chromatograms (
            sample_name TEXT PRIMARY KEY,
            bin_width_min REAL NOT NULL,
            rt_values BLOB NOT NULL,
            intensities BLOB NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE traces (
            feature_id TEXT NOT NULL,
            sample_name TEXT NOT NULL,
            peak_height REAL NOT NULL,
            is_representative INTEGER NOT NULL DEFAULT 0,
            rt_values BLOB NOT NULL,
            rt_intensities BLOB NOT NULL,
            mobility_values BLOB,
            mobility_intensities BLOB,
            PRIMARY KEY (feature_id, sample_name)
        )
        """
    )
    metadata = {
        "format_version": TRACE_DB_FORMAT_VERSION,
        "mode": mode,
        "source_xml": source_xml.name,
    }
    metadata.update(
        {str(key): str(value) for key, value in (selection_metadata or {}).items()}
    )
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        sorted(metadata.items()),
    )
    return connection


def _temporary_database_path(destination: Path) -> Path:
    """Return a same-directory build path suitable for atomic publication."""
    return destination.with_name(f".{destination.name}.{uuid4().hex}.building")


def _discard_database_builds(*paths: Path | None) -> None:
    for path in paths:
        if path is not None:
            path.unlink(missing_ok=True)


def _trace_row(trace: FeatureTrace) -> tuple:
    return (
        trace.feature_id,
        trace.sample_name,
        float(trace.peak_height),
        int(trace.is_representative),
        _pack_array(trace.rt_values),
        _pack_array(trace.rt_intensities),
        _pack_array(trace.mobility_values),
        _pack_array(trace.mobility_intensities),
    )


def _insert_trace(connection: sqlite3.Connection, trace: FeatureTrace) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO traces(
            feature_id, sample_name, peak_height, is_representative,
            rt_values, rt_intensities, mobility_values, mobility_intensities
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _trace_row(trace),
    )


def _accumulate_chromatogram(
    totals: dict[str, np.ndarray],
    trace: FeatureTrace,
    bin_width_min: float = SUMMED_CHROMATOGRAM_BIN_WIDTH_MIN,
) -> None:
    rts = np.asarray(trace.rt_values, dtype=float)
    intensities = np.asarray(trace.rt_intensities, dtype=float)
    valid = np.isfinite(rts) & np.isfinite(intensities) & (rts >= 0)
    if not valid.any():
        return
    bins = np.rint(rts[valid] / float(bin_width_min)).astype(int)
    values = np.bincount(
        bins,
        weights=np.maximum(intensities[valid], 0.0),
    )
    current = totals.get(trace.sample_name)
    if current is None:
        totals[trace.sample_name] = values
        return
    if len(current) < len(values):
        current = np.pad(current, (0, len(values) - len(current)))
        totals[trace.sample_name] = current
    current[: len(values)] += values


def _store_summed_chromatograms(
    connection: sqlite3.Connection,
    totals: dict[str, np.ndarray],
    bin_width_min: float = SUMMED_CHROMATOGRAM_BIN_WIDTH_MIN,
) -> None:
    rows = []
    for sample_name, intensities in sorted(totals.items()):
        nonzero = np.flatnonzero(np.asarray(intensities) > 0)
        if not len(nonzero):
            continue
        first = max(int(nonzero[0]) - 1, 0)
        last = min(int(nonzero[-1]) + 2, len(intensities))
        values = np.asarray(intensities[first:last], dtype=float)
        rts = np.arange(first, last, dtype=float) * float(bin_width_min)
        rows.append(
            (
                sample_name,
                float(bin_width_min),
                _pack_array(rts),
                _pack_array(values),
            )
        )
    connection.executemany(
        """
        INSERT OR REPLACE INTO summed_chromatograms(
            sample_name, bin_width_min, rt_values, intensities
        ) VALUES (?, ?, ?, ?)
        """,
        rows,
    )


def _finish_database(
    connection: sqlite3.Connection,
    source_rows: int | None,
    rows_seen: int,
    traces_written: int,
) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [
            ("source_rows", str(source_rows or "")),
            ("rows_seen", str(rows_seen)),
            ("traces_written", str(traces_written)),
        ],
    )
    connection.execute(
        "CREATE INDEX traces_feature_idx ON traces(feature_id, is_representative)"
    )
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


def _publish_database_builds(
    representative_db: sqlite3.Connection,
    representative_build_path: Path,
    representative_path: Path,
    all_db: sqlite3.Connection | None,
    all_build_path: Path | None,
    all_path: Path | None,
    *,
    source_rows: int | None,
    rows_seen: int,
    representative_written: int,
    all_written: int,
    summed_chromatograms: dict[str, np.ndarray],
) -> None:
    published = False
    try:
        _finish_database(
            representative_db, source_rows, rows_seen, representative_written
        )
        if all_db is not None:
            _store_summed_chromatograms(all_db, summed_chromatograms)
            _finish_database(all_db, source_rows, rows_seen, all_written)
        if all_build_path is not None and all_path is not None:
            os.replace(all_build_path, all_path)
        os.replace(representative_build_path, representative_path)
        published = True
    finally:
        if not published:
            try:
                representative_db.close()
            except sqlite3.Error:
                pass
            if all_db is not None:
                try:
                    all_db.close()
                except sqlite3.Error:
                    pass
            _discard_database_builds(representative_build_path, all_build_path)


def _complete_database_build(
    *,
    representative_db: sqlite3.Connection,
    representative_build_path: Path,
    representative_path: Path,
    all_db: sqlite3.Connection | None,
    all_build_path: Path | None,
    all_path: Path | None,
    source_path: Path,
    source_rows: int | None,
    rows_seen: int,
    traces_seen: int,
    representative_written: int,
    all_written: int,
    summed_chromatograms: dict[str, np.ndarray],
    allowed_feature_ids: set[str] | None,
) -> dict[str, int | str]:
    if source_rows is not None and rows_seen != source_rows:
        representative_db.close()
        if all_db is not None:
            all_db.close()
        _discard_database_builds(representative_build_path, all_build_path)
        raise ValueError(
            f"Expected {source_rows:,} rows but parsed {rows_seen:,} from {source_path}"
        )
    _publish_database_builds(
        representative_db,
        representative_build_path,
        representative_path,
        all_db,
        all_build_path,
        all_path,
        source_rows=source_rows,
        rows_seen=rows_seen,
        representative_written=representative_written,
        all_written=all_written,
        summed_chromatograms=summed_chromatograms,
    )
    return {
        "source_xml": str(source_path),
        "rows_seen": rows_seen,
        "traces_seen": traces_seen,
        "representative_traces": representative_written,
        "all_sample_traces": all_written,
        "requested_feature_ids": len(allowed_feature_ids or ()),
    }


def build_feature_trace_databases(
    source_xml: str | Path,
    representative_output: str | Path,
    all_samples_output: str | Path | None = None,
    feature_ids: set[str] | set[int] | None = None,
    progress_every: int = 1000,
    selection_metadata: dict[str, str] | None = None,
) -> dict[str, int | str]:
    """Stream feature XML into atomic representative/all-sample trace databases.

    Optional feature IDs limit parsing; selection metadata keeps that choice
    auditable in both outputs.
    """
    source_path = Path(source_xml)
    representative_path = Path(representative_output)
    all_path = Path(all_samples_output) if all_samples_output else None
    representative_build_path = _temporary_database_path(representative_path)
    all_build_path = (
        _temporary_database_path(all_path) if all_path is not None else None
    )
    allowed_feature_ids = (
        {str(feature_id) for feature_id in feature_ids}
        if feature_ids is not None
        else None
    )
    database_metadata = dict(selection_metadata or {})
    database_metadata.setdefault(
        "selection_feature_ids",
        str(len(allowed_feature_ids)) if allowed_feature_ids is not None else "all",
    )
    representative_db = _connect_for_write(
        representative_build_path,
        "representative",
        source_path,
        database_metadata,
    )
    all_db = None
    if all_build_path is not None:
        all_db = _connect_for_write(
            all_build_path,
            "all_samples",
            source_path,
            database_metadata,
        )

    source_rows = None
    rows_seen = 0
    traces_seen = 0
    representative_written = 0
    all_written = 0
    current_row_id: str | None = None
    include_current_row = False
    current_feature: dict | None = None
    representative: FeatureTrace | None = None
    in_series = False
    in_summed_mobilogram = False
    in_frame_mobilogram = False
    summed_chromatograms: dict[str, np.ndarray] = {}

    try:
        with source_path.open("r", encoding="utf-8", errors="strict") as source:
            for line_number, line in enumerate(source, start=1):
                stripped = line.strip()
                if source_rows is None and "<featurelist " in stripped:
                    root_match = _ROOT_ROWS.search(stripped)
                    source_rows = int(root_match.group(1)) if root_match else None

                row_match = _ROW_START.search(stripped)
                if row_match:
                    current_row_id = row_match.group(1)
                    include_current_row = (
                        allowed_feature_ids is None
                        or current_row_id in allowed_feature_ids
                    )
                    representative = None
                    continue

                feature_match = _FEATURE_START.search(stripped)
                if feature_match and current_row_id is not None and include_current_row:
                    current_feature = {
                        "feature_id": current_row_id,
                        "sample_name": feature_match.group(1),
                        "peak_height": np.nan,
                        "rt_values": np.empty(0, dtype=np.float32),
                        "rt_intensities": np.empty(0, dtype=np.float32),
                        "mobility_values": np.empty(0, dtype=np.float32),
                        "mobility_intensities": np.empty(0, dtype=np.float32),
                    }
                    in_series = False
                    in_summed_mobilogram = False
                    in_frame_mobilogram = False
                    continue

                if current_feature is not None:
                    height_match = _HEIGHT.search(stripped)
                    if height_match and not in_series:
                        current_feature["peak_height"] = float(height_match.group(1))

                    if (
                        "<simpleiontimeseries>" in stripped
                        or "<simpleionmobilogramtimeseries>" in stripped
                    ):
                        in_series = True
                        continue
                    if "<summedmobilogram " in stripped:
                        in_summed_mobilogram = True
                        continue
                    if "</summedmobilogram>" in stripped:
                        in_summed_mobilogram = False
                        continue
                    if "<ionmobilityseries " in stripped:
                        in_frame_mobilogram = True
                        continue
                    if "</ionmobilityseries>" in stripped:
                        in_frame_mobilogram = False
                        continue

                    if in_series and not in_frame_mobilogram:
                        array_match = _ARRAY.search(stripped)
                        if array_match:
                            tag, values_text = array_match.groups()
                            values = _parse_array(values_text)
                            if in_summed_mobilogram:
                                if tag == "mobilities":
                                    current_feature["mobility_values"] = values
                                elif tag == "intensities":
                                    current_feature["mobility_intensities"] = values
                            elif tag == "rts":
                                current_feature["rt_values"] = values
                            elif tag == "intensities" and not len(
                                current_feature["rt_intensities"]
                            ):
                                current_feature["rt_intensities"] = values

                    if (
                        "</simpleiontimeseries>" in stripped
                        or "</simpleionmobilogramtimeseries>" in stripped
                    ):
                        in_series = False
                        continue

                    if stripped == "</feature>":
                        rt_values = current_feature["rt_values"]
                        rt_intensities = current_feature["rt_intensities"]
                        mobility_values = current_feature["mobility_values"]
                        mobility_intensities = current_feature["mobility_intensities"]
                        if len(rt_values) != len(rt_intensities):
                            raise ValueError(
                                f"RT array length mismatch for feature {current_row_id} "
                                f"at XML line {line_number}"
                            )
                        if len(mobility_values) != len(mobility_intensities):
                            raise ValueError(
                                f"Mobility array length mismatch for feature "
                                f"{current_row_id} at XML line {line_number}"
                            )
                        if len(rt_values):
                            peak_height = current_feature["peak_height"]
                            if not np.isfinite(peak_height):
                                peak_height = float(np.nanmax(rt_intensities))
                            trace = FeatureTrace(
                                feature_id=current_feature["feature_id"],
                                sample_name=current_feature["sample_name"],
                                peak_height=float(peak_height),
                                rt_values=rt_values,
                                rt_intensities=rt_intensities,
                                mobility_values=mobility_values,
                                mobility_intensities=mobility_intensities,
                            )
                            traces_seen += 1
                            if all_db is not None:
                                _insert_trace(all_db, trace)
                                _accumulate_chromatogram(summed_chromatograms, trace)
                                all_written += 1
                            if (
                                representative is None
                                or trace.peak_height > representative.peak_height
                            ):
                                representative = trace
                        current_feature = None
                        continue

                if stripped == "</row>" and current_row_id is not None:
                    rows_seen += 1
                    if representative is not None:
                        representative.is_representative = True
                        _insert_trace(representative_db, representative)
                        representative_written += 1
                        if all_db is not None:
                            all_db.execute(
                                """
                                UPDATE traces SET is_representative=1
                                WHERE feature_id=? AND sample_name=?
                                """,
                                (representative.feature_id, representative.sample_name),
                            )
                    if rows_seen % progress_every == 0:
                        if all_db is not None:
                            all_db.commit()
                        representative_db.commit()
                        print(
                            f"{source_path.name}: {rows_seen:,}/"
                            f"{source_rows or 0:,} rows, {traces_seen:,} traces",
                            flush=True,
                        )
                    current_row_id = None
                    include_current_row = False
                    representative = None
    except Exception:
        representative_db.close()
        if all_db is not None:
            all_db.close()
        _discard_database_builds(representative_build_path, all_build_path)
        raise

    return _complete_database_build(
        representative_db=representative_db,
        representative_build_path=representative_build_path,
        representative_path=representative_path,
        all_db=all_db,
        all_build_path=all_build_path,
        all_path=all_path,
        source_path=source_path,
        source_rows=source_rows,
        rows_seen=rows_seen,
        traces_seen=traces_seen,
        representative_written=representative_written,
        all_written=all_written,
        summed_chromatograms=summed_chromatograms,
        allowed_feature_ids=allowed_feature_ids,
    )


def load_feature_traces(
    database: str | Path,
    feature_id: str | int,
) -> list[FeatureTrace]:
    """Load all stored sample traces for one feature, representative first."""
    path = Path(database)
    if not path.exists():
        return []
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT feature_id, sample_name, peak_height, is_representative,
                   rt_values, rt_intensities, mobility_values, mobility_intensities
            FROM traces
            WHERE feature_id=?
            ORDER BY is_representative DESC, peak_height DESC, sample_name
            """,
            (str(feature_id),),
        ).fetchall()
    finally:
        connection.close()
    return [
        FeatureTrace(
            feature_id=row[0],
            sample_name=row[1],
            peak_height=float(row[2]),
            is_representative=bool(row[3]),
            rt_values=_unpack_array(row[4]),
            rt_intensities=_unpack_array(row[5]),
            mobility_values=_unpack_array(row[6]),
            mobility_intensities=_unpack_array(row[7]),
        )
        for row in rows
    ]


def load_summed_feature_chromatograms(
    database: str | Path,
) -> list[SummedFeatureChromatogram]:
    """Load feature-derived TICs, with a compatibility fallback for v2 DBs."""
    path = Path(database)
    if not path.exists():
        return []
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        rows = []
        if "summed_chromatograms" in tables:
            rows = connection.execute(
                """
                SELECT sample_name, rt_values, intensities
                FROM summed_chromatograms ORDER BY sample_name
                """
            ).fetchall()
        if rows:
            return [
                SummedFeatureChromatogram(
                    sample_name=row[0],
                    rt_values=_unpack_array(row[1]),
                    intensities=_unpack_array(row[2]),
                )
                for row in rows
            ]

        # Existing v2 databases remain readable. Aggregate once at report time;
        # new managed runs store the result during XML conversion.
        totals: dict[str, np.ndarray] = {}
        for sample_name, rt_blob, intensity_blob in connection.execute(
            "SELECT sample_name, rt_values, rt_intensities FROM traces"
        ):
            trace = FeatureTrace(
                feature_id="",
                sample_name=sample_name,
                peak_height=np.nan,
                rt_values=_unpack_array(rt_blob),
                rt_intensities=_unpack_array(intensity_blob),
                mobility_values=np.empty(0),
                mobility_intensities=np.empty(0),
            )
            _accumulate_chromatogram(totals, trace)
        result = []
        for sample_name, intensities in sorted(totals.items()):
            nonzero = np.flatnonzero(intensities > 0)
            if not len(nonzero):
                continue
            first = max(int(nonzero[0]) - 1, 0)
            last = min(int(nonzero[-1]) + 2, len(intensities))
            result.append(
                SummedFeatureChromatogram(
                    sample_name=sample_name,
                    rt_values=(
                        np.arange(first, last, dtype=float)
                        * SUMMED_CHROMATOGRAM_BIN_WIDTH_MIN
                    ),
                    intensities=np.asarray(intensities[first:last], dtype=float),
                )
            )
        return result
    finally:
        connection.close()


def feature_trace_database_metadata(database: str | Path) -> dict[str, str]:
    """Return trace-database provenance without loading any trace arrays."""
    path = Path(database)
    if not path.exists():
        return {}
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        return dict(connection.execute("SELECT key, value FROM metadata"))
    finally:
        connection.close()
