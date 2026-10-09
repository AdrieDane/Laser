"""Per-stream precursor-mass calibration for standard lipidomics."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from annotator.matching import apply_mass_calibration, is_mass_calibration_model
from output_io import atomic_write_bytes


STANDARD_MASS_CALIBRATION_STREAMS = (
    "NP_Neg",
    "NP_Pos",
    "RP_Neg",
    "RP_Pos",
)

CALIBRATED_XCMS_COLUMNS = (
    "LASER_original_mz",
    "LASER_mass_calibration_model",
    "LASER_mass_calibration_coefficients",
    "LASER_mass_calibration_slope",
    "LASER_mass_calibration_intercept",
    "LASER_mass_calibration_n_used",
    "LASER_mass_calibration_rmse_before_ppm",
    "LASER_mass_calibration_rmse_after_ppm",
)
_XCMS_MZ_COLUMNS = {
    "mzmed", "mz", "m/z", "mzmedian", "mz_med", "row m/z", "average mz",
}


def normalize_mass_calibration(calibration: object) -> dict | None:
    """Return one validated model from either UI or frozen config data."""
    if not isinstance(calibration, Mapping):
        return None
    values = dict(calibration)
    if str(values.get("mode", "")).casefold() == "none":
        return None
    if str(values.get("mode", "")).casefold() == "fixed":
        values["kind"] = str(values.get("model", values.get("kind", "linear")))
    coefficients = values.get("coefficients")
    if coefficients is None:
        coefficients = (values.get("slope", 1.0), values.get("intercept", 0.0))
    try:
        values["coefficients"] = [float(value) for value in coefficients]
        values["slope"] = float(values.get("slope", values["coefficients"][-2]))
        values["intercept"] = float(
            values.get("intercept", values["coefficients"][-1])
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Mass-calibration coefficients must be numeric") from exc
    if not is_mass_calibration_model(values):
        raise ValueError(
            f"Unsupported standard-lipidomics mass calibration: {values.get('kind')!r}"
        )
    probe = apply_mass_calibration(np.asarray([50.0, 500.0, 2000.0]), values)
    if not np.isfinite(probe).all() or not np.all(np.diff(probe) > 0):
        raise ValueError("Mass-calibration model must be finite and increasing")
    return values


def normalize_mass_calibrations(value: object) -> dict[str, dict]:
    """Validate a stream-to-model mapping and discard explicit skips."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("mass_calibrations must be a mapping keyed by XCMS stream")
    normalized: dict[str, dict] = {}
    for raw_stream, calibration in value.items():
        stream = str(raw_stream)
        if stream not in STANDARD_MASS_CALIBRATION_STREAMS:
            raise ValueError(f"Unknown standard-lipidomics stream: {stream!r}")
        model = normalize_mass_calibration(calibration)
        if model is not None:
            normalized[stream] = model
    return normalized


def freeze_mass_calibration(calibration: object) -> dict:
    """Return a JSON-safe fixed model for saved/headless configurations."""
    model = normalize_mass_calibration(calibration)
    if model is None:
        return {"mode": "none", "on_failure": "disable"}
    anchors = model.get("anchors")
    standard_features = (
        ";".join(anchors["feat_feature_id"].astype(str).drop_duplicates())
        if isinstance(anchors, pd.DataFrame)
        and "feat_feature_id" in anchors
        else str(model.get("standard_features", ""))
    )
    return {
        "mode": "fixed",
        "model": str(model["kind"]),
        "coefficients": [float(value) for value in model["coefficients"]],
        "slope": float(model["slope"]),
        "intercept": float(model["intercept"]),
        "n_used": int(model.get("n_used", 0)),
        "rmse_before_ppm": float(model.get("rmse_before_ppm", np.nan)),
        "rmse_after_ppm": float(model.get("rmse_after_ppm", np.nan)),
        "standard_features": standard_features,
        "source": "UI-reviewed per-stream calibration",
        "on_failure": "error",
    }


def mass_calibration_provenance(calibration: object) -> dict[str, object]:
    """Return the annotation-export fields for one frozen stream model."""
    model = freeze_mass_calibration(calibration)
    if model["mode"] != "fixed":
        return {}
    return {
        "mass_calibration_standard_features": model["standard_features"],
        "mass_calibration_model": model["model"],
        "mass_calibration_coefficients": str(model["coefficients"]),
        "mass_calibration_slope": model["slope"],
        "mass_calibration_intercept": model["intercept"],
        "mass_calibration_rmse_before_ppm": model["rmse_before_ppm"],
        "mass_calibration_rmse_after_ppm": model["rmse_after_ppm"],
    }


def apply_standard_mass_calibration(
    features: pd.DataFrame,
    calibration: object,
) -> pd.DataFrame:
    """Correct one XCMS feature table while retaining its original m/z."""
    model = normalize_mass_calibration(calibration)
    output = features.copy()
    output.attrs = dict(features.attrs)
    if model is None:
        return output
    measured = pd.to_numeric(output["mz"], errors="coerce")
    output["mz_raw"] = pd.to_numeric(
        output.get("mz_raw", measured), errors="coerce"
    )
    output["mz"] = apply_mass_calibration(output["mz_raw"], model)
    output.attrs["mass_calibration"] = freeze_mass_calibration(model)
    return output


def restore_calibrated_xcms_provenance(
    features: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Restore raw m/z and fixed-model metadata from a calibrated XCMS export."""
    columns = {str(column).casefold(): str(column) for column in raw.columns}
    raw_mz_column = columns.get("laser_original_mz")
    model_column = columns.get("laser_mass_calibration_model")
    coefficients_column = columns.get("laser_mass_calibration_coefficients")
    if (
        raw_mz_column is None
        or model_column is None
        or coefficients_column is None
        or len(raw) != len(features)
    ):
        return features
    output = features.copy()
    output.attrs = dict(features.attrs)
    output["mz_raw"] = pd.to_numeric(raw[raw_mz_column], errors="coerce")
    first = raw.iloc[0]

    def number(name: str, default: float) -> float:
        column = columns.get(name.casefold())
        if column is None:
            return default
        try:
            return float(first[column])
        except (TypeError, ValueError):
            return default

    coefficients = [
        float(value)
        for value in str(first[coefficients_column]).split(";")
        if str(value).strip()
    ]
    output.attrs["mass_calibration"] = {
        "mode": "fixed",
        "model": str(first[model_column]),
        "coefficients": coefficients,
        "slope": number("LASER_mass_calibration_slope", 1.0),
        "intercept": number("LASER_mass_calibration_intercept", 0.0),
        "n_used": int(number("LASER_mass_calibration_n_used", 0.0)),
        "rmse_before_ppm": number(
            "LASER_mass_calibration_rmse_before_ppm", float("nan")
        ),
        "rmse_after_ppm": number(
            "LASER_mass_calibration_rmse_after_ppm", float("nan")
        ),
        "standard_features": "",
        "source": "calibrated XCMS export",
        "on_failure": "error",
    }
    return output


def mass_calibration_summary(
    calibrations: Mapping[str, object] | None,
) -> pd.DataFrame:
    """Build a compact audit with one row for every standard stream."""
    values = calibrations or {}
    rows: list[dict[str, Any]] = []
    for stream in STANDARD_MASS_CALIBRATION_STREAMS:
        model = normalize_mass_calibration(values.get(stream))
        rows.append({
            "stream": stream,
            "status": "applied" if model is not None else "not_applied",
            "model": str(model.get("kind", "none")) if model else "none",
            "confirmed_standards": int(model.get("n_used", 0)) if model else 0,
            "rmse_before_ppm": model.get("rmse_before_ppm", np.nan) if model else np.nan,
            "rmse_after_ppm": model.get("rmse_after_ppm", np.nan) if model else np.nan,
            "slope": model.get("slope", 1.0) if model else 1.0,
            "intercept": model.get("intercept", 0.0) if model else 0.0,
            "coefficients": (
                ";".join(f"{float(item):.16g}" for item in model["coefficients"])
                if model else ""
            ),
        })
    return pd.DataFrame(rows)


def mass_calibration_anchors(
    calibrations: Mapping[str, object] | None,
) -> pd.DataFrame:
    """Combine UI-confirmed calibration anchors across all four streams."""
    frames = []
    for stream, calibration in (calibrations or {}).items():
        model = normalize_mass_calibration(calibration)
        anchors = model.get("anchors") if model else None
        if not isinstance(anchors, pd.DataFrame) or anchors.empty:
            continue
        table = anchors.copy()
        table.insert(0, "stream", str(stream))
        frames.append(table)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def attach_matlab_mass_calibration_result(
    result: dict,
    *,
    phase: str,
    calibrations: Mapping[str, object] | None,
) -> None:
    """Attach phase-local models and audit tables to a MATLAB result."""
    phase_calibrations = {
        stream: calibration
        for stream, calibration in (calibrations or {}).items()
        if str(stream).startswith(f"{phase}_")
    }
    summary = mass_calibration_summary(phase_calibrations)
    result["matlab_mass_calibration_audit"] = summary.loc[
        summary["stream"].str.startswith(f"{phase}_")
    ].reset_index(drop=True)
    result["matlab_mass_calibration_anchors"] = mass_calibration_anchors(
        phase_calibrations
    )
    result["mass_calibrations"] = {
        stream: freeze_mass_calibration(calibration)
        for stream, calibration in phase_calibrations.items()
    }


def _source_bytes(source: str | Path | bytes | Any) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    data = source.read()
    return data.encode("utf-8") if isinstance(data, str) else bytes(data)


def calibrate_standard_xcms_bytes(
    source: str | Path | bytes | Any,
    calibration: object,
) -> bytes:
    """Return a row-preserving XCMS table with calibrated precursor m/z.

    Only the detected XCMS m/z field is replaced. Every original field remains
    byte-for-byte unchanged and explicit ``LASER_*`` provenance columns are
    appended so a later annotation run can recover raw m/z and the fixed model
    without recalibrating the data.
    """
    model = normalize_mass_calibration(calibration)
    raw = _source_bytes(source)
    if model is None:
        return raw
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise ValueError("Cannot calibrate an empty XCMS feature table")

    def content_and_ending(line: bytes) -> tuple[bytes, bytes]:
        if line.endswith(b"\r\n"):
            return line[:-2], b"\r\n"
        if line.endswith((b"\n", b"\r")):
            return line[:-1], line[-1:]
        return line, b""

    header, header_ending = content_and_ending(lines[0])
    delimiter = b"\t" if header.count(b"\t") >= header.count(b",") else b","
    header_fields = header.decode("utf-8-sig", errors="replace").split(
        delimiter.decode("ascii")
    )
    if any(str(field).casefold().startswith("laser_mass_calibration_")
           for field in header_fields):
        raise ValueError(
            "This XCMS table already contains LASER mass-calibration provenance. "
            "Use the original XCMS export to avoid applying calibration twice."
        )
    mz_index = next(
        (
            index
            for index, field in enumerate(header_fields)
            if str(field).strip().casefold() in _XCMS_MZ_COLUMNS
        ),
        None,
    )
    if mz_index is None:
        raise ValueError(
            "Could not detect the XCMS m/z column. Expected mzmed, mz, or m/z."
        )

    frozen = freeze_mass_calibration(model)
    coefficients = ";".join(
        f"{float(value):.16g}" for value in frozen["coefficients"]
    )
    provenance = (
        str(frozen["model"]),
        coefficients,
        f"{float(frozen['slope']):.16g}",
        f"{float(frozen['intercept']):.16g}",
        str(int(frozen["n_used"])),
        f"{float(frozen['rmse_before_ppm']):.16g}",
        f"{float(frozen['rmse_after_ppm']):.16g}",
    )
    output = [
        header
        + delimiter
        + delimiter.join(column.encode("utf-8") for column in CALIBRATED_XCMS_COLUMNS)
        + header_ending
    ]
    for line in lines[1:]:
        content, ending = content_and_ending(line)
        if not content:
            output.append(line)
            continue
        fields = content.split(delimiter)
        if mz_index >= len(fields):
            output.append(
                content
                + (delimiter * len(CALIBRATED_XCMS_COLUMNS))
                + ending
            )
            continue
        original = fields[mz_index].decode("utf-8", errors="replace").strip()
        try:
            calibrated = float(apply_mass_calibration(
                np.asarray([float(original)], dtype=float), model
            )[0])
        except ValueError:
            calibrated = None
        if calibrated is not None and math.isfinite(calibrated):
            fields[mz_index] = f"{calibrated:.12g}".encode("utf-8")
        metadata = (original, *provenance)
        output.append(
            delimiter.join(fields)
            + delimiter
            + delimiter.join(value.encode("utf-8") for value in metadata)
            + ending
        )
    return b"".join(output)


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_calibrated_xcms_files(
    sources: Mapping[str, str | Path | bytes | Any],
    calibrations: Mapping[str, object],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export annotation-ready XCMS files and one calibration audit manifest.

    Reviewed streams are calibrated. Explicitly skipped streams are copied
    unchanged so the output directory remains a complete, directly selectable
    annotation input folder; the manifest records their ``not_applied`` state.
    """
    destination = Path(output_directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    planned = [
        destination / f"peakgrouplist_{stream}.txt"
        for stream in STANDARD_MASS_CALIBRATION_STREAMS
        if stream in sources
    ]
    planned.append(destination / "LASER_standard_mass_calibration.json")
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Calibration output already exists: "
            + ", ".join(path.name for path in existing)
            + ". Choose a new output folder."
        )
    paths: dict[str, Path] = {}
    manifest_streams: dict[str, dict[str, object]] = {}
    for stream in STANDARD_MASS_CALIBRATION_STREAMS:
        if stream not in sources:
            continue
        source = sources[stream]
        model = normalize_mass_calibration(calibrations.get(stream))
        path = destination / f"peakgrouplist_{stream}.txt"
        payload = (
            calibrate_standard_xcms_bytes(source, model)
            if model is not None
            else _source_bytes(source)
        )
        atomic_write_bytes(path, payload)
        paths[stream] = path
        manifest_streams[stream] = {
            "source": str(Path(source).resolve())
            if isinstance(source, (str, Path)) else "uploaded_bytes",
            "output": str(path.resolve()),
            "status": "applied" if model is not None else "not_applied",
            "calibration": freeze_mass_calibration(model),
        }
    if not paths:
        raise ValueError("No standard-lipidomics XCMS streams were supplied")
    manifest = {
        "schema_version": 1,
        "module": "standard_xcms_mass_calibration",
        "annotation_ready_directory": str(destination.resolve()),
        "streams": manifest_streams,
    }
    atomic_write_bytes(
        destination / "LASER_standard_mass_calibration.json",
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False).encode("utf-8")
        + b"\n",
    )
    return paths


__all__ = [
    "CALIBRATED_XCMS_COLUMNS",
    "STANDARD_MASS_CALIBRATION_STREAMS",
    "apply_standard_mass_calibration",
    "attach_matlab_mass_calibration_result",
    "calibrate_standard_xcms_bytes",
    "freeze_mass_calibration",
    "mass_calibration_anchors",
    "mass_calibration_provenance",
    "mass_calibration_summary",
    "normalize_mass_calibration",
    "normalize_mass_calibrations",
    "restore_calibrated_xcms_provenance",
    "write_calibrated_xcms_files",
]
