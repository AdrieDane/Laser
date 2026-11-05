"""XCMS export adapter for the RT models from LASER-4D.

4D maps library RT to observed RT. XCMS exports use its inverse to express
observed RT on the library scale, allowing both standard annotation engines to
consume the selected export. Original values are retained as audit columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.rt_calibration import is_rt_calibration_model, _apply_rt_arrays


def freeze_rt_calibration(model: dict | None) -> dict | None:
    if model is None:
        return None
    if not is_rt_calibration_model(model):
        raise ValueError("The RT calibration is invalid or unsupported.")
    def plain(value):
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    # Convert non-finite diagnostic values to JSON null without changing a model.
    def finite(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite(item) for item in value]
        return value
    return finite(json.loads(json.dumps(model, default=plain)))


def inverse_rt(values, model: dict) -> np.ndarray:
    """Express measured RT on the reference scale using a validated inverse."""
    if not is_rt_calibration_model(model):
        raise ValueError("The RT calibration is invalid or unsupported.")
    values = np.asarray(values, dtype=float)
    if model["kind"] == "monotonic_piecewise":
        return _apply_rt_arrays(
            values, kind="monotonic_piecewise", slope=1.0, intercept=0.0,
            reference_knots=np.asarray(model["adjusted_knots"], dtype=float),
            observed_knots=np.asarray(model["reference_knots"], dtype=float))
    return (values - float(model["intercept"])) / float(model["slope"])


def calibrate_standard_rt_bytes(source: str | Path | bytes, model: dict | None) -> bytes:
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    if model is None:
        return raw
    frozen = freeze_rt_calibration(model)
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise ValueError("Cannot calibrate an empty XCMS table.")
    def split(line):
        ending = b"\r\n" if line.endswith(b"\r\n") else line[-1:] if line.endswith((b"\n", b"\r")) else b""
        return (line[:-len(ending)] if ending else line), ending
    header, ending = split(lines[0])
    delimiter = b"\t" if header.count(b"\t") >= header.count(b",") else b","
    fields = header.decode("utf-8-sig").split(delimiter.decode())
    if any(name.casefold().startswith("laser_rt_calibration") for name in fields):
        raise ValueError("This table is already RT-calibrated. Select the original XCMS input.")
    indices = [index for index, name in enumerate(fields)
               if name.strip().casefold() in {"rtmed", "rt", "retention_time", "rtmin", "rtmax"}]
    if not any(fields[index].casefold() in {"rtmed", "rt", "retention_time"} for index in indices):
        raise ValueError("Could not detect the XCMS retention-time column.")
    extra = [f"LASER_original_{fields[index]}" for index in indices]
    # Hex encodes the audit JSON so both comma and tab source formats stay valid.
    extra += ["LASER_rt_calibration_hex"]
    audit = {key: value for key, value in frozen.items()
             if key in {"kind", "model", "slope", "intercept", "reference_knots", "adjusted_knots", "n_used"}}
    encoded = json.dumps(audit, allow_nan=False, separators=(",", ":")).encode().hex().encode()
    output = [header + delimiter + delimiter.join(name.encode() for name in extra) + ending]
    for line in lines[1:]:
        content, ending = split(line)
        if not content:
            output.append(line)
            continue
        row = content.split(delimiter)
        if len(row) != len(fields):
            raise ValueError("An XCMS row has an unexpected number of fields.")
        original = [row[index] for index in indices]
        for index in indices:
            try:
                value = float(row[index])
            except ValueError:
                continue
            if np.isfinite(value):
                row[index] = f"{inverse_rt([value], model)[0]:.12g}".encode()
        output.append(delimiter.join([*row, *original, encoded]) + ending)
    return b"".join(output)


def restore_rt_provenance(features: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    columns = {str(name).casefold(): name for name in raw.columns}
    encoded = columns.get("laser_rt_calibration_hex")
    if encoded is None or raw.empty:
        return features
    model = json.loads(bytes.fromhex(str(raw.iloc[0][encoded])).decode())
    if not is_rt_calibration_model(model):
        raise ValueError("The XCMS RT-calibration provenance is invalid.")
    output = features.copy()
    for name in ("rtmed", "rt", "retention_time"):
        if f"laser_original_{name}" in columns:
            output["rt_raw"] = pd.to_numeric(raw[columns[f"laser_original_{name}"]], errors="coerce")
            break
    output.attrs["rt_calibration"] = model
    return output
