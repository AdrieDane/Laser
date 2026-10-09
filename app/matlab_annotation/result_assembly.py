"""Result-table assembly shared by automatic MATLAB annotation phases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def assemble_result(
    streams: dict[str, dict],
    outcomes: list[Any],
    annotation_dir: Path,
    reference_root: Path,
) -> dict:
    """Combine phase streams into the dictionary standard mode renders."""
    exports = []
    for phase, stream in streams.items():
        annotations = stream["annotations"]
        if annotations.empty:
            continue
        export = annotations.copy()
        export.insert(0, "stream", phase)
        exports.append(export)
    all_candidates = (
        pd.concat(exports, ignore_index=True, sort=False) if exports else pd.DataFrame()
    )
    if all_candidates.empty:
        automatic = identified = equivalents = all_candidates.copy()
    else:
        automatic = all_candidates.copy()
        # One quantitative primary per lipid species: the monoisotopic peak of
        # the most intense feature. Isotopologues and repeats remain equivalents.
        automatic["_monoisotopic"] = (
            automatic["isotopologue"].astype(str).eq("").map({True: 0, False: 1})
        )
        automatic = automatic.sort_values(
            ["name", "_monoisotopic", "feat_intensity"],
            ascending=[True, True, False],
            kind="stable",
        )
        automatic["quantitative_role"] = np.where(
            automatic.groupby("name").cumcount().eq(0), "primary", "equivalent"
        )
        automatic = automatic.drop(columns="_monoisotopic")
        identified = automatic[
            automatic["quantitative_role"].eq("primary")
        ].reset_index(drop=True)
        equivalents = automatic[
            automatic["quantitative_role"].eq("equivalent")
        ].reset_index(drop=True)
    audit = pd.DataFrame([
        {
            "phase": item.phase,
            "compound": item.compound,
            "definition": item.template,
            "status": item.status,
            "annotations": item.annotations,
            "detail": item.detail,
        }
        for item in outcomes
    ])
    return {
        "streams": streams,
        "all_candidates": all_candidates,
        "automatic_annotations": automatic.reset_index(drop=True),
        "identified_lipids": identified,
        "equivalent_lipids": equivalents,
        "policy_audit": pd.DataFrame(),
        "np_rt_audit": pd.DataFrame(),
        "matlab_policy_applied": True,
        "np_rt_reference_applied": False,
        "engine": "matlab",
        "matlab_project_dir": str(annotation_dir),
        "matlab_reference_dir": str(reference_root),
        "matlab_compound_audit": audit,
    }


def concat_result_tables(results: list[dict], key: str) -> pd.DataFrame:
    """Concatenate an optional audit/result table across phase-local runs."""
    frames = []
    for result in results:
        table = result.get(key)
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        table = table.copy()
        phase = str(result["matlab_phase"])
        if "phase" not in table:
            table.insert(0, "phase", phase)
        else:
            table["phase"] = table["phase"].fillna(phase)
        frames.append(table)
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )
