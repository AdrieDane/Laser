"""Strict ORIGINAL-feature proposals from independently aligned raw mass series.

Unused research helper. Produces candidates, never automatic full annotations.
Already independently aligned classes are not cold-started again by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .automatic import matlab_annotation_adducts


def match_raw_grid_candidates(
    targets,
    features,
    automatic,
    *,
    stream,
    declared_standard_masses=(),
    ppm=3.0,
    rt_seconds=10.0,
):
    if not 0 < ppm <= 3 or not 0 < rt_seconds <= 10:
        raise ValueError("Raw-grid candidates cannot exceed3ppm/10s")
    required = {
        "name",
        "class",
        "adduct",
        "carbon",
        "db",
        "ref_mz",
        "ref_rt",
        "raw_grid_rt_seconds",
        "raw_grid_status",
        "atlas_local_support_count",
    }
    if not required.issubset(targets):
        raise ValueError("Raw-grid targets require independent RT/alignment metadata")
    if not {"feature_id", "mz", "rt", "intensity"}.issubset(features):
        raise ValueError("Original features require ID/mass/RT/intensity")
    if features.feature_id.astype(str).duplicated().any():
        raise ValueError("Original feature IDs must be unique within a stream")
    chosen = targets[
        targets.raw_grid_status.eq("independent_mass_series_shift")
        & pd.to_numeric(targets.atlas_local_support_count, errors="coerce").lt(4)
        & targets.adduct.isin(matlab_annotation_adducts(stream[:2]))
        & ~targets.name.astype(str).str.endswith(" (IS)")
        & ~targets.name.astype(str).str.startswith("IS: ")
    ].drop_duplicates(["name", "adduct"])
    mass = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    rt = pd.to_numeric(features.rt, errors="coerce").to_numpy(float)
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    ids = features.feature_id.astype(str).to_numpy()
    valid = np.isfinite(mass) & (mass > 0) & np.isfinite(rt) & (intensity > 0)
    spikes = np.asarray(list(declared_standard_masses), dtype=float)
    spikes = np.sort(spikes[np.isfinite(spikes) & (spikes > 0)])
    if len(spikes):
        near = np.clip(np.searchsorted(spikes, mass), 0, len(spikes) - 1)
        prev = np.clip(near - 1, 0, len(spikes) - 1)
        difference = np.minimum(
            abs(mass - spikes[near]) / spikes[near],
            abs(mass - spikes[prev]) / spikes[prev],
        )
        valid &= difference >= 3e-6
    standards = (
        automatic.get("is_is", pd.Series(False, index=automatic.index))
        .fillna(False)
        .astype(bool)
    )
    protected = (
        set(automatic.loc[standards, "feat_feature_id"].astype(str))
        if "feat_feature_id" in automatic
        else set()
    )
    valid &= ~np.isin(ids, list(protected))
    order = np.flatnonzero(valid)
    order = order[np.argsort(mass[order], kind="stable")]
    sorted_mass = mass[order]
    rows = []
    for target in chosen.to_dict("records"):
        mz, expected = (
            float(target["ref_mz"]),
            float(target["raw_grid_rt_seconds"]) / 60,
        )
        if not np.isfinite(mz) or mz <= 0 or not np.isfinite(expected) or expected <= 0:
            continue
        lo, hi = np.searchsorted(
            sorted_mass, [mz * (1 - ppm * 1e-6), mz * (1 + ppm * 1e-6)]
        )
        for pos in order[lo:hi]:
            error = (mass[pos] - mz) / mz * 1e6
            residual = (rt[pos] - expected) * 60
            if abs(error) >= ppm or abs(residual) > rt_seconds:
                continue
            row = dict(target)
            row.update(
                stream=stream,
                baseline_ref_rt=target["ref_rt"],
                ref_rt=expected,
                feat_feature_id=ids[pos],
                feat_mz=mass[pos],
                feat_rt=rt[pos],
                feat_intensity=intensity[pos],
                rt_expected=expected,
                rt_error=rt[pos] - expected,
                ppm_error=error,
                atlas_available=np.isfinite(
                    float(target.get("atlas_rt_seconds", np.nan))
                ),
                isotopologue="",
                is_is=False,
                automatic_accepted=False,
                automatic_candidate=True,
                q_atlas_source=0.0,
                q_fitted_source=0.0,
                q_previously_selected=0.0,
                q_raw_grid_source=1.0,
                q_raw_grid_RT_error_seconds=residual,
                q_absolute_raw_grid_RT_error_seconds=abs(residual),
                auto_annotation_status="independent_raw_grid_candidate",
            )
            rows.append(row)
    return pd.DataFrame(rows)
