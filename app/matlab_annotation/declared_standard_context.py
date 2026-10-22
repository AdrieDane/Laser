"""Unused research: independently measured, declared-spike scoring context.

No query curation, new endogenous labels, mass/RT-coordinate changes or file
reads. Reference models must exclude query/test/calibration projects upstream.
Measured spikes are evidence, not proof of endogenous identity or quantitation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .standard_completion import complete_declared_standards
from .standard_rt_library import ion_key

PREFIX = "q_declared_standard_"
COLUMNS = [
    "spike_feature",
    "same_class_compounds",
    "same_adduct_compounds",
    "RT_difference_seconds",
    "carbon_difference",
    "db_difference",
    "log_intensity_difference",
    "log_over_nM_response",
    "quantity_nM",
    "same_adduct",
    "anchor_ppm_error",
    "relative_ppm_error",
    "anchor_RT_error_seconds",
    "training_cohorts",
    "transfer_only",
]


def independent_standard_measurements(baseline_result, declared, model):
    """Detect from raw features with EMPTY proposed annotations, never bootstrap.

    Baseline AUTOMATIC annotations supply only the original frozen alignment.
    Every measured spike must have ONE original feature inside the declared
    mass/predicted RT window BEFORE occupancy exclusions. Existing annotations
    do not force a chosen spike peak. No endogenous proposal list is an input.
    """
    parts, audits = [], []
    for stream, payload in baseline_result["streams"].items():
        automatic = payload["annotations"]
        if {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
            "manual_feature_id",
        } & set(automatic):
            raise ValueError(
                "Only blind automatic alignment, not curation flags, is allowed"
            )
        for c in ("manual_confirmed", "manually_confirmed", "manual_annotation"):
            if c in automatic and automatic[c].fillna(False).astype(bool).any():
                raise ValueError(
                    "Curated annotations cannot calibrate standard context"
                )
        selected, _, audit = complete_declared_standards(
            automatic.iloc[:0].copy(),
            payload["features"],
            declared[declared.stream.eq(stream)],
            model["library"],
            model["atlas"],
            model["transfer"],
            model["channels"],
            stream=stream,
            alignment_annotations=automatic,
        )
        if len(selected):
            if not selected.is_is.fillna(False).astype(bool).all():
                raise AssertionError(
                    "Independent context detector emitted a non-standard"
                )
            parts.append(selected.assign(stream=stream))
        audits.append(audit.assign(stream=stream))
    measured = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    return measured, audit


def _qualified_measurements(measured):
    fields = {
        "stream",
        "class",
        "standard_compound_id",
        "adduct",
        "feat_feature_id",
        "feat_mz",
        "ref_mz",
        "feat_rt",
        "feat_intensity",
        "carbon",
        "db",
        "is_is",
        "internal_standard_quantity",
        "internal_standard_unit",
        "rt_expected",
    }
    if not fields.issubset(measured):
        raise ValueError(
            "Explicit chemical/declaration/raw standard measurements required"
        )
    data = measured.copy()
    for c in (
        "feat_mz",
        "ref_mz",
        "feat_rt",
        "feat_intensity",
        "carbon",
        "db",
        "internal_standard_quantity",
        "rt_expected",
    ):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    finite = np.isfinite(
        data[
            [
                "feat_mz",
                "ref_mz",
                "feat_rt",
                "feat_intensity",
                "carbon",
                "db",
                "internal_standard_quantity",
                "rt_expected",
            ]
        ]
    ).all(axis=1)
    ppm = (data.feat_mz / data.ref_mz - 1) * 1e6
    error = (data.feat_rt - data.rt_expected) * 60
    valid = (
        finite
        & data.is_is.fillna(False).astype(bool)
        & data.ref_mz.gt(0)
        & data.feat_mz.gt(0)
        & data.feat_rt.gt(0)
        & data.feat_intensity.gt(0)
        & data.rt_expected.gt(0)
        & data.standard_compound_id.notna()
        & data.standard_compound_id.astype(str).str.strip().ne("")
        & data.internal_standard_quantity.gt(0)
        & data.carbon.gt(0)
        & data.db.ge(0)
        & ppm.abs().lt(3)
        & error.abs().le(10)
    )
    data = data.loc[valid].copy()
    data["_fid"] = data.feat_feature_id.astype(str)
    data["_class"] = data["class"].map(canonical_lipid_class)
    data["_ion"] = data.adduct.map(ion_key)
    data["_ppm"] = ppm.loc[data.index]
    data["_RT_error"] = error.loc[data.index]
    data = data.drop_duplicates(
        ["stream", "standard_compound_id", "_ion", "_fid", "ref_mz"]
    )
    conflicts = (
        data.groupby(["stream", "_fid"]).standard_compound_id.transform("nunique").gt(1)
    )
    data = data[~conflicts].copy()
    # Different adducts of one added chemical must coelute, not vote as
    # independent standards or anchor incompatible peaks.
    spread = data.groupby(["stream", "standard_compound_id"]).feat_rt.transform(
        "max"
    ) - data.groupby(["stream", "standard_compound_id"]).feat_rt.transform("min")
    return data[spread.le(3 / 60)].copy()


def declared_standard_context_evidence(pool, measurements):
    """Numerical exact-class context; missing/uncertain values remain unknown.

    Never merge ether/positional classes into their parents. Choose a same-ion
    spike first, then nearest C/DB composition, then stronger signal. Repeated
    adducts count as ONE chemical. A query feature cannot use itself OR companion
    ions of its own measured standard compound as an independent anchor.
    """
    required = {
        "stream",
        "class",
        "carbon",
        "db",
        "adduct",
        "feat_feature_id",
        "feat_rt",
        "feat_intensity",
        "ppm_error",
    }
    if not required.issubset(pool) or not pool.index.is_unique:
        raise ValueError(
            "Unique physical query rows with class/composition/ion/coordinates required"
        )
    output = pd.DataFrame(
        np.nan, index=pool.index, columns=[PREFIX + c for c in COLUMNS]
    )
    for c in ("spike_feature", "same_class_compounds", "same_adduct_compounds"):
        output[PREFIX + c] = 0.0
    if pool.empty or measurements.empty:
        return output
    standards = _qualified_measurements(measurements)
    if standards.empty:
        return output
    keys = pool["class"].map(canonical_lipid_class)
    for (stream, cls), indices in pool.groupby(
        [pool.stream, keys], sort=False
    ).groups.items():
        all_stream = standards[standards.stream.eq(stream)]
        if all_stream.empty:
            continue
        rows = pool.loc[indices]
        feature = rows.feat_feature_id.astype(str)
        lookup = dict(zip(all_stream._fid, all_stream.standard_compound_id))
        output.loc[indices, PREFIX + "spike_feature"] = feature.isin(lookup).to_numpy(
            float
        )
        anchor = all_stream[all_stream._class.eq(cls)].sort_values(
            ["standard_compound_id", "_ion", "_fid"], kind="stable"
        )
        if anchor.empty:
            continue
        blocked = feature.map(lookup).fillna("").to_numpy(str)
        compounds = anchor.standard_compound_id.to_numpy(str)
        eligible = blocked[:, None] != compounds[None, :]
        carbon = pd.to_numeric(rows.carbon, errors="coerce").to_numpy(float)
        db = pd.to_numeric(rows.db, errors="coerce").to_numpy(float)
        eligible &= (np.isfinite(carbon) & np.isfinite(db))[:, None]
        same = (
            rows.adduct.map(ion_key).to_numpy(str)[:, None]
            == anchor._ion.to_numpy(str)[None, :]
        )
        for label, available in (
            ("same_class_compounds", eligible),
            ("same_adduct_compounds", eligible & same),
        ):
            counts = np.zeros(len(rows))
            for compound in np.unique(compounds):
                counts += available[:, compounds == compound].any(axis=1)
            output.loc[indices, PREFIX + label] = counts
        distance = np.abs(
            carbon[:, None] - anchor.carbon.to_numpy(float)[None, :]
        ) + 2 * np.abs(db[:, None] - anchor.db.to_numpy(float)[None, :])
        shape = eligible.shape
        order = np.lexsort(
            (
                np.broadcast_to(np.arange(len(anchor)), shape),
                np.broadcast_to(
                    -np.log1p(anchor.feat_intensity.to_numpy(float)), shape
                ),
                distance,
                ~same,
                ~eligible,
            ),
            axis=1,
        )[:, 0]
        valid_rows = eligible.any(axis=1)
        chosen = anchor.iloc[order].reset_index(drop=True)
        selected_indices = pd.Index(indices)[valid_rows]
        intensity = pd.to_numeric(rows.feat_intensity, errors="coerce").to_numpy(float)
        time = pd.to_numeric(rows.feat_rt, errors="coerce").to_numpy(float)
        mass_error = pd.to_numeric(rows.ppm_error, errors="coerce").to_numpy(float)
        quantity = chosen.internal_standard_quantity.to_numpy(float)
        nm = (
            chosen.internal_standard_unit.astype(str)
            .str.strip()
            .str.casefold()
            .isin(["nm", "nmol/l"])
            .to_numpy()
        )
        calibration = np.full(len(rows), np.nan)
        calibration[nm] = np.log1p(np.maximum(intensity[nm], 0)) - np.log(
            chosen.feat_intensity.to_numpy(float)[nm] / quantity[nm]
        )
        quantities = np.where(nm, quantity, np.nan)
        values = dict(
            RT_difference_seconds=(time - chosen.feat_rt.to_numpy(float)) * 60,
            carbon_difference=carbon - chosen.carbon.to_numpy(float),
            db_difference=db - chosen.db.to_numpy(float),
            log_intensity_difference=np.log1p(np.maximum(intensity, 0))
            - np.log1p(chosen.feat_intensity.to_numpy(float)),
            log_over_nM_response=calibration,
            quantity_nM=quantities,
            same_adduct=same[np.arange(len(rows)), order].astype(float),
            anchor_ppm_error=chosen._ppm.to_numpy(float),
            relative_ppm_error=mass_error - chosen._ppm.to_numpy(float),
            anchor_RT_error_seconds=chosen._RT_error.to_numpy(float),
            training_cohorts=pd.to_numeric(
                chosen.get(
                    "standard_training_cohorts", pd.Series(np.nan, index=chosen.index)
                ),
                errors="coerce",
            ).to_numpy(float),
            transfer_only=chosen.get(
                "standard_rt_reference_source", pd.Series("", index=chosen.index)
            )
            .eq("independent_endogenous_class_transfer")
            .to_numpy(float),
        )
        for name, array in values.items():
            output.loc[selected_indices, PREFIX + name] = array[valid_rows]
    return output.replace([np.inf, -np.inf], np.nan)
