"""Other-project RP isomer gaps + a frozen automatic partner, not annotations.

Different from native RTDB pair ordering/local offsets: absolute target RT is
anchored to a strong CURRENT automatic partner and an empirically paired gap
learned on OTHER projects. No RTDB/curator fitted RT or query labels read here.
"""

import numpy as np
import pandas as pd

FORMAT = "laser-other-project-paired-RP-gap-v1"
FAMILIES = {
    "LPC": ("RP_Pos", "[M+H]+", "1-acyl LPC", "2-acyl LPC", 0, -1),
    "LPE": ("RP_Neg", "[M-H]-", "1-acyl LPE", "2-acyl LPE", 0, -1),
    "PC_ether": ("RP_Pos", "[M+H]+", "PC[O]", "PC[P]", 1, None),
    "PE_ether": ("RP_Neg", "[M-H]-", "PE[O]", "PE[P]", 1, None),
}
FORBIDDEN = {
    "same_manual_peak",
    "manual_feature_id",
    "training_positive",
    "training_label_known",
    "calibration_identity_in_manual",
}
PREDICTION_COLUMNS = [
    "stream",
    "family",
    "name",
    "class",
    "adduct",
    "carbon",
    "db",
    "ref_mz",
    "partner_feature_id",
    "partner_name",
    "partner_rt_seconds",
    "partner_raw_ppm",
    "partner_score",
    "predicted_rt_seconds",
    "signed_gap_seconds",
    "training_projects",
    "training_MAD_seconds",
    "training_p90_seconds",
]
HIT_COLUMNS = [
    "prediction_index",
    "feat_feature_id",
    "feat_mz",
    "feat_rt",
    "feat_intensity",
    "ppm_error",
    "paired_rt_error_seconds",
    "observed_gap_seconds",
    "distinct_partner_feature",
    "rp_order_supported",
    "physical_partner_eligible",
]


def decorate(frame, *, class_column="lipid_class"):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key

    result = frame.copy()
    result["lipid_class"] = result[class_column].map(canonical_lipid_class)
    result["adduct"] = result.adduct.map(ion_key)
    result["family"] = ""
    result["side"] = -1
    result["total_db"] = pd.to_numeric(result.db, errors="coerce")
    for family, (stream, ion, a, b, offset, _) in FAMILIES.items():
        for side, cls in enumerate((a, b)):
            use = (
                result.stream.eq(stream)
                & result.adduct.eq(ion)
                & result.lipid_class.eq(cls)
            )
            result.loc[use, "family"] = family
            result.loc[use, "side"] = side
            result.loc[use, "total_db"] = result.loc[use, "db"] + (
                offset if side else 0
            )
    return result.loc[result.side.ge(0)].copy()


def fit_gaps(observations, manual_support, *, training_groups, excluded_groups):
    training, excluded = set(training_groups), set(excluded_groups)
    if not training or not excluded or training & excluded:
        raise ValueError("Explicit disjoint permitted and excluded projects required")
    required = {
        "dataset",
        "cohort",
        "stream",
        "lipid_class",
        "carbon",
        "db",
        "adduct",
        "feature_id",
        "rt_seconds",
        "reference_mz",
        "observed_mz",
    }
    if not required.issubset(observations) or not {
        "cohort",
        "stream",
        "lipid_class",
        "manual_rows",
    }.issubset(manual_support):
        raise ValueError("Only authorized original-feature RT observations/support")
    if (set(observations.cohort) | set(manual_support.cohort)) - training:
        raise ValueError("Excluded/unknown project entered paired-gap fitting")
    if FORBIDDEN & set(observations):
        raise ValueError("Do not pass annotation outcome flags to RT fitting")
    from annotator.composition_ranges import canonical_lipid_class

    support = manual_support.assign(
        lipid_class=manual_support.lipid_class.map(canonical_lipid_class)
    )
    eligible = []
    for family, (stream, _, a, b, _, _) in FAMILIES.items():
        if all(
            int(
                (
                    part := support.loc[
                        support.stream.eq(stream) & support.lipid_class.eq(cls)
                    ]
                ).manual_rows.sum()
            )
            >= 100
            and part.cohort.nunique() >= 3
            for cls in (a, b)
        ):
            eligible.append(family)
    numeric = observations[
        ["carbon", "db", "rt_seconds", "reference_mz", "observed_mz"]
    ].apply(pd.to_numeric, errors="coerce")
    valid = (
        np.isfinite(numeric).all(axis=1)
        & numeric.carbon.gt(0)
        & numeric.db.ge(0)
        & numeric.rt_seconds.gt(0)
        & numeric.reference_mz.gt(0)
        & numeric.observed_mz.gt(0)
    )
    valid &= ((numeric.observed_mz / numeric.reference_mz - 1) * 1e6).abs().lt(3)
    valid &= (
        numeric[["carbon", "db"]].eq(np.floor(numeric[["carbon", "db"]])).all(axis=1)
    )
    rows = decorate(
        observations.loc[valid].assign(**numeric.loc[valid].to_dict("series"))
    )
    pairs = []
    keys = ["dataset", "cohort", "family", "carbon", "total_db", "adduct"]
    for key, part in rows.groupby(keys, sort=True):
        if (
            len(part) != 2
            or set(part.side) != {0, 1}
            or part.feature_id.astype(str).nunique() != 2
        ):
            continue
        a, b = (part.loc[part.side.eq(side)].iloc[0] for side in (0, 1))
        if not np.isclose(a.reference_mz, b.reference_mz, rtol=0, atol=1e-6):
            continue
        gap = float(b.rt_seconds - a.rt_seconds)
        if not np.isfinite(gap) or not 3 <= abs(gap) <= 60:
            continue
        pairs.append(dict(zip(keys, key)) | dict(gap_seconds=gap))
    paired = pd.DataFrame(pairs, columns=[*keys, "gap_seconds"])
    models = {}
    group_keys = ["family", "carbon", "total_db", "adduct"]
    for key, part in paired.groupby(group_keys, sort=True):
        family, carbon, total_db, _ = key
        if family not in eligible:
            continue
        # Related tissues/acquisitions from one project never multiply votes.
        values = part.groupby("cohort", sort=True).gap_seconds.median()
        n = len(values)
        gap = float(values.median())
        deviations = np.abs(values.to_numpy(float) - gap)
        mad, p90 = float(np.median(deviations)), float(np.quantile(deviations, 0.9))
        fraction = float(np.mean(np.sign(values) == np.sign(gap)))
        required_sign = FAMILIES[family][-1]
        if n < 3 or mad > 3 or p90 > 6 or fraction < 0.8 or not 3 <= abs(gap) <= 60:
            continue
        if required_sign is not None and np.sign(gap) != required_sign:
            continue
        models[f"{family}|{float(carbon):g}|{float(total_db):g}"] = dict(
            gap_seconds=gap,
            training_projects=n,
            training_MAD_seconds=mad,
            training_p90_seconds=p90,
            sign_agreement=fraction,
            contributing_groups=sorted(values.index),
            paired_dataset_rows=len(part),
        )
    return dict(
        format=FORMAT,
        training_groups=sorted(training),
        excluded_groups=sorted(excluded),
        eligible_families=eligible,
        models=models,
    ), paired


def predict_partners(anchors, targets, model, *, query_group, declared_masses):
    if (
        model.get("format") != FORMAT
        or query_group in model["training_groups"]
        or query_group not in model["excluded_groups"]
    ):
        raise ValueError("Query must be explicitly excluded from paired-gap model")
    if FORBIDDEN & (set(anchors) | set(targets)):
        raise ValueError("No query gold/outcome flags in partner prediction")
    if {"ref_rt", "rt_expected", "atlas_rt_seconds"} & set(targets):
        raise ValueError("Chemical targets only, not fitted/reference RT inputs")
    if not anchors.is_is.isin([True, False]).all():
        raise ValueError("Explicit parsed standard flags required")
    source = decorate(anchors, class_column="class")
    chemical = decorate(targets, class_column="class")
    source = source.loc[
        ~source.is_is.fillna(False).astype(bool)
        & ~source["name"].astype(str).str.endswith(" (IS)")
        & ~source["name"].astype(str).str.startswith("IS: ")
        & pd.to_numeric(source.class_selection_score, errors="coerce").between(0.95, 1)
    ]
    output = []
    keys = ["family", "carbon", "total_db", "adduct"]
    lookup = {key: part for key, part in chemical.groupby(keys, sort=True)}
    for key, part in source.groupby(keys, sort=True):
        family, carbon, total_db, _ = key
        fit = model["models"].get(f"{family}|{float(carbon):g}|{float(total_db):g}")
        reference = lookup.get(key)
        if fit is None or reference is None:
            continue
        if set(fit["contributing_groups"]) - set(model["training_groups"]):
            raise ValueError("Unpermitted paired-gap contributors")
        reference = reference.drop_duplicates(["name", "adduct", "side", "ref_mz"])
        if len(reference) != 2 or set(reference.side) != {0, 1}:
            continue
        a, b = (reference.loc[reference.side.eq(side)].iloc[0] for side in (0, 1))
        if not np.isclose(a.ref_mz, b.ref_mz, rtol=0, atol=1e-6):
            continue
        for side, group in part.groupby("side", sort=True):
            if len(group) != 1:
                continue
            anchor = group.iloc[0]
            native, target = (a, b) if side == 0 else (b, a)
            mz, rt = float(anchor.feat_mz), float(anchor.feat_rt) * 60
            if not np.isfinite([mz, rt]).all() or mz <= 0 or rt <= 0:
                continue
            ppm = (mz / float(native.ref_mz) - 1) * 1e6
            if abs(ppm) >= 1.5:
                continue
            if str(anchor.stream) not in declared_masses:
                raise ValueError("Explicit declared-spike mass protection required")
            spikes = np.asarray(declared_masses[str(anchor.stream)], float)
            if not np.isfinite(spikes).all() or not (spikes > 0).all():
                raise ValueError("Finite positive declared actual masses required")
            if len(spikes) and (
                (abs(spikes / target.ref_mz - 1) * 1e6 < 3).any()
                or (abs(spikes / mz - 1) * 1e6 < 3).any()
            ):
                continue
            gap = fit["gap_seconds"] * (1 if side == 0 else -1)
            expected = rt + gap
            if expected <= 0:
                continue
            output.append(
                dict(
                    stream=target.stream,
                    family=family,
                    name=target["name"],
                    **{"class": target["class"]},
                    adduct=target.adduct,
                    carbon=float(target.carbon),
                    db=float(target.db),
                    ref_mz=float(target.ref_mz),
                    partner_feature_id=str(anchor.feat_feature_id),
                    partner_name=anchor["name"],
                    partner_rt_seconds=rt,
                    partner_raw_ppm=ppm,
                    partner_score=float(anchor.class_selection_score),
                    predicted_rt_seconds=expected,
                    signed_gap_seconds=gap,
                    training_projects=fit["training_projects"],
                    training_MAD_seconds=fit["training_MAD_seconds"],
                    training_p90_seconds=fit["training_p90_seconds"],
                )
            )
    result = pd.DataFrame(output, columns=PREDICTION_COLUMNS)
    if len(result):
        result = result.sort_values(
            ["stream", "family", "carbon", "db", "name", "partner_feature_id"],
            kind="stable",
        ).reset_index(drop=True)
    return result


def raw_candidates(predictions, features_by_stream):
    """Complete3ppm/10s graph before occupancy/identity filtering; no acceptance."""
    output = []
    for index, target in predictions.iterrows():
        features = features_by_stream[target.stream]
        if not features.feature_id.astype(str).is_unique:
            raise ValueError("Original feature IDs must be unique")
        ids = features.feature_id.astype(str).to_numpy()
        mz, rt, intensity = (
            pd.to_numeric(features[c], errors="coerce").to_numpy(float)
            for c in ("mz", "rt", "intensity")
        )
        original_rt = rt
        rt = rt * 60
        valid = np.isfinite(mz) & np.isfinite(rt) & (mz > 0) & (rt > 0)
        order = np.flatnonzero(valid)
        order = order[np.argsort(mz[order], kind="stable")]
        lo, hi = np.searchsorted(
            mz[order], [target.ref_mz * (1 - 3e-6), target.ref_mz * (1 + 3e-6)]
        )
        for pos in order[lo:hi]:
            ppm = (mz[pos] / target.ref_mz - 1) * 1e6
            error = rt[pos] - target.predicted_rt_seconds
            if abs(ppm) >= 3 or abs(error) > 10:
                continue
            observed_gap = rt[pos] - target.partner_rt_seconds
            distinct = ids[pos] != target.partner_feature_id
            ordered = bool(
                np.sign(observed_gap) == np.sign(target.signed_gap_seconds)
                and abs(observed_gap) >= 3
            )
            output.append(
                dict(
                    target.to_dict(),
                    prediction_index=index,
                    feat_feature_id=ids[pos],
                    feat_mz=mz[pos],
                    feat_rt=original_rt[pos],
                    feat_intensity=intensity[pos],
                    ppm_error=ppm,
                    paired_rt_error_seconds=error,
                    observed_gap_seconds=observed_gap,
                    distinct_partner_feature=distinct,
                    rp_order_supported=ordered,
                    physical_partner_eligible=distinct and ordered,
                )
            )
    return pd.DataFrame(output, columns=[*PREDICTION_COLUMNS, *HIT_COLUMNS])
