"""Separate scalar paired-gap references/predictions and unsorted raw joins.

Shares the declared chemical family mapping/native naming conventions, not the
producer's grouping, fitting, prediction or sorted-mass enumeration functions.
"""

from collections import defaultdict
from statistics import median

import numpy as np
import pandas as pd

from paired_retention_gap import FAMILIES, FORMAT, PREDICTION_COLUMNS, HIT_COLUMNS


def member(stream, ion, label):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key

    cls, adduct = canonical_lipid_class(label), ion_key(ion)
    for family, (mode, channel, a, b, offset, _) in FAMILIES.items():
        if stream == mode and adduct == channel and cls in (a, b):
            side = int(cls == b)
            return family, side, offset if side else 0, adduct
    return None


def percentile90(values):
    values = sorted(map(float, values))
    location = 0.9 * (len(values) - 1)
    lower, upper = int(np.floor(location)), int(np.ceil(location))
    fraction = location - lower
    # Match linear quantile arithmetic while independently enumerating support.
    difference = values[upper] - values[lower]
    return (
        values[upper] - difference * (1 - fraction)
        if fraction >= 0.5
        else values[lower] + difference * fraction
    )


def fit_reference(rows, support, *, training_groups, excluded_groups):
    from annotator.composition_ranges import canonical_lipid_class

    training, excluded = set(training_groups), set(excluded_groups)
    if (
        not training
        or not excluded
        or training & excluded
        or (set(rows.cohort) | set(support.cohort)) - training
    ):
        raise ValueError("Explicit authorized other-project rows required")
    counts, groups = defaultdict(int), defaultdict(set)
    for r in support.to_dict("records"):
        key = (r["stream"], canonical_lipid_class(r["lipid_class"]))
        counts[key] += int(r["manual_rows"])
        groups[key].add(r["cohort"])
    eligible = [
        family
        for family, (stream, _, a, b, _, _) in FAMILIES.items()
        if all(
            counts[(stream, c)] >= 100 and len(groups[(stream, c)]) >= 3 for c in (a, b)
        )
    ]
    raw = defaultdict(list)
    for r in rows.to_dict("records"):
        identity = member(r["stream"], r["adduct"], r["lipid_class"])
        if identity is None:
            continue
        c, d, rt, mass, observed = (
            float(r[k])
            for k in ("carbon", "db", "rt_seconds", "reference_mz", "observed_mz")
        )
        if (
            not np.isfinite([c, d, rt, mass, observed]).all()
            or min(c, rt, mass, observed) <= 0
            or d < 0
            or c != int(c)
            or d != int(d)
        ):
            continue
        if abs((observed / mass - 1) * 1e6) >= 3:
            continue
        family, side, offset, ion = identity
        key = (r["dataset"], r["cohort"], family, int(c), int(d) + offset, ion)
        raw[key].append((side, str(r["feature_id"]), rt, mass))
    pairs = []
    keys = ["dataset", "cohort", "family", "carbon", "total_db", "adduct"]
    for key in sorted(raw):
        group = raw[key]
        if (
            len(group) != 2
            or {r[0] for r in group} != {0, 1}
            or len({r[1] for r in group}) != 2
        ):
            continue
        a, b = sorted(group)
        if abs(a[3] - b[3]) > 1e-6:
            continue
        gap = b[2] - a[2]
        if not 3 <= abs(gap) <= 60:
            continue
        pairs.append(dict(zip(keys, key)) | dict(gap_seconds=gap))
    project_values = defaultdict(lambda: defaultdict(list))
    for r in pairs:
        project_values[(r["family"], r["carbon"], r["total_db"], r["adduct"])][
            r["cohort"]
        ].append(r["gap_seconds"])
    models = {}
    for key in sorted(project_values):
        family, c, total_db, _ = key
        if family not in eligible:
            continue
        grouped = project_values[key]
        values = [median(grouped[g]) for g in sorted(grouped)]
        gap = float(median(values))
        errors = [abs(v - gap) for v in values]
        mad, p90 = float(median(errors)), percentile90(errors)
        fraction = sum(np.sign(v) == np.sign(gap) for v in values) / len(values)
        if (
            len(values) < 3
            or mad > 3
            or p90 > 6
            or fraction < 0.8
            or not 3 <= abs(gap) <= 60
        ):
            continue
        expected_sign = FAMILIES[family][-1]
        if expected_sign is not None and np.sign(gap) != expected_sign:
            continue
        models[f"{family}|{c:g}|{total_db:g}"] = dict(
            gap_seconds=gap,
            training_projects=len(values),
            training_MAD_seconds=mad,
            training_p90_seconds=p90,
            sign_agreement=fraction,
            contributing_groups=sorted(grouped),
            paired_dataset_rows=sum(map(len, grouped.values())),
        )
    return dict(
        format=FORMAT,
        training_groups=sorted(training),
        excluded_groups=sorted(excluded),
        eligible_families=eligible,
        models=models,
    ), pd.DataFrame(pairs, columns=[*keys, "gap_seconds"])


def predict_reference(anchors, targets, model, *, query_group, declared_masses):
    if (
        model["format"] != FORMAT
        or query_group in model["training_groups"]
        or query_group not in model["excluded_groups"]
    ):
        raise ValueError("Heldout query must not train its own reference")
    chemistry = defaultdict(dict)
    for r in targets.to_dict("records"):
        identity = member(r["stream"], r["adduct"], r["class"])
        if identity is None:
            continue
        family, side, offset, ion = identity
        key = (family, r["carbon"], r["db"] + offset, ion)
        # Full multiplicity semantics of native chemical aliases, not a mass set.
        unique = (r["name"], ion, side, float(r["ref_mz"]))
        chemistry[key].setdefault(unique, dict(r, side=side, adduct=ion))
    sources = defaultdict(lambda: defaultdict(list))
    for r in anchors.to_dict("records"):
        identity = member(r["stream"], r["adduct"], r["class"])
        if (
            identity is None
            or bool(r["is_is"])
            or str(r["name"]).startswith("IS: ")
            or str(r["name"]).endswith(" (IS)")
        ):
            continue
        score = float(r["class_selection_score"])
        if not 0.95 <= score <= 1:
            continue
        family, side, offset, ion = identity
        sources[(family, r["carbon"], r["db"] + offset, ion)][side].append(r)
    output = []
    for key in sorted(sources):
        family, c, total_db, _ = key
        fitted = model["models"].get(f"{family}|{c:g}|{total_db:g}")
        target_rows = list(chemistry.get(key, {}).values())
        if (
            fitted is None
            or len(target_rows) != 2
            or {r["side"] for r in target_rows} != {0, 1}
        ):
            continue
        if set(fitted["contributing_groups"]) - set(model["training_groups"]):
            raise ValueError("Forbidden paired-gap reference contributor")
        by_side = {r["side"]: r for r in target_rows}
        if abs(by_side[0]["ref_mz"] - by_side[1]["ref_mz"]) > 1e-6:
            continue
        for side in sorted(sources[key]):
            group = sources[key][side]
            if len(group) != 1:
                continue
            anchor, target, own = group[0], by_side[1 - side], by_side[side]
            mz, rt = float(anchor["feat_mz"]), float(anchor["feat_rt"]) * 60
            if not np.isfinite([mz, rt]).all() or mz <= 0 or rt <= 0:
                continue
            ppm = (mz / own["ref_mz"] - 1) * 1e6
            if abs(ppm) >= 1.5:
                continue
            spikes = list(declared_masses[target["stream"]])
            if any(
                abs(float(m) / target["ref_mz"] - 1) * 1e6 < 3
                or abs(float(m) / mz - 1) * 1e6 < 3
                for m in spikes
            ):
                continue
            shift = fitted["gap_seconds"] if side == 0 else -fitted["gap_seconds"]
            if rt + shift <= 0:
                continue
            output.append(
                dict(
                    stream=target["stream"],
                    family=family,
                    name=target["name"],
                    **{"class": target["class"]},
                    adduct=target["adduct"],
                    carbon=float(target["carbon"]),
                    db=float(target["db"]),
                    ref_mz=float(target["ref_mz"]),
                    partner_feature_id=str(anchor["feat_feature_id"]),
                    partner_name=anchor["name"],
                    partner_rt_seconds=rt,
                    partner_raw_ppm=ppm,
                    partner_score=float(anchor["class_selection_score"]),
                    predicted_rt_seconds=rt + shift,
                    signed_gap_seconds=shift,
                    training_projects=fitted["training_projects"],
                    training_MAD_seconds=fitted["training_MAD_seconds"],
                    training_p90_seconds=fitted["training_p90_seconds"],
                )
            )
    result = pd.DataFrame(output, columns=PREDICTION_COLUMNS)
    if len(result):
        result = result.sort_values(
            ["stream", "family", "carbon", "db", "name", "partner_feature_id"],
            kind="stable",
        ).reset_index(drop=True)
    return result


def raw_reference(predictions, features_by_stream):
    output = []
    for index, r in predictions.iterrows():
        features = features_by_stream[r.stream]
        ids = features.feature_id.astype(str).to_list()
        mz = features.mz.to_numpy(float)
        rawrt = features.rt.to_numpy(float)
        seconds = rawrt * 60
        ppm = (mz / r.ref_mz - 1) * 1e6
        delta = seconds - r.predicted_rt_seconds
        mask = (
            np.isfinite(mz)
            & np.isfinite(seconds)
            & (mz > 0)
            & (seconds > 0)
            & (np.abs(ppm) < 3)
            & (np.abs(delta) <= 10)
        )
        for pos in np.flatnonzero(mask):
            feature = features.iloc[pos]
            observed_gap = seconds[pos] - r.partner_rt_seconds
            distinct = ids[pos] != r.partner_feature_id
            order = bool(
                np.sign(observed_gap) == np.sign(r.signed_gap_seconds)
                and abs(observed_gap) >= 3
            )
            output.append(
                dict(
                    r.to_dict(),
                    prediction_index=index,
                    feat_feature_id=ids[pos],
                    feat_mz=mz[pos],
                    feat_rt=rawrt[pos],
                    feat_intensity=float(feature.intensity),
                    ppm_error=ppm[pos],
                    paired_rt_error_seconds=delta[pos],
                    observed_gap_seconds=observed_gap,
                    distinct_partner_feature=distinct,
                    rp_order_supported=order,
                    physical_partner_eligible=distinct and order,
                )
            )
    return pd.DataFrame(output, columns=[*PREDICTION_COLUMNS, *HIT_COLUMNS])
