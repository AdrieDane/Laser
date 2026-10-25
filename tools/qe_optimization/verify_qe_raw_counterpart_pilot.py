"""Independent NumPy reconstruction of ALL32 raw-pilot fields, no curation.

Does NOT call the evidence/RT measurement helper, _profiles, _comparison, or its
leave-one-out implementation. Shares validated metadata roles/native ion tables
but reconstructs charge algebra, profiles, comparisons and RT application.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import rankdata

from context import REPO, SOURCE
from frozen_resources import configure_frozen_resources
import prepare_qe_raw_counterpart_pilot as prep
import run_qe_add_only_class_experiment as common

OUT = common.QE / "raw_counterpart_feature_independent_check_v1"


def profiles(frame, mapping):
    from annotator.peak_evidence import sample_roles
    from matlab_annotation.specimen_pairs import _group

    columns, roles, _ = sample_roles(frame)
    specimens, labels = {}, {}
    for c in columns:
        if c in mapping and roles[c] == "biological":
            sid = str(mapping[c])
            group = _group(frame.attrs["sample_group_map"][c])
            assert group and (sid not in labels or labels[sid] == group)
            labels[sid] = group
            specimens.setdefault(sid, []).append(c)
    data, centers = {}, {}
    for sid, cols in specimens.items():
        values = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=True)
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            data[sid] = np.log(np.nanmean(values, axis=1))
            centers[sid] = np.nanmedian(data[sid])
    return pd.DataFrame(data, index=frame.feature_id.astype(str)), centers, labels


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or np.std(a) < 0.05 or np.std(b) < 0.05:
        return np.nan
    x, y = a - np.mean(a), b - np.mean(b)
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def loo(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 8:
        return np.nan
    # Remove each centered point using the mean-update identity, rather than
    # copying the producer's raw-sum subtraction formula.
    x, y = a - a.mean(), b - b.mean()
    correction = n / (n - 1)
    vx, vy = np.dot(x, x) - correction * x**2, np.dot(y, y) - correction * y**2
    den = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    if not np.isfinite(den).all() or np.any(den <= 1e-12):
        return np.nan
    return float(np.clip((np.dot(x, y) - correction * x * y) / den, -1, 1).min())


def comparison(a, b, ca, cb, labels):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 8:
        return None
    x, y, cx, cy, groups = a[valid], b[valid], ca[valid], cb[valid], labels[valid]
    raw = corr(x, y)
    if not np.isfinite(raw):
        return None
    finite = np.isfinite(cx) & np.isfinite(cy)
    nx, ny = x - cx, y - cy
    normalized = corr(nx[finite], ny[finite]) if finite.sum() >= 8 else np.nan
    gx, gy, dof = [], [], 0
    for label in sorted(set(groups)):
        keep = groups == label
        if keep.sum() >= 2:
            gx.extend(x[keep] - x[keep].mean())
            gy.extend(y[keep] - y[keep].mean())
            dof += int(keep.sum()) - 1
    within = corr(gx, gy) if len(gx) >= 8 and dof >= 4 else np.nan
    ratio = y - x
    return dict(
        raw=raw,
        normalized=normalized,
        within=within,
        spearman=corr(rankdata(x), rankdata(y)),
        leave_one_out=loo(x, y),
        normalized_leave_one_out=loo(nx, ny),
        log_ratio_location=float(np.median(ratio)),
        normalized_log_ratio_location=float(np.median(ny[finite] - nx[finite]))
        if finite.sum() >= 8
        else np.nan,
        mad=float(np.median(abs(ratio - np.median(ratio)))),
        n=len(x),
        dof=dof,
    )


def check_one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from matlab_annotation.raw_counterpart_evidence import FIELDS, PREFIX
    from matlab_annotation.adduct_family_evidence import native_adduct_families
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.specimen_pairs import explicit_specimen_pairs
    from matlab_annotation.vendor.compound_model import PROTON_MASS
    from annotator.composition_ranges import canonical_lipid_class

    baseline = pd.read_pickle(common.QE / "baseline/runs" / d / "full_annotation.pkl")
    features = {s: v["features"] for s, v in baseline["streams"].items()}
    metadata = loadmat(
        SOURCE / d / "MS_metadata.mat",
        variable_names=["MS_metadata"],
        simplify_cells=True,
    )["MS_metadata"]
    try:
        maps, _ = explicit_specimen_pairs(metadata, features)
    except ValueError:
        maps = {}
    matrices = {s: profiles(f, maps.get(s, {})) for s, f in features.items()}
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    pool = common.common.load_predictors(d, {fold})["pool"]
    evidence = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))["evidence"]
    physical_count = 0
    for stream in common.MODES:
        target = stream[:3] + ("Neg" if stream.endswith("Pos") else "Pos")
        part = pool.loc[pool.stream.eq(stream)].copy()
        part["canonical"] = part["class"].map(canonical_lipid_class)
        part["ion"] = part.adduct.map(ion_key)
        raw = features[target]
        mass, rt, abundance = (
            pd.to_numeric(raw[c], errors="coerce").to_numpy(float)
            for c in ("mz", "rt", "intensity")
        )
        valid = np.flatnonzero(
            np.isfinite(mass)
            & np.isfinite(rt)
            & np.isfinite(abundance)
            & (mass > 0)
            & (rt > 0)
            & (abundance > 0)
        )
        order = valid[np.argsort(mass[valid], kind="stable")]
        ordered = mass[order]
        identifiers = raw.feature_id.astype(str).to_numpy()
        left, lc, ll = matrices[stream]
        right, rc, rl = matrices[target]
        shared = sorted(set(ll) & set(rl))
        assert all(ll[s] == rl[s] for s in shared)
        ca, cb = np.array([lc[s] for s in shared]), np.array([rc[s] for s in shared])
        labels = np.array([ll[s] for s in shared])
        links = json.loads(
            (prep.RT / "private" / (d + "_" + stream[:2] + "_links.json")).read_text()
        )
        links = links["positive" if stream.endswith("Pos") else "negative"]
        assert links["source_stream"] == stream and links["target_stream"] == target
        for (cls, ion, ref, fid), group in part.groupby(
            ["canonical", "ion", "ref_mz", "feat_feature_id"], sort=False
        ):
            physical_count += 1
            family = families.get((stream[:2], cls), {})
            if not family and stream.startswith("NP") and cls.endswith("[O+P]"):
                a, b = (
                    families.get(("NP", cls[:-5] + suffix)) for suffix in ("[O]", "[P]")
                )
                if a and a == b:
                    family = a
            configured = ion in family
            expected = []
            if configured:
                offset, charge = family[ion]
                assert (charge > 0) == stream.endswith("Pos")
                backbone = abs(charge) * float(ref) - charge * PROTON_MASS - offset
                for other_offset, other_charge in family.values():
                    if other_charge and (other_charge > 0) == target.endswith("Pos"):
                        expected.append(
                            (backbone + other_offset) / abs(other_charge)
                            + np.sign(other_charge) * PROTON_MASS
                        )
            hits, hit_ions = {}, 0
            for mz in expected:
                a = np.searchsorted(ordered, mz - mz * 3e-6)
                b = np.searchsorted(ordered, mz + mz * 3e-6, side="right")
                local = {}
                for i in order[a:b]:
                    ppm = abs((mass[i] - mz) / mz * 1e6)
                    if ppm < 3:
                        local[int(i)] = ppm
                        hits[int(i)] = min(hits.get(int(i), np.inf), ppm)
                hit_ions += bool(local)
            values = {c: np.nan for c in FIELDS}
            values.update(
                source_configured=float(configured),
                pairing_available=float(bool(maps)),
                expected_native_ions=len(expected),
                mass_hit_ions=hit_ions,
                distinct_mass_hits=len(hits),
                profile_measurement_available=0.0,
                RT_map_available=0.0,
            )
            if maps:
                values["paired_bio_specimens"] = len(shared)
            source_rt = float(group.feat_rt.iloc[0]) * 60
            mapping = (
                links["global_map"]
                if stream.startswith("RP")
                else links["class_maps"].get(cls)
            )
            if (
                mapping is not None
                and not mapping["minimum_source_seconds"] - 30
                <= source_rt
                <= mapping["maximum_source_seconds"] + 30
            ):
                mapping = None
            if mapping is not None:
                predicted_rt = (
                    mapping["slope"] * source_rt + mapping["intercept_seconds"]
                )
                values.update(
                    RT_map_available=1.0,
                    RT_map_anchor_compounds=mapping["anchor_compounds"],
                    RT_map_mad_seconds=mapping["mad_seconds"],
                )
            comparisons = []
            if maps and len(shared) >= 8:
                x = left.loc[str(fid), shared].to_numpy(float)
                for i, ppm in hits.items():
                    y = right.loc[identifiers[i], shared].to_numpy(float)
                    result = comparison(x, y, ca, cb, labels)
                    if result is not None:
                        result.update(
                            ppm=ppm,
                            raw_rt_delta=rt[i] * 60 - source_rt,
                            rt_residual=rt[i] * 60 - predicted_rt
                            if mapping is not None
                            else np.nan,
                        )
                        comparisons.append(result)
            if comparisons:
                comparisons.sort(
                    key=lambda r: (
                        r["raw"],
                        r["normalized"] if np.isfinite(r["normalized"]) else -2,
                        -r["ppm"],
                    ),
                    reverse=True,
                )
                best = comparisons[0]
                values.update(
                    profile_measurement_available=1.0,
                    valid_profile_hits=len(comparisons),
                    ambiguous_profile_hits=max(
                        0, sum(r["raw"] >= 0.8 for r in comparisons) - 1
                    ),
                )
                for field, statistic in (
                    ("best_raw_correlation", "raw"),
                    ("best_normalized_correlation", "normalized"),
                    ("best_within_group_correlation", "within"),
                    ("best_spearman_correlation", "spearman"),
                    ("best_leave_one_out_correlation", "leave_one_out"),
                    (
                        "best_normalized_leave_one_out_correlation",
                        "normalized_leave_one_out",
                    ),
                    ("best_log_ratio_location", "log_ratio_location"),
                    (
                        "best_normalized_log_ratio_location",
                        "normalized_log_ratio_location",
                    ),
                    ("best_log_ratio_mad", "mad"),
                    ("best_abs_ppm", "ppm"),
                    ("best_raw_RT_delta_seconds", "raw_rt_delta"),
                    ("best_profile_pairs", "n"),
                    ("best_within_group_dof", "dof"),
                ):
                    values[field] = best[statistic]
                if len(comparisons) > 1:
                    values["profile_margin"] = best["raw"] - comparisons[1]["raw"]
                if mapping is not None:
                    values["best_RT_residual_seconds"] = best["rt_residual"]
                    support = [r for r in comparisons if abs(r["rt_residual"]) <= 10]
                    values["RT_supported_profile_hits"] = len(support)
                    if support:
                        for field, statistic in (
                            ("correlation", "raw"),
                            ("normalized_correlation", "normalized"),
                            ("within_group_correlation", "within"),
                            ("abs_ppm", "ppm"),
                        ):
                            values["best_RT_supported_" + field] = support[0][statistic]
            actual = evidence.loc[group.index, [PREFIX + c for c in FIELDS]].to_numpy(
                float
            )
            expected_values = np.tile([values[c] for c in FIELDS], (len(group), 1))
            np.testing.assert_allclose(
                actual,
                expected_values,
                rtol=2e-6,
                atol=2e-7,
                equal_nan=True,
                err_msg=f"{d}/{stream}/{cls}/{ion}/{fid}",
            )
    print(
        "INDEPENDENT RAW MASS/PROFILE/RT ALL32 FIELDS CHECKED",
        d,
        len(pool),
        physical_count,
        flush=True,
    )
    return dict(
        dataset=d,
        candidate_rows=len(pool),
        physical_queries=physical_count,
        numeric_cells=len(pool) * 32,
    )


def main():
    receipt_path = prep.OUT / "blind_physical_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["all_passed"] and receipt["all33_physical_tables_frozen"]
    assert receipt["manual_files_opened"] == 0
    for section in ("source_input_sha256", "physical_sha256"):
        assert all(common.sha(Path(p)) == h for p, h in receipt[section].items())
    datasets, folds = common.inventory()
    source_sha = common.sha(Path(__file__))
    OUT.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(check_one, [(d, folds[d]) for d in datasets]))
    assert len(rows) == 33
    for section in ("source_input_sha256", "physical_sha256"):
        assert all(common.sha(Path(p)) == h for p, h in receipt[section].items())
    assert source_sha == common.sha(Path(__file__))
    pd.DataFrame(rows).to_csv(OUT / "by_dataset.csv", index=False)
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            modes=132,
            manual_files_opened=0,
            annotation_changes=False,
            classifier_fits=0,
            numeric_fields=32,
            candidate_rows=sum(r["candidate_rows"] for r in rows),
            numeric_cells=sum(r["numeric_cells"] for r in rows),
            verifier_sha256=source_sha,
            blind_receipt_sha256=common.sha(receipt_path),
            note="IndependentALL32raw-pilot field reconstruction with separate charge algebra/NumPy profiles/correlation/centeredLOO/RTmapapplication. Shared native ion tables and validated sample roles. NO curation, annotations, learning, model selection or performance gains.",
        ),
    )
    print("ALL33/132 INDEPENDENT NUMERIC CHECK PASS", flush=True)


if __name__ == "__main__":
    main()
