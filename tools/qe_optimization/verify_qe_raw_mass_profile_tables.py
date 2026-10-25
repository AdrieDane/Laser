"""Independent ALL33 raw-union/ALL363 numeric lookup/schema/scope audit.

No curation/model fits/annotation selection. Never calls the measurement
builder, its mass/RT/profile helpers, or its physical lookup implementation.
Uses the separately tested independent NumPy profile/comparison primitives.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from context import REPO, SOURCE
from frozen_resources import configure_frozen_resources
import prepare_qe_raw_mass_profile_tables as prep
import run_qe_add_only_class_experiment as common
from verify_qe_raw_counterpart_pilot import profiles, comparison

OUT = common.QE / "raw_mass_profile_independent_check_v1"


def independent_numeric(queries, features, maps):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.adduct_family_evidence import native_adduct_families
    from matlab_annotation.raw_counterpart_evidence import FIELDS, PREFIX
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.vendor.compound_model import PROTON_MASS

    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    matrices = {s: profiles(f, maps.get(s, {})) for s, f in features.items()}
    fields = FIELDS[:23]
    result = pd.DataFrame(
        np.nan, index=queries.index, columns=[PREFIX + f for f in fields]
    )
    for stream in common.MODES:
        target = stream[:3] + ("Neg" if stream.endswith("Pos") else "Pos")
        target_raw = features[target]
        mass, rt, abundance = (
            pd.to_numeric(target_raw[c], errors="coerce").to_numpy(float)
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
        identifiers = target_raw.feature_id.astype(str).to_numpy()
        left, lc, ll = matrices[stream]
        right, rc, rl = matrices[target]
        shared = sorted(set(ll) & set(rl))
        assert all(ll[s] == rl[s] for s in shared)
        ca, cb = np.array([lc[s] for s in shared]), np.array([rc[s] for s in shared])
        labels = np.array([ll[s] for s in shared])
        for idx, query in queries.loc[queries.stream.eq(stream)].iterrows():
            cls, ion, ref, fid = (
                canonical_lipid_class(query["class"]),
                ion_key(query.adduct),
                float(query.ref_mz),
                str(query.feat_feature_id),
            )
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
                backbone = abs(charge) * ref - charge * PROTON_MASS - offset
                for other_offset, other_charge in family.values():
                    if other_charge and (other_charge > 0) == target.endswith("Pos"):
                        expected.append(
                            (backbone + other_offset) / abs(other_charge)
                            + np.sign(other_charge) * PROTON_MASS
                        )
            hits, hit_ions = {}, 0
            for mz in expected:
                lo = np.searchsorted(ordered, mz - mz * 3e-6)
                hi = np.searchsorted(ordered, mz + mz * 3e-6, side="right")
                local = 0
                for i in order[lo:hi]:
                    ppm = abs((mass[i] - mz) / mz * 1e6)
                    if ppm < 3:
                        local += 1
                        hits[int(i)] = min(hits.get(int(i), np.inf), ppm)
                hit_ions += bool(local)
            row = {f: np.nan for f in fields}
            row.update(
                source_configured=float(configured),
                pairing_available=float(bool(maps)),
                expected_native_ions=len(expected),
                mass_hit_ions=hit_ions,
                distinct_mass_hits=len(hits),
                profile_measurement_available=0.0,
            )
            if maps:
                row["paired_bio_specimens"] = len(shared)
            comparisons = []
            if maps and len(shared) >= 8:
                x = left.loc[fid, shared].to_numpy(float)
                for i, ppm in hits.items():
                    y = right.loc[identifiers[i], shared].to_numpy(float)
                    measured = comparison(x, y, ca, cb, labels)
                    if measured is not None:
                        measured.update(
                            ppm=ppm, raw_rt_delta=rt[i] * 60 - float(query.feat_rt) * 60
                        )
                        comparisons.append(measured)
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
                row.update(
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
                    row[field] = best[statistic]
                if len(comparisons) > 1:
                    row["profile_margin"] = best["raw"] - comparisons[1]["raw"]
            result.loc[idx, :] = [row[f] for f in fields]
    return result


def check_one(job):
    d, own, folds = job
    configure_frozen_resources(REPO)
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.specimen_pairs import explicit_specimen_pairs
    from matlab_annotation.evidence_training import validate_feature_scope

    baseline = pd.read_pickle(common.QE / "baseline/runs" / d / "full_annotation.pkl")
    features = {s: p["features"] for s, p in baseline["streams"].items()}
    metadata = loadmat(
        SOURCE / d / "MS_metadata.mat",
        variable_names=["MS_metadata"],
        simplify_cells=True,
    )["MS_metadata"]
    try:
        maps, _ = explicit_specimen_pairs(metadata, features)
    except ValueError:
        maps = {}
    queries = pd.read_pickle(prep.OUT / "union" / (d + "_queries.pkl"))
    measured = pd.read_pickle(prep.OUT / "union" / (d + "_measurements.pkl"))
    assert (
        queries.index.equals(measured.index) and not queries.duplicated(prep.KEYS).any()
    )
    assert set(queries) == set(prep.KEYS)
    for stream, part in queries.groupby("stream"):
        source = features[stream].copy()
        source.index = source.feature_id.astype(str)
        assert source.index.is_unique
        original = source.loc[part.feat_feature_id.astype(str)]
        for observed, raw_column in (("feat_rt", "rt"), ("feat_mz", "mz")):
            np.testing.assert_allclose(
                part[observed].to_numpy(float),
                original[raw_column].to_numpy(float),
                rtol=8 * np.finfo(float).eps,
                atol=0,
                equal_nan=True,
            )
    independent = independent_numeric(queries, features, maps)
    np.testing.assert_allclose(
        measured.to_numpy(float),
        independent.to_numpy(float),
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
        err_msg=d,
    )
    index = pd.MultiIndex.from_frame(queries[prep.KEYS])
    rows = []
    for mask in prep.masks_for(own):
        old = common.common.load_predictors(d, mask)
        current = pd.read_pickle(prep.prepared_path(d, mask))
        pd.testing.assert_frame_equal(old["pool"], current["pool"], check_exact=True)
        pd.testing.assert_frame_equal(
            old["evidence"],
            current["evidence"].loc[:, old["evidence"].columns],
            check_exact=True,
        )
        assert (
            len(old["evidence"].columns) == 108
            and len(current["evidence"].columns) == 131
        )
        assert list(current["evidence"].columns[108:]) == list(independent)
        for field in (
            "query_dataset",
            "query_group",
            "reference_training_groups",
            "reference_excluded_groups",
            "standard_reference_training_groups",
            "standard_reference_excluded_groups",
            "standard_reference_sha256",
            "excluded_folds",
        ):
            assert current[field] == old[field], (d, mask, field)
        validate_feature_scope(
            current,
            {p[:7] for p in folds if folds[p] not in mask},
            {p[:7] for p in folds if folds[p] in mask},
            query_is_training=False,
        )
        wanted = old["pool"][prep.KEYS].copy()
        wanted["class"] = wanted["class"].map(canonical_lipid_class)
        wanted["adduct"] = wanted.adduct.map(ion_key)
        wanted["feat_feature_id"] = wanted.feat_feature_id.astype(str)
        positions = index.get_indexer(pd.MultiIndex.from_frame(wanted))
        assert (positions >= 0).all() and len(positions) == len(old["pool"])
        # Every one of363 table lookups is bit-exact to its saved raw-only
        # measurement and numerically agrees with independent calculations.
        actual = current["evidence"].iloc[:, 108:].to_numpy(float)
        np.testing.assert_array_equal(actual, measured.iloc[positions].to_numpy(float))
        np.testing.assert_allclose(
            actual,
            independent.iloc[positions].to_numpy(float),
            rtol=2e-6,
            atol=2e-7,
            equal_nan=True,
        )
        provenance = current["raw_mass_profile_provenance"]
        assert provenance["mask"] == sorted(mask)
        assert (
            provenance["fitted_RT_links_used"] is False
            and provenance["curation_runtime_input"] is False
        )
        assert provenance["original_prepared_sha256"] == common.sha(
            common.common.prepared_path(d, mask)
        )
        assert provenance["union_queries_sha256"] == common.sha(
            prep.OUT / "union" / (d + "_queries.pkl")
        )
        assert provenance["union_measurements_sha256"] == common.sha(
            prep.OUT / "union" / (d + "_measurements.pkl")
        )
        rows.append(dict(dataset=d, mask=prep.mask_name(mask), rows=len(old["pool"])))
    print(
        "INDEPENDENT ALL RAW UNION VALUES AND11 ORIGINAL MASK LOOKUPS PASS",
        d,
        len(queries),
        flush=True,
    )
    return rows, dict(
        dataset=d, union_queries=len(queries), numeric_cells=len(queries) * 23
    )


def main():
    receipt_path = prep.OUT / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["all_passed"] and receipt["unique_nested_tables"] == 363
    for field in ("source_input_sha256", "artifact_sha256"):
        assert all(common.sha(Path(p)) == h for p, h in receipt[field].items())
    own_hash = common.sha(Path(__file__))
    primitive_path = Path(__file__).with_name("verify_qe_raw_counterpart_pilot.py")
    primitive_hash = common.sha(primitive_path)
    previous = json.loads(
        (
            common.QE / "raw_counterpart_feature_independent_check_v1/receipt.json"
        ).read_text()
    )
    assert previous["all_passed"] and previous["verifier_sha256"] == primitive_hash
    datasets, folds = common.inventory()
    OUT.mkdir(exist_ok=False)
    rows, counts = [], []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for tables, summary in executor.map(
            check_one, [(d, folds[d], folds) for d in datasets]
        ):
            rows.extend(tables)
            counts.append(summary)
    assert len(rows) == 363 and len(counts) == 33
    assert sum(r["union_queries"] for r in counts) == receipt["union_physical_queries"]
    for field in ("source_input_sha256", "artifact_sha256"):
        assert all(common.sha(Path(p)) == h for p, h in receipt[field].items())
    assert own_hash == common.sha(Path(__file__)) and primitive_hash == common.sha(
        primitive_path
    )
    pd.DataFrame(rows).to_csv(OUT / "by_dataset_mask.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "by_dataset_union.csv", index=False)
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            unique_nested_tables=363,
            training_tables=330,
            outer_tables=33,
            numeric_fields=23,
            original_predictors_unchanged=108,
            total_predictors=131,
            union_numeric_cells=sum(r["numeric_cells"] for r in counts),
            query_lookup_rows=sum(r["rows"] for r in rows),
            classifier_fits=0,
            annotation_changes=False,
            manual_files_opened=0,
            verifier_sha256=own_hash,
            independent_primitive_sha256=primitive_hash,
            preparation_receipt_sha256=common.sha(receipt_path),
            note="ALL33independent raw-union mass/profile/normalized/group/LOO fields;ALL363exactlookup/old108/pool/STDrefscope/provenance. No builder/lookup/RT/profile implementation calls. Shared validatedroles/nativeiontables and separately tested NumPycomparison primitives. No fitpermission untilfull25parent/model/exportpreflight/runner/checker;no annotation gains/FDR/deploymentclaim.",
        ),
    )
    print("ALL363 RAW MASS/PROFILE CONSTRUCTION AUDIT COMPLETE PASS", flush=True)


if __name__ == "__main__":
    main()
