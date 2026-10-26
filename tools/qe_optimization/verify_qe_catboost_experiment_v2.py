"""048 independent selection/count/assembly reconstruction and provenance checks.

Native CatBoost prediction, unchanged005/027 scientific helpers and the existing
034 MIP solver are shared numerical dependencies, not independently rewritten
chemistry or optimization solvers. Peak competition, count construction, model
input reconstruction and output assembly are separate from the producer.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import catboost_experiment_data as data
import catboost_evidence_model as backend
import catboost_experiment_bindings as bindings
from frozen_resources import configure_frozen_resources


def fixed_winners(pool, scores):
    """Scalar graph traversal preserving original stable tie order."""
    assert pool.index.equals(scores.index) and pool.index.is_unique
    assert pool.stream.nunique() <= 1
    values = scores.to_numpy(float)
    assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()
    intensity = pool.feat_intensity.to_numpy(float)
    intensity = np.where(np.isnan(intensity), -np.inf, intensity)
    features = pool.feat_feature_id.to_list()
    names = pool["name"].to_list()
    ions = pool.adduct.to_list()
    ranked = sorted(
        range(len(pool)),
        key=lambda j: (-values[j], -intensity[j]),
    )
    distinct, feature_rows = set(), {}
    for j in ranked:
        key = (features[j], names[j], ions[j])
        if key in distinct:
            continue
        distinct.add(key)
        feature_rows.setdefault(features[j], []).append(j)
    clear = set()
    for rows in feature_rows.values():
        if len(rows) == 1 or values[rows[0]] - values[rows[1]] >= 0.05:
            clear.add(rows[0])
    identities, picked = set(), []
    for j in ranked:
        if j not in clear:
            continue
        identity = (names[j], ions[j])
        if identity not in identities:
            identities.add(identity)
            picked.append(j)
    result = pool.iloc[picked].copy()
    result["class_selection_score"] = scores.iloc[picked].to_numpy()
    return result


def select(pool, scores, base, classes):
    from annotator.composition_ranges import canonical_lipid_class

    winners = fixed_winners(pool, scores)
    cuts = winners["class"].map(canonical_lipid_class).map(classes).fillna(base)
    return winners.loc[cuts.lt(1) & winners.class_selection_score.ge(cuts)].copy()


def counts(rows, threshold):
    correct = extra = wrong = 0
    for row in rows.itertuples():
        if threshold == 1 or row.class_selection_score < threshold:
            continue
        correct += int(row.training_positive)
        extra += int(not row.calibration_identity_in_manual)
        wrong += int(
            row.calibration_identity_in_manual
            and row.training_label_known
            and not row.training_positive
        )
    return dict(correct=correct, extras=extra, different_peak=wrong)


def scientific_keys(frame):
    from standard_lipidomics.validation import _species, canonical_feature_id
    from matlab_annotation.standard_rt_library import ion_key

    return [
        (str(s), _species(n), ion_key(a), canonical_feature_id(f), bool(i))
        for s, n, a, f, i in frame[
            ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        ].itertuples(index=False, name=None)
    ]


def assert_scientific_equal(left, right):
    a, b = scientific_keys(left), scientific_keys(right)
    assert len(a) == len(set(a)) == len(b) == len(set(b))
    assert set(a) == set(b), (len(set(a) - set(b)), len(set(b) - set(a)))
    columns = ["feat_mz", "feat_rt", "feat_intensity"]
    lookup = {key: i for i, key in enumerate(b)}
    np.testing.assert_allclose(
        left[columns].to_numpy(float),
        right.iloc[[lookup[key] for key in a]][columns].to_numpy(float),
        rtol=8 * np.finfo(float).eps,
        atol=0,
        equal_nan=True,
    )


def reconstruct_annotations(dataset, outer, raw, scores, modecuts, classcuts, snapshot):
    """Independent whole-core assembly, with shared fixed005/027 chemistry."""
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

    baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
    std = json.loads(
        (data.STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    catalogue = declared_standard_targets(
        snapshot / "reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    mass = pd.read_pickle(data.MASS / "runs" / dataset / "private_mass_candidates.pkl")
    anchors = pd.concat(
        [p["annotations"].assign(stream=s) for s, p in baseline["streams"].items()],
        ignore_index=True,
    )
    parts = []
    for s, payload in baseline["streams"].items():
        pool = raw["pool"].loc[raw["pool"].stream.eq(s)]
        endogenous = select(
            pool, scores.loc[pool.index], modecuts[s], classcuts.get(s, {})
        )
        protected = (
            payload["annotations"]
            .loc[payload["annotations"].is_is.fillna(False).astype(bool)]
            .assign(stream=s)
        )
        corrected, _, _ = complete_declared_standards(
            pd.concat([endogenous, protected], ignore_index=True),
            payload["features"],
            catalogue.loc[catalogue.stream.eq(s)],
            std["library"],
            std["atlas"],
            std["transfer"],
            std["channels"],
            stream=s,
            alignment_annotations=payload["annotations"],
        )
        local = mass.loc[mass.stream.eq(s)] if len(mass) else mass
        added = select_calibrated_additions(local, anchors, corrected, stream=s)
        parts.append(pd.concat([corrected, added], ignore_index=True).assign(stream=s))
    return pd.concat(parts, ignore_index=True)


def model_for(outer, inner):
    path = data.parent_path(outer, inner)
    complete = json.loads(path.with_suffix(".complete.json").read_text())
    assert complete["model_sha256"] == data.sha(path)
    assert complete["protocol_sha256"] == data.sha(data.OUT / "protocol.json")
    return pd.read_pickle(path)


def native_scores(raw, fitted):
    """Direct estimator interface, no producer predictor or matrix helper."""
    from matlab_annotation.evidence_training import validate_feature_scope

    validate_feature_scope(
        raw,
        fitted["training_cohorts"],
        fitted["excluded_groups"],
        query_is_training=False,
    )
    parts = []
    for stream, expert in fitted["experts"].items():
        indices = raw["pool"].index[raw["pool"].stream.eq(stream)]
        x = raw["evidence"].loc[indices].copy()
        assert list(x) == expert["columns"]
        for c in x:
            if c in expert["categorical_columns"]:
                x[c] = x[c].map(str)
            else:
                a = pd.to_numeric(x[c], errors="coerce").to_numpy(float, copy=True)
                a[~np.isfinite(a)] = np.nan
                x[c] = a
        parts.append(
            pd.Series(
                expert["estimator"].predict_proba(x, thread_count=1)[:, 1],
                index=indices,
            )
        )
    result = pd.concat(parts).reindex(raw["pool"].index)
    assert np.isfinite(result).all() and result.between(0, 1).all()
    pd.testing.assert_series_equal(
        result, backend.predict_four_modes(raw, fitted), check_exact=True
    )
    return result


def check_parent(item):
    outer, inner = item
    configure_frozen_resources(data.OUT / "source_snapshot")
    fitted = model_for(outer, inner)
    records, training, excluded = data.training_records(outer, inner)
    assert (
        fitted["format"] == backend.FORMAT
        and fitted["parameters"] == backend.PARAMETERS
    )
    assert fitted["runtime"] == backend.environment()
    assert (
        set(fitted["training_cohorts"]) == training
        and set(fitted["excluded_groups"]) == excluded
    )
    assert fitted["training_feature_masks"] == {
        d: r["excluded_folds"] for d, r in records.items()
    }
    expected = pd.read_csv(bindings.CONSTRUCTION / "all25_parent_mode_matrices.csv")
    expected = expected.loc[
        expected.outer.eq(outer)
        & (expected.inner.isna() if inner is None else expected.inner.eq(inner))
    ].set_index("stream")
    assert set(fitted["experts"]) == set(data.MODES)
    for mode, expert in fitted["experts"].items():
        x, y, groups, weights, cats = data.independent_fitting_matrix(
            records, training, excluded, mode
        )
        assert (
            expert["fitting_matrix_sha256"]
            == backend.fingerprint(x, y, groups, weights)
            == expected.loc[mode, "fitting_matrix_sha256"]
        )
        assert expert["columns"] == list(x) and expert["categorical_columns"] == cats
        assert expert["training_pairs"] == len(x) and expert["positives"] == int(
            y.sum()
        )
        assert (
            set(expert["training_cohorts"]) == training
            and set(expert["excluded_groups"]) == excluded
        )
        estimator = expert["estimator"]
        assert estimator.tree_count_ == 800 and estimator.classes_.tolist() == [0, 1]
        params = estimator.get_params()
        assert all(params[k] == value for k, value in backend.PARAMETERS.items())
        assert expert["native_parameters"] == estimator.get_all_params()
    for d, raw in records.items():
        assert fitted["standard_training_feature_scopes"][d] == dict(
            training=raw["standard_reference_training_groups"],
            excluded=raw["standard_reference_excluded_groups"],
            reference_sha256=raw["standard_reference_sha256"],
        )
    print("048 MODEL/SCOPE/MATRIX CHECK", outer, inner, flush=True)
    return len(records)


def check_calibration(outer):
    configure_frozen_resources(data.OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.operating_point import choose_balanced_training_points
    from matlab_annotation.guarded_class_operating_point import (
        choose_class_training_points,
    )
    import catboost_operating_points as operating

    datasets, folds = data.inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    folder = data.OUT / "calibration" / f"fold_{outer}"
    saved = json.loads((folder / "operating_points.json").read_text())
    assert saved["protocol_sha256"] == data.sha(data.OUT / "protocol.json")
    for p, h in saved["calibration_artifacts_sha256"].items():
        assert data.sha(p) == h
    wins, manuals = [], []
    for d in datasets:
        inner = folds[d]
        if inner == outer:
            continue
        raw = data.load_predictors(d, {outer, inner})
        scores = native_scores(raw, model_for(outer, inner))
        pd.testing.assert_series_equal(
            scores, pd.read_pickle(folder / (d + "_blind_scores.pkl")), check_exact=True
        )
        labelled, manual = data.authorize_labels(raw, allowed, excluded)
        manuals.append(manual.assign(project=d[:7]))
        for _, part in labelled["pool"].groupby("stream", sort=False):
            w = fixed_winners(part, scores.loc[part.index])
            wins.append(
                w.assign(
                    lipid_class=w["class"].map(canonical_lipid_class), project=d[:7]
                )
            )
    winners, manual = (
        pd.concat(wins, ignore_index=True),
        pd.concat(manuals, ignore_index=True),
    )
    table = pd.read_csv(
        folder / "training_operating_points.csv", float_precision="round_trip"
    )
    assert len(table) == 164 and set(table.threshold) == set(operating.THRESHOLDS)
    for row in table.itertuples():
        part = winners.loc[winners.stream.eq(row.stream)]
        assert row.manual == int(manual.stream.eq(row.stream).sum())
        assert counts(part, row.threshold) == {
            k: getattr(row, k) for k in ("correct", "extras", "different_peak")
        }
    cuts = choose_balanced_training_points(table, extras_per_manual=0.095, target=0.9)
    assert cuts == saved["fallback_thresholds"]
    curves = operating.class_curves(
        winners,
        manual,
        cuts,
        training_groups=allowed,
        excluded_groups=excluded,
        minimum_manual=100,
        minimum_projects=3,
        minimum_positives=20,
    )
    pd.testing.assert_frame_equal(
        curves,
        pd.read_csv(
            folder / "training_class_operating_points.csv", float_precision="round_trip"
        ),
        check_exact=True,
    )
    for row in curves.itertuples():
        part = winners.loc[
            winners.stream.eq(row.stream) & winners.lipid_class.eq(row.lipid_class)
        ]
        assert counts(part, row.threshold) == {
            k: getattr(row, k) for k in ("correct", "extras", "different_peak")
        }
    selected = curves.loc[saved["selected_point_indices"]]
    pd.testing.assert_frame_equal(
        selected.reset_index(drop=True),
        pd.read_csv(
            folder / "selected_class_training_points.csv", float_precision="round_trip"
        ),
        check_exact=True,
    )
    baseline = curves.loc[curves.baseline].set_index(["stream", "lipid_class"])
    chosen = selected.set_index(["stream", "lipid_class"]).reindex(baseline.index)
    assert chosen.correct.ge(baseline.correct).all()
    assert chosen.loc[~baseline.eligible, "threshold"].equals(
        baseline.loc[~baseline.eligible, "threshold"]
    )
    assert selected.extras.sum() <= saved["training_extras_budget"]
    assert all(
        chosen.loc[s].different_peak.sum() <= baseline.loc[s].different_peak.sum()
        for s in data.MODES
    )
    assert saved["training_groups"] == sorted(allowed) and saved[
        "excluded_groups"
    ] == sorted(excluded)
    if saved["solver_fallback"]:
        assert chosen.threshold.equals(baseline.threshold)
        assert any(not a["optimal"] for a in saved["solver_audit"])
    else:
        assert all(a["optimal"] and a["status"] == 0 for a in saved["solver_audit"])
        repeated = choose_class_training_points(
            curves,
            training_groups=allowed,
            excluded_groups=excluded,
            extras_per_manual=0.095,
            target=0.9,
            time_limit=120,
        )
        assert not repeated["solver_fallback"]
        # Multiple equally optimal point sets can exist. Compare each fixed
        # lexicographic objective, not solver-dependent row ordering.
        other = curves.loc[repeated["selected_point_indices"]]

        def objectives(t):
            e = t.loc[t.eligible]
            base = baseline.threshold.reindex(
                pd.MultiIndex.from_frame(e[["stream", "lipid_class"]])
            ).to_numpy()
            return [
                np.minimum(100 * e.correct / e.manual, 90).sum(),
                t.correct.sum(),
                t.different_peak.sum(),
                t.extras.sum(),
                np.abs(e.threshold.to_numpy() - base).sum(),
            ]

        np.testing.assert_allclose(
            objectives(selected), objectives(other), rtol=0, atol=1e-6
        )
    print("048 INNER SCORES/ALL COUNT CURVES/CUTS CHECK", outer, flush=True)
    return len(manuals)


def check_outer(dataset):
    configure_frozen_resources(data.OUT / "source_snapshot")
    from catboost_pipeline_assembly import check_original_coordinates
    from standard_lipidomics.validation import compare_manual_annotations
    from evaluate_qe_baseline import adduct

    _, folds = data.inventory()
    outer = folds[dataset]
    folder = data.OUT / "runs" / dataset
    complete = json.loads((folder / "complete.json").read_text())
    for p, h in complete["artifact_sha256"].items():
        assert data.sha(p) == h
    assert complete["model_sha256"] == data.sha(data.parent_path(outer, None))
    cutpath = data.OUT / "calibration" / f"fold_{outer}" / "operating_points.json"
    assert complete["calibration_sha256"] == data.sha(cutpath)
    cuts = json.loads(cutpath.read_text())
    raw = data.load_predictors(dataset, {outer})
    scores = native_scores(raw, model_for(outer, None))
    pd.testing.assert_series_equal(
        scores, pd.read_pickle(folder / "scoring_audit.pkl"), check_exact=True
    )
    expected = reconstruct_annotations(
        dataset,
        outer,
        raw,
        scores,
        cuts["fallback_thresholds"],
        cuts["thresholds"],
        data.OUT / "source_snapshot",
    )
    actual = pd.read_csv(
        folder / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    assert_scientific_equal(actual, expected)
    assert actual.automatic_accepted.all() and not actual.automatic_candidate.any()
    baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
    check_original_coordinates(actual, baseline)
    candidate = pd.read_pickle(folder / "candidates.pkl")
    assert candidate.automatic_candidate.equals(~candidate.automatic_accepted)
    assert set(scientific_keys(actual)) == set(
        scientific_keys(candidate.loc[candidate.automatic_accepted])
    )
    reference = pd.read_pickle(
        data.QE / "manual_reference_v2" / dataset / "curated_feature_crosswalk.pkl"
    )
    evaluated, candidate_eval = actual.copy(), candidate.copy()
    evaluated["adduct"], candidate_eval["adduct"] = (
        actual.adduct.map(adduct),
        candidate.adduct.map(adduct),
    )
    fresh = compare_manual_annotations(
        reference, evaluated, candidates=candidate_eval, dataset=dataset
    )
    for field in ("manual", "laser"):
        pd.testing.assert_frame_equal(
            getattr(fresh, field),
            pd.read_pickle(
                data.OUT / "evaluation_v2" / dataset / (field + "_comparison.pkl")
            ),
            check_exact=True,
        )
    digest = hashlib.sha256(reference.to_csv(index=False).encode()).hexdigest()
    for root in (data.OUT, data.PREVIOUS, data.BASE):
        assert (
            json.loads(
                (
                    root / "evaluation_v2" / dataset / "evaluation_identity.json"
                ).read_text()
            )["manual_sha256"]
            == digest
        )
    print("048 OUTER WHOLE/RAW COORDINATES/FRESH GOLD CHECK", dataset, flush=True)
    return 4


def extension_inputs():
    failure = data.OUT / "verification_memory_copy_v1_failure"
    files = [
        Path(__file__),
        Path(__file__).with_name("test_catboost_checker_v2.py"),
        bindings.REPO / "docs/validation/next_catboost_checker_v2.txt",
        failure / "failure.log",
        failure / "failure_receipt.json",
    ]
    return {str(p): data.sha(p) for p in files}


def main():
    configure_frozen_resources(data.OUT / "source_snapshot")
    protocol = json.loads((data.OUT / "protocol.json").read_text())
    assert protocol["source_input_sha256"] == bindings.source_inputs()
    original = Path(__file__).with_name("verify_qe_catboost_experiment.py")
    assert protocol["planned_actual_verifier_sha256"] == data.sha(original)
    extension = extension_inputs()
    ledger = data.OUT / "verification_memory_copy_v2"
    ledger.mkdir(exist_ok=False)
    data.dump(ledger / "planned_extension_inputs.json", extension)
    assert not (data.OUT / "integrity_receipt.json").exists()
    assert protocol["runtime"] == backend.environment()
    for relative, h in protocol["source_snapshot_sha256"].items():
        assert data.sha(data.OUT / "source_snapshot" / relative) == h
    produced = json.loads((data.OUT / "producer_complete.json").read_text())
    assert produced["all_blind_outputs_complete"] and produced[
        "protocol_sha256"
    ] == data.sha(data.OUT / "protocol.json")
    datasets, _ = data.inventory()
    frozen = json.loads((data.OUT / "all_calibrations_frozen.json").read_text())
    assert frozen == {
        str(o): data.sha(
            data.OUT / "calibration" / f"fold_{o}" / "operating_points.json"
        )
        for o in range(5)
    }
    for d in datasets:
        assert produced["dataset_complete_sha256"][d] == data.sha(
            data.OUT / "runs" / d / "complete.json"
        )
    with ProcessPoolExecutor(max_workers=2) as executor:
        parents = list(executor.map(check_parent, data.parents()))
    assert len(parents) == 25 and sum(parents) == 528
    with ProcessPoolExecutor(max_workers=2) as executor:
        calibration = list(executor.map(check_calibration, range(5)))
    assert sum(calibration) == 132
    with ProcessPoolExecutor(max_workers=2) as executor:
        modes = list(executor.map(check_outer, datasets))
    assert sum(modes) == 132
    summary = json.loads(
        (
            data.OUT
            / "comparison_with_experiment_041/target_scope_without_X4/summary.json"
        ).read_text()
    )
    assert (
        summary["target_manual_rows"] == 92100
        and summary["unresolved_manual_after"] == 28
    )
    assert bindings.source_inputs() == protocol["source_input_sha256"]
    assert extension == extension_inputs()
    artifacts = {
        str(p): data.sha(p)
        for p in data.OUT.rglob("*")
        if p.is_file()
        and p.name != "integrity_receipt.json"
        and "__pycache__" not in p.parts
    }
    data.dump(
        data.OUT / "integrity_receipt.json",
        dict(
            all_passed=True,
            new_parents=25,
            new_classifiers=100,
            training_query_uses=528,
            inner_score_reproductions=132,
            outer_score_reproductions=33,
            dataset_modes=132,
            original_coordinates_checked=True,
            same_manual_reference=True,
            source_input_sha256=protocol["source_input_sha256"],
            artifact_sha256=artifacts,
            verifier_sha256=data.sha(Path(__file__)),
            original_planned_verifier_sha256=data.sha(original),
            original_planned_verifier_completed=False,
            verification_extension_files_sha256={
                Path(p).name: h for p, h in extension.items()
            },
            memory_copy_v2_only=True,
            note="Original planned verifier passed25modelparents,then hit Pandas3read-only view assignment. Original source/failure preserved; v2 ONLY owns a numeric copy before identical normalization. Alloriginal sources/producer/models/cuts/annotations unchanged. Separate scalar competition/counts/whole-assembly and direct native prediction. Shared CatBoost native library,005/027chemistry,034MIPsolver; not independent chemical confirmation,fresh deployment or external validation.",
        ),
    )
    print("048 V2 MEMORY-OWNERSHIP CHECKER ALL INTEGRITY CHECKS PASS", flush=True)


if __name__ == "__main__":
    main()
