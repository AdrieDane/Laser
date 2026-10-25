"""040 whole-experiment audit,not an accuracy or deployment claim.

All25training parents/528uses,132INNERscore tables/five grids,33outer full
exports/candidates,ALLoldcells,originalquantification andsame33manual hashes.
No master PASS until all components and source/input/artifact fingerprints pass.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import run_qe_np_context_bmp_experiment as runner

OUT, QE = runner.OUT, runner.QE
NP_MODES = ("NP_Pos", "NP_Neg")


def normalized_csv(frame):
    return pd.read_csv(
        StringIO(frame.to_csv(index=False)),
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )


def verify_candidate_flags(candidates, previous_candidates):
    """Preserve legacy provenance flags; require new acceptances to clear them."""
    from standard_lipidomics.validation import _boolean

    def keys(frame, both=False):
        accepted = frame.automatic_accepted.map(_boolean).fillna(False).astype(bool)
        if both:
            accepted &= frame.automatic_candidate.map(_boolean).fillna(False).astype(bool)
        return set(runner.accepted_keys(frame.loc[accepted]))

    old_accepted = keys(previous_candidates)
    old_legacy = keys(previous_candidates, both=True)
    actual_accepted = keys(candidates)
    actual_legacy = keys(candidates, both=True)
    assert old_accepted.issubset(actual_accepted)
    assert actual_legacy == old_legacy, "Inherited provenance changed or new call still marked candidate"
    assert not (actual_accepted - old_accepted) & actual_legacy
    return len(old_legacy)


def verify_parent(job):
    configure_frozen_resources(OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.evidence_training import validate_feature_scope

    outer, inner, projects, folds, protocol = job
    blocked = {outer} if inner is None else {outer, inner}
    path = OUT / (
        f"fold_{outer}_nested_model.pkl"
        if inner is None
        else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
    )
    model = pd.read_pickle(path)
    allowed = {d[:7] for d in projects if folds[d] not in blocked}
    excluded = {d[:7] for d in projects if folds[d] in blocked}
    assert (
        set(model["training_cohorts"]) == allowed
        and set(model["excluded_groups"]) == excluded
    )
    assert model["format"] == "laser-NP-wide-context-BMP-completion-v1"
    assert set(model["experts"]) == set(NP_MODES) and len(model["columns"]) == 108
    expected_datasets = {d for d in projects if folds[d] not in blocked}
    for key in (
        "training_feature_masks",
        "partial_supervision_audit",
        "training_feature_scopes",
    ):
        assert set(model[key]) == expected_datasets
    totals = {s: Counter() for s in NP_MODES}
    bmp_projects = {s: set() for s in NP_MODES}
    categorical_values = {s: {} for s in NP_MODES}
    for dataset in sorted(expected_datasets):
        mask = blocked | {folds[dataset]}
        raw = runner.load_predictors(dataset, mask)
        validate_feature_scope(raw, allowed, excluded, query_is_training=True)
        labeled, _, audit = runner.label_allowed_query(
            raw, allowed, excluded, partial=True
        )
        assert model["partial_supervision_audit"][dataset] == audit
        assert set(model["training_feature_masks"][dataset]) == mask
        assert model["training_feature_scopes"][dataset] == dict(
            training=raw["reference_training_groups"],
            excluded=raw["reference_excluded_groups"],
            standard_training=raw["standard_reference_training_groups"],
            standard_excluded=raw["standard_reference_excluded_groups"],
        )
        pd.testing.assert_frame_equal(
            raw["evidence"], labeled["evidence"], check_exact=True
        )
        pool = labeled["pool"]
        for s in NP_MODES:
            rows = pool.stream.eq(s) & pool.training_label_known
            bmp = pool["class"].map(canonical_lipid_class).eq(
                "BMP"
            ) & ~pool.is_is.astype(bool)
            positive = rows & pool.training_positive
            totals[s]["known"] += int(rows.sum())
            totals[s]["positives"] += int(positive.sum())
            totals[s]["BMP_positives"] += int((positive & bmp).sum())
            totals[s]["context"] += int((rows & ~bmp).sum())
            if (positive & bmp).any():
                bmp_projects[s].add(dataset[:7])
            for c in raw["evidence"]:
                if not pd.api.types.is_numeric_dtype(raw["evidence"][c]):
                    categorical_values[s].setdefault(c, set()).update(
                        raw["evidence"].loc[rows, c].astype(str)
                    )
    preflight = QE / "np_context_BMP_preflight_v2"
    receipt = json.loads((preflight / "receipt.json").read_text())
    assert (
        runner.sha(preflight / "parent_training_counts.json")
        == receipt["parent_training_counts_sha256"]
    )
    counts = json.loads((preflight / "parent_training_counts.json").read_text())
    fresh = 0
    for s in NP_MODES:
        proof = next(
            r
            for r in counts
            if r["outer"] == outer and r["inner"] == inner and r["mode"] == s
        )
        assert (
            totals[s]["known"] == proof["known"]
            and totals[s]["positives"] == proof["positives"]
        )
        assert (
            totals[s]["BMP_positives"] == proof["BMP_positives"]
            and sorted(bmp_projects[s]) == proof["BMP_positive_projects"]
        )
        supported = totals[s]["BMP_positives"] >= 20 and len(bmp_projects[s]) >= 3
        assert supported == proof["completion_supported"]
        assert model["support"][s] == dict(
            known_BMP_positive_pairs=totals[s]["BMP_positives"],
            BMP_positive_projects=sorted(bmp_projects[s]),
            other_class_known_context_rows=totals[s]["context"],
        )
        expert = model["experts"][s]
        if not supported:
            assert expert is None
            continue
        fresh += 1
        assert (
            expert["training_pairs"] == totals[s]["known"]
            and expert["positives"] == totals[s]["positives"]
        )
        assert (
            set(expert["training_cohorts"]) == allowed
            and set(expert["excluded_groups"]) == excluded
        )
        assert expert["columns"] == model["columns"]
        assert expert["categories"] == {
            c: {v: i for i, v in enumerate(sorted(values))}
            for c, values in categorical_values[s].items()
        }
        for key, value in runner.estimator_parameters(protocol["classifier"]).items():
            assert expert["estimator"].get_params()[key] == value
        assert expert["estimator"].n_iter_ == protocol["classifier"]["max_iter"] == 180
    assert model["fresh_mode_classifier_fits"] == fresh
    assert model["training_pairs"] == sum(
        totals[s]["known"] for s in NP_MODES if model["experts"][s] is not None
    )
    assert model["positives"] == sum(
        totals[s]["positives"] for s in NP_MODES if model["experts"][s] is not None
    )
    print(
        "040 MODEL ALLLABELS/SCOPES/CONTEXT/SUPPORT PASS",
        outer,
        inner,
        fresh,
        flush=True,
    )
    return dict(
        training_queries=len(expected_datasets), fresh_NP_mode_classifiers=fresh
    )


def verify_inner(job):
    configure_frozen_resources(OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.np_context_bmp_completion import (
        predict_np_context,
        select_context_bmp_additions,
    )
    from matlab_annotation.operating_point import (
        select_operating_candidates,
        choose_balanced_training_points,
    )

    outer, projects, folds, protocol = job
    folder = OUT / "calibration" / f"fold_{outer}"
    training = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in projects if folds[d] == outer}
    core_folder = runner.PREVIOUS / "calibration" / f"fold_{outer}"
    core_cutoffs = json.loads((core_folder / "operating_points.json").read_text())[
        "thresholds"
    ]
    cases, manual_counts, base_correct = [], Counter(), Counter()
    for inner in sorted(set(folds.values()) - {outer}):
        model = pd.read_pickle(folder / f"inner_block_{inner}_model.pkl")
        for dataset in (d for d in projects if folds[d] == inner):
            raw = runner.load_predictors(dataset, {outer, inner})
            scores = predict_np_context(raw, model)
            pd.testing.assert_series_equal(
                scores,
                pd.read_pickle(folder / (dataset + "_blind_scores.pkl")),
                check_exact=True,
            )
            labeled, manual, _ = runner.label_allowed_query(
                raw, training, excluded, partial=False
            )
            manual_counts.update(
                manual.loc[
                    manual.lipid_class.map(canonical_lipid_class).eq("BMP")
                    & manual.stream.isin(NP_MODES),
                    "stream",
                ]
                .value_counts()
                .to_dict()
            )
            core_scores = pd.read_pickle(core_folder / (dataset + "_blind_scores.pkl"))
            assert core_scores.index.equals(raw["pool"].index)
            baseline = pd.read_pickle(
                runner.BASE / "runs" / dataset / "full_annotation.pkl"
            )
            for s in NP_MODES:
                part = labeled["pool"].loc[labeled["pool"].stream.eq(s)]
                core = select_operating_candidates(
                    part,
                    core_scores.loc[part.index],
                    threshold=core_cutoffs[s],
                    ambiguity_margin=0.05,
                )
                base_correct[s] += int(
                    core.loc[
                        core["class"].map(canonical_lipid_class).eq("BMP"),
                        "training_positive",
                    ].sum()
                )
                auto = baseline["streams"][s]["annotations"]
                occupancy = pd.concat(
                    [
                        core,
                        auto.loc[auto.is_is.fillna(False).astype(bool)].assign(
                            stream=s
                        ),
                    ],
                    ignore_index=True,
                )
                protection = runner.physical_protection(dataset, {outer, inner}, s, raw)
                cases.append(
                    (
                        s,
                        part,
                        scores.loc[part.index],
                        occupancy,
                        protection,
                        model["experts"][s] is not None,
                    )
                )
    records = []
    for threshold in protocol["threshold_grid"]:
        counts = {s: Counter() for s in NP_MODES}
        for s, part, scores, occupancy, protection, supported in cases:
            chosen = (
                select_context_bmp_additions(
                    part, scores, occupancy, stream=s, threshold=threshold, **protection
                )
                if supported
                else part.iloc[:0].copy()
            )
            counts[s]["added_correct"] += int(chosen.training_positive.sum())
            counts[s]["extras"] += int((~chosen.calibration_identity_in_manual).sum())
            counts[s]["different_peak"] += int(
                (
                    chosen.calibration_identity_in_manual
                    & chosen.training_label_known
                    & ~chosen.training_positive
                ).sum()
            )
        for s in NP_MODES:
            records.append(
                dict(
                    stream=s,
                    threshold=threshold,
                    manual=manual_counts[s],
                    core_correct=base_correct[s],
                    correct=base_correct[s] + counts[s]["added_correct"],
                    **dict(counts[s]),
                )
            )
    expected = pd.DataFrame(records)
    actual = pd.read_csv(
        folder / "training_operating_points.csv", float_precision="round_trip"
    )
    pd.testing.assert_frame_equal(normalized_csv(expected), actual, check_exact=True)
    feasible = expected.loc[
        expected.different_peak.le(
            protocol["additional_different_features_per_np_mode_manual"]
            * expected.manual
        )
    ]
    cutoffs = choose_balanced_training_points(
        feasible,
        extras_per_manual=protocol["additional_extras_per_endogenous_np_bmp_manual"],
        target=0.9,
    )
    saved = json.loads((folder / "operating_points.json").read_text())
    assert cutoffs is not None and cutoffs == saved["thresholds"]
    assert (
        set(saved["training_groups"]) == training
        and set(saved["outer_excluded"]) == excluded
    )
    assert saved["training_table_sha256"] == runner.sha(
        folder / "training_operating_points.csv"
    )
    selected = expected.loc[
        [
            t == cutoffs[s]
            for s, t in expected[["stream", "threshold"]].itertuples(
                index=False, name=None
            )
        ]
    ]
    pd.testing.assert_frame_equal(
        normalized_csv(selected.reset_index(drop=True)),
        pd.read_csv(
            folder / "selected_training_points.csv", float_precision="round_trip"
        ),
        check_exact=True,
    )
    print(
        "040 ALLINNER SCORES/FULLGRID/CUTOFFS PASS", outer, len(cases) // 2, flush=True
    )
    return len(cases) // 2


def verify_dataset(job):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.np_context_bmp_completion import (
        predict_np_context,
        select_context_bmp_additions,
    )
    from standard_lipidomics.validation import _boolean

    dataset, outer = job
    folder = OUT / "runs" / dataset
    model = pd.read_pickle(OUT / f"fold_{outer}_nested_model.pkl")
    thresholds = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    raw = runner.load_predictors(dataset, {outer})
    scores = predict_np_context(raw, model)
    pd.testing.assert_series_equal(
        scores, pd.read_pickle(folder / "scoring_audit.pkl"), check_exact=True
    )
    previous = runner.read_previous(dataset)
    parts = []
    for s in NP_MODES:
        part = raw["pool"].loc[raw["pool"].stream.eq(s)]
        parts.append(
            select_context_bmp_additions(
                part,
                scores.loc[part.index],
                previous.loc[previous.stream.eq(s)],
                stream=s,
                threshold=thresholds[s],
                **runner.physical_protection(dataset, {outer}, s, raw),
            )
            if model["experts"][s] is not None
            else part.iloc[:0].copy()
        )
    added = runner.fill_original_features(dataset, pd.concat(parts, ignore_index=True))
    pd.testing.assert_frame_equal(
        added, pd.read_pickle(folder / "private_additions.pkl"), check_exact=True
    )
    expected, candidates = runner.assemble_exports(dataset, previous, added)
    actual = pd.read_csv(
        folder / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    pd.testing.assert_frame_equal(actual, normalized_csv(expected), check_exact=True)
    pd.testing.assert_frame_equal(
        candidates, pd.read_pickle(folder / "candidates.pkl"), check_exact=True
    )
    text = pd.read_csv(
        folder / "automatic_annotations.csv", dtype=str, keep_default_na=False
    )
    pd.testing.assert_frame_equal(
        text.iloc[: len(previous)][list(previous)].reset_index(drop=True),
        previous.reset_index(drop=True),
        check_exact=True,
    )
    flags = candidates.automatic_accepted.map(_boolean).fillna(False).astype(bool)
    assert set(runner.accepted_keys(candidates.loc[flags])) == set(
        runner.accepted_keys(actual)
    )
    assert len(candidates.loc[flags]) == len(actual)
    inherited_candidate_flags = verify_candidate_flags(
        candidates,
        pd.read_pickle(runner.PREVIOUS / "runs" / dataset / "candidates.pkl"),
    )
    assert set(runner.accepted_keys(previous)).issubset(
        set(runner.accepted_keys(actual))
    )
    baseline = pd.read_pickle(runner.BASE / "runs" / dataset / "full_annotation.pkl")
    rounding, available, inherited = 0, 0, 0
    for s, payload in baseline["streams"].items():
        rows = actual.loc[actual.stream.eq(s)]
        assert not rows.feat_feature_id.astype(str).duplicated().any()
        original = payload["features"].copy()
        original.index = original.feature_id.astype(str)
        source = original.loc[rows.feat_feature_id.astype(str)]
        for dest, origin in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            a, b = rows[dest].to_numpy(float), source[origin].to_numpy(float)
            np.testing.assert_allclose(
                a, b, rtol=8 * np.finfo(float).eps, atol=0, equal_nan=True
            )
            rounding += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        for c in original:
            col = "feat_" + c
            is_new = rows.index.to_numpy() >= len(previous)
            if col not in text:
                assert not is_new.any(), (dataset, s, col, "missing new original field")
                continue
            values = text.loc[rows.index, col]
            measured = source[c].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(original[c]):
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
                wanted = measured.to_numpy(float)
                populated = values.ne("").to_numpy()
                check = populated | is_new
                np.testing.assert_allclose(
                    numeric[check],
                    wanted[check],
                    rtol=8 * np.finfo(float).eps,
                    atol=0,
                    equal_nan=True,
                    err_msg=col,
                )
                inherited += int((~populated & np.isfinite(wanted) & ~is_new).sum())
                available += int(populated.sum())
            else:
                wanted = measured.fillna("").astype(str).to_numpy()
                actual_text = values.astype(str).to_numpy()
                populated = actual_text != ""
                assert (
                    actual_text[populated | is_new] == wanted[populated | is_new]
                ).all(), col
                inherited += int(((~populated) & (wanted != "") & ~is_new).sum())
                available += int(populated.sum())
    identities = [
        json.loads(
            (
                QE / v / "evaluation_v2" / dataset / "evaluation_identity.json"
            ).read_text()
        )
        for v in ("baseline", "experiment_029", OUT.name)
    ]
    assert len({i["manual_sha256"] for i in identities}) == 1
    assert identities[-1]["selected_sha256"] == runner.sha(
        folder / "automatic_annotations.csv"
    )
    assert identities[-1]["candidates_sha256"] == runner.sha(folder / "candidates.pkl")
    complete = json.loads((folder / "complete.json").read_text())
    assert (
        complete["added_annotations"] == len(added)
        and complete["previous_calls_immutable"] is True
    )
    assert complete["thresholds"] == thresholds
    assert complete["counts"] == actual.groupby("stream").size().to_dict()
    print(
        "040 ALL4MODES/OLDALLCELLS/NEWRAWQUANT/EXPORT/GOLD PASS",
        dataset,
        len(added),
        flush=True,
    )
    return dict(
        dataset=dataset,
        added_annotations=len(added),
        raw_stream_checks=4,
        roundoff=rounding,
        available_measurement_cells_checked=available,
        inherited_unpopulated_old_measurement_cells=inherited,
        inherited_accepted_candidate_provenance_flags=inherited_candidate_flags,
    )


def main():
    protocol, projects = runner.initialize()
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    files = {
        OUT / "protocol.json",
        OUT / "runner_source.py",
        Path(__file__),
        QE / "np_context_BMP_preflight_v2/receipt.json",
        QE / "np_context_BMP_preflight_v2/parent_training_counts.json",
    }
    files.update(OUT.glob("fold_*_nested_model.pkl"))
    for name in ("calibration", "runs", "evaluation_v2", "reports_v2"):
        files.update(p for p in (OUT / name).rglob("*") if p.is_file())
    artifacts = {str(p): runner.sha(p) for p in sorted(files)}
    jobs = [
        (o, i, projects, folds, protocol)
        for o in range(5)
        for i in [None, *sorted(set(folds.values()) - {o})]
    ]
    with ProcessPoolExecutor(max_workers=3) as executor:
        models = list(executor.map(verify_parent, jobs))
        inner = list(
            executor.map(
                verify_inner, [(o, projects, folds, protocol) for o in range(5)]
            )
        )
        outputs = list(executor.map(verify_dataset, [(d, folds[d]) for d in projects]))
    assert len(models) == 25 and sum(m["training_queries"] for m in models) == 528
    assert (
        sum(inner) == 132
        and len(outputs) == 33
        and sum(r["raw_stream_checks"] for r in outputs) == 132
    )
    for p, h in {**protocol["input_sha256"], **artifacts}.items():
        assert runner.sha(Path(p)) == h, p
    for p, h in protocol["source_snapshot_sha256"].items():
        assert runner.sha(OUT / "source_snapshot" / p) == h, p
    runner.dump(
        OUT / "integrity_receipt.json",
        dict(
            all_passed=True,
            source_sha256=runner.sha(Path(__file__)),
            planned_verifier_sha256=protocol["planned_verifier_sha256"],
            source_hash_checks=len(protocol["source_snapshot_sha256"]),
            input_hash_checks=len(protocol["input_sha256"]),
            bound_verification_artifacts=len(artifacts),
            fresh_context_parents=25,
            fresh_NP_mode_classifiers=sum(
                m["fresh_NP_mode_classifiers"] for m in models
            ),
            training_query_checks=528,
            independently_prefit_label_checked_unique_tables=330,
            inner_score_table_checks=132,
            calibration_grid_checks=5,
            outer_datasets=33,
            original_coordinate_stream_checks=132,
            old_allcell_export_preservation_datasets=33,
            candidate_export_parity_checks=33,
            inherited_accepted_candidate_provenance_flags=sum(
                r["inherited_accepted_candidate_provenance_flags"] for r in outputs
            ),
            identical_manual_reference_hash_checks=33,
            added_annotations=sum(r["added_annotations"] for r in outputs),
            preexisting_roundoff_cells=sum(r["roundoff"] for r in outputs),
            available_measurement_cells_checked=sum(
                r["available_measurement_cells_checked"] for r in outputs
            ),
            inherited_unpopulated_old_measurement_cells=sum(
                r["inherited_unpopulated_old_measurement_cells"] for r in outputs
            ),
            note="Complete engineering replay/label/scope/originalmeasurement audit,not analytical FDR/identification/deployment approval. ALLoldacceptedannotationcells exactlypreserved,ALLnew original quantification fields checked. All33datasets/132modes andsamegoldreference. No posthoc classwinner splice or threshold relaxation.",
        ),
    )


if __name__ == "__main__":
    main()
