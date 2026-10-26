"""049 complete model/scopes/matrices/INNER curves/outer/raw/gold verification.

Reconstruct selection separately. Fixed previously audited native005/027/041
physical/chemical components remain shared scientific definitions, not caches
of old outputs. No fitting, changes, tolerance adaptation or class splicing.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import matrix_experiment_data as data
import matrix_experiment_bindings as bindings
import run_qe_matrix_context_experiment as runner
import matrix_operating_points as operating


def select_scalar(pool, scores, cut):
    ranked = pool.assign(_score=scores).sort_values(
        ["_score", "feat_intensity"], ascending=False, kind="stable"
    )
    groups, rows, duplicates = {}, [], set()
    for index, fid, name, ion, score in ranked[
        ["feat_feature_id", "name", "adduct", "_score"]
    ].itertuples(index=True, name=None):
        key = (fid, name, ion)
        if key in duplicates:
            continue
        duplicates.add(key)
        groups.setdefault(fid, []).append((index, name, ion, float(score)))
        rows.append((index, fid, name, ion, float(score)))
    selected, identities = [], set()
    for index, fid, name, ion, score in rows:
        siblings = groups[fid]
        if siblings[0][0] != index or score < cut:
            continue
        if len(siblings) > 1 and score - siblings[1][3] < 0.05:
            continue
        if (name, ion) in identities:
            continue
        identities.add((name, ion))
        selected.append(index)
    result = pool.loc[selected].copy()
    result["class_selection_score"] = scores.loc[selected]
    result["automatic_accepted"] = True
    result["auto_annotation_status"] = "class_rt_atlas_selected"
    return result


def choose_scalar(table):
    modes = sorted(table.stream.unique())
    parts = [
        table.loc[table.stream.eq(s)].sort_values("threshold", ascending=False)
        for s in modes
    ]
    budget = 0.095 * sum(int(p.manual.iloc[0]) for p in parts)
    best, chosen = None, None
    for indices in product(*(range(len(p)) for p in parts)):
        rows = [p.iloc[i] for p, i in zip(parts, indices)]
        extra = sum(float(r.extras) for r in rows)
        if extra > budget:
            continue
        key = (
            min(0.9, min(float(r.correct) / float(r.manual) for r in rows)),
            sum(float(r.correct) for r in rows),
            -sum(float(r.different_peak) for r in rows),
            -extra,
        )
        if best is None or key > best:
            best = key
            chosen = {s: float(r.threshold) for s, r in zip(modes, rows)}
    assert chosen is not None
    return chosen


def check_parent(item):
    configure_frozen_resources(data.OUT / "source_snapshot")
    from matlab_annotation.evidence_training import evidence_matrix
    from matrix_mode_model import digest

    outer, inner = item
    datasets, folds = data.inventory()
    blocked = {outer} | (set() if inner is None else {inner})
    allowed = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    model = runner.read_model(outer, inner)
    assert (
        set(model["training_cohorts"]) == allowed
        and set(model["excluded_groups"]) == excluded
    )
    records = {}
    for dataset in datasets:
        if folds[dataset] in blocked:
            continue
        raw = data.load_predictors(dataset, blocked | {folds[dataset]})
        labeled, _ = data.authorize_labels(raw, allowed, excluded)
        records[dataset] = labeled
        expected = dict(
            training=raw["matrix_reference_training_groups"],
            excluded=raw["matrix_reference_excluded_groups"],
            reference_sha256=raw["matrix_reference_sha256"],
            layer_sha256=raw["matrix_layer_sha256"],
            excluded_folds=raw["excluded_folds"],
        )
        assert model["matrix_feature_scopes"][dataset] == expected
    assert set(records) == set(model["matrix_feature_scopes"])
    stage = json.loads((bindings.STAGE / "training_matrices.json").read_text())
    stage = next(r for r in stage if (r["outer"], r["inner"]) == item)
    for mode in data.MODES:
        expert = model["experts"][mode]
        estimator = expert["estimator"]
        assert (
            estimator.n_iter_ == len(estimator._predictors) == 180
            and estimator.n_features_in_ == 116
        )
        actual_parameters = estimator.get_params()
        assert all(actual_parameters[k] == v for k, v in data.parameters().items())
        assert (
            actual_parameters["warm_start"] is False
            and "sample_weight_policy" not in expert
        )
        frames, labels, groups = [], [], []
        for raw in records.values():
            use = raw["pool"].stream.eq(mode) & raw["pool"].training_label_known
            frames.append(raw["evidence"].loc[use])
            labels.extend(raw["pool"].loc[use, "training_positive"].astype(int))
            groups.extend([raw["query_group"]] * int(use.sum()))
        x = pd.concat(frames, ignore_index=True)
        categories = {
            c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
            for c in x
            if not pd.api.types.is_numeric_dtype(x[c])
        }
        assert (
            categories == expert["categories"]
            and list(x) == expert["columns"] == model["columns"]
        )
        np.testing.assert_array_equal(
            actual_parameters["categorical_features"], [c in categories for c in x]
        )
        matrix = evidence_matrix(x, expert)
        frequency = Counter(groups)
        weights = np.asarray(
            [len(groups) / len(frequency) / frequency[g] for g in groups]
        )
        ledger = model["independent_fitting_ledger"][mode]
        assert ledger == stage["ledgers"][mode]
        assert (
            digest(matrix) == ledger["matrix_sha256"]
            and digest(np.asarray(labels, int)) == ledger["labels_sha256"]
            and digest(weights) == ledger["weights_sha256"]
        )
        assert expert["training_pairs"] == len(x) and expert["positives"] == sum(labels)
        assert set(groups) == allowed
    print("049 INDEPENDENT PARENT/MATRICES/SCOPES", item, len(records), flush=True)
    return len(records)


def check_completion_curves(proposals, manual, base_counts, table):
    from annotator.composition_ranges import canonical_lipid_class

    p = proposals.copy()
    p["lipid_class"] = p.lipid_class.map(canonical_lipid_class)
    m = manual.copy()
    m["lipid_class"] = m.lipid_class.map(canonical_lipid_class)
    cuts = {s: {} for s in data.MODES}
    for (stream, cls), part in table.groupby(["stream", "lipid_class"], sort=True):
        q = p.loc[p.stream.eq(stream) & p.lipid_class.eq(cls)]
        ref = m.loc[m.stream.eq(stream) & m.lipid_class.eq(cls)]
        n, base = len(ref), int(base_counts.get((stream, cls), 0))
        assert part.manual_projects.eq(ref.project.nunique()).all()
        eligible = (
            n >= 100 and ref.project.nunique() >= 3 and base < 0.9 * n and cls != "X4"
        )
        positives = q.loc[q.training_positive]
        levels = {1.0}
        if eligible and len(positives):
            values = sorted(positives.class_selection_score, reverse=True)
            levels.update(
                float(values[int(np.ceil(f * len(values))) - 1])
                for f in np.arange(0.05, 1.001, 0.05)
            )
        assert set(part.threshold) == levels
        candidates = []
        for row in part.itertuples():
            selected = (
                q.loc[q.class_selection_score.ge(row.threshold)]
                if row.threshold < 1
                else q.iloc[:0]
            )
            correct = int(selected.training_positive.sum())
            extra = int((~selected.calibration_identity_in_manual).sum())
            wrong = int(
                (
                    selected.calibration_identity_in_manual
                    & selected.training_label_known
                    & ~selected.training_positive
                ).sum()
            )
            grouped = [
                (int(sub.training_positive.sum()), len(sub))
                for _, sub in selected.groupby("project", sort=True)
            ]
            leave = min(
                (
                    (correct - t) / (len(selected) - count)
                    for t, count in grouped
                    if len(selected) > count
                ),
                default=0.0,
            )
            worst = min((t / count for t, count in grouped if count >= 3), default=0.0)
            projects = selected.loc[selected.training_positive, "project"].nunique()
            feasible = row.threshold == 1 or bool(
                eligible
                and correct >= 10
                and projects >= 3
                and wrong == 0
                and extra <= 0.2 * correct
                and extra <= 0.02 * n
                and correct >= 0.8 * len(selected)
                and leave >= 0.8
                and worst >= 0.5
            )
            assert (
                row.manual,
                row.core_correct,
                row.eligible,
                row.added_correct,
                row.extras,
                row.different_peak,
                row.added_calls,
                row.positive_projects,
                row.feasible,
            ) == (
                n,
                base,
                eligible,
                correct,
                extra,
                wrong,
                len(selected),
                projects,
                feasible,
            )
            np.testing.assert_allclose(
                [
                    row.minimum_leave_one_project_agreement,
                    row.worst_substantial_project_agreement,
                ],
                [leave, worst],
                rtol=0,
                atol=1e-12,
            )
            gain = min(correct, max(0, int(np.ceil(0.9 * n)) - base))
            assert row.capped_gain == gain
            if feasible:
                candidates.append((gain, -extra, -len(selected), row.threshold))
        cuts[stream][cls] = float(max(candidates)[-1])
    return cuts


def independent_calibration_cases(outer):
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts

    datasets, folds = data.inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    cases, manuals = {}, []
    for dataset in datasets:
        inner = folds[dataset]
        if inner == outer:
            continue
        raw = data.load_predictors(dataset, {outer, inner})
        model = runner.read_model(outer, inner)
        data.validate_record(
            raw,
            model["training_cohorts"],
            model["excluded_groups"],
            query_is_training=False,
        )
        score = predict_mode_evidence_experts(raw, model)
        saved = (
            data.OUT / "calibration" / f"fold_{outer}" / (dataset + "_blind_scores.pkl")
        )
        pd.testing.assert_series_equal(score, pd.read_pickle(saved), check_exact=True)
        labeled, manual = data.authorize_labels(raw, allowed, excluded)
        cases[dataset] = dict(
            raw=raw,
            pool=labeled["pool"],
            scores=score,
            inner=inner,
            project=dataset[:7],
        )
        manuals.append(manual.assign(project=dataset[:7]))
    return cases, pd.concat(manuals, ignore_index=True), allowed, excluded


def check_calibration(outer):
    configure_frozen_resources(data.OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.add_only_class_completion import free_completion_winners
    import run_qe_np_context_bmp_experiment as protection

    folder = data.OUT / "calibration" / f"fold_{outer}"
    saved = json.loads((folder / "operating_points.json").read_text())
    for name, h in saved["artifact_sha256"].items():
        assert data.nested.pilot.sha(folder / name) == h
    cases, manual, allowed, excluded = independent_calibration_cases(outer)
    assert (
        set(saved["training_groups"]) == allowed
        and set(saved["outer_excluded"]) == excluded
    )
    rows = []
    for cut in operating.GRID:
        counts = {s: Counter(correct=0, extras=0, different_peak=0) for s in data.MODES}
        for case in cases.values():
            for stream, pool in case["pool"].groupby("stream", sort=False):
                chosen = select_scalar(pool, case["scores"].loc[pool.index], cut)
                counts[stream]["correct"] += int(chosen.training_positive.sum())
                counts[stream]["extras"] += int(
                    (~chosen.calibration_identity_in_manual).sum()
                )
                counts[stream]["different_peak"] += int(
                    (
                        chosen.calibration_identity_in_manual
                        & chosen.training_label_known
                        & ~chosen.training_positive
                    ).sum()
                )
        rows.extend(
            dict(
                stream=s,
                threshold=cut,
                manual=int(manual.stream.eq(s).sum()),
                **dict(counts[s]),
            )
            for s in data.MODES
        )
    curve = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(
        curve,
        pd.read_csv(
            folder / "training_operating_points.csv", float_precision="round_trip"
        ),
        check_dtype=False,
        check_exact=True,
    )
    modecuts = choose_scalar(curve)
    assert modecuts == saved["mode_thresholds"]
    free, base_counts = [], Counter()
    for dataset, case in cases.items():
        baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
        occupied = []
        for stream in data.MODES:
            pool = case["pool"].loc[case["pool"].stream.eq(stream)]
            core = select_scalar(pool, case["scores"].loc[pool.index], modecuts[stream])
            base_counts.update(
                {
                    (stream, c): int(n)
                    for c, n in core.assign(
                        _cls=core["class"].map(canonical_lipid_class)
                    )
                    .groupby("_cls")
                    .training_positive.sum()
                    .items()
                }
            )
            native = baseline["streams"][stream]["annotations"]
            occupied.extend(
                [
                    core,
                    native.loc[native.is_is.fillna(False).astype(bool)].assign(
                        stream=stream
                    ),
                ]
            )
        occupied = pd.concat(occupied, ignore_index=True)
        labeled = dict(case["raw"], pool=case["pool"])
        for stream in data.MODES:
            pool = case["pool"].loc[case["pool"].stream.eq(stream)]
            q = free_completion_winners(
                pool,
                case["scores"].loc[pool.index],
                occupied.loc[occupied.stream.eq(stream)],
                stream=stream,
                **protection.physical_protection(
                    dataset, {outer, case["inner"]}, stream, labeled
                ),
            )
            free.append(q.assign(project=dataset[:7]))
    proposals = pd.concat(free, ignore_index=True)
    keys = ["stream", "name", "adduct", "feat_feature_id", "project"]
    before = pd.read_pickle(folder / "private_completion_proposals.pkl")
    # Selection/scientific fields identical; auxiliary selector status differs.
    compare = [
        *keys,
        "class_selection_score",
        "training_positive",
        "training_label_known",
        "calibration_identity_in_manual",
    ]
    pd.testing.assert_frame_equal(
        proposals[compare].sort_values(keys).reset_index(drop=True),
        before[compare].sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    table = pd.read_csv(
        folder / "training_completion_points.csv", float_precision="round_trip"
    )
    assert (
        check_completion_curves(proposals, manual, base_counts, table)
        == saved["completion_thresholds"]
    )
    print("049 INDEPENDENT FULL MODE/COMPLETION CURVES", outer, len(cases), flush=True)
    return len(cases)


def reconstruct(dataset, outer, raw, scores, cuts, *, snapshot=None):
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions
    from matlab_annotation.add_only_class_completion import free_completion_winners
    from annotator.composition_ranges import canonical_lipid_class
    import run_qe_np_context_bmp_experiment as protection

    baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
    std = json.loads(
        (data.old.STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    scientific_root = (
        data.OUT / "source_snapshot" if snapshot is None else Path(snapshot)
    )
    catalogue = declared_standard_targets(
        scientific_root / "reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    mass = pd.read_pickle(
        data.old.MASS / "runs" / dataset / "private_mass_candidates.pkl"
    )
    automatic = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    parts = []
    for stream, payload in baseline["streams"].items():
        pool = raw["pool"].loc[raw["pool"].stream.eq(stream)]
        core = select_scalar(
            pool, scores.loc[pool.index], cuts["mode_thresholds"][stream]
        )
        native = (
            payload["annotations"]
            .loc[payload["annotations"].is_is.fillna(False).astype(bool)]
            .assign(stream=stream)
        )
        previous = pd.concat([core, native], ignore_index=True)
        corrected, _, _ = complete_declared_standards(
            previous,
            payload["features"],
            catalogue.loc[catalogue.stream.eq(stream)],
            std["library"],
            std["atlas"],
            std["transfer"],
            std["channels"],
            stream=stream,
            alignment_annotations=payload["annotations"],
        )
        local = mass.loc[mass.stream.eq(stream)] if len(mass) else mass
        added = select_calibrated_additions(local, automatic, corrected, stream=stream)
        whole_core = pd.concat([corrected, added], ignore_index=True).assign(
            stream=stream
        )
        free = free_completion_winners(
            pool,
            scores.loc[pool.index],
            whole_core,
            stream=stream,
            **protection.physical_protection(dataset, {outer}, stream, raw),
        )
        keep = []
        for index, row in free.iterrows():
            threshold = (
                cuts["completion_thresholds"]
                .get(stream, {})
                .get(canonical_lipid_class(row["class"]), 1.0)
            )
            if threshold < 1 and row.class_selection_score >= threshold:
                keep.append(index)
        parts.extend([whole_core, free.loc[keep]])
    return pd.concat(parts, ignore_index=True), baseline


def legacy_control(dataset):
    """NO-FIT independent whole041 selector/chemical policy engineering check."""
    from context import REPO

    configure_frozen_resources(REPO)
    from verify_qe_catboost_experiment import assert_scientific_equal

    _, folds = data.inventory()
    outer = folds[dataset]
    raw = data.load_predictors(dataset, {outer})
    scores = pd.read_pickle(data.CORE / "runs" / dataset / "scoring_audit.pkl")
    modecuts = json.loads(
        (data.CORE / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    completioncuts = json.loads(
        (
            data.PREVIOUS / "calibration" / f"fold_{outer}/operating_points.json"
        ).read_text()
    )["thresholds"]
    expected, _ = reconstruct(
        dataset,
        outer,
        raw,
        scores,
        dict(mode_thresholds=modecuts, completion_thresholds=completioncuts),
        snapshot=REPO,
    )
    previous = pd.read_csv(
        data.PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    assert_scientific_equal(expected, previous)
    print("049 FINAL NOFIT INDEPENDENT WHOLE041 CONTROL", dataset, flush=True)
    return dataset


def check_outer(dataset):
    configure_frozen_resources(data.OUT / "source_snapshot")
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from verify_qe_catboost_experiment import assert_scientific_equal
    from annotator.peak_evidence import sample_roles
    from standard_lipidomics.validation import (
        canonical_feature_id,
        compare_manual_annotations,
    )
    from run_qe_atlas_experiment import adduct

    _, folds = data.inventory()
    outer = folds[dataset]
    folder = data.OUT / "runs" / dataset
    done = json.loads((folder / "complete.json").read_text())
    for name, h in done["artifact_sha256"].items():
        assert data.nested.pilot.sha(folder / name) == h
    raw = data.load_predictors(dataset, {outer})
    model = runner.read_model(outer, None)
    data.validate_record(
        raw,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    scores = predict_mode_evidence_experts(raw, model)
    pd.testing.assert_series_equal(
        scores, pd.read_pickle(folder / "scoring_audit.pkl"), check_exact=True
    )
    cuts = json.loads(
        (data.OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )
    expected, baseline = reconstruct(dataset, outer, raw, scores, cuts)
    actual = pd.read_csv(
        folder / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    assert_scientific_equal(actual, expected)
    assert actual.automatic_accepted.all() and not actual.automatic_candidate.any()
    cells = 0
    for stream, part in actual.groupby("stream", sort=False):
        f = baseline["streams"][stream]["features"]
        assert not part.feat_feature_id.map(canonical_feature_id).duplicated().any()
        source = f.set_index(f.feature_id.map(canonical_feature_id)).loc[
            part.feat_feature_id.map(canonical_feature_id)
        ]
        for column in sample_roles(f)[0]:
            np.testing.assert_array_equal(
                pd.to_numeric(part["feat_" + column], errors="coerce").to_numpy(float),
                pd.to_numeric(source[column], errors="coerce").to_numpy(float),
            )
            cells += len(part)
    assert cells == done["original_sample_cells_exact"]
    candidate = pd.read_pickle(folder / "candidates.pkl")
    import run_qe_np_context_bmp_experiment as normalization

    assert set(normalization.accepted_keys(actual)) == set(
        normalization.accepted_keys(candidate.loc[candidate.automatic_accepted])
    )
    assert candidate.automatic_candidate.equals(~candidate.automatic_accepted)
    # Same reader as evaluation stage, independent authoritative crosswalk.
    evaluated = pd.read_csv(
        folder / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
    )
    reference = pd.read_pickle(
        data.QE / "manual_reference_v2" / dataset / "curated_feature_crosswalk.pkl"
    )
    fresh = compare_manual_annotations(
        reference,
        evaluated.assign(adduct=evaluated.adduct.map(adduct)),
        candidates=candidate.assign(adduct=candidate.adduct.map(adduct)),
        dataset=dataset,
    )
    for name in ("manual", "laser"):
        pd.testing.assert_frame_equal(
            getattr(fresh, name),
            pd.read_pickle(
                data.OUT / "evaluation_v2" / dataset / (name + "_comparison.pkl")
            ),
            check_exact=True,
        )
    identity = json.loads(
        (data.OUT / "evaluation_v2" / dataset / "evaluation_identity.json").read_text()
    )
    before = json.loads(
        (
            data.PREVIOUS / "evaluation_v2" / dataset / "evaluation_identity.json"
        ).read_text()
    )
    assert identity["manual_sha256"] == before["manual_sha256"]
    print("049 INDEPENDENT OUTER/SAMPLE CELLS/FRESH GOLD", dataset, cells, flush=True)
    return cells


def main():
    configure_frozen_resources(data.OUT / "source_snapshot")
    p = runner.protocol()
    assert p["planned_actual_verifier_sha256"] == data.nested.pilot.sha(Path(__file__))
    assert p["source_input_sha256"] == bindings.source_inputs()
    produced = json.loads((data.OUT / "producer_complete.json").read_text())
    assert produced["all_blind_outputs_complete"] and produced["datasets"] == 33
    assert produced["protocol_sha256"] == data.nested.pilot.sha(
        data.OUT / "protocol.json"
    )
    frozen_path = data.OUT / "all_calibrations_frozen.json"
    assert produced["calibration_freeze_sha256"] == data.nested.pilot.sha(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    for outer in range(5):
        assert frozen[str(outer)] == data.nested.pilot.sha(
            data.OUT / "calibration" / f"fold_{outer}/operating_points.json"
        )
    for relative, h in p["source_snapshot_sha256"].items():
        assert data.nested.pilot.sha(data.OUT / "source_snapshot" / relative) == h
    with ProcessPoolExecutor(max_workers=2) as executor:
        uses = list(executor.map(check_parent, data.parents()))
    assert len(uses) == 25 and sum(uses) == 528
    with ProcessPoolExecutor(max_workers=2) as executor:
        calibration = list(executor.map(check_calibration, range(5)))
    assert sum(calibration) == 132
    datasets, _ = data.inventory()
    with ProcessPoolExecutor(max_workers=2) as executor:
        cells = list(executor.map(check_outer, datasets))
    assert p["source_input_sha256"] == bindings.source_inputs()
    data.nested.pilot.dump(
        data.OUT / "integrity_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            parent_models=25,
            experts=100,
            training_query_uses=528,
            inner_score_checks=132,
            outer_score_checks=33,
            original_sample_cells_exact=sum(cells),
            mode_and_completion_calibrations=5,
            whole_original_assignments_reconstructed=33,
            actual_verifier_sha256=data.nested.pilot.sha(Path(__file__)),
            source_input_sha256=p["source_input_sha256"],
            note="Full049 source/matrix/labels/weights/scopes/nativeHGB/INNERcurves/wholeouter/rawsample/freshsamegold checks. No source/tolerance/default changes. Accuracy/wholedecision separate;notdeployment/FDR.",
        ),
    )
    print("049 COMPLETE INDEPENDENT INTEGRITY ALLPASS", flush=True)


if __name__ == "__main__":
    main()
