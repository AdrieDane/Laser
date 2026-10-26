"""StageA NO-FIT controls; cannot authorize actual049 classifier training.

All363 independent gate,528old supervision uses,25x4 consumer matrices,165old
scores and33WHOLE041 scientific controls. Future actual runner/verifier still
need complete final source binding and a separate explicit fit-permission gate.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import matrix_experiment_data as data
import matrix_mode_model as backend
import matrix_pipeline_assembly as assembly

OUT = data.QE / "matrix_context_stage_a_v1"


def inputs():
    paths = {
        Path(__file__),
        data.PLAN,
        data.nested.CHECK / "receipt.json",
        data.nested.OUT / "blind_receipt.json",
        data.CORE / "protocol.json",
        data.PREVIOUS / "protocol.json",
        data.PREVIOUS / "integrity_receipt.json",
    }
    for name in (
        "matrix_experiment_data.py",
        "matrix_mode_model.py",
        "matrix_pipeline_assembly.py",
        "test_matrix_mode_model.py",
        "test_matrix_pipeline_assembly.py",
        "preflight_qe_catboost_models.py",
        "verify_qe_catboost_experiment.py",
        "catboost_pipeline_assembly.py",
        "catboost_operating_points.py",
        "run_qe_np_context_bmp_experiment.py",
        "run_qe_add_only_class_experiment.py",
        "run_qe_atlas_experiment.py",
        "frozen_resources.py",
        "context.py",
    ):
        paths.add(Path(__file__).with_name(name))
    hashes = data.nested.inputs()
    nested = json.loads((data.nested.OUT / "blind_receipt.json").read_text())
    hashes.update(nested["artifact_sha256"])
    datasets, folds = data.inventory()
    for outer, inner in data.parents():
        paths.add(data.old.parent_path(outer, inner, root=data.CORE))
    for outer in range(5):
        paths.update(
            (
                data.CORE / "calibration" / f"fold_{outer}/operating_points.json",
                data.PREVIOUS / "calibration" / f"fold_{outer}/operating_points.json",
                data.old.STD / "models" / f"fold_{outer}_standard_models.json",
            )
        )
    for dataset in datasets:
        paths.update(
            (
                data.CORE / "runs" / dataset / "automatic_annotations.csv",
                data.PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
                data.PREVIOUS / "runs" / dataset / "candidates.pkl",
                data.old.MASS / "runs" / dataset / "private_mass_candidates.pkl",
            )
        )
        for outer in range(5):
            paths.add(
                data.CORE
                / (
                    f"runs/{dataset}/scoring_audit.pkl"
                    if outer == folds[dataset]
                    else f"calibration/fold_{outer}/{dataset}_blind_scores.pkl"
                )
            )
    for name in (
        "nested_standard_development_v1",
        "standard_context_preparation_v1/measurements",
    ):
        paths.update(
            p
            for p in (data.QE / name).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
        )
    paths.add(data.CORE / "runner_source.py")
    hashes.update({str(p): data.nested.pilot.sha(p) for p in sorted(paths)})
    assert all(data.nested.pilot.sha(Path(p)) == h for p, h in hashes.items())
    return hashes


def training(item):
    configure_frozen_resources(REPO)
    from preflight_qe_catboost_models import original_runner
    from matlab_annotation.evidence_training import evidence_matrix

    outer, inner = item
    legacy = original_runner()
    records, allowed, excluded = data.training_records(outer, inner)
    datasets, folds = data.inventory()
    assert set(records) == {d for d in datasets if d[:7] in allowed}
    for dataset, record in records.items():
        mask = {outer, folds[dataset]} | (set() if inner is None else {inner})
        raw = data.old.load_predictors(dataset, mask)
        old, manual, identities = legacy.with_training_labels(
            raw, permitted_groups=allowed, excluded_groups=excluded
        )
        for column in ("training_positive", "training_label_known"):
            pd.testing.assert_series_equal(
                record["pool"][column], old["pool"][column], check_exact=True
            )
        _, current_manual = data.authorize_labels(
            data.load_predictors(dataset, mask), allowed, excluded
        )
        pd.testing.assert_frame_equal(manual, current_manual, check_exact=True)
        from run_qe_atlas_experiment import species, adduct

        expected = [
            (s, species(n), adduct(a)) in identities
            for s, n, a in raw["pool"][["stream", "name", "adduct"]].itertuples(
                index=False, name=None
            )
        ]
        np.testing.assert_array_equal(
            record["pool"].calibration_identity_in_manual, expected
        )
        pd.testing.assert_frame_equal(
            record["evidence"][raw["evidence"].columns],
            raw["evidence"],
            check_exact=True,
        )
    model = pd.read_pickle(data.old.parent_path(outer, inner, root=data.CORE))
    prototype = dict(
        model,
        experts={
            s: dict(expert, columns=list(next(iter(records.values()))["evidence"]))
            for s, expert in model["experts"].items()
        },
    )
    ledgers = backend.fitting_ledger(records, prototype)
    for mode, expert in prototype["experts"].items():
        frames, labels, groups = [], [], []
        for record in records.values():
            use = record["pool"].stream.eq(mode) & record["pool"].training_label_known
            frames.append(record["evidence"].loc[use])
            labels.extend(record["pool"].loc[use, "training_positive"].astype(int))
            groups.extend([record["query_group"]] * int(use.sum()))
        frame = pd.concat(frames, ignore_index=True)
        native = evidence_matrix(frame, expert)
        counts = pd.Series(groups).value_counts().to_dict()
        weights = np.asarray([len(groups) / len(counts) / counts[g] for g in groups])
        assert backend.digest(native) == ledgers[mode]["matrix_sha256"]
        assert backend.digest(np.asarray(labels, int)) == ledgers[mode]["labels_sha256"]
        assert backend.digest(weights) == ledgers[mode]["weights_sha256"]
        assert len(frame) == model["experts"][mode]["training_pairs"]
        assert sum(labels) == model["experts"][mode]["positives"]
    print(
        "049 STAGEA ORIGINAL LABELS/NATIVE MATRIX EXACT",
        outer,
        inner,
        len(records),
        flush=True,
    )
    return dict(outer=outer, inner=inner, query_uses=len(records), ledgers=ledgers)


def query(dataset):
    configure_frozen_resources(REPO)
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from verify_qe_catboost_experiment import assert_scientific_equal

    _, folds = data.inventory()
    own = folds[dataset]
    outer_raw, outer_scores = None, None
    for outer in range(5):
        raw = data.load_predictors(dataset, {outer, own})
        original = data.old.load_predictors(dataset, {outer, own})
        model = pd.read_pickle(
            data.old.parent_path(outer, None if outer == own else own, root=data.CORE)
        )
        data.validate_record(
            raw,
            model["training_cohorts"],
            model["excluded_groups"],
            query_is_training=False,
        )
        pd.testing.assert_frame_equal(raw["pool"], original["pool"], check_exact=True)
        pd.testing.assert_frame_equal(
            raw["evidence"][original["evidence"].columns],
            original["evidence"],
            check_exact=True,
        )
        stripped = dict(raw, evidence=raw["evidence"][original["evidence"].columns])
        score = predict_mode_evidence_experts(stripped, model)
        saved = data.CORE / (
            f"runs/{dataset}/scoring_audit.pkl"
            if outer == own
            else f"calibration/fold_{outer}/{dataset}_blind_scores.pkl"
        )
        pd.testing.assert_series_equal(score, pd.read_pickle(saved), check_exact=True)
        if outer == own:
            outer_raw, outer_scores = raw, score
    modecuts = json.loads(
        (data.CORE / "calibration" / f"fold_{own}/operating_points.json").read_text()
    )["thresholds"]
    core = assembly.assemble_core(dataset, own, outer_raw, outer_scores, modecuts, REPO)
    reference = pd.read_csv(
        data.CORE / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    assert_scientific_equal(core["annotations"], reference)
    completioncuts = json.loads(
        (
            data.PREVIOUS / "calibration" / f"fold_{own}/operating_points.json"
        ).read_text()
    )["thresholds"]
    result = assembly.complete(
        dataset, own, outer_raw, outer_scores, core, completioncuts
    )
    reference = pd.read_csv(
        data.PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    assert_scientific_equal(result["annotations"], reference)
    disabled, candidates = assembly.whole_disabled(dataset)
    pd.testing.assert_frame_equal(
        disabled,
        pd.read_csv(
            data.PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        ),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        candidates,
        pd.read_pickle(data.PREVIOUS / "runs" / dataset / "candidates.pkl"),
        check_exact=True,
    )
    print("049 STAGEA FIVE OLD SCORES/WHOLE041 CONTROL PASS", dataset, flush=True)
    return dict(
        dataset=dataset,
        old_score_checks=5,
        whole029_core005027=True,
        whole041_recomputed_completion=True,
        whole_disabled041_exact=True,
    )


def main():
    configure_frozen_resources(REPO)
    assert not data.OUT.exists(), "No actual049 outputs before final fit permission"
    checked = json.loads((data.nested.CHECK / "receipt.json").read_text())
    assert checked["all_passed"] and checked["query_tables"] == 363
    hashes = inputs()
    OUT.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(training, data.parents()))
    assert len(rows) == 25 and sum(r["query_uses"] for r in rows) == 528
    datasets, _ = data.inventory()
    with ProcessPoolExecutor(max_workers=2) as executor:
        cases = list(executor.map(query, datasets))
    assert len(cases) == 33 and sum(c["old_score_checks"] for c in cases) == 165
    assert hashes == inputs()
    data.nested.pilot.dump(OUT / "training_matrices.json", rows)
    data.nested.pilot.dump(OUT / "old_controls.json", cases)
    data.nested.pilot.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            classifier_fits=0,
            actual049_fit_permission=False,
            source_input_sha256=hashes,
            original_training_query_uses=528,
            original_score_checks=165,
            whole041_control_datasets=33,
            prototype_columns=116,
            matrix_nested_receipt_sha256=data.nested.pilot.sha(
                data.nested.CHECK / "receipt.json"
            ),
            note="Compatibility/sourceinput stageA ONLY; no actual049 fits/calibration/newannotations. Complete actualdriver/checker/sourcebinding and separate finalpreflightpermission stillrequired.",
        ),
    )
    print("049 STAGEA ALLPASS; STILL NO ACTUAL FIT PERMISSION", flush=True)


if __name__ == "__main__":
    main()
