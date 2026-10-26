"""NO-FIT stage A: all363 blind tables, then25authorized matrix reconstructions.

This is NOT the final model/export preflight and cannot authorize real fits.
It does not select cutoffs, score queries, assemble or change annotations.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import catboost_experiment_data as data
import catboost_evidence_model as model

OUT = data.QE / "catboost_input_compatibility_v1"


def inputs():
    datasets, folds = data.inventory()
    paths = {
        Path(__file__),
        Path(data.__file__),
        Path(model.__file__),
        data.PLAN,
        Path(__file__).with_name("requirements-catboost.txt"),
        Path(__file__).with_name("frozen_resources.py"),
        Path(__file__).with_name("context.py"),
        data.BASE / "run_manifest.json",
        data.QE / "rt_atlas_development/fold_assignment.json",
        data.PREPARED / "protocol.json",
        data.PREPARED / "summary.json",
        data.CORE / "protocol.json",
        data.CORE / "integrity_receipt.json",
    }
    paths.update(
        p
        for folder in ("app", "reference_data", "methods")
        for p in (REPO / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    for d in datasets:
        checkpoint = data.PREPARED / "checkpoints" / (d + ".json")
        paths.add(checkpoint)
        prepared = json.loads(checkpoint.read_text())["prepared_sha256"]
        for mask in data.masks_for(folds[d]):
            path = data.prepared_path(d, mask)
            assert data.sha(path) == prepared[str(path)]
            paths.add(path)
        paths.add(data.BASE / "evaluation_v2" / d / "manual_comparison.pkl")
    for outer, inner in data.parents():
        paths.add(data.parent_path(outer, inner, root=data.CORE))
    return {str(p): data.sha(p) for p in sorted(paths)}


def blind_tables(item):
    d, own = item
    configure_frozen_resources(REPO)
    rows = []
    for mask in data.masks_for(own):
        raw = data.load_predictors(d, mask)
        assert not {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
        } & set(raw["pool"])
        assert raw["pool"].is_is.isin([True, False]).all()
        assert raw["pool"].stream.equals(raw["evidence"].stream)
        assert list(raw["evidence"].select_dtypes(exclude=["number", "bool"])) == [
            "stream",
            "class",
            "adduct",
            "stream_class",
        ]
        rows.append(
            dict(
                dataset=d,
                excluded_folds=sorted(mask),
                candidate_options=len(raw["pool"]),
                predictors=len(raw["evidence"].columns),
                original_prepared_sha256=data.sha(data.prepared_path(d, mask)),
            )
        )
    print("ALL11 ORIGINAL108 BLIND SCOPES CHECKED", d, flush=True)
    return rows


def parent_matrix(item):
    outer, inner = item
    configure_frozen_resources(REPO)
    from matlab_annotation.mode_evidence_experts import _subset

    records, training, excluded = data.training_records(outer, inner)
    old = pd.read_pickle(data.parent_path(outer, inner, root=data.CORE))
    assert (
        set(old["training_cohorts"]) == training
        and set(old["excluded_groups"]) == excluded
    )
    rows = []
    for mode in data.MODES:
        subset = {d: _subset(raw, mode) for d, raw in records.items()}
        x, y, groups, weights, categorical = model.fitting_matrix(
            subset, training, excluded
        )
        expected = data.independent_fitting_matrix(records, training, excluded, mode)
        pd.testing.assert_frame_equal(x, expected[0], check_exact=True)
        for a, b in zip((y, groups, weights), expected[1:4], strict=True):
            np.testing.assert_array_equal(a, b)
        assert categorical == expected[4] == list(old["experts"][mode]["categories"])
        native = old["experts"][mode]
        assert (
            list(x) == native["columns"]
            and len(x) == native["training_pairs"]
            and int(y.sum()) == native["positives"]
        )
        for c in categorical:
            assert set(x[c]) == set(native["categories"][c])
        rows.append(
            dict(
                outer=outer,
                inner=inner,
                stream=mode,
                training_projects=len(training),
                training_queries=len(records),
                training_pairs=len(x),
                positives=int(y.sum()),
                predictors=len(x.columns),
                fitting_matrix_sha256=model.fingerprint(x, y, groups, weights),
                original_hgb_model_sha256=data.sha(
                    data.parent_path(outer, inner, root=data.CORE)
                ),
            )
        )
    print(
        "AUTHORIZED CATBOOST INPUT MATRIX PARITY",
        outer,
        inner,
        len(records),
        flush=True,
    )
    return rows, len(records)


def main():
    configure_frozen_resources(REPO)
    datasets, folds = data.inventory()
    hashes = inputs()
    runtime = model.environment()
    assert json.loads((data.CORE / "integrity_receipt.json").read_text())["all_passed"]
    OUT.mkdir(exist_ok=False)
    data.dump(OUT / "planned_inputs.json", hashes)
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(blind_tables, [(d, folds[d]) for d in datasets]))
    blind = [row for part in results for row in part]
    assert len(blind) == 363 and hashes == inputs()
    data.dump(
        OUT / "all363_blind_receipt.json",
        dict(
            all_passed=True,
            unique_nested_tables=363,
            manual_tables_decoded_before_all363_gate=0,
            source_input_sha256=hashes,
            classifier_fits=0,
            annotation_changes=False,
            thresholds_selected=False,
        ),
    )
    # Only NOW reconstruct authorized TRAINING labels/matrices. No heldout
    # query gold can choose a model, hyperparameter, threshold or stopping time.
    with ProcessPoolExecutor(max_workers=2) as executor:
        matrices = list(executor.map(parent_matrix, data.parents()))
    rows = [row for part, _ in matrices for row in part]
    uses = sum(n for _, n in matrices)
    assert len(rows) == 100 and len(matrices) == 25 and uses == 528
    predictions = []
    for outer in range(5):
        for d in datasets:
            own = folds[d]
            predictions.append(
                dict(
                    dataset=d,
                    outer=outer,
                    kind="outer" if own == outer else "inner_calibration",
                    excluded_folds=sorted({outer, own}),
                    prepared_sha256=data.sha(data.prepared_path(d, {outer, own})),
                )
            )
    assert (
        len(predictions) == 165 and sum(r["kind"] == "outer" for r in predictions) == 33
    )
    pd.DataFrame(blind).to_csv(OUT / "all363_blind_tables.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "all25_parent_mode_matrices.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUT / "all165_prediction_contexts.csv", index=False
    )
    assert hashes == inputs() and runtime == model.environment()
    data.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            unique_nested_tables=363,
            predictors=108,
            parent_preparations=25,
            training_query_uses=uses,
            mode_matrices=100,
            inner_calibration_prediction_uses=132,
            outer_prediction_uses=33,
            source_input_sha256=hashes,
            runtime=runtime,
            matrix_table_sha256=data.sha(OUT / "all25_parent_mode_matrices.csv"),
            classifier_fits=0,
            annotation_changes=False,
            thresholds_selected=False,
            actual_training_fit_authorized=False,
            note="StageAinputcompatibility ONLY. All363 unlabelled input scopes checked before authorized training matrix reconstruction. Exactoriginal029 rows/labels/projectweights/categoryvocabulary; scalar/positional reconstruction for100modematrices. DoesNOTreplace final model/export preflight or actualindependent verifier; NO QEclassifierfit or annotation benchmark.",
        ),
    )


if __name__ == "__main__":
    main()
