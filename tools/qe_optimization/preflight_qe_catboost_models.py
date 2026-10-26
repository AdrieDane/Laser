"""Final048 NO-FIT gate: original supervision/scores and whole export parity.

Must finish before the048 runner can create its experiment directory. Reuses
old029 fitted models ONLY to test compatibility, never as new CatBoost fits.
"""

from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import catboost_experiment_data as data
import catboost_evidence_model as backend
import catboost_experiment_bindings as bindings
from catboost_pipeline_assembly import assemble, disabled_replay
from verify_qe_catboost_experiment import fixed_winners, assert_scientific_equal


def original_runner():
    path = data.CORE / "runner_source.py"
    spec = importlib.util.spec_from_file_location(
        "legacy029_readonly_compatibility", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.BASE == data.BASE and module.OUT == data.CORE
    return module


def label_parity(item):
    outer, inner = item
    configure_frozen_resources(REPO)
    original = original_runner()
    records, training, excluded = data.training_records(outer, inner)
    _, folds = data.inventory()
    for d, labelled in records.items():
        mask = {outer, folds[d]} | (set() if inner is None else {inner})
        raw = data.load_predictors(d, mask)
        prior, manual, identities = original.with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded
        )
        for column in ("training_positive", "training_label_known"):
            pd.testing.assert_series_equal(
                labelled["pool"][column], prior["pool"][column], check_exact=True
            )
        from run_qe_atlas_experiment import species, adduct

        keys = [
            (s, species(n), adduct(a))
            for s, n, a in raw["pool"][["stream", "name", "adduct"]].itertuples(
                index=False, name=None
            )
        ]
        np.testing.assert_array_equal(
            labelled["pool"].calibration_identity_in_manual,
            [k in identities for k in keys],
        )
        _, new_manual = data.authorize_labels(raw, training, excluded)
        pd.testing.assert_frame_equal(new_manual, manual, check_exact=True)
    print(
        "048 PREFLIGHT ORIGINAL ROW LABELS EXACT",
        outer,
        inner,
        len(records),
        flush=True,
    )
    return len(records)


def legacy_predictions(dataset):
    configure_frozen_resources(REPO)
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.guarded_class_operating_point import clear_identity_winners
    from matlab_annotation.operating_point import select_operating_candidates
    from catboost_operating_points import select_candidates

    _, folds = data.inventory()
    own = folds[dataset]
    outer_scores = outer_raw = None
    for outer in range(5):
        raw = data.load_predictors(dataset, {outer, own})
        inner = None if own == outer else own
        model = pd.read_pickle(data.parent_path(outer, inner, root=data.CORE))
        scores = predict_mode_evidence_experts(raw, model)
        path = data.CORE / (
            f"runs/{dataset}/scoring_audit.pkl"
            if inner is None
            else f"calibration/fold_{outer}/{dataset}_blind_scores.pkl"
        )
        pd.testing.assert_series_equal(scores, pd.read_pickle(path), check_exact=True)
        for _, pool in raw["pool"].groupby("stream", sort=False):
            sub = scores.loc[pool.index]
            a, b = (
                fixed_winners(pool, sub),
                clear_identity_winners(pool, sub, ambiguity_margin=0.05),
            )
            assert a.index.equals(b.index)
            pd.testing.assert_series_equal(
                a.class_selection_score, b.class_selection_score, check_exact=True
            )
            # Existing valid domain, not a new threshold choice/benchmark.
            for cut in (0.05, 0.2, 0.5, 0.95):
                assert select_candidates(
                    pool, sub, baseline_threshold=cut
                ).index.equals(
                    select_operating_candidates(
                        pool, sub, threshold=cut, ambiguity_margin=0.05
                    ).index
                )
        if inner is None:
            outer_raw, outer_scores = raw, scores
    cuts = json.loads(
        (
            data.CORE / "calibration" / f"fold_{own}" / "operating_points.json"
        ).read_text()
    )["thresholds"]
    result = assemble(
        dataset,
        own,
        outer_raw,
        outer_scores,
        cuts,
        {s: {} for s in data.MODES},
        REPO,
        status="legacy029_control_replay",
    )
    reference = pd.read_csv(
        data.CORE / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    assert_scientific_equal(result["annotations"], reference)
    # No automatic-candidate metadata parity claim:048 normalizes final output
    # flags consistently, without changing calls, raw values or old source files.
    disabled, candidates = disabled_replay(dataset)
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
    print(
        "048 PREFLIGHT FIVE OLD SCORES/WHOLE029/DISABLED041 PASS", dataset, flush=True
    )
    return dict(
        dataset=dataset,
        old_score_reproductions=5,
        whole029_replay=True,
        disabled_whole041=True,
        old_calls=len(reference),
    )


def main():
    configure_frozen_resources(REPO)
    assert not data.OUT.exists(), "No actual048 outputs may precede final preflight"
    hashes, runtime = bindings.source_inputs(), backend.environment()
    data.PREFLIGHT.mkdir(exist_ok=False)
    data.dump(data.PREFLIGHT / "planned_inputs.json", hashes)
    datasets, _ = data.inventory()
    with ProcessPoolExecutor(max_workers=2) as executor:
        labels = list(executor.map(label_parity, data.parents()))
    assert len(labels) == 25 and sum(labels) == 528
    with ProcessPoolExecutor(max_workers=2) as executor:
        predictions = list(executor.map(legacy_predictions, datasets))
    assert (
        len(predictions) == 33
        and sum(p["old_score_reproductions"] for p in predictions) == 165
    )
    pd.DataFrame(predictions).to_csv(
        data.PREFLIGHT / "whole_control_replays.csv", index=False
    )
    assert hashes == bindings.source_inputs() and runtime == backend.environment()
    data.dump(
        data.PREFLIGHT / "receipt.json",
        dict(
            all_passed=True,
            actual_training_fit_authorized=True,
            classifier_fits=0,
            new_threshold_selections=0,
            annotation_changes=False,
            whole029_replays=33,
            disabled_whole041_datasets=33,
            legacy_label_row_parity_uses=528,
            legacy_inner_score_reproductions=132,
            legacy_outer_score_reproductions=33,
            runtime=runtime,
            planned_actual_verifier_sha256=data.sha(
                Path(__file__).with_name("verify_qe_catboost_experiment.py")
            ),
            source_input_sha256=hashes,
            note="Compatibility/export controls ONLY. All363/100matrix stageA bound; original029 row labels exact,165 old scores exact,whole029 scientific calls/raw coordinates and disabled041 exact. No new QE fits,cutoffs or accuracy claims. Planned048sources/verifier frozen by this receipt.",
        ),
    )
    print("048 FINAL NO-FIT PREFLIGHT ALL PASS; SOURCES FROZEN", flush=True)


if __name__ == "__main__":
    main()
