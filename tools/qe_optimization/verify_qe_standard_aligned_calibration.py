"""Supplemental033 replay: identical gold references and ALL inner calibration.

Read-only predictions/count reconstruction, never a new fit or annotation.
Separate from the already-running full discovery verifier.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "experiment_033"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replay(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    import run_qe_standard_aligned_experiment as runner
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.operating_point import (
        select_operating_candidates,
        choose_balanced_training_points,
    )

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    folder = OUT / "calibration" / f"fold_{outer}"
    excluded = {d[:7] for d in projects if folds[d] == outer}
    allowed = {d[:7] for d in projects if folds[d] != outer}
    cases, scores, manual_counts, fingerprints = {}, {}, Counter(), {}
    for inner in range(5):
        if inner == outer:
            continue
        model_path = folder / f"inner_block_{inner}_model.pkl"
        model = pd.read_pickle(model_path)
        fingerprints[model_path.relative_to(OUT).as_posix()] = sha(model_path)
        assert set(model["training_cohorts"]) == {
            d[:7] for d in projects if folds[d] not in {outer, inner}
        }
        assert set(model["excluded_groups"]) == {
            d[:7] for d in projects if folds[d] in {outer, inner}
        }
        for dataset in projects:
            if folds[dataset] != inner:
                continue
            data = runner.load_predictors(dataset, {outer, inner})
            predicted = predict_mode_evidence_experts(data, model)
            score_path = folder / (dataset + "_blind_scores.pkl")
            pd.testing.assert_series_equal(
                predicted, pd.read_pickle(score_path), check_exact=True
            )
            fingerprints[score_path.relative_to(OUT).as_posix()] = sha(score_path)
            labeled, manual, identities = runner.with_training_labels(
                data,
                permitted_groups=allowed,
                excluded_groups=excluded,
            )
            pool = labeled["pool"]
            pool["calibration_identity_in_manual"] = [
                (s, runner.species(n), runner.adduct(a)) in identities
                for s, n, a in pool[["stream", "name", "adduct"]].itertuples(
                    index=False, name=None
                )
            ]
            cases[dataset], scores[dataset] = pool, predicted
            manual_counts.update(manual.stream.value_counts().to_dict())
    assert set(cases) == {d for d in projects if folds[d] != outer}
    records = []
    for threshold in protocol["threshold_grid"]:
        counts = {s: Counter() for s in manual_counts}
        for dataset, pool in cases.items():
            for stream, part in pool.groupby("stream", sort=False):
                selected = select_operating_candidates(
                    part,
                    scores[dataset].loc[part.index],
                    threshold=threshold,
                    ambiguity_margin=protocol["ambiguity_margin"],
                )
                counts[stream]["correct"] += int(selected.training_positive.sum())
                counts[stream]["extras"] += int(
                    (~selected.calibration_identity_in_manual).sum()
                )
                counts[stream]["different_peak"] += int(
                    (
                        selected.calibration_identity_in_manual
                        & selected.training_label_known
                        & ~selected.training_positive
                    ).sum()
                )
        records.extend(
            dict(stream=s, threshold=threshold, manual=n, **dict(counts[s]))
            for s, n in manual_counts.items()
        )
    table = pd.DataFrame(records).fillna(0)
    actual = pd.read_csv(
        folder / "training_operating_points.csv", float_precision="round_trip"
    )
    pd.testing.assert_frame_equal(table, actual, check_exact=True, check_dtype=False)
    fitted = choose_balanced_training_points(
        table,
        extras_per_manual=protocol["extras_budget_per_endogenous_manual"],
        target=0.9,
    )
    recorded = json.loads((folder / "operating_points.json").read_text())
    assert fitted == recorded["thresholds"]
    assert set(recorded["training_groups"]) == allowed
    assert set(recorded["outer_excluded"]) == excluded
    for name in (
        "operating_points.json",
        "training_operating_points.csv",
        "selected_training_points.csv",
    ):
        path = folder / name
        fingerprints[path.relative_to(OUT).as_posix()] = sha(path)
    print(
        "ALL INNER SCORES, COUNTS AND OPERATING-POINT CHOICE REPRODUCED",
        outer,
        len(cases),
        flush=True,
    )
    return dict(
        outer=outer,
        blind_dataset_score_checks=len(cases),
        operating_point_rows=len(table),
        artifact_sha256=fingerprints,
    )


def main():
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    assert len(projects) == 33
    hashes = {}
    for dataset in projects:
        identities = []
        for name in ("baseline", "experiment_029", "experiment_033"):
            path = QE / name / "evaluation_v2" / dataset / "evaluation_identity.json"
            identities.append(json.loads(path.read_text())["manual_sha256"])
            hashes[str(path)] = sha(path)
        assert len(set(identities)) == 1, dataset
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).with_name(
        "run_qe_standard_aligned_experiment.py"
    ).read_bytes()
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(replay, range(5)))
    assert sum(r["blind_dataset_score_checks"] for r in results) == 132
    result = dict(
        all_passed=True,
        annotation_changes=False,
        new_fits=False,
        source_sha256=sha(Path(__file__)),
        identical_manual_reference_hash_checks=33,
        inner_heldout_dataset_score_reproductions=132,
        complete_training_count_grid_reproductions=5,
        training_only_operating_point_selection_reproductions=5,
        manual_reference_identity_inputs_sha256=hashes,
        fold_results=results,
        note="Supplemental engineering replay,not new annotation performance. Original full discovery/pipeline integrity receipt is separately required. No outer evaluated labels choose cutoffs; all inner data references and score models exclude outer and inner validation groups.",
    )
    (OUT / "calibration_integrity_receipt.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"manual_reference_identity_inputs_sha256", "fold_results"}
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
