"""Read-only: new physical opportunities in033/036preparations, NOT their scores.

All33datasets; no classifier fitting/annotation changes or outer winner splice.
Counts only mechanical upper bounds, never claimed as annotation improvements.
"""

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources
from run_qe_standard_mode_experiment import sha, dump, adduct

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OLD, BASE = QE / "experiment_029", QE / "baseline"
OUT = QE / "discovery_union_opportunities_029_v1"
PREPARATIONS = {
    "old029": "standard_context_preparation_v1",
    "standard033": "standard_aligned_candidate_preparation_v2",
    "empirical036": "empirical_RT_candidate_preparation_v3",
}


def keys(pool):
    from standard_lipidomics.validation import _normalise

    frame = pool.copy()
    frame["adduct"] = frame.adduct.map(adduct)
    normalized = _normalise(frame, manual=False)
    return set(
        normalized[
            ["stream", "species", "adduct", "internal_standard", "feature_id"]
        ].itertuples(index=False, name=None)
    )


def main():
    configure_frozen_resources(REPO)
    from matlab_annotation.evidence_training import validate_feature_scope

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    fold_path = QE / "rt_atlas_development/fold_assignment.json"
    folds = json.loads(fold_path.read_text())
    sources = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path(__file__).with_name("run_qe_standard_mode_experiment.py"),
            REPO / "app/standard_lipidomics/validation.py",
            REPO / "app/matlab_annotation/evidence_training.py",
        )
    }
    inputs, cache = {str(fold_path): sha(fold_path)}, {}
    counts, universes = Counter(), []
    for dataset in projects:
        outer = folds[dataset]
        model_path = OLD / f"fold_{outer}_nested_model.pkl"
        if outer not in cache:
            cache[outer] = pd.read_pickle(model_path)
        model = cache[outer]
        inputs[str(model_path)] = sha(model_path)
        candidates = {}
        for label, folder in PREPARATIONS.items():
            path = QE / folder / "prepared" / f"exclude_{outer}" / (dataset + ".pkl")
            inputs[str(path)] = sha(path)
            data = pd.read_pickle(path)
            assert data["query_dataset"] == dataset and set(data["excluded_folds"]) == {
                outer
            }
            validate_feature_scope(
                data,
                model["training_cohorts"],
                model["excluded_groups"],
                query_is_training=False,
            )
            candidates[label] = keys(data["pool"])
        candidates["union033036"] = (
            candidates["standard033"] | candidates["empirical036"]
        )
        for label, values in candidates.items():
            assert candidates["old029"].issubset(values), (dataset, label)
            universes.append(
                dict(
                    dataset=dataset,
                    preparation=label,
                    distinct_physical_label_pairs=len(values),
                    added_physical_label_pairs=len(values - candidates["old029"]),
                )
            )
        manual_path = OLD / "evaluation_v2" / dataset / "manual_comparison.pkl"
        called_path = OLD / "runs" / dataset / "automatic_annotations.csv"
        for p in (manual_path, called_path):
            inputs[str(p)] = sha(p)
        manual = pd.read_pickle(manual_path)
        called = keys(
            pd.read_csv(
                called_path,
                dtype={"feat_feature_id": str},
                float_precision="round_trip",
                low_memory=False,
            )
        )
        identities = {key[:4] for key in called}
        occupied = {(key[0], str(key[4])) for key in called}
        missing = manual[
            manual.mapped
            & ~manual.internal_standard
            & ~manual.same_manual_peak
            & manual.lipid_class.ne("X4")
        ]
        for row in missing.itertuples():
            key = (row.stream, row.species, row.adduct, False, str(row.feature_id))
            for label in ("standard033", "empirical036", "union033036"):
                if key not in candidates[label] - candidates["old029"]:
                    continue
                counts[
                    (
                        dataset,
                        row.stream,
                        row.lipid_class,
                        label,
                        "new_correct_feature_candidate_for_missing_manual_entry",
                    )
                ] += 1
                if key[:4] in identities:
                    reason = "blocked_identity_already_called_on_another_peak"
                elif (row.stream, str(row.feature_id)) in occupied:
                    reason = "blocked_original_feature_already_occupied"
                else:
                    reason = "mechanically_free_for_identity_preserving_additions"
                counts[(dataset, row.stream, row.lipid_class, label, reason)] += 1
        print("READONLY PHYSICAL DISCOVERY UNION CHECK", dataset, flush=True)
    assert len(projects) == 33
    assert all(sha(Path(p)) == h for p, h in {**sources, **inputs}.items())
    OUT.mkdir(exist_ok=False)
    table = pd.DataFrame(
        [
            dict(
                dataset=d,
                stream=s,
                lipid_class=c,
                preparation=p,
                reason=r,
                manual_entries=n,
            )
            for (d, s, c, p, r), n in sorted(counts.items())
        ]
    )
    table.to_csv(OUT / "by_dataset_mode_class_reason.csv", index=False)
    summary = table.groupby(
        ["preparation", "stream", "lipid_class", "reason"], as_index=False
    ).manual_entries.sum()
    summary.to_csv(OUT / "by_mode_class_reason.csv", index=False)
    totals = table.groupby(
        ["preparation", "reason"], as_index=False
    ).manual_entries.sum()
    totals.to_csv(OUT / "by_preparation_reason.csv", index=False)
    pd.DataFrame(universes).to_csv(OUT / "candidate_universe_counts.csv", index=False)
    dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            classifier_fits=0,
            annotation_changes=False,
            source_sha256=sources,
            input_sha256=inputs,
            note="Mechanical retrospective opportunity counts ONLY,not accepted annotations or gains. Wholelabel-free033/036physicalpreparations inspected,NOrejectedscore/model/outputclasswinners used. Old029identities/initialoccupiedfeatures protected inadd-onlyupperbound. Mappedendogenousdiagnosticscope doesNOTchangefullmanualbenchmarkN/unresolved/STDroles. No querycurationusedbyruntime.",
        ),
    )
    print(totals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
