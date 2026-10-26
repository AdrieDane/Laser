"""All33 RP partner-gap numeric pilot, no annotation/classifier/cutoff changes.

Training RTs may use ONLY authorized other-project curation. Query gold is
never decoded in this producer. An independent checker must precede diagnosis.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import paired_gap_pilot_data as data
import paired_retention_gap as gap

OUT = data.OUT
PLAN = REPO / "docs/validation/next_paired_isomer_gap_pilot.txt"
SOURCES = (
    "paired_retention_gap.py",
    "paired_gap_pilot_data.py",
    "prepare_qe_paired_gap_pilot.py",
    "paired_gap_independent.py",
    "verify_qe_paired_gap_pilot.py",
    "diagnose_qe_paired_gap_pilot.py",
    "test_paired_retention_gap.py",
    "test_paired_gap_independent.py",
    "frozen_resources.py",
    "context.py",
)


def source_inputs():
    datasets, folds = data.inventory()
    paths = {Path(__file__).with_name(n) for n in SOURCES}
    paths.update(
        (
            PLAN,
            data.BASE / "run_manifest.json",
            data.QE / "rt_atlas_development/fold_assignment.json",
            data.REFERENCE / "integrity_receipt.json",
            data.REFERENCE / "protocol.json",
            data.QE / "experiment_048/decision.json",
            data.QE / "experiment_048/integrity_receipt.json",
        )
    )
    for d in datasets:
        paths.update(
            (
                data.BASE / "runs" / d / "full_annotation.pkl",
                data.BASE / "evaluation_v2" / d / "manual_comparison.pkl",
                data.REFERENCE / "runs" / d / "automatic_annotations.csv",
                data.REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl",
                data.QE
                / "standard_context_preparation_v1/prepared"
                / f"exclude_{folds[d]}"
                / (d + ".pkl"),
            )
        )
    for root in ("app", "methods", "reference_data"):
        paths.update(
            p
            for p in (REPO / root).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
        )
    return {str(p): data.sha(p) for p in sorted(paths)}


def train(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    rows, support, training, excluded, reads = data.training_rows(outer)
    model, pairs = gap.fit_gaps(
        rows, support, training_groups=training, excluded_groups=excluded
    )
    folder = OUT / "models" / f"fold_{outer}"
    folder.mkdir(parents=True, exist_ok=False)
    rows.to_pickle(folder / "authorized_training_rows.pkl")
    support.to_pickle(folder / "authorized_training_support.pkl")
    pairs.to_pickle(folder / "training_pairs.pkl")
    pd.DataFrame(reads).to_csv(folder / "training_role_audit.csv", index=False)
    data.dump(folder / "model.json", model)
    print(
        "OTHER-PROJECT PAIRED-GAP REFERENCE",
        outer,
        len(reads),
        len(pairs),
        len(model["models"]),
        flush=True,
    )
    return dict(
        outer=outer,
        training_dataset_uses=len(reads),
        paired_rows=len(pairs),
        supported_exact_gaps=len(model["models"]),
    )


def identity_keys(table, *, feature=False):
    from standard_lipidomics.validation import _species, canonical_feature_id
    from matlab_annotation.standard_rt_library import ion_key

    columns = ["stream", "name", "adduct", "is_is"] + (
        ["feat_feature_id"] if feature else []
    )
    output = set()
    for row in table[columns].itertuples(index=False, name=None):
        key = (str(row[0]), _species(row[1]), ion_key(row[2]), bool(row[3]))
        output.add((*key, canonical_feature_id(row[4])) if feature else key)
    return output


def old_pool(dataset, outer):
    from matlab_annotation.evidence_training import validate_feature_scope

    datasets, folds = data.inventory()
    path = (
        data.QE
        / "standard_context_preparation_v1/prepared"
        / f"exclude_{outer}"
        / (dataset + ".pkl")
    )
    raw = pd.read_pickle(path)
    training = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    assert raw["query_dataset"] == dataset and set(raw["excluded_folds"]) == {outer}
    assert not gap.FORBIDDEN & set(raw["pool"])
    validate_feature_scope(raw, training, excluded, query_is_training=False)
    return raw["pool"], path


def query(item):
    configure_frozen_resources(OUT / "source_snapshot")
    from standard_lipidomics.validation import _species, canonical_feature_id
    from matlab_annotation.standard_rt_library import ion_key

    dataset, outer = item
    model_path = OUT / "models" / f"fold_{outer}" / "model.json"
    model = json.loads(model_path.read_text())
    anchors, targets, masses, baseline, accepted = data.query_inputs(dataset, outer)
    # No query manual comparison, saved curator RT fit or prior pilot proposals.
    predictions = gap.predict_partners(
        anchors, targets, model, query_group=dataset[:7], declared_masses=masses
    )
    candidates = gap.raw_candidates(
        predictions, {s: p["features"] for s, p in baseline["streams"].items()}
    )
    pool, pool_path = old_pool(dataset, outer)
    original = identity_keys(pool, feature=True)
    occupied = {
        (str(s), canonical_feature_id(f))
        for s, f in accepted[["stream", "feat_feature_id"]].itertuples(
            index=False, name=None
        )
    }
    identities = identity_keys(accepted)
    if len(candidates):
        keys = [
            (
                r.stream,
                _species(r["name"]),
                ion_key(r.adduct),
                False,
                canonical_feature_id(r.feat_feature_id),
            )
            for _, r in candidates.iterrows()
        ]
        candidates["novel_to_old029_core_pool"] = [key not in original for key in keys]
        candidates["identity_is_free"] = [key[:4] not in identities for key in keys]
        candidates["feature_is_free"] = [
            (key[0], key[4]) not in occupied for key in keys
        ]
    else:
        for field in (
            "novel_to_old029_core_pool",
            "identity_is_free",
            "feature_is_free",
        ):
            candidates[field] = pd.Series(dtype=bool)
    folder = OUT / "private" / dataset
    folder.mkdir(parents=True, exist_ok=False)
    predictions.to_pickle(folder / "predictions.pkl")
    candidates.to_pickle(folder / "candidate_graph.pkl")
    data.dump(
        folder / "receipt.json",
        dict(
            dataset=dataset,
            outer=outer,
            query_group=dataset[:7],
            query_gold_used=False,
            model_sha256=data.sha(model_path),
            old_pool_sha256=data.sha(pool_path),
            predictions=len(predictions),
            graph_rows=len(candidates),
            artifact_sha256={p.name: data.sha(p) for p in folder.iterdir()},
        ),
    )
    print(
        "BLIND PARTNER-GAP NUMERIC OUTPUT",
        dataset,
        len(predictions),
        len(candidates),
        flush=True,
    )
    return dict(
        dataset=dataset,
        outer=outer,
        predictions=len(predictions),
        raw_graph_rows=len(candidates),
    )


def main():
    datasets, folds = data.inventory()
    assert json.loads((data.REFERENCE / "integrity_receipt.json").read_text())[
        "all_passed"
    ]
    assert json.loads((data.QE / "experiment_048/integrity_receipt.json").read_text())[
        "all_passed"
    ]
    assert (
        json.loads((data.QE / "experiment_048/decision.json").read_text())[
            "retain_whole_method"
        ]
        is False
    )
    hashes = (
        source_inputs()
    )  # Also requires the independent checker/diagnostic to exist.
    OUT.mkdir(exist_ok=False)
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir()
    for folder in ("app", "methods", "reference_data"):
        shutil.copytree(
            REPO / folder,
            snapshot / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    resource = configure_frozen_resources(
        snapshot
    )  # FINAL root before any app imports.
    data.dump(
        OUT / "protocol.json",
        dict(
            purpose="READ-ONLY OTHER-PROJECT PAIRED RP GAP PILOT; not049annotationexperiment",
            source_input_sha256=hashes,
            numeric_runtime=data.environment(),
            resource_receipt=resource,
            families=gap.FAMILIES,
            minimum_manual_per_class=100,
            minimum_projects=3,
            gap_MAD_seconds=3,
            gap_p90_seconds=6,
            sign_fraction=0.8,
            gap_range_seconds=[3, 60],
            partner_score=0.95,
            partner_raw_ppm=1.5,
            final_raw_ppm=3,
            final_RT_seconds=10,
            numerical_workers=2,
            annotation_changes=False,
            annotation_classifier_fits=0,
            heldout_query_gold_used_in_own_predictions=False,
            reference_policy="Curated OTHERPROJECT identities select ORIGINAL feature RTs;NOTsavedRTfits. Per-dataset unique paired endpoints/primaryion;one medianvoteperproject. Query041partners frozen,notnewproposals. ChemicaltargetRTcolumns dropped. ScopecheckBEFOREtrainingmanualfileopen. All33blindgraphs beforequerygoldanalysis.",
        ),
    )
    with ProcessPoolExecutor(max_workers=2) as executor:
        fitted = list(executor.map(train, range(5)))
    assert len(fitted) == 5 and sum(r["training_dataset_uses"] for r in fitted) == 132
    pd.DataFrame(fitted).to_csv(OUT / "by_outer_reference.csv", index=False)
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(query, [(d, folds[d]) for d in datasets]))
    assert len(results) == 33
    pd.DataFrame(results).to_csv(OUT / "by_dataset_blind_availability.csv", index=False)
    assert hashes == source_inputs()
    assert (
        json.loads((OUT / "protocol.json").read_text())["numeric_runtime"]
        == data.environment()
    )
    artifacts = {
        str(p): data.sha(p)
        for p in OUT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    data.dump(
        OUT / "blind_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            outer_reference_models=5,
            authorized_other_project_training_table_uses=132,
            heldout_query_gold_used_in_own_predictions=False,
            query_gold_validation_tables_opened=0,
            annotation_outputs=0,
            annotation_changes=False,
            annotation_classifier_fits=0,
            acceptance_cutoffs_selected=False,
            source_input_sha256=hashes,
            artifact_sha256=artifacts,
            note="Numericcandidateopportunities ONLY,not acceptedgains. Independently verify ALL33 before decodingquerygold for availability/peakcontrast diagnostics. Outer-onlypilot cannotenterfutureINNERscores; newactualmethodrequiresmatchingnestedreferenceexclusions.",
        ),
    )
    print(
        "ALL33 PAIRED-GAP BLIND NUMERIC OUTPUTS FROZEN; INDEPENDENT CHECK REQUIRED",
        flush=True,
    )


if __name__ == "__main__":
    main()
