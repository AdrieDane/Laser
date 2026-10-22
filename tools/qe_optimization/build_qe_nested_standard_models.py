"""Prepare25query-excluded standard CONTEXT reference bundles, not annotations."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from context import REPO, WORKSPACE
from frozen_resources import configure_frozen_resources
from build_qe_nested_references import same

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "nested_standard_development_v1"
REFERENCES = QE / "nested_reference_development_v2"
OLD = QE / "standard_rt_development_v3"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def initialize():
    projects = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
    assert json.loads((REFERENCES / "summary.json").read_text())["all_complete"]
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        receipt = configure_frozen_resources(snapshot)
        from matlab_annotation.declared_standards import declared_standard_targets

        staged = OUT / "automatic_inputs"
        staged.mkdir()
        paths = [
            Path(__file__).with_name("frozen_resources.py"),
            Path(__file__).with_name("build_qe_nested_references.py"),
            QE / "rt_atlas_development/fold_assignment.json",
            OLD / "observations.pkl",
            OLD / "manual_standard_entries.pkl",
            REFERENCES / "summary.json",
        ]
        paths.extend(REFERENCES.glob("exclude_*.json"))
        for fold in range(5):
            paths.append(
                QE / "experiment_005/models" / f"fold_{fold}_standard_models.json"
            )
        for dataset in projects:
            baseline = QE / "baseline/runs" / dataset / "full_annotation.pkl"
            legacy_catalog = OLD / "declared_targets" / (dataset + ".pkl")
            result = pd.read_pickle(baseline)
            catalogue = declared_standard_targets(
                snapshot / "reference_data/matlab_pipeline",
                result["internal_standards_expected"],
            )
            old_catalogue = pd.read_pickle(legacy_catalog)
            keys = ["stream", "name", "adduct", "ref_mz"]
            pd.testing.assert_frame_equal(
                catalogue.sort_values(keys).reset_index(drop=True),
                old_catalogue.reindex(columns=catalogue.columns)
                .sort_values(keys)
                .reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
            automatic = {}
            for stream, payload in result["streams"].items():
                # Fixed blind automatic alignment columns only, not full raw
                # sample matrices/curated comparisons or proposed endo calls.
                columns = [
                    "name",
                    "class",
                    "carbon",
                    "db",
                    "adduct",
                    "feat_feature_id",
                    "feat_mz",
                    "feat_rt",
                    "feat_intensity",
                    "ppm_error",
                    "post_regression_confidence",
                    "is_is",
                ]
                automatic[stream] = payload["annotations"][
                    [c for c in columns if c in payload["annotations"]]
                ].copy()
            dest = staged / (dataset + ".pkl")
            pd.to_pickle(dict(automatic=automatic, declared=catalogue), dest)
            paths.extend([baseline, legacy_catalog, dest])
        protocol = dict(
            purpose="NUMERICAL STANDARD CONTEXT REFERENCE PREPARATION ONLY. No endo annotations, score fitting, acceptance changes or accuracy claims.",
            datasets=33,
            project_groups=30,
            expected_masks=25,
            numerical_workers=3,
            resource_receipt=receipt,
            source_sha256={
                p.relative_to(snapshot).as_posix(): sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
            input_sha256={str(p): sha(p) for p in paths},
            scope="Same25single/pair/triple exclusion masks as022. Filter standard RT/report/adduct-channel observations AND transfer residual references before fitting. Endogenous transfer atlas has identical exclusion mask. Query/outer/inner groups absent from new learned references. Standard library/channel/transfer/atlas for5singlemasks must reproduce005. Existing blind baseline anchors retain prior exposure.",
            privacy="Raw automatic/catalog staging and all fitted models outsideGit. Original feature IDs never fitter inputs/parameters. Legacy before_same_peak flags stripped. Full explicit frozen scientific assets and paths.",
        )
        (OUT / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for path, digest in protocol["source_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest, path
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    from matlab_annotation.nested_feature_folds import plan_nested_feature_folds

    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    return protocol, list(
        plan_nested_feature_folds(folds, {d: d[:7] for d in folds})["models"]
    )


def fit(mask):
    path = OUT / ("exclude_" + "_".join(map(str, mask)) + ".json")
    if path.exists():
        return json.loads(path.read_text())
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.nested_standard_models import fit_nested_standard_bundle

    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    reference_path = REFERENCES / path.name
    reference = json.loads(reference_path.read_text())
    auto, declared = {}, {}
    for dataset in reference["training_datasets"]:
        data = pd.read_pickle(OUT / "automatic_inputs" / (dataset + ".pkl"))
        auto[dataset], declared[dataset] = data["automatic"], data["declared"]
    bundle = fit_nested_standard_bundle(
        pd.read_pickle(OLD / "observations.pkl"),
        pd.read_pickle(OLD / "manual_standard_entries.pkl"),
        auto,
        declared,
        reference,
        folds,
        {d: d[:7] for d in folds},
        mask,
    )
    bundle["reference_bundle_sha256"] = sha(reference_path)
    if len(mask) == 1:
        old = json.loads(
            (
                QE / "experiment_005/models" / f"fold_{mask[0]}_standard_models.json"
            ).read_text()
        )
        for key in ("library", "atlas", "transfer", "channels"):
            same(bundle[key], old[key])
        assert bundle["training_groups"] == old["training_groups"]
        bundle["original_single_fold_005_parity"] = True
    path.write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    return bundle


def main():
    protocol, masks = initialize()
    rows = []
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed([executor.submit(fit, m) for m in masks]):
            m = future.result()
            rows.append(
                dict(
                    excluded_folds="_".join(map(str, m["excluded_folds"])),
                    training_groups=len(m["training_groups"]),
                    standard_observation_rows=m["standard_observation_rows"],
                    channel_observation_rows=m["channel_observation_rows"],
                    transfer_reference_rows=m["transfer_reference_rows"],
                    library_models=len(m["library"]["models"]),
                    transfer_models=len(m["transfer"]["models"]),
                    channel_models=len(m["channels"]["models"]),
                    original_single_fold_005_parity=m.get(
                        "original_single_fold_005_parity", False
                    ),
                )
            )
            print("STANDARD CONTEXT BUNDLE", rows[-1], flush=True)
    assert len(rows) == 25
    pd.DataFrame(rows).sort_values("excluded_folds").to_csv(
        OUT / "by_exclusion_mask.csv", index=False
    )
    summary = dict(
        reference_bundles=25,
        all_complete=True,
        single_fold_005_parity_checks=5,
        chemical_catalogue_parity_checks=33,
        annotation_changes=False,
        note=protocol["purpose"],
        model_sha256={p.name: sha(p) for p in OUT.glob("exclude_*.json")},
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
