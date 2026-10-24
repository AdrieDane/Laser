"""Read-only029 common-class discovery limits, not new annotation accuracy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_atlas_experiment import species, adduct

QE = WORKSPACE / "outputs/qe_optimization_20261022"
EXP = QE / "experiment_029"
OUT = QE / "common_discovery_diagnosis_029_v1"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    configure_frozen_resources(EXP / "source_snapshot")
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.retention_atlas import predict_aligned_retention_atlas

    assert json.loads((EXP / "integrity_receipt.json").read_text())["all_passed"]
    projects = json.loads((EXP / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    OUT.mkdir(exist_ok=False)
    sources, manuals = {}, []
    for dataset in projects:
        path = EXP / "evaluation_v2" / dataset / "manual_comparison.pkl"
        sources[str(path)] = sha(path)
        frame = pd.read_pickle(path).assign(dataset=dataset, project_group=dataset[:7])
        manuals.append(frame)
    all_manual = pd.concat(manuals, ignore_index=True)
    assert len(all_manual) == 92102
    support = all_manual.groupby(["stream", "lipid_class"]).agg(
        manual=("dataset", "size"), groups=("project_group", "nunique")
    )
    common = set(support[support.manual.ge(100) & support.groups.ge(3)].index)
    rows = []
    for dataset in projects:
        base_path = QE / "baseline/runs" / dataset / "full_annotation.pkl"
        pool_path = EXP / "runs" / dataset / "candidates.pkl"
        reference_path = (
            QE / "nested_reference_development_v2" / f"exclude_{folds[dataset]}.json"
        )
        for path in (base_path, pool_path, reference_path):
            sources[str(path)] = sha(path)
        reference_bundle = json.loads(reference_path.read_text())
        assert dataset[:7] in reference_bundle["excluded_groups"]
        assert dataset[:7] not in reference_bundle["training_groups"]
        assert set(reference_bundle["excluded_folds"]) == {folds[dataset]}
        baseline = pd.read_pickle(base_path)
        pool = pd.read_pickle(pool_path)
        available = {
            (s, species(n), adduct(a), str(fid), bool(std))
            for s, n, a, fid, std in pool[
                ["stream", "name", "adduct", "feat_feature_id", "is_is"]
            ].itertuples(index=False, name=None)
        }
        missing = all_manual[
            all_manual.dataset.eq(dataset) & ~all_manual.same_manual_peak
        ].copy()
        missing = missing[
            [
                ((s, c) in common)
                for s, c in missing[["stream", "lipid_class"]].itertuples(
                    index=False, name=None
                )
            ]
        ]
        for stream, group in missing.groupby("stream", sort=False):
            payload = baseline["streams"][stream]
            targets, _, _ = compile_additional_targets(
                payload["reference"],
                reference_bundle["composition_envelope"],
                stream=stream,
            )
            targets = pd.concat([payload["reference"], targets], ignore_index=True)
            targets["species_key"] = targets.name.map(species)
            targets["ion_key"] = targets.adduct.map(adduct)
            needed = set(
                group[["species", "adduct"]].itertuples(index=False, name=None)
            )
            targets = targets[
                [
                    k in needed
                    for k in targets[["species_key", "ion_key"]].itertuples(
                        index=False, name=None
                    )
                ]
            ].copy()
            predictions = predict_aligned_retention_atlas(
                targets,
                payload["annotations"],
                reference_bundle["atlas"],
                stream=stream,
            )
            targets = targets.join(predictions)
            targets["discovery_rt"] = targets.atlas_rt_seconds.div(60).fillna(
                targets.ref_rt
            )
            target_groups = {
                k: g for k, g in targets.groupby(["species_key", "ion_key"], sort=False)
            }
            raw = payload["features"].set_index(
                payload["features"].feature_id.astype(str)
            )
            assert raw.index.is_unique
            for entry in group.itertuples():
                row = dict(
                    dataset=dataset,
                    stream=stream,
                    lipid_class=entry.lipid_class,
                    internal_standard=entry.internal_standard,
                )
                key = (
                    stream,
                    species(entry.species),
                    adduct(entry.adduct),
                    str(entry.feature_id),
                    bool(entry.internal_standard),
                )
                if not entry.mapped:
                    reason = "unresolved_manual_feature"
                elif key in available:
                    reason = "correct_feature_candidate_already_available"
                elif entry.internal_standard:
                    reason = "standard_specialist_gap"
                elif (entry.species, entry.adduct) not in target_groups:
                    reason = "identity_absent_from_current_theoretical_grid"
                else:
                    ref = target_groups[(entry.species, entry.adduct)]
                    feature = raw.loc[str(entry.feature_id)]
                    mass_errors = (float(feature.mz) / ref.ref_mz - 1) * 1e6
                    rt_errors = (float(feature.rt) - ref.discovery_rt) * 60
                    valid_mass = mass_errors.abs().lt(3)
                    row["minimum_abs_mass_error_ppm"] = float(mass_errors.abs().min())
                    row["minimum_abs_RT_error_seconds"] = (
                        float(rt_errors[valid_mass].abs().min())
                        if valid_mass.any()
                        else np.nan
                    )
                    if not valid_mass.any():
                        reason = "mass_outside_3ppm"
                    elif not rt_errors[valid_mass].notna().any():
                        reason = "no_RT_prediction"
                    elif rt_errors[valid_mass].abs().gt(10).all():
                        reason = "RT_outside_10seconds"
                    else:
                        reason = "range_adduct_protection_or_other_preparation_gate"
                rows.append(dict(row, reason=reason))
        print("COMMON DISCOVERY LIMITS CHECKED", dataset, flush=True)
    frame = pd.DataFrame(rows)
    # Private diagnostic details contain no names/FIDs but remain outsideGit.
    frame.to_pickle(OUT / "diagnostic_rows.pkl")
    for name, keys in (
        ("by_mode_reason", ["stream", "reason"]),
        ("by_mode_class_reason", ["stream", "lipid_class", "reason"]),
        (
            "by_dataset_mode_class_reason",
            ["dataset", "stream", "lipid_class", "reason"],
        ),
    ):
        frame.groupby(keys).size().rename(
            "missing_manual_entries"
        ).reset_index().to_csv(OUT / (name + ".csv"), index=False)
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    summary = dict(
        annotation_changes=False,
        models_fitted=0,
        datasets=33,
        priority_class_modes=len(common),
        priority_manual_entries=int(support.loc[list(common), "manual"].sum()),
        priority_missing_entries=len(frame),
        reason_counts=frame.reason.value_counts().to_dict(),
        source_sha256=sha(Path(__file__)),
        input_sha256=sources,
        note="READ-ONLY common-class diagnosis of retained029missing calls,not proposed/accepted new annotations. "
        ">=100manual and>=3independent projects perclass/mode. Full92102historical/92100target denominators unchanged. "
        "Query-excluded outer reference;manual usedONLYto diagnose evaluation failures,never refit a query. "
        "RT/mass gate categories are diagnostics,not guarantees that wider windows would be safe or sufficient.",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
