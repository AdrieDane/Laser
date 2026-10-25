"""Blind full physical joins FIRST, then read-only mechanical upper bounds.

No annotation acceptance, classifier fits or outer-class score splicing. Uses
the WHOLE competing-class numerical RT rule, never a successful-class subset.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_standard_mode_experiment import sha, dump
from diagnose_qe_discovery_unions import keys as physical_keys
import diagnose_qe_competing_class_rt as pilot

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "competing_class_RT_opportunities_v1"
OLD = QE / "experiment_029"


def prepare(dataset):
    checkpoint = OUT / "checkpoints" / (dataset + ".json")
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        assert sha(Path(saved["path"])) == saved["sha256"]
        return saved
    configure_frozen_resources(pilot.OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.coherent_retention import _protected_mass

    folds = json.loads(pilot.FOLDS.read_text())
    bundle = json.loads((pilot.REFS / f"exclude_{folds[dataset]}.json").read_text())
    assert (
        dataset[:7] in bundle["excluded_groups"]
        and dataset[:7] not in bundle["training_groups"]
    )
    raw = pd.read_pickle(pilot.BASE / "runs" / dataset / "full_annotation.pkl")
    staged = pd.read_pickle(
        QE / "nested_standard_development_v1/automatic_inputs" / (dataset + ".pkl")
    )
    measured = pd.read_pickle(
        pilot.STD / "measurements" / f"exclude_{folds[dataset]}" / (dataset + ".pkl")
    )
    predictions = pd.read_pickle(pilot.OUT / "predictions" / (dataset + ".pkl"))
    columns = [
        "stream",
        "name",
        "class",
        "adduct",
        "carbon",
        "db",
        "ref_mz",
        "feat_feature_id",
        "feat_mz",
        "feat_rt",
        "feat_intensity",
        "coherent_rt_seconds",
        "coherent_shift_seconds",
    ]
    rows = []
    for stream, payload in raw["streams"].items():
        declared = staged["declared"][staged["declared"].stream.eq(stream)].copy()
        generated, _, _ = compile_additional_targets(
            payload["reference"],
            bundle["composition_envelope"],
            stream=stream,
            declared_standards=declared,
        )
        targets = pd.concat(
            [payload["reference"], generated], ignore_index=True
        ).drop_duplicates(["name", "adduct"])
        targets = targets[~targets.name.astype(str).str.endswith(" (IS)")].copy()
        targets = targets[
            targets.get("isotopologue", pd.Series("", index=targets.index))
            .fillna("")
            .astype(str)
            .eq("")
        ].copy()
        targets["class"] = targets["class"].map(canonical_lipid_class)
        prediction = predictions[
            predictions.stream.eq(stream) & predictions.coherent_available.astype(bool)
        ]
        targets = targets.merge(
            prediction[
                [
                    "class",
                    "carbon",
                    "db",
                    "adduct",
                    "coherent_rt_seconds",
                    "coherent_shift_seconds",
                ]
            ],
            on=["class", "carbon", "db", "adduct"],
            validate="many_to_one",
        )
        protected = set()
        auto = payload["annotations"]
        standard = (
            auto.get("is_is", pd.Series(False, index=auto.index))
            .fillna(False)
            .astype(bool)
        )
        protected.update(auto.loc[standard, "feat_feature_id"].dropna().astype(str))
        if len(measured):
            protected.update(
                measured.loc[measured.stream.eq(stream), "feat_feature_id"]
                .dropna()
                .astype(str)
            )
        features = payload["features"][["feature_id", "mz", "rt", "intensity"]].copy()
        valid = np.isfinite(features[["mz", "rt", "intensity"]]).all(axis=1)
        valid &= features.mz.gt(0) & features.rt.gt(0) & features.intensity.gt(0)
        valid &= ~features.feature_id.astype(str).isin(protected)
        valid &= ~_protected_mass(
            features.mz.to_numpy(float), declared.ref_mz.to_numpy(float)
        )
        features = features[valid].sort_values("mz", kind="stable")
        masses = features.mz.to_numpy(float)
        for target in targets.to_dict("records"):
            mass = float(target["ref_mz"])
            lo, hi = np.searchsorted(masses, [mass * (1 - 3e-6), mass * (1 + 3e-6)])
            for hit in features.iloc[lo:hi].itertuples():
                if (
                    abs((hit.mz / mass - 1) * 1e6) >= 3
                    or abs(hit.rt * 60 - target["coherent_rt_seconds"]) > 10
                ):
                    continue
                rows.append(
                    dict(
                        target,
                        stream=stream,
                        feat_feature_id=str(hit.feature_id),
                        feat_mz=float(hit.mz),
                        feat_rt=float(hit.rt),
                        feat_intensity=float(hit.intensity),
                    )
                )
    pool = (
        pd.DataFrame(rows, columns=columns)
        .drop_duplicates(["stream", "name", "adduct", "feat_feature_id"])
        .reset_index(drop=True)
    )
    path = OUT / "blind_physical_proposals" / (dataset + ".pkl")
    pool.to_pickle(path)
    saved = dict(
        dataset=dataset,
        streams=4,
        path=str(path),
        sha256=sha(path),
        rows=len(pool),
        manual_evaluation_loaded=False,
        accepted_annotations=0,
    )
    dump(checkpoint, saved)
    print("BLIND PHYSICAL RT PROPOSALS ONLY", dataset, len(pool), flush=True)
    return saved


def main():
    protocol, projects, folds = pilot.initialize()
    integrity = json.loads((pilot.OUT / "integrity_receipt.json").read_text())
    assert (
        integrity["all_passed"]
        and integrity["full_blind_stream_prediction_reproductions"] == 132
    )
    assert integrity["protocol_sha256"] == sha(pilot.OUT / "protocol.json")
    ready = json.loads((pilot.OUT / "blind_predictions_complete.json").read_text())
    assert ready["all_33_complete"] and ready["dataset_modes"] == 132
    for p, h in ready["prediction_sha256"].items():
        assert sha(Path(p)) == h
    sources = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path(__file__).with_name("diagnose_qe_discovery_unions.py"),
        )
    }
    inputs = {
        **protocol["input_sha256"],
        **ready["prediction_sha256"],
        str(pilot.OUT / "integrity_receipt.json"): sha(
            pilot.OUT / "integrity_receipt.json"
        ),
    }
    OUT.mkdir(exist_ok=True)
    source_receipt = OUT / "source_binding.json"
    binding = dict(source_sha256=sources, input_sha256=inputs)
    if source_receipt.exists():
        assert json.loads(source_receipt.read_text()) == binding
    else:
        dump(source_receipt, binding)
    for name in ("checkpoints", "blind_physical_proposals"):
        (OUT / name).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as executor:
        completed = [
            f.result()
            for f in as_completed([executor.submit(prepare, d) for d in projects])
        ]
    assert len(completed) == 33 and {r["dataset"] for r in completed} == set(projects)
    artifacts = {r["path"]: r["sha256"] for r in completed}
    dump(
        OUT / "blind_all33_physical_proposals_complete.json",
        dict(
            datasets=33,
            dataset_modes=132,
            manual_evaluation_loaded=False,
            annotation_changes=False,
            physical_pool_sha256=artifacts,
        ),
    )
    # First manual-comparison access occurs only after ALL physical pools freeze.
    configure_frozen_resources(pilot.OUT / "source_snapshot")
    counts, sizes = Counter(), []
    for dataset in projects:
        path = OUT / "blind_physical_proposals" / (dataset + ".pkl")
        new = physical_keys(pd.read_pickle(path))
        old_path = (
            pilot.STD / "prepared" / f"exclude_{folds[dataset]}" / (dataset + ".pkl")
        )
        old = pd.read_pickle(old_path)
        bundle = json.loads((pilot.REFS / f"exclude_{folds[dataset]}.json").read_text())
        for name in ("reference", "standard_reference"):
            assert set(old[name + "_training_groups"]) == set(bundle["training_groups"])
            assert set(old[name + "_excluded_groups"]) == set(bundle["excluded_groups"])
        old_keys = physical_keys(old["pool"])
        fresh = new - old_keys
        sizes.append(
            dict(
                dataset=dataset,
                blind_coherent_pairs=len(new),
                old_core_pairs=len(old_keys),
                new_pairs=len(fresh),
            )
        )
        manual_path = OLD / "evaluation_v2" / dataset / "manual_comparison.pkl"
        called_path = OLD / "runs" / dataset / "automatic_annotations.csv"
        for p in (old_path, manual_path, called_path):
            inputs[str(p)] = sha(p)
        manual = pd.read_pickle(manual_path)
        called = physical_keys(
            pd.read_csv(
                called_path,
                dtype={"feat_feature_id": str},
                float_precision="round_trip",
                low_memory=False,
            )
        )
        identities = {k[:4] for k in called}
        occupied = {(k[0], k[4]) for k in called}
        missing = manual[
            manual.mapped
            & ~manual.internal_standard
            & ~manual.same_manual_peak
            & manual.lipid_class.ne("X4")
        ]
        for row in missing.itertuples():
            key = (row.stream, row.species, row.adduct, False, str(row.feature_id))
            if key not in new:
                continue
            counts[
                (
                    dataset,
                    row.stream,
                    row.lipid_class,
                    "missing_manual_correct_feature_in_blind_coherent_pool",
                )
            ] += 1
            if key not in fresh:
                counts[
                    (
                        dataset,
                        row.stream,
                        row.lipid_class,
                        "already_available_in_old_core_not_new_discovery",
                    )
                ] += 1
                continue
            counts[
                (
                    dataset,
                    row.stream,
                    row.lipid_class,
                    "new_correct_feature_candidate_absent_old_core",
                )
            ] += 1
            if key[:4] in identities:
                reason = "blocked_identity_already_called_on_another_peak"
            elif (key[0], key[4]) in occupied:
                reason = "blocked_original_feature_already_occupied"
            else:
                reason = "mechanically_free_if_all_old_calls_preserved"
            counts[(dataset, row.stream, row.lipid_class, reason)] += 1
    pilot.check_binding(protocol)
    for p, h in {**sources, **inputs, **artifacts}.items():
        assert sha(Path(p)) == h
    table = pd.DataFrame(
        [
            dict(dataset=d, stream=s, lipid_class=c, reason=r, manual_entries=n)
            for (d, s, c, r), n in sorted(counts.items())
        ]
    )
    table.to_csv(OUT / "by_dataset_mode_class_reason.csv", index=False)
    table.groupby(
        ["stream", "lipid_class", "reason"], as_index=False
    ).manual_entries.sum().to_csv(OUT / "by_mode_class_reason.csv", index=False)
    table.groupby("reason", as_index=False).manual_entries.sum().to_csv(
        OUT / "totals.csv", index=False
    )
    pd.DataFrame(sizes).to_csv(OUT / "physical_universe_sizes.csv", index=False)
    dump(
        OUT / "receipt.json",
        dict(
            datasets=33,
            dataset_modes=132,
            annotation_changes=False,
            classifier_fits=0,
            manual_opened_only_after_all_blind_pools=True,
            source_sha256=sources,
            input_sha256=inputs,
            physical_pool_sha256=artifacts,
            note="Mechanical upper bounds ONLY,NOT annotation gains/accepted annotations/FDR. Entire class-agnostic RT rule applied,no successful-class subset. No candidate acceptance or baseline change.",
        ),
    )
    print(table.groupby("reason").manual_entries.sum().to_string(), flush=True)


if __name__ == "__main__":
    main()
