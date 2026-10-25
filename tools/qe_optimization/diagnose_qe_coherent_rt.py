"""Numerical-only all33 RT pilot; freeze ALL blind predictions before truth.

Run --predict, then --evaluate. No candidates, classifiers, acceptance or exports
change. Five existing grouped RT atlases are deliberately reused, not refitted.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources
from run_qe_standard_mode_experiment import sha, dump

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "coherent_RT_diagnostic_v2"
BASE = QE / "baseline"
REFS = QE / "nested_reference_development_v2"
STD = QE / "standard_context_preparation_v1"
FOLDS = QE / "rt_atlas_development/fold_assignment.json"
OBS = QE / "rt_atlas_development/observations.pkl"


def check_binding(protocol):
    assert sha(Path(__file__)) == protocol["runner_sha256"]
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest, path
    for path, digest in protocol["source_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest, path


def initialize():
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads(FOLDS.read_text())
    assert len(projects) == 33 and len({d[:7] for d in projects}) == 30
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "methods", "reference_data"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        inputs = [
            BASE / "run_manifest.json",
            FOLDS,
            Path(__file__).with_name("frozen_resources.py"),
            Path(__file__).with_name("run_qe_standard_mode_experiment.py"),
            REPO / "docs/validation/next_coherent_RT_discovery.txt",
        ]
        inputs.extend(REFS / f"exclude_{f}.json" for f in range(5))
        for d in projects:
            inputs.extend(
                [
                    BASE / "runs" / d / "full_annotation.pkl",
                    QE
                    / "nested_standard_development_v1/automatic_inputs"
                    / (d + ".pkl"),
                    STD / "measurements" / f"exclude_{folds[d]}" / (d + ".pkl"),
                    STD / "prepared" / f"exclude_{folds[d]}" / (d + ".pkl"),
                ]
            )
        dump(
            OUT / "protocol.json",
            dict(
                purpose="Numeric RT diagnostic ONLY. No annotation changes, candidates, classifier fits, cutoffs or exports. RT proximity is NOT identification accuracy/FDR. Saved blind raw-input payloads with prior development exposure,not untouched validation.",
                policy="Frozen 5 OTHER-group references; exact expected exclusion scopes/fingerprints; independent mask-matched spike measurements. Top intensity decile raw features,even-C anchors ONLY,native ions,<1.5ppm,maximum90s,bw2s,>=6features/compositions,>=3Ccounts/2DBcounts/3neighborlinks,peakratio>=1.25. Coherent offsets leave out each target's ENTIRE formula-equivalent chemical-coordinate block (C//2+3totalDB)%5 across every isomer branch. RP P-class totalDB=writtenDB+1,so O(C:D) and P(C:D-1) are held out TOGETHER. NPexactclasses retain their own composition convention. RPonly explicit isomer families. Preserve base branch RT differences. Training>=3projects/MAD<=5s,trained domain+existingrange envelope. Spike mass5.01ppm/FIDs protected. Unsupported remainsbaseline;shift only near supporting CDB domain. NOquerycuration used to construct/choose offsets. These fixed settings are developmental proposals,not analytical validation.",
                supersedes="v1 completed33blind numeric predictions but NEVER opened manual evaluation. Source review caught O/P writtenDB-versus-formula holdout issue BEFORE evaluation. Allv1artifacts/source preserved,not claimed as accuracy or used to tune settings. v2 corrects formula holdout grouping only;all other numeric thresholds remain fixed.",
                stages="All33/132blind prediction artifacts AND fingerprint receipt saved before any manual observation workbook/table is opened. Evaluate only frozen class/CDB/ion lookup. No tuning inside this pilot.",
                annotation_changes=False,
                classifier_fits=0,
                numerical_reference_fits=0,
                reused_numerical_reference_bundles=5,
                datasets=33,
                modes=132,
                groups=30,
                runner_sha256=sha(Path(__file__)),
                source_sha256={
                    p.relative_to(snapshot).as_posix(): sha(p)
                    for p in snapshot.rglob("*")
                    if p.is_file()
                },
                input_sha256={str(p): sha(p) for p in inputs},
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    check_binding(protocol)
    return protocol, projects, folds


def predict(dataset):
    checkpoint = OUT / "checkpoints" / (dataset + ".json")
    if checkpoint.exists():
        receipt = json.loads(checkpoint.read_text())
        for p, h in receipt["prediction_sha256"].items():
            assert sha(Path(p)) == h
        return receipt
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.coherent_retention import coherent_retention_predictions
    from annotator.composition_ranges import canonical_lipid_class

    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads(FOLDS.read_text())
    outer = folds[dataset]
    bundle = json.loads((REFS / f"exclude_{outer}.json").read_text())
    allowed = {d for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in projects if folds[d] == outer}
    assert set(bundle["training_datasets"]) == allowed
    assert set(bundle["training_groups"]) == {d[:7] for d in allowed}
    old = pd.read_pickle(STD / "prepared" / f"exclude_{outer}" / (dataset + ".pkl"))
    for name in ("reference", "standard_reference"):
        assert set(old[name + "_training_groups"]) == set(bundle["training_groups"])
        assert set(old[name + "_excluded_groups"]) == excluded
    raw = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    staged = pd.read_pickle(
        QE / "nested_standard_development_v1/automatic_inputs" / (dataset + ".pkl")
    )
    declared = staged["declared"]
    measurements = pd.read_pickle(
        STD / "measurements" / f"exclude_{outer}" / (dataset + ".pkl")
    )
    frames, audits = [], []
    keys = ["class", "carbon", "db", "adduct"]
    for stream, payload in raw["streams"].items():
        spikes = declared[declared.stream.eq(stream)].copy()
        generated, ranges, _ = compile_additional_targets(
            payload["reference"],
            bundle["composition_envelope"],
            stream=stream,
            declared_standards=spikes,
        )
        targets = (
            pd.concat([payload["reference"], generated], ignore_index=True)
            .drop_duplicates(["name", "adduct"])
            .reset_index(drop=True)
        )
        # Monoisotopic endogenous chemistry only; not manual target identities.
        eligible = ~targets.name.astype(str).str.endswith(" (IS)")
        eligible &= (
            targets.get("isotopologue", pd.Series("", index=targets.index))
            .fillna("")
            .astype(str)
            .eq("")
        )
        targets = targets[eligible].reset_index(drop=True)
        prediction, audit = coherent_retention_predictions(
            targets,
            payload["features"],
            payload["annotations"],
            spikes,
            measurements[measurements.stream.eq(stream)].copy()
            if len(measurements)
            else measurements.copy(),
            bundle,
            stream=stream,
            query_group=dataset[:7],
            expected_excluded_groups=excluded,
            ranges=ranges,
        )
        part = targets[keys].copy()
        part["class"] = part["class"].map(canonical_lipid_class)
        part = part.join(prediction)
        # Numeric RT by class/CDB/ion must not depend on chemical-name aliases.
        columns = list(prediction)
        for _, g in part.groupby(keys, sort=False):
            for c in columns:
                v = g[c].to_numpy(float)
                assert np.isnan(v).all() or (
                    np.isfinite(v).all() and np.ptp(v) <= 1e-8
                ), (dataset, stream, c)
        frames.append(part.drop_duplicates(keys).assign(dataset=dataset, stream=stream))
        audits.append(audit.assign(dataset=dataset))
        print(
            "BLIND NUMERIC RT",
            dataset,
            stream,
            "supported_targets",
            int(prediction.coherent_available.sum()),
            flush=True,
        )
    path = OUT / "predictions" / (dataset + ".pkl")
    audit_path = OUT / "audits" / (dataset + ".pkl")
    pd.concat(frames, ignore_index=True).to_pickle(path)
    pd.concat(audits, ignore_index=True).to_pickle(audit_path)
    receipt = dict(
        dataset=dataset,
        outer=outer,
        annotation_changes=False,
        manual_evaluation_loaded=False,
        streams=4,
        prediction_sha256={str(p): sha(p) for p in (path, audit_path)},
    )
    dump(checkpoint, receipt)
    return receipt


def predict_all():
    protocol, projects, _ = initialize()
    for name in ("checkpoints", "predictions", "audits"):
        (OUT / name).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(predict, d): d for d in projects}
        for f in as_completed(futures):
            receipt = f.result()
            print("BLIND RT CASE COMPLETE", receipt["dataset"], flush=True)
    check_binding(protocol)
    artifacts = {}
    for d in projects:
        path = OUT / "checkpoints" / (d + ".json")
        saved = json.loads(path.read_text())
        assert saved["manual_evaluation_loaded"] is False
        for p, h in saved["prediction_sha256"].items():
            assert sha(Path(p)) == h
            artifacts[p] = h
        artifacts[str(path)] = sha(path)
    dump(
        OUT / "blind_predictions_complete.json",
        dict(
            all_33_complete=True,
            dataset_modes=132,
            annotation_changes=False,
            manual_evaluation_loaded=False,
            protocol_sha256=sha(OUT / "protocol.json"),
            prediction_sha256=artifacts,
        ),
    )


def evaluate():
    protocol, projects, _ = initialize()
    receipt_path = OUT / "blind_predictions_complete.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["all_33_complete"] and receipt["manual_evaluation_loaded"] is False
    assert receipt["protocol_sha256"] == sha(OUT / "protocol.json")
    for p, h in receipt["prediction_sha256"].items():
        assert sha(Path(p)) == h
    configure_frozen_resources(OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class

    obs_hash = sha(OBS)
    # First access to manual targets occurs ONLY AFTER the all33 blind receipt.
    observations = pd.read_pickle(OBS)
    assert observations.cohort.eq(observations.dataset.str[:7]).all()
    records = []
    keys = ["class", "carbon", "db", "adduct"]
    for d in projects:
        predictions = pd.read_pickle(OUT / "predictions" / (d + ".pkl"))
        query = observations[
            observations.dataset.eq(d) & observations.mapped.astype(bool)
        ].copy()
        query["class"] = query["class"].map(canonical_lipid_class)
        query = query[~query["class"].eq("X4")]
        raw = pd.read_pickle(BASE / "runs" / d / "full_annotation.pkl")
        for stream, part in query.groupby("stream", sort=False):
            lookup = (
                raw["streams"][stream]["features"]
                .assign(feature_id=lambda f: f.feature_id.astype(str))
                .set_index("feature_id")
            )
            np.testing.assert_array_equal(
                lookup.loc[part.original_feature_id.astype(str), "rt"].to_numpy(float)
                * 60,
                part.rt_seconds.to_numpy(float),
            )
        detail = query.merge(
            predictions,
            on=["dataset", "stream", *keys],
            how="left",
            validate="many_to_one",
        )
        for side, c in (
            ("before", "baseline_rt_seconds"),
            ("after", "coherent_rt_seconds"),
        ):
            detail["absolute_error_" + side] = (detail[c] - detail.rt_seconds).abs()
            detail["within_10_" + side] = detail["absolute_error_" + side].le(10)
        detail["gained_within_10"] = ~detail.within_10_before & detail.within_10_after
        detail["lost_within_10"] = detail.within_10_before & ~detail.within_10_after
        records.append(detail)
    all_rows = pd.concat(records, ignore_index=True)
    all_rows.to_pickle(OUT / "private_numerical_evaluation.pkl")
    for name, keys in (
        ("by_dataset_mode", ["dataset", "stream"]),
        ("by_mode_class", ["stream", "class"]),
        ("by_dataset_mode_class", ["dataset", "stream", "class"]),
        ("by_mode", ["stream"]),
    ):
        table = all_rows.groupby(keys, as_index=False).agg(
            assessed_original_RTs=("mapped", "size"),
            blind_lookup_available=("baseline_rt_seconds", "count"),
            coherent_RT_available=("coherent_available", "sum"),
            within_10_seconds_before=("within_10_before", "sum"),
            within_10_seconds_after=("within_10_after", "sum"),
            newly_within_10_seconds=("gained_within_10", "sum"),
            no_longer_within_10_seconds=("lost_within_10", "sum"),
            median_absolute_RT_error_before=("absolute_error_before", "median"),
            median_absolute_RT_error_after=("absolute_error_after", "median"),
        )
        table.to_csv(OUT / (name + ".csv"), index=False)
    check_binding(protocol)
    assert sha(OBS) == obs_hash
    for p, h in receipt["prediction_sha256"].items():
        assert sha(Path(p)) == h
    dump(
        OUT / "summary.json",
        dict(
            annotation_changes=False,
            classifier_fits=0,
            numerical_reference_fits=0,
            datasets=33,
            dataset_modes=132,
            assessed_original_RTs=len(all_rows),
            blind_lookup_available=int(all_rows.baseline_rt_seconds.notna().sum()),
            coherent_RT_available=int(
                all_rows.coherent_available.fillna(False).astype(bool).sum()
            ),
            within_10_seconds_before=int(all_rows.within_10_before.sum()),
            within_10_seconds_after=int(all_rows.within_10_after.sum()),
            gained=int(all_rows.gained_within_10.sum()),
            lost=int(all_rows.lost_within_10.sum()),
            blind_all33_receipt_sha256=sha(receipt_path),
            evaluated_manual_observations_sha256=obs_hash,
            source_hash_checks=len(protocol["source_sha256"]),
            input_hash_checks=len(protocol["input_sha256"]),
            purpose=protocol["purpose"],
            note="Mapped endogenous RT-only assessment excludes X4;missing theoretical/native-ion lookups remain in this RT denominator. Unresolved/manualSTD rows are not RT assessable here but remain in full annotation benchmark. No annotation performance/gain is claimed.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    predict_all() if args.predict else evaluate()
