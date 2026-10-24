"""Experiment027: fixed conservative mass-calibrated additions to immutable022."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources

QE = WORKSPACE / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
OLD = QE / "experiment_022"
REFERENCES = QE / "nested_reference_development_v2"
STD = QE / "nested_standard_development_v1"
PREPARED = QE / "nested_candidate_preparation_v2"
OUT = QE / "experiment_027"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    assert json.loads((STD / "summary.json").read_text())["all_complete"]
    assert json.loads((OLD / "reports_v2/summary.json").read_text())["complete"]
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
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
        paths = [
            Path(__file__).with_name("frozen_resources.py"),
            BASE / "run_manifest.json",
            OLD / "run_manifest.json",
            OLD / "reports_v2/summary.json",
            STD / "summary.json",
            QE / "rt_atlas_development/fold_assignment.json",
        ]
        for outer in range(5):
            paths.extend(
                [REFERENCES / f"exclude_{outer}.json", STD / f"exclude_{outer}.json"]
            )
        for dataset in projects:
            paths.extend(
                [
                    BASE / "runs" / dataset / "full_annotation.pkl",
                    OLD / "runs" / dataset / "automatic_annotations.csv",
                    OLD / "runs" / dataset / "candidates.pkl",
                    PREPARED
                    / "prepared"
                    / f"exclude_{folds[dataset]}"
                    / (dataset + ".pkl"),
                ]
            )
        protocol = dict(
            name="027-fixed-standard-calibrated-mass-discovery",
            baseline=str(BASE),
            previous=str(OLD),
            production_enabled=False,
            curation_runtime_input=False,
            numerical_workers=3,
            folds=5,
            score_refitting=False,
            threshold_fitting=False,
            candidate_policy="Same existing chemical reference+group-trained composition envelope; existing range settings binding. Only new rawabsppm>=3pairs may be added. Correctedabsppm<3 and atlasRT<=10s,>=3RTtraininggroups,in-domain,MAD<=10s. Leave-target-out standard calibration>=6distinct actualneutralmasses,MAD<=.6ppm,p90<=1.5ppm,absmedian<=2ppm. Whole declared-spike mass aliases within5.01ppm withheld. All coarse alternatives compete BEFORE novelty/fine gates/occupancy.",
            acceptance_policy="FIXED, no test-label tuning: unique feature per identity AND identity per feature in coarse pool,RTerror<=3s,atlasMAD<=5s,strong exact canonicalclass/CDB automaticanchor in ANOTHERstream(confidence>=.9,rawabsppm<1.5,unambiguousadductRT). Existing022identities/features/standards ALLimmutable. No replacements or fallback promotion. No learned scores, probabilities or FDR claims.",
            standard_policy="Independently re-detect declared actual isotope spikes from raw features with EMPTY endogenous proposals,matching query-excluded outer standard/RTmodels. Positive quantities and rawunique3ppm/10s evidence; no own-neutral mass anchor. Does NOT reuse rejected025score predictions or026weights/024labels. Final022standard calls immutable.",
            validation="All33datasets/30projectgroups using SAMEouterblocks; relevant learned RT/envelope/standard references exclude queryproject. Fixed physical rule has no inner cutoff/score fitting. Cached baseline/original-feature replay with known inherited baseline/vendor development exposure; NOT untouched external whole-pipeline validation or analyticalFDR.",
            model_fit_note="No new score fit. Existing022calls retained exactly. RT/standard reference models exclude outer/querygroups. Physical mass centering uses measured query-run chemical standards only,not manual annotations or own feature/neutral identities. No new formula guessed; unknown classes remain unsupported.",
            run_kind="Saved022immutable calls plus fixed-rule independent standard-calibrated new original-feature discovery. Raw observed m/z,RT,I unchanged;calibratedmasserror is a separate descriptor. Not fresh production raw-input runs.",
            source_snapshot_sha256={
                p.relative_to(snapshot).as_posix(): sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
            resource_receipt=configure_frozen_resources(snapshot),
            input_sha256={str(p): sha(p) for p in paths},
        )
        dump(OUT / "protocol.json", protocol)
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=projects,
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"] + " " + protocol["validation"],
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for path, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / path) == digest
    for path, digest in protocol["input_sha256"].items():
        assert sha(Path(path)) == digest
    return protocol, projects


def run_dataset(dataset):
    dest = OUT / "runs" / dataset
    if (dest / "complete.json").exists():
        return json.loads((dest / "complete.json").read_text())
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.declared_standard_context import (
        independent_standard_measurements,
    )
    from matlab_annotation.calibrated_mass_discovery import (
        independently_measured_mass_anchors,
        calibrated_mass_candidates,
        select_calibrated_additions,
    )
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.retention_atlas import predict_aligned_retention_atlas
    from matlab_annotation.standard_rt_library import ion_key

    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    outer = folds[dataset]
    reference = json.loads((REFERENCES / f"exclude_{outer}.json").read_text())
    standard = json.loads((STD / f"exclude_{outer}.json").read_text())
    assert (
        dataset in reference["excluded_datasets"]
        and dataset in standard["excluded_datasets"]
    )
    assert set(reference["training_groups"]) == set(standard["training_groups"])
    assert set(reference["excluded_groups"]) == set(standard["excluded_groups"])
    assert dataset[:7] not in reference["training_groups"]
    result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    old = pd.read_csv(
        OLD / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
    )
    original = pd.read_pickle(
        PREPARED / "prepared" / f"exclude_{outer}" / (dataset + ".pkl")
    )
    assert set(original["reference_training_groups"]) == set(
        reference["training_groups"]
    )
    catalogue = declared_standard_targets(
        OUT / "source_snapshot/reference_data/matlab_pipeline",
        result["internal_standards_expected"],
    )
    measured, standard_audit = independent_standard_measurements(
        result, catalogue, standard
    )
    anchors = independently_measured_mass_anchors(measured)
    automatic = pd.concat(
        [
            payload["annotations"].copy().assign(stream=s)
            for s, payload in result["streams"].items()
        ],
        ignore_index=True,
    )
    pools, additions, audits = [], [], []
    for stream, payload in result["streams"].items():
        native = payload["reference"]
        declared = catalogue[catalogue.stream.eq(stream)]
        generated, ranges, _ = compile_additional_targets(
            native,
            reference["composition_envelope"],
            stream=stream,
            declared_standards=declared,
        )
        targets = (
            pd.concat([native, generated], ignore_index=True)
            .drop_duplicates(["name", "adduct"])
            .reset_index(drop=True)
        )
        allowed = [
            ranges.decide(r["class"], r["carbon"], r["db"], species=r["name"]).allowed
            for r in targets.to_dict("records")
        ]
        targets = targets.loc[allowed].reset_index(drop=True)
        predictions = predict_aligned_retention_atlas(
            targets, payload["annotations"], reference["atlas"], stream=stream
        )
        targets = targets.drop(columns=list(predictions), errors="ignore").join(
            predictions
        )
        previous_pool = original["pool"][original["pool"].stream.eq(stream)]
        pool = calibrated_mass_candidates(
            targets,
            payload["features"],
            anchors,
            previous_pool,
            stream=stream,
            declared_standard_masses=declared.ref_mz,
        )
        before = old[old.stream.eq(stream)]
        added = select_calibrated_additions(pool, automatic, before, stream=stream)
        pools.append(pool)
        additions.append(added)
        audits.append(
            dict(
                dataset=dataset,
                stream=stream,
                outer=outer,
                calibration_anchor_ions=int(anchors.stream.eq(stream).sum()),
                coarse_candidate_rows=len(pool),
                novel_mass_pair_rows=int(pool.discovery_novel_mass_pair.sum())
                if len(pool)
                else 0,
                added_annotations=len(added),
                original_annotations=len(before),
                original_calls_immutable=True,
            )
        )
    added = pd.concat(additions, ignore_index=True)
    final = pd.concat([old, added], ignore_index=True)
    pd.testing.assert_frame_equal(
        final.iloc[: len(old)][old.columns].reset_index(drop=True),
        old.reset_index(drop=True),
        check_dtype=False,
    )
    old_identities = set(zip(old.stream, old.name.astype(str), old.adduct.map(ion_key)))
    if len(added):
        new_identities = list(
            zip(added.stream, added.name.astype(str), added.adduct.map(ion_key))
        )
        assert len(new_identities) == len(set(new_identities))
        assert not set(new_identities) & old_identities
    assert not final[["stream", "feat_feature_id"]].astype(str).duplicated().any()
    existing_candidates = pd.read_pickle(OLD / "runs" / dataset / "candidates.pkl")
    candidates = pd.concat([existing_candidates, *pools], ignore_index=True)
    keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    candidates = candidates.drop_duplicates(keys, keep="last").reset_index(drop=True)
    accepted = set(final[keys].itertuples(index=False, name=None))
    candidates["automatic_accepted"] = [
        k in accepted for k in candidates[keys].itertuples(index=False, name=None)
    ]
    dest.mkdir(parents=True, exist_ok=True)
    final.to_csv(dest / "automatic_annotations.csv", index=False)
    candidates.to_pickle(dest / "candidates.pkl")
    added.to_pickle(dest / "private_additions.pkl")
    pd.concat(pools, ignore_index=True).to_pickle(dest / "private_mass_candidates.pkl")
    anchors.to_pickle(dest / "private_mass_anchors.pkl")
    standard_audit.to_pickle(dest / "private_standard_audit.pkl")
    pd.DataFrame(audits).to_csv(dest / "discovery_audit.csv", index=False)
    complete = dict(
        dataset=dataset,
        outer=outer,
        status="complete",
        production_enabled=False,
        added_annotations=len(added),
        previous_calls_immutable=True,
        counts=final.groupby("stream").size().to_dict(),
    )
    dump(dest / "complete.json", complete)
    return complete


def main():
    protocol, projects = initialize()
    executor = ProcessPoolExecutor(max_workers=protocol["numerical_workers"])
    remaining = iter(projects)
    pending = {
        executor.submit(run_dataset, next(remaining))
        for _ in range(protocol["numerical_workers"])
    }
    try:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            complete = [future.result() for future in done]
            for result in complete:
                print(
                    "CALIBRATED MASS DISCOVERY",
                    result["dataset"],
                    result["added_annotations"],
                    "additions",
                    flush=True,
                )
                dataset = next(remaining, None)
                if dataset is not None:
                    pending.add(executor.submit(run_dataset, dataset))
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    print(
        "EXPERIMENT027 COMPLETE;FULL132TABLE REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
