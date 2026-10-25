"""046 reuses ALL25/50fixed044models; new INNERcontinuous-admission operatingpoints.

No classifierfits, model blending or adoption/splicing of044outerannotations.
Whole041immutable comparator. Complete no-cut preflight before newchoices.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
from immutable_exports import assemble_immutable
import run_qe_bmp_context_experiment as source
import bmp_project_headroom as headroom
from bmp_continuous_admission import proposals

QE, BASE, CORE, PREVIOUS = source.QE, source.BASE, source.CORE, source.PREVIOUS
original = source.original
completion = source.completion
OUT = QE / "experiment_046"
PREFLIGHT = QE / "BMP_continuous_admission_preflight_v1"
PLAN = REPO / "docs/validation/next_BMP_continuous_admission_experiment.txt"
MODES, NP_MODES = source.MODES, source.NP_MODES
sha, dump, inventory = source.sha, source.dump, source.inventory
METHOD = "continuous_evidence_BMP_completion"
load_predictors, previous_annotations, parent_path = (
    source.load_predictors,
    source.previous_annotations,
    source.parent_path,
)


def project_curves(cases, allowed, excluded):
    rows = []
    for case in cases:
        source.preparation.validate_scope(
            case["raw"], allowed, excluded, query_is_training=True
        )
        if case["dataset"][:7] not in allowed or case["dataset"][:7] in excluded:
            raise ValueError("Only authorized TRAINING projects can supply curves")
        potential = proposals(
            case["dataset"],
            case["mask"],
            case["raw"],
            case["scores"],
            case["occupied"],
            dict.fromkeys(NP_MODES, min(source.completion.LEVELS)),
            case["supported"],
        )
        previous = source.prior_counts(case["occupied"], case["manual"])
        for t in source.completion.LEVELS:
            chosen = source.threshold_subset(potential, t)
            for s in NP_MODES:
                counts = source.addition_counts(
                    chosen.loc[chosen.stream.eq(s)], case["manual"]
                )
                rows.append(
                    dict(
                        dataset=case["dataset"],
                        project=case["dataset"][:7],
                        stream=s,
                        threshold=t,
                        **previous[s],
                        **counts,
                    )
                )
    folders = pd.DataFrame(rows)
    projects = headroom.combine_project_folders(folders)
    pooled = headroom.pooled_curves(
        projects, training_groups=allowed, excluded_groups=excluded
    )
    return folders, projects, pooled


def cases_and_curves(outer):
    datasets, folds = inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    protocol = json.loads((source.OUT / "protocol.json").read_text())
    cases = source.calibration_cases(outer, protocol, datasets, folds, fit=False)
    for case in cases:
        pd.testing.assert_series_equal(
            case["scores"],
            pd.read_pickle(
                source.OUT
                / "calibration"
                / f"fold_{outer}"
                / (case["dataset"] + "_blind_scores.pkl")
            ),
            check_exact=True,
        )
    folders, projects, pooled = project_curves(cases, allowed, excluded)
    return cases, folders, projects, pooled, allowed, excluded


def assemble(d, previous, selected):
    filled = source.original.inherited.common.fill_original_features(d, selected)
    if len(filled):
        filled.rename(
            columns={"context_completion_score": "BMP_continuous_agreement_rank"},
            inplace=True,
        )
        filled["auto_annotation_status"] = METHOD
    candidates = pd.read_pickle(PREVIOUS / "runs" / d / "candidates.pkl")
    annotations, candidates = assemble_immutable(
        previous, candidates, filled, method=METHOD
    )
    return annotations, candidates, filled


def source_inputs():
    master_path = source.OUT / "integrity_receipt.json"
    master = json.loads(master_path.read_text())
    assert (
        master["all_passed"]
        and master["parents"] == 25
        and master["fresh_classifiers"] == 50
        and master["inner_query_cases"] == 132
    )
    inputs = {**master["source_input_sha256"], **master["artifact_sha256"]}
    diagnostic = json.loads(
        (QE / "BMP_admission_ceiling_041_v1/receipt.json").read_text()
    )
    assert diagnostic["all_passed"] and diagnostic["datasets"] == 33
    assert diagnostic["annotation_changes"] is False
    inputs.update(diagnostic["source_input_sha256"])
    inputs.update(diagnostic["artifact_sha256"])
    assert all(sha(Path(p)) == h for p, h in inputs.items())
    paths = set(map(Path, inputs))
    paths.update(
        {
            master_path,
            source.OUT / "protocol.json",
            Path(source.__file__),
            Path(__file__),
            Path(headroom.__file__),
            Path(__file__).with_name("bmp_continuous_admission.py"),
            Path(__file__).with_name("diagnose_qe_bmp_admission_ceiling.py"),
            QE / "BMP_admission_ceiling_041_v1/receipt.json",
            PLAN,
            Path(__file__).with_name("preflight_qe_bmp_continuous.py"),
            Path(__file__).with_name("verify_qe_bmp_continuous_experiment.py"),
            Path(__file__).with_name("postprocess_qe_bmp_continuous.py"),
            Path(__file__).with_name("report_qe_bmp_continuous_tradeoff.py"),
            source.OUT / "prospective_tradeoff_decision/receipt.json",
            source.OUT / "prospective_tradeoff_decision/decision.json",
            QE / "experiment_045/prospective_tradeoff_decision/decision.json",
            QE / "experiment_045/prospective_tradeoff_decision/receipt.json",
        }
    )
    assert not json.loads(
        (source.OUT / "prospective_tradeoff_decision/decision.json").read_text()
    )["retain_whole_research_method"]
    return {str(p): sha(p) for p in sorted(paths)}


def initialize():
    gate = json.loads((PREFLIGHT / "receipt.json").read_text())
    assert (
        gate["all_passed"]
        and gate["inner_query_cases"] == 132
        and gate["reused_parents"] == 25
        and gate["reused_experts"] == 50
        and gate["new_classifier_fits"] == 0
        and gate["new_operating_points_selected"] is False
    )
    assert gate["source_input_sha256"] == source_inputs()
    assert all(sha(Path(p)) == h for p, h in gate["curve_sha256"].items())
    verifier = Path(__file__).with_name("verify_qe_bmp_continuous_experiment.py")
    datasets, _ = inventory()
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "methods", "reference_data"):
            shutil.copytree(
                source.OUT / "source_snapshot" / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        old = json.loads((source.OUT / "protocol.json").read_text())
        hashes = {
            p.relative_to(snapshot).as_posix(): sha(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        assert hashes == old["source_snapshot_sha256"]
        dump(
            OUT / "protocol.json",
            dict(
                name="046-continuous-admission-BMP-operatingpoints",
                previous=str(PREVIOUS),
                source_models=str(source.OUT),
                source_master_sha256=sha(source.OUT / "integrity_receipt.json"),
                reused_parents=25,
                reused_experts=50,
                new_classifier_fits=0,
                production_enabled=False,
                curation_runtime_input=False,
                planned_verifier_sha256=sha(verifier),
                preflight_sha256=sha(PREFLIGHT / "receipt.json"),
                source_input_sha256=gate["source_input_sha256"],
                source_snapshot_sha256=hashes,
                resource_receipt=configure_frozen_resources(snapshot),
                threshold_grid=list(source.completion.LEVELS),
                plan=PLAN.read_text(),
                run_kind="Previously exposed saved-input grouped development. ALLfixed25/50NPcontextmodels reused;newINNERoriginal044jointcuts,profilevaluescontinuousnotveto;ALL041callsimmutable. No044outeroutputsplicing/newfits/deployment/externalvalidation/FDR.",
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=datasets,
                fresh_raw_input_runs=False,
                evaluation_note="New grouped continuous-admission cuts on explicitly reused044models;WHOLE041immutable comparator,notfreshappdeployment.",
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["source_input_sha256"] == source_inputs()
    assert protocol["planned_verifier_sha256"] == sha(verifier)
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    assert all(
        sha(OUT / "source_snapshot" / p) == h
        for p, h in protocol["source_snapshot_sha256"].items()
    )
    return protocol, datasets


def calibrate(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    cases, folders, projects, pooled, allowed, excluded = cases_and_curves(outer)
    before = PREFLIGHT / f"fold_{outer}"
    pd.testing.assert_frame_equal(
        projects, pd.read_pickle(before / "project_curves.pkl"), check_exact=True
    )
    thresholds, grid = completion.choose_joint_grid(pooled)
    destination = OUT / "calibration" / f"fold_{outer}"
    destination.mkdir(parents=True, exist_ok=False)
    folders.to_csv(destination / "by_folder_mode_threshold.csv", index=False)
    projects.to_csv(destination / "by_project_mode_threshold.csv", index=False)
    pooled.to_csv(destination / "training_operating_points.csv", index=False)
    grid.to_csv(destination / "complete_joint_grid.csv", index=False)
    for case in cases:
        case["scores"].to_pickle(destination / (case["dataset"] + "_blind_scores.pkl"))
    dump(
        destination / "operating_points.json",
        dict(
            thresholds=thresholds,
            training_groups=sorted(allowed),
            outer_excluded=sorted(excluded),
            project_curves_sha256=sha(destination / "by_project_mode_threshold.csv"),
            joint_grid_sha256=sha(destination / "complete_joint_grid.csv"),
        ),
    )
    print("046 CONTINUOUS-ADMISSION INNER GRID COMPLETE", outer, thresholds, flush=True)


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    datasets, folds = inventory()
    modelpath = parent_path(outer, None)
    assert (
        sha(modelpath)
        == json.loads(modelpath.with_suffix(".complete.json").read_text())[
            "model_sha256"
        ]
    )
    model = pd.read_pickle(modelpath)
    cuts = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    for d in datasets:
        if folds[d] != outer:
            continue
        raw = load_predictors(d, {outer})
        scores = source.completion.predict_context(raw, model)
        pd.testing.assert_series_equal(
            scores,
            pd.read_pickle(source.OUT / "runs" / d / "scoring_audit.pkl"),
            check_exact=True,
        )
        previous = previous_annotations(d)
        selected = proposals(
            d,
            {outer},
            raw,
            scores,
            previous,
            cuts,
            {s: model["experts"][s] is not None for s in NP_MODES},
        )
        annotations, candidates, filled = assemble(d, previous, selected)
        path = OUT / "runs" / d
        path.mkdir(parents=True, exist_ok=False)
        annotations.to_csv(path / "automatic_annotations.csv", index=False)
        candidates.to_pickle(path / "candidates.pkl")
        filled.to_pickle(path / "private_additions.pkl")
        scores.to_pickle(path / "scoring_audit.pkl")
        dump(
            path / "complete.json",
            dict(
                dataset=d,
                fold=outer,
                status="complete",
                previous_calls_immutable=True,
                production_enabled=False,
                added_annotations=len(filled),
                counts=annotations.groupby("stream").size().to_dict(),
            ),
        )
        print(
            "046 OUTER CONTINUOUS-ADMISSION BMP ADDITIONS", d, len(filled), flush=True
        )


def main():
    initialize()
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(calibrate, range(5)))
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(run_fold, range(5)))
    print("046 COMPLETE;FULL132chat/wholeintegrity/decision BEFORE046", flush=True)


if __name__ == "__main__":
    main()
