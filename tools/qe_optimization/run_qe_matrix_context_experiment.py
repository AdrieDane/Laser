"""049 fresh HGB/context experiment, gated until COMPLETE final preflight.

All25fresh modelparents -> ALL5INNERmode+041completion calibrations ->33blind
whole outputs. No outer query validation curation, evaluation or auto-next-run.
Bounded dispatch supports explicit user pause at completed atomic boundaries.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import shutil

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import matrix_experiment_data as data
import matrix_experiment_bindings as bindings
import matrix_mode_model as backend
import matrix_operating_points as operating


def atomic_json(path, value):
    path = Path(path)
    assert path.is_relative_to(data.OUT)
    temporary = path.with_name(path.name + ".tmp")
    data.nested.pilot.dump(temporary, value)
    temporary.replace(path)


def protocol():
    result = json.loads((data.OUT / "protocol.json").read_text())
    assert result["numeric_runtime"] == bindings.environment()
    assert result["planned_actual_verifier_sha256"] == data.nested.pilot.sha(
        bindings.CHECKER
    )
    return result


def initialize():
    permission = bindings.require_fit_permission()  # BEFORE experiment mkdir.
    if not data.OUT.exists():
        data.OUT.mkdir()
        snapshot = data.OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "methods", "reference_data"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        resources = configure_frozen_resources(snapshot)
        native_hashes = {
            str(p.relative_to(snapshot)): data.nested.pilot.sha(p)
            for p in snapshot.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        atomic_json(
            data.OUT / "protocol.json",
            dict(
                name="049-native-HGB-other-family-matrix-context",
                production_enabled=False,
                source_input_sha256=permission["source_input_sha256"],
                numeric_runtime=bindings.environment(),
                planned_actual_verifier_sha256=data.nested.pilot.sha(bindings.CHECKER),
                source_snapshot_sha256=native_hashes,
                resource_receipt=resources,
                classifier=data.parameters(),
                mode_threshold_grid=list(operating.GRID),
                ambiguity_margin=0.05,
                core_extras_per_manual=0.095,
                numerical_workers=2,
                fits="25fresh nativeHGBparents/100modeexperts/180iterationsEACH",
                completion="EXACT041conservative policy RECOMPUTED usingnewINNERscores+nativeblindSTDoccupancy;notoldoutercalls",
                standards="fixed005",
                mass="fixed027 AFTERnewcore/std occupancy",
                input_schema="108old+8nestedmatrix=116; no newphysicalcandidates/hardlimits",
                outer_evaluation_curation_used=False,
                plan=data.PLAN.read_text(),
                run_kind="Cached original-feature grouped development replay; prior dataset exposure,not fresh production/externalvalidation/chemicalFDR. All33/30projects/132cases,primary92100includingstandards/unresolved,X4onlyexcluded.",
            ),
        )
        datasets, _ = data.inventory()
        atomic_json(
            data.OUT / "run_manifest.json",
            dict(
                datasets=datasets,
                fresh_raw_input_runs=False,
                evaluation_note="Grouped cached development replay; new8context predictors/freshnativeHGBmodels,original005/027/041algorithm. No appdefaults. Not externalvalidation/FDR.",
            ),
        )
    p = protocol()
    assert p["source_input_sha256"] == permission["source_input_sha256"]
    snapshot = data.OUT / "source_snapshot"
    configure_frozen_resources(snapshot)
    for relative, h in p["source_snapshot_sha256"].items():
        assert data.nested.pilot.sha(snapshot / relative) == h
    return p


def model_completion(path):
    return path.with_suffix(".complete.json")


def read_model(outer, inner):
    path = data.model_path(outer, inner)
    done = json.loads(model_completion(path).read_text())
    assert done["protocol_sha256"] == data.nested.pilot.sha(data.OUT / "protocol.json")
    assert done["model_sha256"] == data.nested.pilot.sha(path)
    return pd.read_pickle(path)


def fit_parent(item):
    configure_frozen_resources(data.OUT / "source_snapshot")
    outer, inner = item
    path = data.model_path(outer, inner)
    if model_completion(path).exists():
        model = read_model(outer, inner)
        return dict(
            outer=outer,
            inner=inner,
            query_uses=len(model["matrix_feature_scopes"]),
            resumed_completed_model=True,
        )
    assert not path.exists(), (
        "Uncommitted model artifact: preserve/inspect, do not overwrite"
    )
    records, allowed, excluded = data.training_records(outer, inner)
    expected = json.loads((bindings.STAGE / "training_matrices.json").read_text())
    expected = next(r for r in expected if (r["outer"], r["inner"]) == item)
    model = backend.fit(
        records,
        training_groups=allowed,
        excluded_groups=excluded,
        parameters=protocol()["classifier"],
    )
    assert model["independent_fitting_ledger"] == expected["ledgers"]
    assert len(records) == expected["query_uses"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pkl")
    pd.to_pickle(model, temporary)
    temporary.replace(path)
    atomic_json(
        model_completion(path),
        dict(
            outer=outer,
            inner=inner,
            model_sha256=data.nested.pilot.sha(path),
            protocol_sha256=data.nested.pilot.sha(data.OUT / "protocol.json"),
            query_uses=len(records),
            fresh_experts=4,
            n_iter_each=180,
            reference_scopes_checked=True,
            fitting_ledger=model["independent_fitting_ledger"],
        ),
    )
    print("049 FOUR FRESH NATIVE HGB EXPERTS", outer, inner, len(records), flush=True)
    return dict(
        outer=outer, inner=inner, query_uses=len(records), resumed_completed_model=False
    )


def calibration_cases(outer, *, save):
    datasets, folds = data.inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    folder = data.OUT / "calibration" / f"fold_{outer}"
    cases, manuals = {}, []
    for inner in range(5):
        if inner == outer:
            continue
        model = read_model(outer, inner)
        for dataset in datasets:
            if folds[dataset] != inner:
                continue
            raw = data.load_predictors(dataset, {outer, inner})
            scores = backend.predict(raw, model)
            if save:
                scores.to_pickle(folder / (dataset + "_blind_scores.pkl"))
            # Only NOW authorized TRAINING calibration truth, not ownoutergold.
            labeled, manual = data.authorize_labels(raw, allowed, excluded)
            cases[dataset] = dict(
                project=dataset[:7],
                raw=raw,
                pool=labeled["pool"],
                scores=scores,
                inner=inner,
            )
            manuals.append(manual.assign(project=dataset[:7]))
    assert set(cases) == {d for d in datasets if folds[d] != outer}
    return cases, pd.concat(manuals, ignore_index=True), allowed, excluded


def completion_inputs(outer, cases, modecuts):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.operating_point import select_operating_candidates
    from matrix_pipeline_assembly import proposals

    rows, counts = [], Counter()
    for dataset, case in cases.items():
        baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
        occupancy = []
        for stream in data.MODES:
            pool = case["pool"].loc[case["pool"].stream.eq(stream)]
            core = select_operating_candidates(
                pool,
                case["scores"].loc[pool.index],
                threshold=modecuts[stream],
                ambiguity_margin=0.05,
            )
            counts.update(
                {
                    (stream, c): int(n)
                    for c, n in core.assign(
                        _cls=core["class"].map(canonical_lipid_class)
                    )
                    .groupby("_cls")
                    .training_positive.sum()
                    .items()
                }
            )
            native = baseline["streams"][stream]["annotations"]
            occupancy.extend(
                [
                    core,
                    native.loc[native.is_is.fillna(False).astype(bool)].assign(
                        stream=stream
                    ),
                ]
            )
        labeled = dict(case["raw"], pool=case["pool"])
        free = proposals(
            dataset,
            {outer, case["inner"]},
            labeled,
            case["scores"],
            pd.concat(occupancy, ignore_index=True),
        )
        rows.append(free.assign(project=dataset[:7]))
    return pd.concat(rows, ignore_index=True), counts


def calibrate(outer):
    configure_frozen_resources(data.OUT / "source_snapshot")
    folder = data.OUT / "calibration" / f"fold_{outer}"
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / "operating_points.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        assert result["protocol_sha256"] == data.nested.pilot.sha(
            data.OUT / "protocol.json"
        )
        for name, h in result["artifact_sha256"].items():
            assert data.nested.pilot.sha(folder / name) == h
        return outer
    cases, manual, allowed, excluded = calibration_cases(outer, save=True)
    modecurve = operating.mode_curves(
        cases, manual, training_groups=allowed, excluded_groups=excluded
    )
    modecuts = operating.choose_modes(modecurve)
    free, basecounts = completion_inputs(outer, cases, modecuts)
    classcurve, completioncuts, selected = operating.completion_curves(
        free, manual, basecounts, training_groups=allowed, excluded_groups=excluded
    )
    modecurve.to_csv(folder / "training_operating_points.csv", index=False)
    classcurve.to_csv(folder / "training_completion_points.csv", index=False)
    selected.to_csv(folder / "selected_completion_points.csv", index=False)
    free.to_pickle(folder / "private_completion_proposals.pkl")
    artifacts = {
        p.name: data.nested.pilot.sha(p)
        for p in folder.iterdir()
        if p.is_file() and p.name != "operating_points.json"
    }
    atomic_json(
        completed,
        dict(
            mode_thresholds=modecuts,
            completion_thresholds=completioncuts,
            training_groups=sorted(allowed),
            outer_excluded=sorted(excluded),
            protocol_sha256=data.nested.pilot.sha(data.OUT / "protocol.json"),
            artifact_sha256=artifacts,
            inner_queries=len(cases),
            original029modepolicy=True,
            original041completionpolicy=True,
        ),
    )
    print(
        "049 NEW INNER MODE/COMPLETION CALIBRATION",
        outer,
        modecuts,
        int(selected.threshold.lt(1).sum()),
        flush=True,
    )
    return outer


def freeze_calibrations():
    datasets, folds = data.inventory()
    frozen = {}
    for outer in range(5):
        path = data.OUT / "calibration" / f"fold_{outer}/operating_points.json"
        p = json.loads(path.read_text())
        assert set(p["training_groups"]) == {
            d[:7] for d in datasets if folds[d] != outer
        }
        assert set(p["outer_excluded"]) == {
            d[:7] for d in datasets if folds[d] == outer
        }
        frozen[str(outer)] = data.nested.pilot.sha(path)
    atomic_json(data.OUT / "all_calibrations_frozen.json", frozen)


def annotate(dataset):
    configure_frozen_resources(data.OUT / "source_snapshot")
    from matrix_pipeline_assembly import (
        assemble_core,
        complete,
        check_original_sample_cells,
    )

    _, folds = data.inventory()
    outer = folds[dataset]
    frozen = json.loads((data.OUT / "all_calibrations_frozen.json").read_text())
    calibration_path = data.OUT / "calibration" / f"fold_{outer}/operating_points.json"
    assert frozen[str(outer)] == data.nested.pilot.sha(calibration_path)
    destination = data.OUT / "runs" / dataset
    done = destination / "complete.json"
    if done.exists():
        p = json.loads(done.read_text())
        assert p["protocol_sha256"] == data.nested.pilot.sha(data.OUT / "protocol.json")
        for name, h in p["artifact_sha256"].items():
            assert data.nested.pilot.sha(destination / name) == h
        return dataset
    cuts = json.loads(calibration_path.read_text())
    model = read_model(outer, None)
    raw = data.load_predictors(dataset, {outer})
    scores = backend.predict(raw, model)
    core = assemble_core(
        dataset,
        outer,
        raw,
        scores,
        cuts["mode_thresholds"],
        data.OUT / "source_snapshot",
    )
    result = complete(dataset, outer, raw, scores, core, cuts["completion_thresholds"])
    destination.mkdir(parents=True, exist_ok=True)
    result["annotations"].to_csv(destination / "automatic_annotations.csv", index=False)
    result["candidates"].to_pickle(destination / "candidates.pkl")
    result["completion_additions"].to_pickle(destination / "private_additions.pkl")
    scores.to_pickle(destination / "scoring_audit.pkl")
    result["standard_audit"].to_csv(destination / "standard_audit.csv", index=False)
    result["mass_audit"].to_csv(destination / "mass_discovery_audit.csv", index=False)
    baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
    cells = check_original_sample_cells(result["annotations"], baseline)
    artifacts = {
        p.name: data.nested.pilot.sha(p)
        for p in destination.iterdir()
        if p.is_file() and p.name != "complete.json"
    }
    atomic_json(
        done,
        dict(
            dataset=dataset,
            fold=outer,
            status="complete",
            production_enabled=False,
            protocol_sha256=data.nested.pilot.sha(data.OUT / "protocol.json"),
            calibration_sha256=frozen[str(outer)],
            artifact_sha256=artifacts,
            counts=result["annotations"].groupby("stream").size().to_dict(),
            original_sample_cells_exact=cells,
            own_outer_curation_used=False,
        ),
    )
    print("049 BLIND WHOLE OUTPUT", dataset, len(result["annotations"]), flush=True)
    return dataset


def execute(tasks, worker, stage_name):
    """At most two in flight; explicit pause stops dispatch, not atomic writes."""
    remaining = iter(tasks)
    paused = False
    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        pending = {}
        while True:
            paused = paused or (data.OUT / "pause_requested").exists()
            while not paused and len(pending) < 2:
                item = next(remaining, None)
                if item is None:
                    break
                pending[executor.submit(worker, item)] = item
            if not pending:
                break
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                del pending[future]
    if paused:
        atomic_json(
            data.OUT / "pause_status.json",
            dict(
                stage=stage_name,
                paused_by_marker=True,
                completed_tasks_this_invocation=len(results),
                all_blind_outputs_complete=False,
            ),
        )
        print("049 EXPLICIT PAUSE AT COMPLETED BOUNDARY", stage_name, flush=True)
        return None
    return results


def main():
    initialize()
    fitted = execute(data.parents(), fit_parent, "fits")
    if fitted is None:
        return
    assert len(fitted) == 25 and sum(r["query_uses"] for r in fitted) == 528
    assert protocol()["source_input_sha256"] == bindings.source_inputs()
    calibrated = execute(range(5), calibrate, "calibration")
    if calibrated is None:
        return
    assert len(calibrated) == 5
    freeze_calibrations()
    datasets, _ = data.inventory()
    outputs = execute(datasets, annotate, "blind_outputs")
    if outputs is None:
        return
    assert set(outputs) == set(datasets) and len(outputs) == 33
    assert protocol()["source_input_sha256"] == bindings.source_inputs()
    atomic_json(
        data.OUT / "producer_complete.json",
        dict(
            all_blind_outputs_complete=True,
            datasets=33,
            dataset_modes=132,
            model_parents=25,
            fresh_unique_experts=100,
            protocol_sha256=data.nested.pilot.sha(data.OUT / "protocol.json"),
            calibration_freeze_sha256=data.nested.pilot.sha(
                data.OUT / "all_calibrations_frozen.json"
            ),
            dataset_complete_sha256={
                d: data.nested.pilot.sha(data.OUT / "runs" / d / "complete.json")
                for d in datasets
            },
            own_outer_evaluation_curation_used=False,
            note="Whole049blindcomplete; manually evaluate/verify/reportfull132 andwholedecision. NOTdeploymentready; noautomaticnextannotationexperiment.",
        ),
    )
    print(
        "049 ALL25/5CAL/33BLIND OUTPUTS COMPLETE; EVALUATION/MASTER REQUIRED",
        flush=True,
    )


if __name__ == "__main__":
    main()
