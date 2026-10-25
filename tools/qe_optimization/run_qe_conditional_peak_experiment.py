"""038: preserve WHOLE029identity acceptance; learn original-peak refinement only."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil

import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources
import run_qe_standard_mode_experiment as legacy

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT, OLD, BASE = QE / "experiment_038", QE / "experiment_029", QE / "baseline"
PREPARED = QE / "standard_context_preparation_v1"
sha, dump = legacy.sha, legacy.dump
MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
CONFIDENCES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def initialize():
    assert json.loads((QE / "experiment_037/integrity_receipt.json").read_text())[
        "all_passed"
    ]
    assert json.loads((OLD / "integrity_receipt.json").read_text())["all_passed"]
    preflight_path = QE / "conditional_peak_preflight_v1/receipt.json"
    preflight = json.loads(preflight_path.read_text())
    assert preflight["all_passed"] and preflight["classifier_fits"] == 0
    for path, digest in {
        **preflight["source_sha256"],
        **preflight["input_sha256"],
    }.items():
        assert sha(Path(path)) == digest, path
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    assert len(projects) == 33
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
        sources = {
            p.relative_to(snapshot).as_posix(): sha(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        files = {
            OLD / "protocol.json",
            OLD / "integrity_receipt.json",
            QE / "experiment_037/integrity_receipt.json",
            BASE / "run_manifest.json",
            QE / "rt_atlas_development/fold_assignment.json",
            PREPARED / "protocol.json",
            PREPARED / "summary.json",
            preflight_path,
            QE / "experiment_007/protocol.json",
            Path(__file__).with_name("run_qe_standard_mode_experiment.py"),
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
            Path(__file__).with_name("context.py"),
            Path(__file__).with_name("frozen_resources.py"),
        }
        files.update(OLD.glob("fold_*_nested_model.pkl"))
        files.update(OLD.glob("calibration/fold_*/inner_block_*_model.pkl"))
        files.update(OLD.glob("calibration/fold_*/*_blind_scores.pkl"))
        files.update(OLD.glob("calibration/fold_*/operating_points.json"))
        for dataset in projects:
            files.update(
                (
                    BASE / "runs" / dataset / "full_annotation.pkl",
                    BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                )
            )
            for name in (
                "automatic_annotations.csv",
                "candidates.pkl",
                "scoring_audit.pkl",
                "complete.json",
            ):
                files.add(OLD / "runs" / dataset / name)
            receipt_path = PREPARED / "checkpoints" / (dataset + ".json")
            files.add(receipt_path)
            for path, digest in json.loads(receipt_path.read_text())[
                "prepared_sha256"
            ].items():
                path = Path(path)
                assert sha(path) == digest
                files.update(
                    (path, PREPARED / "measurements" / path.parent.name / path.name)
                )
        protocol = dict(
            name="038-identity-preserving-conditional-original-peak-refinement",
            baseline=str(BASE),
            previous=str(OLD),
            production_enabled=False,
            curation_runtime_input=False,
            numerical_workers=3,
            classifier=json.loads((QE / "experiment_007/protocol.json").read_text())[
                "classifier"
            ],
            confidence_grid=CONFIDENCES,
            ambiguity_margin=0.05,
            minimum_pair_rows=100,
            minimum_contributing_projects=3,
            minimum_class_projects=3,
            acceptance_policy="WHOLE029accepted lipid/adduct/class/standard list FIXED;allidentitycounts/extras MUSTstayEXACT. Onlyeligibleendogenousoriginalfeaturesmaymove. No037weights,candidates,newmass/RTmodels,outerclasswinnerchoices ornewannotations.",
            pairing_policy="Same108original029predictors ->320conditionaldifference/midpoint/missingness/categoricalsidefields. AuthorizedTRAININGsameidentitycorrect-vs-knownwrongphysicalpeaks ONLY,bothorientations,identity-balanced andequalactualprojectweights. No names/FIDs/manualflags as predictors. Sparsemode/class support fallsbacktooldpeak;actualsupervisedcontributors separate from referencepermissions.",
            threshold_policy="NEW INNERblind conditionalpreferences. PerMODEconfidencegrid .5,.6,.7,.8,.9,.95,1(disable). Maximize meanclasscoverage capped90,thenmodecorrect,thenhighercutoff. EligibleTRAININGclass/modes>=100endomanual,>=3projects. No eligibleclass/modecorrectloss,no totalmodecorrectloss orwrongincrease. No-movesalwaysfeasible. Originalcalibrationtruth unchanged;noouterlabels choosecuts.",
            protection_policy="Onlyoldcorephysicalcandidates withsameidentity/mass,raw<3ppm,existingdualRT<=10s. Neverstandard/namedIS,initiallyoccupiedfeature,independentlymeasuredspikeFID,declaredspikemasswithin5.01ppm,orunsupportedclass/mode. No chain/swaps/vacatedFIDreuse. Ambiguouschallengers/freefeatureconflictswithin.05withheld. Originalrawsamplevalues/bounds copiedbynewFID;oldpeak-onlymetadata cleared.",
            calibration_boundary="INNERbasecoreacceptance fromEXACT029INNERscores/outercalibratedcutoffs,not aninventedINNERfull005/027pipeline. Protectmatchinginner/queryexcludeddeclaredspikemeasurements/masses. Outer-only027masspools/005outerfits NEVERenterINNERcalibration. OUTERusesWHOLE029finalacceptedlistwithallSTD/massoccupiedfeatures;additionaloccupancycanconservativelyblockmoves. No rerunacceptance/STDcleanupafterrefinement.",
            nested_policy="Same5groupedblocks/30projects.25conditionalparents.ALLscoreTRAININGqueryk referencesexcludeown/outer/inner;INNERVAlpredictionsreferencesexcludeouter/inner. Manualtraininglabels openedONLYafterpermissioncheck andphysicalgeneration;querycuration NEVERusedbyruntime. ALLfourmodeexperts fresh180rounds ifsupported. Whole029identitymodels/acceptance reusedexplicitly,not fitted/cached asnewmodels.",
            run_kind="Saved original-feature replay andWHOLE029accepted-identity-preserving refinement,NOTfreshend-to-endproduction. No sourcepeaklistsedited. Samev2manualreference/92102historicalrows;target92100excludesuserX4only,all28unresolvedinprimaryN.",
            validation="ALL33datasets/132modes,groupedheld-out conditionalrankers/newINNERcuts. Priorrepeateddevelopmentandbaseline/vendorreferenceexposure remain;NOTuntouchedexternalvalidation/analyticalFDR/deploymentreadiness.",
            source_snapshot_sha256=sources,
            input_sha256={str(p): sha(p) for p in sorted(files)},
            resource_receipt=configure_frozen_resources(snapshot),
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=projects,
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"] + " " + protocol["validation"],
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for name, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / name) == digest, name
    for name, digest in protocol["input_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    return protocol, projects


def load_predictors(dataset, mask):
    return legacy.load_predictors(dataset, mask)


def with_training_labels(data, *, permitted_groups, excluded_groups):
    return legacy.with_training_labels(
        data, permitted_groups=permitted_groups, excluded_groups=excluded_groups
    )


def fit_score(records, training, excluded, protocol, path):
    from matlab_annotation.conditional_peak_model import fit_conditional_peak_model

    if path.exists():
        model = pd.read_pickle(path)
        assert set(model["training_cohorts"]) == set(training) and set(
            model["excluded_groups"]
        ) == set(excluded)
        return model
    model = fit_conditional_peak_model(
        records,
        training_groups=training,
        excluded_groups=excluded,
        parameters={k: v for k, v in protocol["classifier"].items() if k != "kind"},
        minimum_pair_rows=protocol["minimum_pair_rows"],
        minimum_contributing_projects=protocol["minimum_contributing_projects"],
        minimum_class_projects=protocol["minimum_class_projects"],
    )
    model["training_feature_masks"] = {
        d: v["excluded_folds"] for d, v in records.items()
    }
    model["standard_training_feature_scopes"] = {
        d: dict(
            training=v["standard_reference_training_groups"],
            excluded=v["standard_reference_excluded_groups"],
            reference_sha256=v["standard_reference_sha256"],
        )
        for d, v in records.items()
    }
    base_path = OLD / path.relative_to(OUT)
    base = pd.read_pickle(base_path)
    assert model["original_columns"] == base["columns"]
    assert model["training_feature_masks"] == base["training_feature_masks"]
    assert (
        model["standard_training_feature_scopes"]
        == base["standard_training_feature_scopes"]
    )
    model["base_identity_model_sha256"] = sha(base_path)
    for expert in model["experts"].values():
        if expert["supported"]:
            assert expert["estimator"].n_iter_ == 180
    pd.to_pickle(model, path)
    print(
        "FRESH CONDITIONAL RANKERS",
        path.relative_to(OUT),
        sum(e["supported"] for e in model["experts"].values()),
        "supported modes",
        flush=True,
    )
    return model


def physical_context(dataset, mask):
    from matlab_annotation.declared_standards import declared_standard_targets

    baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    targets = declared_standard_targets(
        OUT / "source_snapshot/reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    name = "exclude_" + "_".join(map(str, sorted(mask)))
    measured = pd.read_pickle(PREPARED / "measurements" / name / (dataset + ".pkl"))
    masses = {
        s: targets.loc[targets.stream.eq(s), "ref_mz"].astype(float).tolist()
        for s in MODES
    }
    features = {
        s: set(measured.loc[measured.stream.eq(s), "feat_feature_id"].astype(str))
        for s in MODES
    }
    originals = {s: v["features"] for s, v in baseline["streams"].items()}
    return features, masses, originals


def base_core(dataset, outer, raw):
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.operating_point import select_operating_candidates

    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    inner = folds[dataset]
    folder = OLD / "calibration" / f"fold_{outer}"
    base = pd.read_pickle(folder / f"inner_block_{inner}_model.pkl")
    scores = predict_mode_evidence_experts(raw, base)
    pd.testing.assert_series_equal(
        scores,
        pd.read_pickle(folder / (dataset + "_blind_scores.pkl")),
        check_exact=True,
    )
    cuts = json.loads((folder / "operating_points.json").read_text())["thresholds"]
    return pd.concat(
        [
            select_operating_candidates(
                p, scores.loc[p.index], threshold=cuts[s], ambiguity_margin=0.05
            )
            for s, p in raw["pool"].groupby("stream", sort=False)
        ],
        ignore_index=True,
    )


def truth_case(manual, accepted):
    from standard_lipidomics.validation import _normalise

    a = accepted.copy()
    a["adduct"] = a.adduct.map(legacy.adduct)
    calls = _normalise(a, manual=False)
    rows = []
    for r in manual.itertuples():
        ident = (r.stream, r.species, r.adduct, bool(r.internal_standard))
        rows.append(
            (ident, str(r.feature_id), bool(r.mapped), (r.stream, r.lipid_class))
        )
    identities = {r[0] for r in rows}
    idkeys = [
        (r.stream, r.species, r.adduct, bool(r.internal_standard))
        for r in calls.itertuples()
    ]
    return dict(
        rows=rows,
        called_identities=idkeys,
        features=calls.feature_id.astype(str).tolist(),
        manual_identities=identities,
    )


def case_counts(case, moves):
    from standard_lipidomics.validation import canonical_feature_id

    fids = list(case["features"])
    for r in moves.itertuples():
        fids[int(r.annotation_index)] = canonical_feature_id(r.new_feature_id)
    selected = {(*ident, fid) for ident, fid in zip(case["called_identities"], fids)}
    chosen_ids = set(case["called_identities"])
    correct, wrong = Counter(), Counter()
    for ident, fid, mapped, cell in case["rows"]:
        same = mapped and (*ident, fid) in selected
        if same:
            correct[cell] += 1
        elif mapped and ident in chosen_ids:
            wrong[cell] += 1
    extras = Counter(
        ident[0]
        for ident in case["called_identities"]
        if ident not in case["manual_identities"]
    )
    return correct, wrong, extras


def call_keys(frame):
    return [
        (str(s), str(n), str(a), bool(std))
        for s, n, a, std in frame[["stream", "name", "adduct", "is_is"]].itertuples(
            index=False, name=None
        )
    ]


def feature_keys(frame):
    assert frame.is_is.isin([True, False]).all()
    return {
        (str(s), str(n), str(a), str(fid), bool(std))
        for s, n, a, fid, std in frame[
            ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        ].itertuples(index=False, name=None)
    }


def updated_candidates(candidate, before, after):
    """Make saved review flags consistent; do not alter candidate measurements."""
    candidate = candidate.copy()
    keys, old_keys = feature_keys(after), feature_keys(before)
    assert call_keys(before) == call_keys(after)
    assert candidate.is_is.isin([True, False]).all()
    records_keys = [
        (str(s), str(n), str(a), str(f), bool(std))
        for s, n, a, f, std in candidate[
            ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        ].itertuples(index=False, name=None)
    ]
    candidate["automatic_accepted"] = [k in keys for k in records_keys]
    changed = pd.Series(
        [k in keys ^ old_keys for k in records_keys], index=candidate.index
    )
    if "automatic_candidate" in candidate:
        candidate.loc[changed, "automatic_candidate"] = ~candidate.loc[
            changed, "automatic_accepted"
        ]
    if "auto_annotation_status" in candidate:
        candidate.loc[
            changed & candidate.automatic_accepted, "auto_annotation_status"
        ] = "conditional_original_peak_refined"
        candidate.loc[
            changed & ~candidate.automatic_accepted, "auto_annotation_status"
        ] = "original_peak_released_by_conditional_refinement"
    assert feature_keys(candidate[candidate.automatic_accepted]) == keys
    assert len(candidate[candidate.automatic_accepted]) == len(after)
    return candidate


def calibrate(protocol, projects, folds, outer):
    from matlab_annotation.conditional_peak_refinement import (
        peak_refinement_preferences,
        select_peak_refinements,
    )

    folder = OUT / "calibration" / f"fold_{outer}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operating_points.json"
    excluded = {d[:7] for d in projects if folds[d] == outer}
    allowed = {d[:7] for d in projects if folds[d] != outer}
    if path.exists():
        record = json.loads(path.read_text())
        assert (
            set(record["training_groups"]) == allowed
            and set(record["outer_excluded"]) == excluded
        )
        return record["thresholds"]
    cases = {}
    manual_counts = Counter()
    support = defaultdict(set)
    for inner in range(5):
        if inner == outer:
            continue
        fitting = [d for d in projects if folds[d] not in {outer, inner}]
        training = {d[:7] for d in fitting}
        forbidden = {d[:7] for d in projects if folds[d] in {outer, inner}}
        records = {}
        for dataset in fitting:
            raw = load_predictors(dataset, {outer, inner, folds[dataset]})
            records[dataset], _, _ = with_training_labels(
                raw, permitted_groups=training, excluded_groups=forbidden
            )
        model = fit_score(
            records,
            training,
            forbidden,
            protocol,
            folder / f"inner_block_{inner}_model.pkl",
        )
        del records
        for dataset in projects:
            if folds[dataset] != inner:
                continue
            raw = load_predictors(dataset, {outer, inner})
            accepted = base_core(dataset, outer, raw)
            protected, masses, _ = physical_context(dataset, {outer, inner})
            preferences = peak_refinement_preferences(
                raw,
                accepted,
                model,
                protected_features_by_stream=protected,
                protected_masses_by_stream=masses,
            )
            # Persist BLIND predictions before opening this calibration truth.
            preferences.to_pickle(folder / (dataset + "_blind_preferences.pkl"))
            accepted.to_pickle(folder / (dataset + "_base_core.pkl"))
            _, manual, _ = with_training_labels(
                raw, permitted_groups=allowed, excluded_groups=excluded
            )
            case = truth_case(manual, accepted)
            case["preferences"] = preferences
            cases[dataset] = case
            for r in manual.itertuples():
                cell = (r.stream, r.lipid_class)
                manual_counts[cell] += 1
                support[cell].add(dataset[:7])
    assert set(cases) == {d for d in projects if folds[d] != outer}
    empty = pd.DataFrame(columns=["annotation_index", "new_feature_id"])
    before_correct, before_wrong, before_extras = Counter(), Counter(), Counter()
    for case in cases.values():
        c, w, e = case_counts(case, empty)
        before_correct.update(c)
        before_wrong.update(w)
        before_extras.update(e)
    common = {
        cell
        for cell, n in manual_counts.items()
        if n >= 100 and len(support[cell]) >= 3
    }
    mode_before = {
        s: sum(n for cell, n in before_correct.items() if cell[0] == s) for s in MODES
    }
    wrong_before = {
        s: sum(n for cell, n in before_wrong.items() if cell[0] == s) for s in MODES
    }
    class_rows, mode_rows = [], []
    for threshold in protocol["confidence_grid"]:
        correct, wrong, extras, moved = Counter(), Counter(), Counter(), Counter()
        for case in cases.values():
            moves, _ = select_peak_refinements(
                case["preferences"],
                {s: threshold for s in MODES},
                ambiguity_margin=protocol["ambiguity_margin"],
            )
            c, w, e = case_counts(case, moves)
            correct.update(c)
            wrong.update(w)
            extras.update(e)
            moved.update(moves.stream.value_counts().to_dict())
        assert extras == before_extras
        class_rows.extend(
            dict(
                stream=s,
                lipid_class=cls,
                threshold=threshold,
                manual=n,
                training_project_groups=len(support[(s, cls)]),
                eligible_priority=(s, cls) in common,
                correct_before=before_correct[(s, cls)],
                correct_after=correct[(s, cls)],
                wrong_before=before_wrong[(s, cls)],
                wrong_after=wrong[(s, cls)],
            )
            for (s, cls), n in manual_counts.items()
        )
        for mode in MODES:
            cells = [cell for cell in sorted(common) if cell[0] == mode]
            after = sum(n for cell, n in correct.items() if cell[0] == mode)
            errors = sum(n for cell, n in wrong.items() if cell[0] == mode)
            no_class_loss = all(correct[cell] >= before_correct[cell] for cell in cells)
            objective = (
                sum(min(0.9, correct[cell] / manual_counts[cell]) for cell in cells)
                / len(cells)
                if cells
                else 0.0
            )
            feasible = (
                no_class_loss
                and after >= mode_before[mode]
                and errors <= wrong_before[mode]
            )
            mode_rows.append(
                dict(
                    stream=mode,
                    threshold=threshold,
                    manual=sum(
                        n for cell, n in manual_counts.items() if cell[0] == mode
                    ),
                    correct_before=mode_before[mode],
                    correct_after=after,
                    wrong_before=wrong_before[mode],
                    wrong_after=errors,
                    extras_before=before_extras[mode],
                    extras_after=extras[mode],
                    moves=moved[mode],
                    priority_class_modes=len(cells),
                    capped90_class_mean=objective,
                    feasible=feasible,
                )
            )
    classes = pd.DataFrame(class_rows)
    modes = pd.DataFrame(mode_rows)
    classes.to_csv(folder / "class_move_calibration.csv", index=False)
    modes.to_csv(folder / "mode_move_calibration.csv", index=False)
    thresholds = {}
    for mode in MODES:
        valid = modes[modes.stream.eq(mode) & modes.feasible]
        assert len(valid) and valid.threshold.eq(1.0).any(), (
            "No-moves must remain feasible"
        )
        chosen = valid.sort_values(
            ["capped90_class_mean", "correct_after", "threshold"],
            ascending=False,
            kind="stable",
        ).iloc[0]
        thresholds[mode] = float(chosen.threshold)
    selected = modes[
        [
            float(t) == thresholds[s]
            for s, t in modes[["stream", "threshold"]].itertuples(
                index=False, name=None
            )
        ]
    ]
    selected.to_csv(folder / "selected_training_points.csv", index=False)
    dump(
        path,
        dict(
            thresholds=thresholds,
            training_groups=sorted(allowed),
            outer_excluded=sorted(excluded),
            class_table_sha256=sha(folder / "class_move_calibration.csv"),
            mode_table_sha256=sha(folder / "mode_move_calibration.csv"),
            note=protocol["threshold_policy"] + " " + protocol["calibration_boundary"],
        ),
    )
    print("INNER CONDITIONAL MOVE CUTS", outer, thresholds, flush=True)
    return thresholds


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.conditional_peak_refinement import (
        peak_refinement_preferences,
        select_peak_refinements,
        apply_peak_refinements,
    )

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects = json.loads((OUT / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    heldout = [d for d in projects if folds[d] == outer]
    if all((OUT / "runs" / d / "complete.json").exists() for d in heldout):
        return outer, "already complete"
    thresholds = calibrate(protocol, projects, folds, outer)
    training = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in heldout}
    records = {}
    for dataset in projects:
        if dataset in heldout:
            continue
        raw = load_predictors(dataset, {outer, folds[dataset]})
        records[dataset], _, _ = with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded
        )
    model = fit_score(
        records, training, excluded, protocol, OUT / f"fold_{outer}_nested_model.pkl"
    )
    del records
    for dataset in heldout:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        data = load_predictors(dataset, {outer})
        before = pd.read_csv(
            OLD / "runs" / dataset / "automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            float_precision="round_trip",
            low_memory=False,
        )
        protected, masses, features = physical_context(dataset, {outer})
        preferences = peak_refinement_preferences(
            data,
            before,
            model,
            protected_features_by_stream=protected,
            protected_masses_by_stream=masses,
        )
        moves, audit = select_peak_refinements(
            preferences, thresholds, ambiguity_margin=protocol["ambiguity_margin"]
        )
        after = apply_peak_refinements(
            before, data["pool"], moves, features_by_stream=features
        )
        assert call_keys(before) == call_keys(after)
        assert (
            before.groupby(["stream", "class"])
            .size()
            .equals(after.groupby(["stream", "class"]).size())
        )
        candidate = updated_candidates(
            pd.read_pickle(OLD / "runs" / dataset / "candidates.pkl"), before, after
        )
        preferences.to_pickle(dest / "blind_preferences.pkl")
        audit.to_pickle(dest / "private_peak_refinement_audit.pkl")
        moves.to_csv(dest / "private_peak_moves.csv", index=False)
        after.to_csv(dest / "automatic_annotations.csv", index=False)
        candidate.to_pickle(dest / "candidates.pkl")
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                fold=outer,
                status="complete",
                production_enabled=False,
                identity_list_exactly_preserved=True,
                thresholds=thresholds,
                moved_peaks=len(moves),
                counts=after.groupby("stream").size().to_dict(),
            ),
        )
        print(
            "IDENTITY-PRESERVING HELDOUT PEAK REFINEMENT",
            dataset,
            len(moves),
            "moves",
            flush=True,
        )
    return outer, "complete"


def main():
    protocol, _ = initialize()
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed([executor.submit(run_fold, f) for f in range(5)]):
            print("CONDITIONAL OUTERFOLD", *future.result(), flush=True)
    print(
        "038COMPLETE;FULL132USER-FACINGTABLE/INTEGRITYREQUIRED BEFORENEXTEXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
