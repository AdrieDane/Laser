"""039research: conditional isomer label resolution on unchanged029corefeatures.

No production defaults. Original029pointwise classifiers explicitly reused;
fresh conditional label classifiers and new INNERconditional cutoffs only.
"""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil

import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources
import run_qe_standard_mode_experiment as legacy
from run_qe_conditional_peak_experiment import truth_case, case_counts

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT, OLD, BASE = QE / "experiment_039", QE / "experiment_029", QE / "baseline"
PREPARED, STD, MASS = legacy.PREPARED, legacy.STD, legacy.MASS
sha, dump = legacy.sha, legacy.dump
MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
CONFIDENCES = [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def initialize():
    assert json.loads((QE / "experiment_038/integrity_receipt.json").read_text())[
        "all_passed"
    ]
    assert json.loads((OLD / "integrity_receipt.json").read_text())["all_passed"]
    preflight_path = QE / "isomer_label_preflight_v1/receipt.json"
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
        # Inherit and RECHECK all original029inputs, not just model filenames.
        old_protocol = json.loads((OLD / "protocol.json").read_text())
        files = {Path(p) for p in old_protocol["input_sha256"]}
        for p, digest in old_protocol["input_sha256"].items():
            assert sha(Path(p)) == digest, p
        files.update(
            {
                OLD / "protocol.json",
                OLD / "integrity_receipt.json",
                BASE / "run_manifest.json",
                QE / "experiment_038/integrity_receipt.json",
                QE / "experiment_007/protocol.json",
                preflight_path,
                QE / "rt_atlas_development/fold_assignment.json",
                PREPARED / "summary.json",
                PREPARED / "protocol.json",
                REPO / "docs/validation/next_isomer_label_experiment.txt",
            }
        )
        files.update(OLD.glob("fold_*_nested_model.pkl"))
        files.update(OLD.glob("calibration/fold_*/inner_block_*_model.pkl"))
        files.update(OLD.glob("calibration/fold_*/*_blind_scores.pkl"))
        files.update(OLD.glob("calibration/fold_*/operating_points.json"))
        files.update((QE / "nested_standard_development_v1").glob("exclude_*.json"))
        for dataset in projects:
            files.update(
                (
                    BASE / "runs" / dataset / "full_annotation.pkl",
                    BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
                    MASS / "runs" / dataset / "private_mass_candidates.pkl",
                )
            )
            files.update(
                OLD / "runs" / dataset / n
                for n in (
                    "automatic_annotations.csv",
                    "candidates.pkl",
                    "scoring_audit.pkl",
                    "complete.json",
                )
            )
            receipt = PREPARED / "checkpoints" / (dataset + ".json")
            files.add(receipt)
            for p, digest in json.loads(receipt.read_text())["prepared_sha256"].items():
                p = Path(p)
                assert sha(p) == digest
                files.update((p, PREPARED / "measurements" / p.parent.name / p.name))
        files.update(STD.glob("models/fold_*_standard_models.json"))
        for n in (
            "run_qe_standard_mode_experiment.py",
            "run_qe_atlas_experiment.py",
            "run_qe_conditional_peak_experiment.py",
            "context.py",
            "frozen_resources.py",
            "preflight_qe_isomer_labels.py",
            "verify_qe_conditional_peak_experiment.py",
        ):
            files.add(Path(__file__).with_name(n))
        protocol = dict(
            name="039-conditional-same-feature-isomer-label-resolution",
            baseline=str(BASE),
            previous=str(OLD),
            production_enabled=False,
            curation_runtime_input=False,
            numerical_workers=3,
            classifier=json.loads((QE / "experiment_007/protocol.json").read_text())[
                "classifier"
            ],
            confidence_grid=CONFIDENCES,
            minimum_pair_rows=100,
            minimum_contributing_projects=3,
            minimum_class_pair_projects=3,
            minimum_rt_advantage_seconds=1.0,
            planned_integrity_tool_sha256=sha(
                Path(__file__).with_name("verify_qe_isomer_label_experiment.py")
            ),
            pairing_policy="Same029strictcorephysicalrows/108predictors. AuthorizedTRAININGmappedlabel versusKNOWNwrongisomerlabel onSAMEoriginalfeature/adduct/theoreticalmass1e-8Da ONLY. Bothorientations,featuretotal1 thenequalactualprojecttotals. No pseudo negatives for entirelymanual-absentfeatures. All4conditionalexperts fresh180rounds ifsupported;actualcontributors separatefullreferencepermissions. Unorderedchemicalclasspairs need>=3actualtrainingprojects,elseunknown.",
            score_policy="Original029pointwiseclassifiers deliberately reused/hashbound. COMPLETE supportedisomercluster only:unique labelmustbeatEVERYalternative bylearnedconditionalcut;otherwiseWHOLEcluster unchanged. UseMAXoriginalpointwise score acrossALLoriginalreferenceversions WITHINcluster asfeature-detectionPROXYforwinner,suppressotherclusterlabels. Keeporiginalscores/conditionalpreferences separate. Nonisomer/adductcompetitors untouched,globalonefeature/oneidentityselection/.05margin unchanged. Proxy/pairpreference NOTanalyticalidentityprobability/FDR.",
            physical_policy="No newcandidates/formulas/ranges/massorRTmodels. Strictoriginalraw<3ppm anddualRT<=10s;STD/namedIS/declaredspikemass5.01ppm/independentlymeasuredspikefeatures immutable. Completecliquemassclusters;inconsistentduplicate mass/role/noncliquemasschains/unknownpairswithheld. No querycurationusedbyruntime;physicalmeasurements/quantification unchanged.",
            calibration_boundary="NEWINNERblindisomerpreferences,old029INNERpointwisescores EXACT andoldoutercalibratedmodeacceptancecutoffs fixed. No outer-only005STDfits/027masspools usedasinnerexcluded. Guard/cutcalibrationENDOcoreonly,notinventedINNERfullpipeline. OUTERnewcore thenEXACT005STDcleanup/fixed027massstage. Extraouterpipeline interactions reportedinfullbenchmark.",
            threshold_policy="PerMODEgrid[.55,.6,.7,.8,.9,.95,1disable]. Maximize commonclassmean capped90,thentotalcorrect,thenfewerwrong/extras,thenhighercut. CommonTRAININGclass/modes>=100endomanual/3projects. No totalMODEcorrectloss,wrongincrease or extrasincrease versus029core. No individualclasszerolossconstraint;allregressionsvisible. Disable1 EXACT029core andalwaysfeasible. Noouterlabelsselectcuts.",
            nested_policy="Same5projectblocks/30groups;25freshconditionalparents. TRAININGqueryk referencesexcludeown/outer/inner,validationqueryreferencesexcludeouter/inner. LabelsopenedONLYafterauthorization/physicalgeneration;allpredictors label-free. No cohort/FID/name/manualflags as predictors. Actualclasspairsupport recomputedINSIDEeachtrainingfit. No class/modewinner splicefromrejected037/038outputs.",
            standard_policy="EXACT029outer-excluded005STDadjudication/models andfixed027masscandidate/rules. No newstandard/massfits orquerycuratedanchors. Originalrawcoordinates retained forALLannotations.",
            run_kind="Saved original-feature candidate replay;25newconditionalisomermodels andnewINNERconditionalcuts;old029pointwiseidentitymodels/acceptancecutoffs deliberatelyreused,NOTclaimedfresh. Full005/027outerpipeline rerun. NOTfreshend-to-endproductioninputs.",
            validation="ALL33activefolders/132modes,groupedheldout. Samev2manual92102historical/92100targetonlyX4excluded,all28unresolvedprimaryN. Repeateddevelopment/vendor/baselineexposure remains;NOTuntouchedexternalvalidation/FDR/deploymentready. Wholealgorithmretainonlyifmorecorrect,noextras/wrongincrease,bettercapped90prioritymean,andnoincreaseprioritybelow90count.",
            source_snapshot_sha256={
                p.relative_to(snapshot).as_posix(): sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
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
    for p, digest in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / p) == digest, p
    for p, digest in protocol["input_sha256"].items():
        assert sha(Path(p)) == digest, p
    assert (
        configure_frozen_resources(OUT / "source_snapshot")
        == protocol["resource_receipt"]
    )
    return protocol, projects


load_predictors, with_training_labels = (
    legacy.load_predictors,
    legacy.with_training_labels,
)


def fit_score(records, training, excluded, protocol, path):
    from matlab_annotation.isomer_label_model import fit_isomer_label_model

    if path.exists():
        model = pd.read_pickle(path)
        assert set(model["training_reference_permissions"]) == set(training) and set(
            model["excluded_groups"]
        ) == set(excluded)
        return model
    model = fit_isomer_label_model(
        records,
        training_groups=training,
        excluded_groups=excluded,
        parameters={k: v for k, v in protocol["classifier"].items() if k != "kind"},
        minimum_pair_rows=protocol["minimum_pair_rows"],
        minimum_contributing_projects=protocol["minimum_contributing_projects"],
        minimum_class_pair_projects=protocol["minimum_class_pair_projects"],
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
    for key in ("training_feature_masks", "standard_training_feature_scopes"):
        assert model[key] == base[key]
    model["base_pointwise_model_sha256"] = sha(base_path)
    for expert in model["experts"].values():
        if expert["supported"]:
            assert expert["estimator"].n_iter_ == 180
    pd.to_pickle(model, path)
    print(
        "FRESH CONDITIONAL ISOMER MODELS",
        path.relative_to(OUT),
        sum(e["supported"] for e in model["experts"].values()),
        "supported modes",
        flush=True,
    )
    return model


def physical_context(dataset, mask, *, snapshot=None):
    from matlab_annotation.declared_standards import declared_standard_targets

    baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    snapshot = OUT / "source_snapshot" if snapshot is None else Path(snapshot)
    catalogue = declared_standard_targets(
        snapshot / "reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    measurements = pd.read_pickle(
        PREPARED
        / "measurements"
        / ("exclude_" + "_".join(map(str, sorted(mask))))
        / (dataset + ".pkl")
    )
    masses = {
        s: catalogue.loc[catalogue.stream.eq(s), "ref_mz"].astype(float).tolist()
        for s in MODES
    }
    features = {
        s: set(
            measurements.loc[measurements.stream.eq(s), "feat_feature_id"].astype(str)
        )
        for s in MODES
    }
    return features, masses


def pointwise_scores(dataset, outer, raw, *, inner=None):
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts

    folder = OLD if inner is None else OLD / "calibration" / f"fold_{outer}"
    model_path = folder / (
        f"fold_{outer}_nested_model.pkl"
        if inner is None
        else f"inner_block_{inner}_model.pkl"
    )
    scores = predict_mode_evidence_experts(raw, pd.read_pickle(model_path))
    recorded = (
        OLD / "runs" / dataset / "scoring_audit.pkl"
        if inner is None
        else folder / (dataset + "_blind_scores.pkl")
    )
    pd.testing.assert_series_equal(scores, pd.read_pickle(recorded), check_exact=True)
    return scores


def select_core(pool, scores, preferences, thresholds, base_thresholds):
    from matlab_annotation.isomer_label_resolution import resolve_isomer_scores
    from matlab_annotation.operating_point import select_operating_candidates

    modified, audit = resolve_isomer_scores(pool, scores, preferences, thresholds)
    parts = []
    for stream, local in pool.groupby("stream", sort=False):
        selected = select_operating_candidates(
            local,
            modified.loc[local.index],
            threshold=base_thresholds[stream],
            ambiguity_margin=0.05,
        )
        selected["operating_threshold"] = base_thresholds[stream]
        selected["original_pointwise_class_selection_score"] = scores.loc[
            selected.index
        ]
        selected["isomer_feature_detection_proxy"] = modified.loc[selected.index]
        selected["isomer_label_preference"] = float("nan")
        selected["isomer_rt_advantage_seconds"] = float("nan")
        selected["isomer_label_operating_threshold"] = thresholds[stream]
        if len(audit):
            for r in audit[
                audit.decision.eq("resolved_isomer_label_detection_proxy")
                & audit.stream.eq(stream)
            ].itertuples():
                winner = pool.loc[r.winner_index]
                hit = (
                    selected.feat_feature_id.astype(str).eq(str(winner.feat_feature_id))
                    & selected.name.eq(winner["name"])
                    & selected.adduct.eq(winner.adduct)
                )
                selected.loc[hit, "isomer_label_preference"] = (
                    r.conditional_min_preference
                )
                selected.loc[hit, "isomer_rt_advantage_seconds"] = (
                    r.minimum_rt_advantage_seconds
                )
                selected.loc[hit, "auto_annotation_status"] = (
                    "isomer_resolved_feature_detection_proxy"
                )
        parts.append(selected)
    return pd.concat(parts, ignore_index=True), modified, audit


def calibrate(protocol, projects, folds, outer):
    from matlab_annotation.isomer_label_resolution import isomer_label_preferences

    folder = OUT / "calibration" / f"fold_{outer}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operating_points.json"
    allowed = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in projects if folds[d] == outer}
    if path.exists():
        saved = json.loads(path.read_text())
        assert (
            set(saved["training_groups"]) == allowed
            and set(saved["outer_excluded"]) == excluded
        )
        return saved["thresholds"]
    base_thresholds = json.loads(
        (OLD / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    cases, totals, support = {}, Counter(), defaultdict(set)
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
            scores = pointwise_scores(dataset, outer, raw, inner=inner)
            protected, masses = physical_context(dataset, {outer, inner})
            preferences = isomer_label_preferences(
                raw,
                model,
                protected_features_by_stream=protected,
                protected_masses_by_stream=masses,
            )
            # Persist blind preferences/scores BEFORE opening calibration truth.
            preferences.to_pickle(folder / (dataset + "_blind_preferences.pkl"))
            scores.to_pickle(folder / (dataset + "_pointwise_scores.pkl"))
            _, manual, _ = with_training_labels(
                raw, permitted_groups=allowed, excluded_groups=excluded
            )
            cases[dataset] = (raw["pool"], scores, preferences, manual)
            for r in manual.itertuples():
                cell = (r.stream, r.lipid_class)
                totals[cell] += 1
                support[cell].add(dataset[:7])
    empty = pd.DataFrame(columns=["annotation_index", "new_feature_id"])
    baseline_correct, baseline_wrong, baseline_extras = Counter(), Counter(), Counter()
    for pool, scores, preferences, manual in cases.values():
        selected, disabled, _ = select_core(
            pool, scores, preferences, {s: 1.0 for s in MODES}, base_thresholds
        )
        pd.testing.assert_series_equal(disabled, scores, check_exact=True)
        c, w, e = case_counts(truth_case(manual, selected), empty)
        baseline_correct.update(c)
        baseline_wrong.update(w)
        baseline_extras.update(e)
    common = {c for c, n in totals.items() if n >= 100 and len(support[c]) >= 3}
    class_rows, mode_rows = [], []
    for threshold in protocol["confidence_grid"]:
        correct, wrong, extras, resolved = Counter(), Counter(), Counter(), Counter()
        for pool, scores, preferences, manual in cases.values():
            selected, _, audit = select_core(
                pool,
                scores,
                preferences,
                {s: threshold for s in MODES},
                base_thresholds,
            )
            c, w, e = case_counts(truth_case(manual, selected), empty)
            correct.update(c)
            wrong.update(w)
            extras.update(e)
            if len(audit):
                resolved.update(
                    audit[audit.decision.eq("resolved_isomer_label_detection_proxy")]
                    .stream.value_counts()
                    .to_dict()
                )
        class_rows.extend(
            dict(
                stream=s,
                lipid_class=cls,
                threshold=threshold,
                manual=n,
                training_project_groups=len(support[(s, cls)]),
                eligible_priority=(s, cls) in common,
                correct_before=baseline_correct[(s, cls)],
                correct_after=correct[(s, cls)],
                wrong_before=baseline_wrong[(s, cls)],
                wrong_after=wrong[(s, cls)],
            )
            for (s, cls), n in totals.items()
        )
        for mode in MODES:
            cells = [c for c in sorted(common) if c[0] == mode]
            cb = sum(n for c, n in baseline_correct.items() if c[0] == mode)
            wb = sum(n for c, n in baseline_wrong.items() if c[0] == mode)
            ca = sum(n for c, n in correct.items() if c[0] == mode)
            wa = sum(n for c, n in wrong.items() if c[0] == mode)
            objective = (
                sum(min(0.9, correct[c] / totals[c]) for c in cells) / len(cells)
                if cells
                else 0.0
            )
            feasible = ca >= cb and wa <= wb and extras[mode] <= baseline_extras[mode]
            mode_rows.append(
                dict(
                    stream=mode,
                    threshold=threshold,
                    manual=sum(n for c, n in totals.items() if c[0] == mode),
                    correct_before=cb,
                    correct_after=ca,
                    wrong_before=wb,
                    wrong_after=wa,
                    extras_before=baseline_extras[mode],
                    extras_after=extras[mode],
                    resolved_clusters=resolved[mode],
                    priority_class_modes=len(cells),
                    capped90_class_mean=objective,
                    feasible=feasible,
                )
            )
    classes, modes = pd.DataFrame(class_rows), pd.DataFrame(mode_rows)
    classes.to_csv(folder / "class_isomer_calibration.csv", index=False)
    modes.to_csv(folder / "mode_isomer_calibration.csv", index=False)
    thresholds = {}
    for mode in MODES:
        valid = modes[modes.stream.eq(mode) & modes.feasible]
        assert len(valid) and valid.threshold.eq(1.0).any()
        chosen = valid.sort_values(
            [
                "capped90_class_mean",
                "correct_after",
                "wrong_after",
                "extras_after",
                "threshold",
            ],
            ascending=[False, False, True, True, False],
            kind="stable",
        ).iloc[0]
        thresholds[mode] = float(chosen.threshold)
    dump(
        path,
        dict(
            thresholds=thresholds,
            base_thresholds=base_thresholds,
            training_groups=sorted(allowed),
            outer_excluded=sorted(excluded),
            class_table_sha256=sha(folder / "class_isomer_calibration.csv"),
            mode_table_sha256=sha(folder / "mode_isomer_calibration.csv"),
            note=protocol["threshold_policy"] + " " + protocol["calibration_boundary"],
        ),
    )
    modes[
        [
            t == thresholds[s]
            for s, t in modes[["stream", "threshold"]].itertuples(
                index=False, name=None
            )
        ]
    ].to_csv(folder / "selected_training_points.csv", index=False)
    print("INNER ISOMER CUTS", outer, thresholds, flush=True)
    return thresholds


def final_pipeline(dataset, outer, raw, selected, *, snapshot=None):
    """Same005standard/fixed027mass stages, no inner-resource pretending."""
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

    baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    snapshot = OUT / "source_snapshot" if snapshot is None else Path(snapshot)
    catalogue = declared_standard_targets(
        snapshot / "reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    standards = json.loads(
        (STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    excluded = {d[:7] for d, f in folds.items() if f == outer}
    assert not set(standards["training_groups"]) & excluded
    mass_pool = pd.read_pickle(MASS / "runs" / dataset / "private_mass_candidates.pkl")
    assert not {"training_positive", "training_label_known", "same_manual_peak"} & set(
        raw["pool"]
    )
    assert not {"training_positive", "training_label_known", "same_manual_peak"} & set(
        mass_pool
    )
    automatic = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    parts, candidates, audits, mass_audits = [], [raw["pool"], mass_pool], [], []
    for stream, payload in baseline["streams"].items():
        core = selected[selected.stream.eq(stream)]
        protected = (
            payload["annotations"][
                payload["annotations"].is_is.fillna(False).astype(bool)
            ]
            .copy()
            .assign(stream=stream)
        )
        combined = pd.concat([core, protected], ignore_index=True)
        assert not combined.feat_feature_id.astype(str).duplicated().any()
        corrected, extra, audit = complete_declared_standards(
            combined,
            payload["features"],
            catalogue[catalogue.stream.eq(stream)],
            standards["library"],
            standards["atlas"],
            standards["transfer"],
            standards["channels"],
            stream=stream,
            alignment_annotations=payload["annotations"],
        )
        local_mass = (
            mass_pool[mass_pool.stream.eq(stream)] if len(mass_pool) else mass_pool
        )
        added = select_calibrated_additions(
            local_mass, automatic, corrected, stream=stream
        )
        final = pd.concat([corrected, added], ignore_index=True)
        assert not final.feat_feature_id.astype(str).duplicated().any()
        parts.append(final.assign(stream=stream))
        audits.append(audit.assign(dataset=dataset, stream=stream))
        mass_audits.append(
            dict(
                dataset=dataset,
                stream=stream,
                added_annotations=len(added),
                current_core_annotations=len(corrected),
                fixed_027_rule=True,
            )
        )
        candidates.extend(
            [
                protected,
                extra.assign(stream=stream),
                corrected[corrected.is_is.fillna(False).astype(bool)].assign(
                    stream=stream
                ),
            ]
        )
    annotations = pd.concat(parts, ignore_index=True)
    candidates.append(annotations)  # actual accepted score/proxy metadata wins
    candidate = pd.concat(candidates, ignore_index=True)
    keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    candidate = candidate.drop_duplicates(keys, keep="last").reset_index(drop=True)
    accepted = set(annotations[keys].itertuples(index=False, name=None))
    candidate["automatic_accepted"] = [
        k in accepted for k in candidate[keys].itertuples(index=False, name=None)
    ]
    candidate["automatic_candidate"] = ~candidate.automatic_accepted
    return (
        annotations,
        candidate,
        pd.concat(audits, ignore_index=True),
        pd.DataFrame(mass_audits),
    )


def run_fold(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.isomer_label_resolution import isomer_label_preferences

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
    base_thresholds = json.loads(
        (OLD / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    for dataset in heldout:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        raw = load_predictors(dataset, {outer})
        scores = pointwise_scores(dataset, outer, raw)
        protected, masses = physical_context(dataset, {outer})
        preferences = isomer_label_preferences(
            raw,
            model,
            protected_features_by_stream=protected,
            protected_masses_by_stream=masses,
        )
        selected, modified, audit = select_core(
            raw["pool"], scores, preferences, thresholds, base_thresholds
        )
        annotations, candidate, standard_audit, mass_audit = final_pipeline(
            dataset, outer, raw, selected
        )
        annotations.to_csv(dest / "automatic_annotations.csv", index=False)
        candidate.to_pickle(dest / "candidates.pkl")
        scores.to_pickle(dest / "scoring_audit.pkl")
        modified.to_pickle(dest / "adjusted_scoring_audit.pkl")
        preferences.to_pickle(dest / "blind_preferences.pkl")
        audit.to_pickle(dest / "private_isomer_resolution_audit.pkl")
        selected.to_pickle(dest / "private_selected_core.pkl")
        standard_audit.to_csv(dest / "standard_audit.csv", index=False)
        mass_audit.to_csv(dest / "mass_discovery_audit.csv", index=False)
        resolved = (
            int(audit.decision.eq("resolved_isomer_label_detection_proxy").sum())
            if len(audit)
            else 0
        )
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                fold=outer,
                status="complete",
                production_enabled=False,
                thresholds=thresholds,
                base_thresholds=base_thresholds,
                resolved_isomer_clusters=resolved,
                counts=annotations.groupby("stream").size().to_dict(),
            ),
        )
        print(
            "HELDOUT ISOMER LABEL RESOLUTION",
            dataset,
            resolved,
            "resolved clusters",
            flush=True,
        )
    return outer, "complete"


def main():
    protocol, _ = initialize()
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed([executor.submit(run_fold, f) for f in range(5)]):
            print("ISOMER OUTERFOLD", *future.result(), flush=True)
    print(
        "039COMPLETE;FULL132TABLE/INTEGRITY/DECISION REQUIRED BEFORE NEXTEXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
