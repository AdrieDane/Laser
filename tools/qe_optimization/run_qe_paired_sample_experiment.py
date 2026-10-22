"""Experiment019: explicit paired-specimen abundance evidence, exact007 pools."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO, SOURCE

QE = ROOT / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
POOL = QE / "experiment_007"
PREVIOUS = QE / "experiment_009"
STD = QE / "experiment_005"
ATLAS = QE / "rt_atlas_development"
OUT = QE / "experiment_019"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_dual_experiment import model_matrix
from run_qe_atlas_experiment import species, adduct


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(parents=True, exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        hashes = {
            p.relative_to(snapshot).as_posix(): sha(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        dependencies = {}
        for d in projects:
            for p in (
                BASE / "runs" / d / "full_annotation.pkl",
                BASE / "evaluation_v2" / d / "manual_comparison.pkl",
                SOURCE / d / "MS_metadata.mat",
            ):
                dependencies[str(p)] = sha(p)
            for f in range(5):
                p = POOL / "prepared" / f"fold_{f}" / (d + ".pkl")
                dependencies[str(p)] = sha(p)
        for f in range(5):
            p = STD / "models" / f"fold_{f}_standard_models.json"
            dependencies[str(p)] = sha(p)
        for p in (
            ATLAS / "fold_assignment.json",
            POOL / "protocol.json",
            PREVIOUS / "protocol.json",
            Path(__file__).with_name("run_qe_dual_experiment.py"),
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
        ):
            dependencies[str(p)] = sha(p)
        old = json.loads((PREVIOUS / "protocol.json").read_text())
        source = json.loads((POOL / "protocol.json").read_text())
        protocol = dict(
            name="019-explicit-paired-specimen-abundance-evidence",
            baseline=str(BASE),
            previous=str(PREVIOUS),
            production_enabled=False,
            curation_runtime_input=False,
            folds=5,
            inner_folds=3,
            numerical_workers=3,
            candidate_universe="EXACT007 physical pools/old evidence, plus8 explicit paired-specimen abundance descriptors. No new physical candidates, wider mass/RT ranges or chemistry guesses.",
            metadata_policy="Load ONLY MS_metadata acquisition fields; exact filename/phase/mode/shared sample_id and validated group agreement. Conflicting/unsupported dataset maps yield unavailable evidence, not guessed filename offsets. Technical replicate intensities averaged per physical ID before log; biological samples only.",
            evidence_policy="OTHER-stream frozen BASELINE automatic endogenous exact canonical class/C/DB anchors,confidence>=.9/absppm<1.5,disagreeing adduct RTs withheld. >=8 positive paired specimens; variable log profiles. Raw/normalized/within-group Pearson evidence,log-ratio MAD,support/specimen/within-group degrees of freedom. Normalization from raw global per-sample log-intensity median. No NP/RP RT transfer or iterative auto-call feedback.",
            classifier=source["classifier"],
            threshold_grid=[round(v, 2) for v in np.arange(0.2, 1.001, 0.05)],
            ambiguity_margin=0.05,
            extras_budget_per_endogenous_manual=0.095,
            threshold_policy="NEW THREE INNER group-excluded score fits per outer fold for NEW019 evidence/scores. EXACT009 joint weakest-mode coverage objective and pooled9.5% endogenous extras/manual TRAINING budget. No evaluated outer-group labels choose cutoffs. Standards separate/same005. Held-out9697 original extras safeguard is checked,not guaranteed/FDR.",
            standard_policy="EXACT005 declared-standard adjudication/models from matching outer fold, unchanged.",
            validation=old["validation"],
            model_fit_note=old["model_fit_note"]
            + " New final and INNER score fits exclude evaluated outer project groups; paired evidence itself label-free. Inner RT/prior features still inherited from OUTER training atlas,NOT inner RT cross-fitting. Baseline prior exposure; not pristine nested/external validation.",
            run_kind="Saved007 physical candidate replay plus explicit paired-specimen evidence, same180-round global HGB with newly inner-training-calibrated joint cutoffs and005 standards. Not a fresh full production/raw-input run.",
            source_snapshot_sha256=hashes,
            dependencies_sha256=dependencies,
        )
        dump(OUT / "protocol.json", protocol)
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
        for name in ("run_qe_dual_experiment.py", "run_qe_atlas_experiment.py"):
            shutil.copy2(Path(__file__).with_name(name), OUT / (name + ".dependency"))
        dump(
            OUT / "run_manifest.json",
            dict(
                datasets=projects,
                fresh_raw_input_runs=False,
                evaluation_note=protocol["run_kind"]
                + " "
                + protocol["validation"]
                + " "
                + protocol["model_fit_note"],
            ),
        )
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, expected in protocol["source_snapshot_sha256"].items():
        assert sha(OUT / "source_snapshot" / name) == expected, name
    for name, expected in protocol["dependencies_sha256"].items():
        assert sha(Path(name)) == expected, name
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for name in (
        "standard_completion.py",
        "standard_rt_library.py",
        "standard_rt_transfer.py",
        "declared_standards.py",
    ):
        p = Path("source_snapshot/app/matlab_annotation") / name
        assert (OUT / p).read_bytes() == (STD / p).read_bytes(), name
    return protocol, projects


def prepare_dataset(dataset):
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    from scipy.io import loadmat
    from matlab_annotation.specimen_pairs import explicit_specimen_pairs
    from matlab_annotation.paired_sample_evidence import paired_sample_evidence

    keys = ["stream", "class", "carbon", "db", "feat_feature_id"]
    if all(
        (OUT / "prepared" / f"fold_{f}" / (dataset + ".pkl")).exists() for f in range(5)
    ):
        return dataset, "already prepared"
    existing = {
        f: pd.read_pickle(POOL / "prepared" / f"fold_{f}" / (dataset + ".pkl"))
        for f in range(5)
    }
    result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    features = {s: p["features"] for s, p in result["streams"].items()}
    metadata_path = SOURCE / dataset / "MS_metadata.mat"
    before = sha(metadata_path)
    metadata = loadmat(
        metadata_path, variable_names=["MS_metadata"], simplify_cells=True
    )["MS_metadata"]
    try:
        mapping, audit = explicit_specimen_pairs(metadata, features)
        status = "explicit_metadata_links_checked"
        error = ""
    except ValueError as exc:
        mapping, audit = {}, {}
        status = "metadata_conflict_or_unsupported"
        error = str(exc)
    assert sha(metadata_path) == before
    automatic = pd.concat(
        [p["annotations"].assign(stream=s) for s, p in result["streams"].items()],
        ignore_index=True,
    )
    queries = (
        pd.concat([p["pool"][keys] for p in existing.values()], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    extra = paired_sample_evidence(queries, features, automatic, mapping)
    extra.index = pd.MultiIndex.from_frame(queries[keys])
    assert extra.index.is_unique
    folder = OUT / "pairing_audits"
    folder.mkdir(exist_ok=True)
    dump(
        folder / (dataset + ".json"),
        dict(
            dataset=dataset,
            status=status,
            error=error,
            metadata_sha256=before,
            metadata_counts=audit,
            queries=len(queries),
            nonmissing=extra.notna().sum().to_dict(),
            note="No annotation selection or curation reads in preparation. Exact explicit specimen links only.",
        ),
    )
    for fold, data in existing.items():
        dest = OUT / "prepared" / f"fold_{fold}"
        dest.mkdir(parents=True, exist_ok=True)
        pool, evidence = data["pool"].copy(), data["evidence"].copy()
        aligned = extra.reindex(pd.MultiIndex.from_frame(pool[keys]))
        for col in extra:
            assert col not in pool and col not in evidence
            pool[col] = aligned[col].to_numpy()
            evidence[col] = aligned[col].to_numpy()
        pd.testing.assert_frame_equal(pool[data["pool"].columns], data["pool"])
        pd.testing.assert_frame_equal(
            evidence[data["evidence"].columns], data["evidence"]
        )
        assert not {
            "name",
            "feat_feature_id",
            "dataset",
            "cohort",
            "training_positive",
            "training_label_known",
        } & set(evidence)
        pd.to_pickle(dict(pool=pool, evidence=evidence), dest / (dataset + ".pkl"))
    return dataset, dict(
        status=status,
        queries=len(queries),
        supported=int(extra.q_paired_best_log_correlation.notna().sum()),
    )


def fit_score_model(data, train, excluded, protocol, path):
    from sklearn.ensemble import HistGradientBoostingClassifier

    if not path.exists():
        xs, ys, groups = [], [], []
        for d in train:
            known = data[d]["pool"].training_label_known.astype(bool)
            xs.append(data[d]["evidence"].loc[known])
            ys.extend(data[d]["pool"].loc[known, "training_positive"].astype(int))
            groups.extend([d[:7]] * int(known.sum()))
        assert not set(groups) & excluded
        x = pd.concat(xs, ignore_index=True)
        categories = {
            c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
            for c in x
            if not pd.api.types.is_numeric_dtype(x[c])
        }
        assert all(len(v) <= 255 for v in categories.values())
        model = dict(
            columns=list(x),
            categories=categories,
            training_cohorts=sorted(set(groups)),
            excluded_groups=sorted(excluded),
            training_pairs=len(x),
        )
        counts = Counter(groups)
        weights = np.asarray([len(groups) / len(counts) / counts[g] for g in groups])
        params = {k: v for k, v in protocol["classifier"].items() if k != "kind"}
        estimator = HistGradientBoostingClassifier(
            **params, categorical_features=[c in categories for c in x]
        )
        estimator.fit(model_matrix(x, model), np.asarray(ys), sample_weight=weights)
        model["estimator"] = estimator
        pd.to_pickle(model, path)
        print("FIT PAIRED MODEL", path.name, len(x), flush=True)
    model = pd.read_pickle(path)
    assert set(model["training_cohorts"]) == {d[:7] for d in train}
    assert not set(model["training_cohorts"]) & excluded
    assert set(model["excluded_groups"]) == excluded
    return model


def calibrate(protocol, projects, folds, fold, data):
    from sklearn.model_selection import GroupKFold
    from matlab_annotation.operating_point import (
        select_operating_candidates,
        choose_balanced_training_points,
    )

    folder = OUT / "calibration" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "operating_points.json"
    train = [d for d in projects if folds[d] != fold]
    excluded = {d[:7] for d in projects if folds[d] == fold}
    if path.exists():
        saved = json.loads(path.read_text())
        assert set(saved["training_groups"]) == {d[:7] for d in train}
        assert set(saved["outer_excluded"]) == excluded
        return saved["thresholds"]
    predictions = {}
    for inner, (ti, vi) in enumerate(
        GroupKFold(n_splits=3).split(train, groups=[d[:7] for d in train])
    ):
        fitting = [train[i] for i in ti]
        validation = [train[i] for i in vi]
        model = fit_score_model(
            data,
            fitting,
            excluded | {d[:7] for d in validation},
            protocol,
            folder / f"inner_{inner}_score_model.pkl",
        )
        for d in validation:
            predictions[d] = pd.Series(
                model["estimator"].predict_proba(
                    model_matrix(data[d]["evidence"], model)
                )[:, 1],
                index=data[d]["pool"].index,
            )
    assert set(predictions) == set(train)
    counts_manual, cases = Counter(), {}
    for d in train:
        # ONLY outer-TRAINING manual references are opened for threshold choice.
        manual = pd.read_pickle(BASE / "evaluation_v2" / d / "manual_comparison.pkl")
        manual = manual[~manual.internal_standard]
        identities = set(
            manual[["stream", "species", "adduct"]].itertuples(index=False, name=None)
        )
        counts_manual.update(manual.stream.value_counts().to_dict())
        pool = data[d]["pool"].copy()
        pool["calibration_identity_in_manual"] = [
            (s, species(n), adduct(a)) in identities
            for s, n, a in pool[["stream", "name", "adduct"]].itertuples(
                index=False, name=None
            )
        ]
        cases[d] = pool
    records = []
    for threshold in protocol["threshold_grid"]:
        counts = {s: Counter() for s in counts_manual}
        for d, pool in cases.items():
            for stream, part in pool.groupby("stream", sort=False):
                selected = select_operating_candidates(
                    part,
                    predictions[d].loc[part.index],
                    threshold=threshold,
                    ambiguity_margin=0.05,
                )
                counts[stream]["correct"] += int(selected.training_positive.sum())
                counts[stream]["extras"] += int(
                    (~selected.calibration_identity_in_manual).sum()
                )
                counts[stream]["different_peak"] += int(
                    (
                        selected.calibration_identity_in_manual
                        & selected.training_label_known
                        & ~selected.training_positive
                    ).sum()
                )
        records.extend(
            dict(stream=s, threshold=threshold, manual=n, **dict(counts[s]))
            for s, n in counts_manual.items()
        )
    table = pd.DataFrame(records).fillna(0)
    table.to_csv(folder / "training_operating_points.csv", index=False)
    thresholds = choose_balanced_training_points(
        table,
        extras_per_manual=protocol["extras_budget_per_endogenous_manual"],
        target=0.9,
    )
    assert thresholds is not None, (
        "No feasible training cutoff; never silently exceed budget"
    )
    selected = table.loc[
        [
            float(t) == thresholds[s]
            for s, t in table[["stream", "threshold"]].itertuples(
                index=False, name=None
            )
        ]
    ]
    selected.to_csv(folder / "selected_training_points.csv", index=False)
    dump(
        path,
        dict(
            thresholds=thresholds,
            training_groups=sorted({d[:7] for d in train}),
            outer_excluded=sorted(excluded),
            training_operating_points_sha256=sha(
                folder / "training_operating_points.csv"
            ),
            extras_budget_per_endogenous_manual=protocol[
                "extras_budget_per_endogenous_manual"
            ],
            note=protocol["model_fit_note"],
        ),
    )
    print("PAIRED TRAINING-ONLY CUTOFFS", fold, thresholds, flush=True)
    return thresholds


def run_fold(fold):
    sys.path.insert(0, str(OUT / "source_snapshot/app"))
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    heldout = [d for d in projects if folds[d] == fold]
    if all((OUT / "runs" / d / "complete.json").exists() for d in heldout):
        return fold, "already complete"
    data = {
        d: pd.read_pickle(OUT / "prepared" / f"fold_{fold}" / (d + ".pkl"))
        for d in projects
    }
    thresholds = calibrate(protocol, projects, folds, fold, data)
    model = fit_score_model(
        data,
        [d for d in projects if folds[d] != fold],
        {d[:7] for d in heldout},
        protocol,
        OUT / f"fold_{fold}_nonlinear_model.pkl",
    )
    std = json.loads((STD / "models" / f"fold_{fold}_standard_models.json").read_text())
    assert not set(std["training_groups"]) & {d[:7] for d in heldout}
    for dataset in heldout:
        dest = OUT / "runs" / dataset
        if (dest / "complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        raw, evidence = data[dataset]["pool"], data[dataset]["evidence"]
        scores = pd.Series(
            model["estimator"].predict_proba(model_matrix(evidence, model))[:, 1],
            index=raw.index,
        )
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        catalogue = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            result["internal_standards_expected"],
        )
        parts, candidates, audits = (
            [],
            [raw.drop(columns=["training_positive", "training_label_known"])],
            [],
        )
        for stream, payload in result["streams"].items():
            pool = raw[raw.stream.eq(stream)]
            selected = select_operating_candidates(
                pool,
                scores.loc[pool.index],
                threshold=thresholds[stream],
                ambiguity_margin=0.05,
            )
            selected = selected.drop(
                columns=["training_positive", "training_label_known"]
            )
            selected["auto_annotation_status"] = (
                "paired_sample_training_operating_point"
            )
            selected["operating_threshold"] = thresholds[stream]
            protected = (
                payload["annotations"][
                    payload["annotations"].is_is.fillna(False).astype(bool)
                ]
                .copy()
                .assign(stream=stream)
            )
            combined = pd.concat([selected, protected], ignore_index=True)
            assert not combined.feat_feature_id.astype(str).duplicated().any(), (
                dataset,
                stream,
            )
            corrected, extra, audit = complete_declared_standards(
                combined,
                payload["features"],
                catalogue[catalogue.stream.eq(stream)],
                std["library"],
                std["atlas"],
                std["transfer"],
                std["channels"],
                stream=stream,
                alignment_annotations=payload["annotations"],
            )
            parts.append(corrected.assign(stream=stream))
            candidates.extend(
                [
                    protected,
                    extra.assign(stream=stream),
                    corrected[corrected.is_is.fillna(False).astype(bool)].assign(
                        stream=stream
                    ),
                ]
            )
            audits.append(audit.assign(dataset=dataset, stream=stream))
        annotations = pd.concat(parts, ignore_index=True)
        all_candidates = pd.concat(candidates, ignore_index=True)
        keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
        all_candidates = all_candidates.drop_duplicates(keys, keep="last").reset_index(
            drop=True
        )
        chosen = set(annotations[keys].itertuples(index=False, name=None))
        all_candidates["automatic_accepted"] = [
            k in chosen for k in all_candidates[keys].itertuples(index=False, name=None)
        ]
        annotations.to_csv(dest / "automatic_annotations.csv", index=False)
        all_candidates.to_pickle(dest / "candidates.pkl")
        scores.to_pickle(dest / "scoring_audit.pkl")
        pd.concat(audits, ignore_index=True).to_csv(
            dest / "standard_audit.csv", index=False
        )
        dump(
            dest / "complete.json",
            dict(
                dataset=dataset,
                fold=fold,
                status="complete",
                production_enabled=False,
                thresholds=thresholds,
                counts=annotations.groupby("stream").size().to_dict(),
            ),
        )
        print("SELECTED PAIRED HELD-OUT", dataset, len(annotations), flush=True)
    return fold, "complete"


def main():
    protocol, projects = initialize()
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed(
            [executor.submit(prepare_dataset, d) for d in projects]
        ):
            print("PREPARED PAIRED", *future.result(), flush=True)
    with ProcessPoolExecutor(max_workers=protocol["numerical_workers"]) as executor:
        for future in as_completed([executor.submit(run_fold, f) for f in range(5)]):
            print("PAIRED FOLD", *future.result(), flush=True)
    print(
        "EXPERIMENT019 COMPLETE; FULL132-ROW REPORT REQUIRED BEFORE NEXT EXPERIMENT",
        flush=True,
    )


if __name__ == "__main__":
    main()
