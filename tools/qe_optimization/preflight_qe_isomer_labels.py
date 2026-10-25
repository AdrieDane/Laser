"""039pre-fit gate: independent pair support/weights plus ALL33disabled parity.

No classifiers fitted, new preferences or annotation outputs. Every disabled
full005STD/fixed027mass outcome must match WHOLE029 original accepted keys.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_qe_isomer_label_experiment as runner
from context import REPO
from frozen_resources import configure_frozen_resources

OUT = runner.QE / "isomer_label_preflight_v1"


def independent_support(records, mode):
    from annotator.composition_ranges import canonical_lipid_class

    rows, features, class_projects = Counter(), Counter(), defaultdict(set)
    for data in records.values():
        pool = data["pool"]
        part = pool[pool.stream.eq(mode)].drop_duplicates(
            ["stream", "feat_feature_id", "name", "adduct"], keep="first"
        )
        standards = (
            part.is_is
            | part.name.astype(str).str.startswith("IS: ")
            | part.name.astype(str).str.endswith(" (IS)")
        )
        part = part[~standards]
        part = part.loc[
            part.groupby(["stream", "feat_feature_id", "adduct"])
            .name.transform("size")
            .ge(2)
        ]
        positive_features = set(part.loc[part.training_positive, "feat_feature_id"])
        for _, group in part[part.feat_feature_id.isin(positive_features)].groupby(
            ["stream", "feat_feature_id", "adduct"], sort=False
        ):
            gold = group[group.training_positive]
            if gold.empty:
                continue
            assert len(gold) == 1
            gold = gold.iloc[0]
            if not np.isfinite(float(gold.ref_mz)) or float(gold.ref_mz) <= 0:
                continue
            negatives = group[
                group.training_label_known
                & ~group.training_positive
                & ~group.name.eq(gold["name"])
                & np.isclose(
                    group.ref_mz.to_numpy(float), float(gold.ref_mz), rtol=0, atol=1e-8
                )
            ]
            if negatives.empty:
                continue
            project = data["query_group"]
            rows[project] += 2 * len(negatives)
            features[project] += 1
            for cls in negatives["class"]:
                pair = tuple(
                    sorted(
                        (
                            canonical_lipid_class(gold["class"]),
                            canonical_lipid_class(cls),
                        )
                    )
                )
                class_projects[pair].add(project)
    return rows, features, {p: sorted(g) for p, g in sorted(class_projects.items())}


def verify_pairs(records, training, excluded):
    from matlab_annotation.isomer_label_pairs import isomer_training_pairs
    from matlab_annotation.mode_evidence_experts import _subset

    summary = {}
    for mode in runner.MODES:
        subsets = {d: _subset(v, mode) for d, v in records.items()}
        subsets = {d: v for d, v in subsets.items() if len(v["pool"])}
        expected, features, classes = independent_support(records, mode)
        if not expected:
            summary[mode] = dict(pair_rows=0, supervised_contributing_projects=[])
            continue
        pairs = isomer_training_pairs(
            subsets, training_groups=training, excluded_groups=excluded
        )
        assert pairs["pair_rows_by_project"] == dict(expected)
        assert pairs["features_by_project"] == dict(features)
        assert pairs["class_pair_projects"] == classes
        assert pairs["supervised_contributing_projects"] == sorted(expected)
        assert pairs["training_reference_permissions"] == sorted(training)
        assert pairs["excluded_groups"] == sorted(excluded)
        assert set(expected).issubset(training) and not set(expected) & excluded
        assert pairs["positive"].sum() * 2 == len(pairs["positive"])
        target = len(pairs["positive"]) / len(expected)
        for project in expected:
            weights = pairs["sample_weight"][pairs["groups"] == project]
            np.testing.assert_allclose(weights.sum(), target, rtol=1e-12)
            pos = pairs["positive"][pairs["groups"] == project].astype(bool)
            np.testing.assert_allclose(weights[pos].sum(), target / 2, rtol=1e-12)
        summary[mode] = dict(
            pair_rows=len(pairs["positive"]),
            pair_rows_by_project=dict(expected),
            features_by_project=dict(features),
            class_pair_projects=classes,
            supervised_contributing_projects=sorted(expected),
            columns=list(pairs["evidence"]),
            categories={
                c: {
                    v: i
                    for i, v in enumerate(
                        sorted(pairs["evidence"][c].astype(str).unique())
                    )
                }
                for c in pairs["evidence"]
                if not pd.api.types.is_numeric_dtype(pairs["evidence"][c])
            },
        )
    return summary


def accepted_keys(frame):
    keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    return set(frame[keys].astype(str).itertuples(index=False, name=None))


def main():
    assert json.loads(
        (runner.QE / "experiment_038/integrity_receipt.json").read_text()
    )["all_passed"]
    configure_frozen_resources(REPO)
    source_paths = [
        Path(__file__),
        Path(runner.__file__),
        *[
            p
            for name in ("app", "reference_data", "methods")
            for p in (REPO / name).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
            and p.name != ".DS_Store"
        ],
        Path(__file__).with_name("run_qe_standard_mode_experiment.py"),
        Path(__file__).with_name("run_qe_conditional_peak_experiment.py"),
        Path(__file__).with_name("verify_qe_conditional_peak_experiment.py"),
        Path(__file__).with_name("frozen_resources.py"),
        Path(__file__).with_name("context.py"),
        REPO / "docs/validation/next_isomer_label_experiment.txt",
    ]
    sources = {str(p): runner.sha(p) for p in source_paths}
    old_inputs = json.loads((runner.OLD / "protocol.json").read_text())["input_sha256"]
    inputs = dict(old_inputs)
    for p, digest in inputs.items():
        assert runner.sha(Path(p)) == digest, p
    model_path = runner.OLD / "calibration/fold_0/inner_block_1_model.pkl"
    inputs[str(model_path)] = runner.sha(model_path)
    base = pd.read_pickle(model_path)
    training, excluded = set(base["training_cohorts"]), set(base["excluded_groups"])
    records = {}
    for dataset, mask in base["training_feature_masks"].items():
        raw = runner.load_predictors(dataset, set(mask))
        records[dataset], _, _ = runner.with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded
        )
        pd.testing.assert_frame_equal(
            records[dataset]["evidence"], raw["evidence"], check_exact=True
        )
        for p in (
            runner.PREPARED
            / "prepared"
            / ("exclude_" + "_".join(map(str, sorted(mask))))
            / (dataset + ".pkl"),
            runner.BASE / "evaluation_v2" / dataset / "manual_comparison.pkl",
        ):
            digest = runner.sha(p)
            assert old_inputs[str(p)] == digest
            inputs[str(p)] = digest
    support = verify_pairs(records, training, excluded)
    del records
    projects = json.loads((runner.BASE / "run_manifest.json").read_text())["datasets"]
    folds_path = runner.QE / "rt_atlas_development/fold_assignment.json"
    folds = json.loads(folds_path.read_text())
    inputs[str(folds_path)] = runner.sha(folds_path)
    from matlab_annotation.isomer_label_resolution import (
        PAIR_COLUMNS,
        isomer_label_preferences,
    )
    from matlab_annotation.isomer_label_model import FORMAT

    empty = pd.DataFrame(columns=PAIR_COLUMNS)
    for dataset in projects:
        outer = folds[dataset]
        cuts_path = runner.OLD / "calibration" / f"fold_{outer}/operating_points.json"
        old_path = runner.OLD / "runs" / dataset / "automatic_annotations.csv"
        for p in (
            cuts_path,
            old_path,
            runner.OLD / f"fold_{outer}_nested_model.pkl",
            runner.OLD / "runs" / dataset / "scoring_audit.pkl",
            runner.STD / "models" / f"fold_{outer}_standard_models.json",
            runner.MASS / "runs" / dataset / "private_mass_candidates.pkl",
        ):
            inputs[str(p)] = runner.sha(p)
        raw = runner.load_predictors(dataset, {outer})
        scores = runner.pointwise_scores(dataset, outer, raw)
        old_scope = pd.read_pickle(runner.OLD / f"fold_{outer}_nested_model.pkl")
        permissions = dict(
            training_reference_permissions=old_scope["training_cohorts"],
            excluded_groups=old_scope["excluded_groups"],
        )
        # Scope-only, ALLexperts explicitlyunsupported: validate all physical
        # pairs/RT/spike guards without fabricating any fitted classifier/score.
        unsupported = dict(
            format=FORMAT,
            original_columns=list(raw["evidence"]),
            experts={s: dict(supported=False, **permissions) for s in runner.MODES},
            minimum_class_pair_projects=3,
            **permissions,
        )
        measured_path = (
            runner.PREPARED / "measurements" / f"exclude_{outer}" / (dataset + ".pkl")
        )
        inputs[str(measured_path)] = runner.sha(measured_path)
        protected, masses = runner.physical_context(dataset, {outer}, snapshot=REPO)
        unscored = isomer_label_preferences(
            raw,
            unsupported,
            protected_features_by_stream=protected,
            protected_masses_by_stream=masses,
        )
        assert unscored.preference.isna().all()
        assert np.isfinite(
            unscored[["left_rt_error_seconds", "right_rt_error_seconds"]].to_numpy(
                float
            )
        ).all()
        cuts = json.loads(cuts_path.read_text())["thresholds"]
        selected, disabled, _ = runner.select_core(
            raw["pool"], scores, empty, {s: 1.0 for s in runner.MODES}, cuts
        )
        pd.testing.assert_series_equal(disabled, scores, check_exact=True)
        annotations, candidate, _, _ = runner.final_pipeline(
            dataset, outer, raw, selected, snapshot=REPO
        )
        old = pd.read_csv(
            old_path,
            dtype={"feat_feature_id": str},
            float_precision="round_trip",
            low_memory=False,
        )
        assert accepted_keys(annotations) == accepted_keys(old), dataset
        assert len(annotations) == len(old)
        assert accepted_keys(candidate[candidate.automatic_accepted]) == accepted_keys(
            old
        )
        print("039 DISABLED FULL005/027 EXACT029PARITY", dataset, flush=True)
    assert len(projects) == 33
    assert all(
        runner.sha(Path(p)) == digest for p, digest in {**sources, **inputs}.items()
    )
    OUT.mkdir(exist_ok=False)
    # Tuple class-pair keys remain PRIVATE pickle, with a JSON-safe public receipt.
    pd.to_pickle(support, OUT / "private_independent_pair_support.pkl")
    runner.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            classifier_fits=0,
            annotation_changes=False,
            training_queries=len(base["training_feature_masks"]),
            independent_pair_weight_class_project_checks=4,
            unscored_physical_pair_RT_spike_construction_checks=33,
            disabled_score_noop_checks=33,
            disabled_full005027_accepted_key_parity_checks=33,
            disabled_candidate_export_parity_checks=33,
            source_sha256=sources,
            input_sha256=inputs,
            support_artifact_sha256=runner.sha(
                OUT / "private_independent_pair_support.pkl"
            ),
            note="Pre-fitengineering ONLY;no fittedclassifiers/scoredpreferences/newannotations. Scope-only unsupportedexperts verifyALL33physicalpair/RT/spikeconstruction;allpreferencesUNKNOWN. Full25parent/528query/132INNER/33OUTER039verification stillmandatory.",
        ),
    )
    print("039 PREFIT PAIR/WEIGHT/ALL33DISABLED FULLPIPELINE PASS", flush=True)


if __name__ == "__main__":
    main()
