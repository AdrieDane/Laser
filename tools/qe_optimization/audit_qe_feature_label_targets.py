"""No-fit construction audit, not the full042run authorization or accuracy test."""

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_add_only_class_experiment as common

OUT = common.QE / "feature_label_target_audit_v2"


def check_case(job):
    configure_frozen_resources(REPO)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    from matlab_annotation.feature_label_examples import (
        feature_evidence,
        feature_training_targets,
    )
    from matlab_annotation.evidence_training import validate_feature_scope
    from matlab_annotation.standard_rt_library import ion_key
    from standard_lipidomics.validation import _normalise

    dataset, mask, datasets, folds = job
    blocked = set(mask) - {folds[dataset]}
    training = {d[:7] for d in datasets if folds[d] not in blocked}
    excluded = {d[:7] for d in datasets if folds[d] in blocked}
    raw = common.common.load_predictors(dataset, mask)
    validate_feature_scope(raw, training, excluded, query_is_training=True)
    assert len(raw["evidence"].columns) == 108
    unlabeled = feature_evidence(raw["pool"], raw["evidence"])
    labeled, manual, _ = common.common.label_allowed_query(
        raw, training, excluded, partial=False
    )
    pool = labeled["pool"]
    pd.testing.assert_frame_equal(
        pool[list(raw["pool"])], raw["pool"], check_exact=True
    )
    pd.testing.assert_frame_equal(
        labeled["evidence"], raw["evidence"], check_exact=True
    )
    pd.testing.assert_frame_equal(
        unlabeled, feature_evidence(pool, labeled["evidence"]), check_exact=True
    )
    target, conditional = feature_training_targets(pool)
    normalized = _normalise(pool, manual=False)
    normalized.index = pool.index
    normalized["adduct"] = normalized.adduct.map(ion_key)
    exact = set(
        manual.loc[
            manual.mapped, ["stream", "species", "adduct", "feature_id"]
        ].itertuples(index=False, name=None)
    )
    unresolved = set(
        manual.loc[~manual.mapped, ["stream", "species", "adduct"]].itertuples(
            index=False, name=None
        )
    )
    keys = list(
        normalized[["stream", "species", "adduct", "feature_id"]].itertuples(
            index=False, name=None
        )
    )
    positive = pd.Series([k in exact for k in keys], index=pool.index)
    known = pd.Series(
        [k[:3] not in unresolved or k in exact for k in keys], index=pool.index
    )
    pd.testing.assert_series_equal(
        positive, pool.training_positive, check_names=False, check_exact=True
    )
    pd.testing.assert_series_equal(
        known, pool.training_label_known, check_names=False, check_exact=True
    )
    # Independent feature-target reconstruction from normalized identities.
    independent = normalized.assign(positive=positive, known=known)
    independent = independent.drop_duplicates(
        ["stream", "feature_id", "species", "adduct"]
    )
    stats = independent.groupby(["stream", "feature_id"], sort=False).agg(
        positive=("positive", "any"),
        all_labels_known=("known", "all"),
        standard=("internal_standard", "any"),
        distinct_labels=("species", "size"),
    )
    stats["known"] = (stats.positive | stats.all_labels_known) & ~stats.standard
    stats["positive"] &= ~stats.standard
    stats.index.names = target.index.names
    pd.testing.assert_frame_equal(target, stats, check_exact=True)
    lookup = stats.reindex(
        pd.MultiIndex.from_frame(normalized[["stream", "feature_id"]])
    )
    expected_conditional = (
        lookup.positive.to_numpy()
        & ~lookup.standard.to_numpy()
        & lookup.distinct_labels.gt(1).to_numpy()
        & known.to_numpy()
    )
    np.testing.assert_array_equal(conditional.to_numpy(), expected_conditional)
    # All numeric aggregate cells independently reconstructed with NumPy, not
    # another call to the pandas aggregation implementation being tested.
    numeric = (
        raw["evidence"]
        .select_dtypes(include=["number", "bool"])
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    groups = normalized.groupby(["stream", "feature_id"], sort=False).indices
    numeric_values = numeric.to_numpy(float)
    expected_values = []
    ordered_keys = list(groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for positions in groups.values():
            matrix = numeric_values[positions]
            expected_values.append(
                np.concatenate(
                    [
                        np.nanmin(matrix, axis=0),
                        np.nanmedian(matrix, axis=0),
                        np.nanmax(matrix, axis=0),
                    ]
                )
            )
    columns = [
        "feature_" + method + "__" + str(c)
        for method in ("min", "median", "max")
        for c in numeric
    ]
    np.testing.assert_array_equal(
        unlabeled.loc[ordered_keys, columns].to_numpy(float),
        np.asarray(expected_values),
    )
    distinct = independent.groupby(["stream", "feature_id"], sort=False)
    for column, expected in (
        ("feature_distinct_label_hypotheses", distinct.size()),
        ("feature_distinct_classes", distinct.lipid_class.nunique()),
        ("feature_distinct_adducts", distinct.adduct.nunique()),
    ):
        np.testing.assert_array_equal(
            unlabeled.loc[list(expected.index), column].to_numpy(), expected.to_numpy()
        )
    result = []
    for stream in common.MODES:
        t = target.loc[target.index.get_level_values("stream") == stream]
        rows = pool.stream.eq(stream)
        result.append(
            dict(
                dataset=dataset,
                mask="_".join(map(str, sorted(mask))),
                stream=stream,
                original_candidates=int(rows.sum()),
                positive_candidates=int(pool.loc[rows, "training_positive"].sum()),
                feature_observations=len(t),
                positive_features=int(t.positive.sum()),
                known_features=int(t.known.sum()),
                unknown_features=int((~t.known).sum()),
                conditional_rows=int((rows & conditional).sum()),
                conditional_positive_rows=int(
                    (rows & conditional & pool.training_positive).sum()
                ),
                numeric_features=len(unlabeled.columns),
            )
        )
    print(
        "FEATURE/LABEL TARGETS +ALLAGGREGATES PASS", dataset, sorted(mask), flush=True
    )
    return result


def main():
    datasets, folds = common.inventory()
    jobs = []
    uses = 0
    for d in datasets:
        own = folds[d]
        for n in (1, 2):
            for parent in combinations(sorted(set(folds.values()) - {own}), n):
                jobs.append((d, {own, *parent}, datasets, folds))
    for outer in range(5):
        for inner in [None, *sorted(set(folds.values()) - {outer})]:
            blocked = {outer} if inner is None else {outer, inner}
            uses += sum(folds[d] not in blocked for d in datasets)
    assert len(jobs) == 330 and uses == 528
    files = {
        Path(__file__),
        Path(common.__file__),
        Path(common.common.__file__),
        Path(__file__).with_name("run_qe_atlas_experiment.py"),
        Path(__file__).with_name("context.py"),
        Path(__file__).with_name("frozen_resources.py"),
        REPO / "docs/validation/next_feature_label_factorization.txt",
        common.OUT / "integrity_receipt.json",
        common.QE / "rt_atlas_development/fold_assignment.json",
    }
    files.update(
        p
        for directory in ("app", "methods", "reference_data")
        for p in (REPO / directory).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    for d, mask, _, _ in jobs:
        files.add(common.common.prepared_path(d, mask))
        files.add(common.BASE / "evaluation_v2" / d / "manual_comparison.pkl")
    hashes = {str(p): common.sha(p) for p in sorted(files)}
    OUT.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(check_case, jobs))
    table = pd.DataFrame([r for part in results for r in part])
    table.to_csv(OUT / "by_dataset_mask_mode.csv", index=False)
    assert len(table) == 1320
    assert all(common.sha(Path(p)) == h for p, h in hashes.items())
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            classifier_fits=0,
            annotation_changes=False,
            unique_training_tables=330,
            training_query_uses=528,
            table_mode_checks=1320,
            independent_manual_key_checks=330,
            independent_numpy_all_numeric_aggregate_checks=330,
            unchanged_original_predictor_checks=330,
            source_input_sha256=hashes,
            note="Construction-only audit, NOT full042fit authorization, annotation performance or deployment approval. Still requires full model/runner/disabled041exports preflight before any example fits. Unique masks/uses overlap, not independent biological observations.",
            engineering_revision="v2 eliminates repeated pandas whole-frame filtering/scalar indexing in independent numeric verification ONLY. Same complete330target/aggregate checks restarted; no v1stdout/partialchecks reused, no model fits or annotation outputs in either version.",
        ),
    )
    print("ALL330 FEATURE/LABEL CONSTRUCTION CHECKS PASS", flush=True)


if __name__ == "__main__":
    main()
