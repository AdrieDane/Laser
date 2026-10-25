"""043 no-fit ALL25 parents/528 uses/ALL33 disabled-full041 export gate."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_raw_mass_profile_experiment as runner


def prepare_parent(job):
    configure_frozen_resources(REPO)
    outer, inner, datasets, folds = job
    data, training, excluded = runner.training_data(outer, inner, datasets, folds)
    old_path = runner.CORE / (
        f"fold_{outer}_nested_model.pkl"
        if inner is None
        else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
    )
    old_model = pd.read_pickle(old_path)
    assert set(old_model["training_cohorts"]) == training
    assert set(old_model["excluded_groups"]) == excluded
    support = {}
    for stream in runner.MODES:
        xs, ys, groups = [], [], []
        schema = None
        for d, raw in data.items():
            pool, evidence = raw["pool"], raw["evidence"]
            assert pool.training_positive.isin([False, True]).all()
            assert pool.training_label_known.isin([False, True]).all()
            selected = pool.stream.eq(stream) & pool.training_label_known
            if schema is None:
                schema = list(evidence)
            assert schema == list(evidence) and len(schema) == 131
            assert raw["query_group"] in training and raw["query_group"] not in excluded
            xs.append(evidence.loc[selected])
            ys.extend(pool.loc[selected, "training_positive"].astype(bool))
            groups.extend([d[:7]] * int(selected.sum()))
        x = pd.concat(xs, ignore_index=True)
        assert len(x) == len(ys) == len(groups)
        assert set(groups) == training and not set(groups) & excluded
        categories = {
            c: {v: i for i, v in enumerate(sorted(x[c].astype(str).unique()))}
            for c in x
            if not pd.api.types.is_numeric_dtype(x[c])
        }
        assert all(c in schema[:108] for c in categories)
        assert all(len(v) <= 255 for v in categories.values())
        old = old_model["experts"][stream]
        # Labels, all original rows and project balance are the original029
        # construction, NOT042 factorized training or a new supervision policy.
        assert old["training_pairs"] == len(ys) and old["positives"] == sum(ys)
        assert old["columns"] == schema[:108] and old["categories"] == categories
        counts = Counter(groups)
        weights = np.array([len(groups) / len(counts) / counts[g] for g in groups])
        totals = pd.Series(weights).groupby(pd.Series(groups)).sum()
        np.testing.assert_allclose(totals, len(groups) / len(counts), rtol=1e-12)
        support[stream] = dict(
            observations=len(ys),
            positives=int(sum(ys)),
            negatives=len(ys) - int(sum(ys)),
            actual_projects=sorted(set(groups)),
            project_counts=dict(counts),
            columns=schema,
            categories=categories,
        )
    print(
        "043 NOFIT ALLPARENT ROWS/LABELS/SCHEMA/BALANCE PASS",
        outer,
        inner,
        len(data),
        flush=True,
    )
    return dict(outer=outer, inner=inner, training_queries=len(data), support=support)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier

    hashes = runner.source_inputs()
    configure_frozen_resources(REPO)
    configuration = json.loads((runner.CORE / "protocol.json").read_text())[
        "classifier"
    ]
    parameters = runner.inherited.common.estimator_parameters(configuration)
    constructor = HistGradientBoostingClassifier(**parameters)
    assert all(constructor.get_params()[k] == v for k, v in parameters.items())
    assert parameters["early_stopping"] is False
    datasets, folds = runner.inventory()
    jobs = [
        (o, i, datasets, folds)
        for o in range(5)
        for i in [None, *sorted(set(folds.values()) - {o})]
    ]
    assert not runner.PREFLIGHT.exists(), "Preserve previous gates/version new attempts"
    with ProcessPoolExecutor(max_workers=2) as executor:
        parents = list(executor.map(prepare_parent, jobs))
    assert len(parents) == 25 and sum(p["training_queries"] for p in parents) == 528
    for d in datasets:
        previous = runner.previous_annotations(d)
        old = pd.read_pickle(runner.PREVIOUS / "runs" / d / "candidates.pkl")
        annotations, candidates, _ = runner.assemble(d, previous, old.iloc[:0])
        pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
        pd.testing.assert_frame_equal(candidates, old, check_exact=True)
    assert hashes == runner.source_inputs()
    runner.PREFLIGHT.mkdir(exist_ok=False)
    runner.dump(runner.PREFLIGHT / "parent_support.json", parents)
    runner.dump(
        runner.PREFLIGHT / "receipt.json",
        dict(
            all_passed=True,
            classifier_fits=0,
            parent_preparations=25,
            training_query_uses=528,
            disabled_full_041_datasets=33,
            original_predictors=108,
            raw_predictors=23,
            total_predictors=131,
            source_input_sha256=hashes,
            parent_support_sha256=runner.sha(runner.PREFLIGHT / "parent_support.json"),
            note="Actual25parent constructions/528queryuses, all modeknownrow/positive/category/schema counts unchangedvs029, cohortbalance/unfittedconstructor, ALL33disabledENTIRE041CSV/candidates exact. IndependentALL363construction required. Not performance/deployment; zero fits.",
        ),
    )
    print("043 FULL MODEL/EXPORT PREFLIGHT PASS", flush=True)


if __name__ == "__main__":
    main()
