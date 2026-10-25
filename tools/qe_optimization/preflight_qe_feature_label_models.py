"""042no-fit gate:25actualprep/528uses +ALL33disabledfull041exports."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_feature_label_experiment as runner


def prepare_parent(job):
    configure_frozen_resources(REPO)
    from matlab_annotation.feature_label_model import prepare_feature_label_training

    outer, inner, datasets, folds = job
    data, training, excluded = runner.training_data(outer, inner, datasets, folds)
    records, audit, original_columns, feature_columns = prepare_feature_label_training(
        data, training_groups=training, excluded_groups=excluded
    )
    assert len(original_columns) == 108 and len(feature_columns) == 315
    assert set(audit) == set(data)
    construction = pd.read_csv(
        runner.CONSTRUCTION / "by_dataset_mask_mode.csv", dtype={"mask": str}
    )
    cases = []
    for d, raw in data.items():
        mask = "_".join(map(str, sorted(raw["excluded_folds"])))
        match = construction.loc[
            construction.dataset.eq(d) & construction["mask"].eq(mask)
        ].set_index("stream")
        assert len(match) == 4
        for stream in runner.MODES:
            actual = audit[d]["modes"][stream]
            reference = match.loc[stream]
            assert actual == dict(
                feature_known=int(reference.known_features),
                feature_positive=int(reference.positive_features),
                conditional_rows=int(reference.conditional_rows),
                conditional_positive=int(reference.conditional_positive_rows),
            )
        cases.append(d)
    support = {}
    for stream, r in records.items():
        totals = {}
        for prefix, kind in (("detector", "feature"), ("label", "conditional")):
            n = len(r[prefix + "_y"])
            positive = sum(r[prefix + "_y"])
            assert set(r[prefix + "_groups"]).issubset(training)
            assert not set(r[prefix + "_groups"]) & excluded
            rows = pd.concat(r[prefix + "_x"], ignore_index=True)
            assert len(rows) == n
            assert n == sum(
                v["modes"][stream][
                    "feature_known" if kind == "feature" else "conditional_rows"
                ]
                for v in audit.values()
            )
            assert positive == sum(
                v["modes"][stream][
                    "feature_positive" if kind == "feature" else "conditional_positive"
                ]
                for v in audit.values()
            )
            totals[prefix] = dict(
                observations=n,
                positives=int(positive),
                negatives=n - int(positive),
                actual_projects=sorted(set(r[prefix + "_groups"])),
            )
        # Independently reconstruct positive class/project support using the
        # already verified original labels, with one physical label vote.
        counts, projects, ccounts, cprojects = (
            Counter(),
            defaultdict(set),
            Counter(),
            defaultdict(set),
        )
        from matlab_annotation.feature_label_examples import feature_training_targets
        from standard_lipidomics.validation import _normalise
        from matlab_annotation.standard_rt_library import ion_key
        from annotator.composition_ranges import canonical_lipid_class

        for d, raw in data.items():
            pool = raw["pool"]
            _, cond = feature_training_targets(pool)
            normalized = _normalise(pool, manual=False)
            normalized.index = pool.index
            normalized["adduct"] = normalized.adduct.map(ion_key)
            normalized["lipid_class"] = normalized.lipid_class.map(
                canonical_lipid_class
            )
            for selected, ct, pr in (
                (pool.stream.eq(stream) & pool.training_positive, counts, projects),
                (
                    pool.stream.eq(stream) & cond & pool.training_positive,
                    ccounts,
                    cprojects,
                ),
            ):
                unique = normalized.loc[
                    selected & ~normalized.internal_standard
                ].drop_duplicates(["stream", "feature_id", "species", "adduct"])
                for c, n in unique.lipid_class.value_counts().items():
                    ct[c] += int(n)
                    pr[c].add(d[:7])
        assert r["class_counts"] == counts and dict(r["class_projects"]) == dict(
            projects
        )
        assert r["conditional_counts"] == ccounts and dict(
            r["conditional_projects"]
        ) == dict(cprojects)
        totals.update(
            class_counts=dict(counts),
            class_projects={k: sorted(v) for k, v in projects.items()},
            conditional_class_counts=dict(ccounts),
            conditional_class_projects={k: sorted(v) for k, v in cprojects.items()},
        )
        support[stream] = totals
    print("042 NOFIT PARENT PREPARATION PASS", outer, inner, len(cases), flush=True)
    return dict(outer=outer, inner=inner, training_queries=len(cases), support=support)


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
    datasets, folds = runner.inventory()
    jobs = [
        (o, i, datasets, folds)
        for o in range(5)
        for i in [None, *sorted(set(folds.values()) - {o})]
    ]
    assert not runner.PREFLIGHT.exists(), "Preserve old receipts/version new attempt"
    with ProcessPoolExecutor(max_workers=3) as executor:
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
            construction_unique_tables=330,
            source_input_sha256=hashes,
            parent_support_sha256=runner.sha(runner.PREFLIGHT / "parent_support.json"),
            note="Complete no-fit parent/preparation/support/constructor and33disabledENTIRE041export gate. Not annotation performance; no scores or exampleclassifier fits.",
        ),
    )
    print("042 FULLMODEL/EXPORT PREFLIGHT PASS", flush=True)


if __name__ == "__main__":
    main()
