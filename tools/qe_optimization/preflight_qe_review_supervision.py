"""No fits: check ALL037training masks, unchanged labels and independent flags."""

from collections import Counter
import json
from pathlib import Path

import pandas as pd

import run_qe_review_tempered_experiment as runner
from context import REPO
from frozen_resources import configure_frozen_resources
from diagnose_qe_negative_supervision import categorize

QE, OLD = runner.QE, runner.OLD
OUT = QE / "review_supervision_preflight_v1"
sha = runner.sha


def main():
    configure_frozen_resources(REPO)
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.evidence_training import validate_feature_scope

    source_paths = [
        Path(__file__),
        Path(runner.__file__),
        Path(__file__).with_name("diagnose_qe_negative_supervision.py"),
        *(
            REPO / "app/matlab_annotation" / n
            for n in (
                "review_aware_supervision.py",
                "evidence_training.py",
                "mode_evidence_experts.py",
            )
        ),
    ]
    sources = {str(p): sha(p) for p in source_paths}
    old_inputs = json.loads((OLD / "protocol.json").read_text())["input_sha256"]
    paths = sorted(
        list(OLD.glob("fold_*_nested_model.pkl"))
        + list(OLD.glob("calibration/fold_*/inner_block_*_model.pkl"))
    )
    assert len(paths) == 25
    OUT.mkdir(exist_ok=False)
    inputs, cache, records = {}, {}, []
    for path in paths:
        inputs[str(path)] = sha(path)
        old = pd.read_pickle(path)
        training, excluded = set(old["training_cohorts"]), set(old["excluded_groups"])
        counts = {m: Counter() for m in old["experts"]}
        for dataset, mask in old["training_feature_masks"].items():
            key = (dataset, tuple(sorted(mask)))
            if key not in cache:
                pp = (
                    runner.PREPARED
                    / "prepared"
                    / ("exclude_" + "_".join(map(str, sorted(mask))))
                    / (dataset + ".pkl")
                )
                mp = runner.BASE / "evaluation_v2" / dataset / "manual_comparison.pkl"
                for p in (pp, mp):
                    digest = sha(p)
                    assert old_inputs[str(p)] == digest
                    inputs[str(p)] = digest
                raw = runner.load_predictors(dataset, set(mask))
                validate_feature_scope(raw, training, excluded, query_is_training=True)
                labeled, _, _ = runner.with_training_labels(
                    raw, permitted_groups=training, excluded_groups=excluded
                )
                original, _, _ = runner.with_training_labels(
                    raw,
                    permitted_groups=training,
                    excluded_groups=excluded,
                    apply_review=False,
                )
                pool = labeled["pool"]
                pd.testing.assert_frame_equal(
                    pool.drop(columns="training_review_uncertain"),
                    original["pool"],
                    check_exact=True,
                )
                pd.testing.assert_frame_equal(
                    labeled["evidence"], raw["evidence"], check_exact=True
                )
                manual = pd.read_pickle(mp)
                expected = {
                    (r["stream"], r["lipid_class"]): r["candidate_rows"]
                    for r in categorize(raw["pool"], manual, canonical_lipid_class)
                    if r["category"]
                    == "negative_class_absent_in_mode_present_in_other_mode"
                }
                actual = (
                    pool[pool.training_review_uncertain]
                    .assign(
                        canonical_class=lambda p: p["class"].map(canonical_lipid_class)
                    )
                    .groupby(["stream", "canonical_class"])
                    .size()
                    .to_dict()
                )
                assert expected == actual
                local = {}
                for mode, part in pool[pool.training_label_known].groupby(
                    "stream", sort=False
                ):
                    assert not (
                        part.training_positive & part.training_review_uncertain
                    ).any()
                    local[mode] = Counter(
                        known=len(part),
                        positive=int(part.training_positive.sum()),
                        uncertain=int(part.training_review_uncertain.sum()),
                    )
                cache[key] = dict(
                    training=sorted(training), excluded=sorted(excluded), counts=local
                )
            case = cache[key]
            assert case["training"] == sorted(training) and case["excluded"] == sorted(
                excluded
            )
            for mode, values in case["counts"].items():
                counts[mode].update(values)
        assert sum(v["known"] for v in counts.values()) == old["training_pairs"]
        assert sum(v["positive"] for v in counts.values()) == old["positives"]
        for mode, values in counts.items():
            assert values["known"] == old["experts"][mode]["training_pairs"]
            assert values["positive"] == old["experts"][mode]["positives"]
        records.append(
            dict(
                model=path.relative_to(OLD).as_posix(),
                queries=len(old["training_feature_masks"]),
                known=old["training_pairs"],
                positive=old["positives"],
                uncertain=sum(v["uncertain"] for v in counts.values()),
            )
        )
        print(
            "037PREFLIGHT:UNCHANGEDLABELS/INDEPENDENTREVIEWFLAGS",
            path.relative_to(OLD),
            flush=True,
        )
    assert len(cache) == 330 and sum(r["queries"] for r in records) == 528
    assert all(sha(Path(p)) == h for p, h in {**sources, **inputs}.items())
    pd.DataFrame(records).to_csv(OUT / "by_parent_scope.csv", index=False)
    result = dict(
        all_passed=True,
        classifier_fits=0,
        annotation_changes=False,
        parent_scopes=25,
        training_query_scope_checks=528,
        unique_full_training_tables=330,
        independent_review_category_checks=330,
        unchanged_positive_known_label_checks=330,
        unchanged_physical_predictor_checks=330,
        source_sha256=sources,
        input_sha256=inputs,
        note="Pre-fit supervision engineering,NOTannotation performance. No candidates/labels/predictors changed. Query,outer andinner references excluded. Original029known/positive counts match ALL25parents/100experts. Independent retrospective category counts match fit-only flags byexactcanonicalclass/mode inALL330unique training tables. No pseudo positives or calibration/evaluation relabeling.",
    )
    runner.dump(OUT / "receipt.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"source_sha256", "input_sha256"}
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
