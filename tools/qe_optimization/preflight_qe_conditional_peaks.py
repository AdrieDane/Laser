"""No fits: independent physical-pair/weight checks on one complete inner parent.

All528 training-query uses remain required in the final experiment verifier.
This pre-fit gate checks engineering, not annotation accuracy or model quality.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_qe_conditional_peak_experiment as runner
from context import REPO
from frozen_resources import configure_frozen_resources

OUT = runner.QE / "conditional_peak_preflight_v1"


def independent_pair_support(records, mode):
    rows, identities, classes = Counter(), Counter(), defaultdict(set)
    for data in records.values():
        pool = data["pool"]
        part = pool[pool.stream.eq(mode)].drop_duplicates(
            ["stream", "name", "adduct", "feat_feature_id"]
        )
        for _, group in part.groupby(["stream", "name", "adduct"], sort=False):
            positive = group[group.training_positive]
            negative = group[group.training_label_known & ~group.training_positive]
            if positive.empty or negative.empty:
                continue
            assert len(positive) == 1
            project = data["query_group"]
            rows[project] += 2 * len(negative)
            identities[project] += 1
            from annotator.composition_ranges import canonical_lipid_class

            classes[canonical_lipid_class(positive.iloc[0]["class"])].add(project)
    return rows, identities, {c: sorted(g) for c, g in sorted(classes.items())}


def verify_pairs(records, training, excluded, *, compare_scalar=False):
    from matlab_annotation.conditional_peak_pairs import conditional_training_pairs
    from matlab_annotation.mode_evidence_experts import MODES, _subset

    output = {}
    for mode in MODES:
        subsets = {d: _subset(v, mode) for d, v in records.items()}
        subsets = {d: v for d, v in subsets.items() if len(v["pool"])}
        expected, identities, classes = independent_pair_support(records, mode)
        if not expected:
            output[mode] = dict(pair_rows=0, supervised_contributing_projects=[])
            continue
        pairs = conditional_training_pairs(
            subsets, training_groups=training, excluded_groups=excluded
        )
        assert pairs["pair_rows_by_project"] == dict(expected)
        assert pairs["identities_by_project"] == dict(identities)
        assert pairs["supervised_contributing_projects"] == sorted(expected)
        assert pairs["training_reference_permissions"] == sorted(training)
        assert pairs["excluded_groups"] == sorted(excluded)
        assert pairs["positive"].sum() * 2 == len(pairs["positive"])
        project_weights = {
            g: pairs["sample_weight"][pairs["groups"] == g].sum() for g in expected
        }
        np.testing.assert_allclose(
            list(project_weights.values()),
            len(pairs["positive"]) / len(expected),
            rtol=1e-12,
        )
        assert set(expected).issubset(training) and not set(expected) & excluded
        if compare_scalar:
            scalar = conditional_training_pairs(
                subsets,
                training_groups=training,
                excluded_groups=excluded,
                vectorized=False,
            )
            pd.testing.assert_frame_equal(
                pairs["evidence"], scalar["evidence"], check_exact=True
            )
            for key in ("positive", "groups", "sample_weight"):
                np.testing.assert_array_equal(pairs[key], scalar[key])
        output[mode] = dict(
            pair_rows=len(pairs["positive"]),
            supervised_contributing_projects=sorted(expected),
            pair_rows_by_project=dict(expected),
            identities_by_project=dict(identities),
            conditional_class_projects=classes,
        )
    return output


def main():
    configure_frozen_resources(REPO)
    paths = [
        Path(__file__),
        Path(runner.__file__),
        *sorted((REPO / "app/matlab_annotation").glob("*.py")),
        REPO / "app/annotator/composition_ranges.py",
    ]
    sources = {str(p): runner.sha(p) for p in paths}
    base_model = runner.OLD / "calibration/fold_0/inner_block_1_model.pkl"
    base = pd.read_pickle(base_model)
    training, excluded = set(base["training_cohorts"]), set(base["excluded_groups"])
    records, inputs = {}, {str(base_model): runner.sha(base_model)}
    old_inputs = json.loads((runner.OLD / "protocol.json").read_text())["input_sha256"]
    for dataset, mask in base["training_feature_masks"].items():
        raw = runner.load_predictors(dataset, set(mask))
        records[dataset], _, _ = runner.with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded
        )
        pd.testing.assert_frame_equal(
            raw["evidence"], records[dataset]["evidence"], check_exact=True
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
        print("CONDITIONAL PREFIT PERMITTED QUERY", dataset, flush=True)
    summary = verify_pairs(records, training, excluded)
    # Exact full feature/label/weight parity against original scalar implementation
    # on one real query, ALL modes; full independent support counts above use ALL.
    first = next(iter(records))
    verify_pairs({first: records[first]}, training, excluded, compare_scalar=True)
    assert all(runner.sha(Path(p)) == h for p, h in {**sources, **inputs}.items())
    OUT.mkdir(exist_ok=False)
    runner.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            classifier_fits=0,
            annotation_changes=False,
            training_queries=len(records),
            complete_inner_parent="fold_0/inner_block_1",
            per_mode=summary,
            scalar_query=first,
            source_sha256=sources,
            input_sha256=inputs,
            note="Engineering pre-fit gate ONLY. Final ALL25parents/528query/scoring/calibration/rawoutput integrity still mandatory. No annotation accuracy measured.",
        ),
    )
    print(
        "CONDITIONAL PREFLIGHT ALL PASS", len(records), "training queries", flush=True
    )


if __name__ == "__main__":
    main()
