"""Complete25parent/528use no-fit gate; independent targets and full exports."""

from collections import Counter
import json

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_bmp_context_experiment as runner

OUT = runner.PREFLIGHT


def independent_case(d, mask, datasets, folds):
    from annotator.composition_ranges import canonical_lipid_class

    parent = set(mask) - {folds[d]}
    assert len(parent) in {1, 2}
    allowed = {p[:7] for p in datasets if folds[p] not in parent}
    excluded = {p[:7] for p in datasets if folds[p] in parent}
    raw = runner.load_predictors(d, mask)
    runner.preparation.validate_scope(raw, allowed, excluded, query_is_training=True)
    # Decode the independently named reference only AFTER authorization.
    manual = pd.read_pickle(runner.BASE / "evaluation_v2" / d / "manual_comparison.pkl")
    manual = manual.loc[~manual.internal_standard]
    common = runner.original.inherited.common
    exact = {
        (
            r.stream,
            common.species(r.species),
            common.adduct(r.adduct),
            str(r.feature_id),
        )
        for r in manual.loc[manual.mapped].itertuples()
    }
    unresolved = {
        (r.stream, common.species(r.species), common.adduct(r.adduct))
        for r in manual.loc[~manual.mapped].itertuples()
    }
    identities = {
        (r.stream, common.species(r.species), common.adduct(r.adduct))
        for r in manual.itertuples()
    }
    keys = [
        (
            r.stream,
            common.species(r.name),
            common.adduct(r.adduct),
            str(r.feat_feature_id),
        )
        for r in raw["pool"].itertuples()
    ]
    positive = pd.Series(
        [k in exact for k in keys], index=raw["pool"].index, name="training_positive"
    )
    known = pd.Series(
        [k[:3] not in unresolved or k in exact for k in keys],
        index=raw["pool"].index,
        name="training_label_known",
    )
    in_manual = pd.Series(
        [k[:3] in identities for k in keys],
        index=raw["pool"].index,
        name="calibration_identity_in_manual",
    )
    bmp = raw["pool"]["class"].map(canonical_lipid_class).eq("BMP")
    present = set(
        manual.loc[manual.lipid_class.map(canonical_lipid_class).eq("BMP"), "stream"]
    ) & set(runner.NP_MODES)
    absent = set(runner.NP_MODES) - present if len(present) == 1 else set()
    withheld = (
        known & bmp & ~raw["pool"].is_is.astype(bool) & raw["pool"].stream.isin(absent)
    )
    assert not (withheld & positive).any()
    expected = (known & ~withheld).rename("training_label_known")
    actual, _, audit = runner.authorize_labels(raw, allowed, excluded, partial=True)
    for c, value in (
        ("training_positive", positive),
        ("training_label_known", expected),
        ("calibration_identity_in_manual", in_manual),
    ):
        pd.testing.assert_series_equal(actual["pool"][c], value, check_exact=True)
    pd.testing.assert_frame_equal(
        actual["pool"][list(raw["pool"])], raw["pool"], check_exact=True
    )
    pd.testing.assert_frame_equal(actual["evidence"], raw["evidence"], check_exact=True)
    assert audit["evaluation_labels_changed"] is False and audit[
        "positive_rows_before"
    ] == audit["positive_rows_after"] == int(positive.sum())
    assert audit["known_rows_before"] == int(known.sum()) and audit[
        "known_rows_after"
    ] == int(expected.sum())
    assert audit["withheld_known_negatives"] == int(withheld.sum())
    assert audit["curated_np_modes"] == sorted(present) and audit[
        "uncurated_opposite_np_modes"
    ] == sorted(absent)
    result = {}
    for s in runner.NP_MODES:
        rows = raw["pool"].stream.eq(s)
        bp = rows & positive & expected & bmp & ~raw["pool"].is_is.astype(bool)
        result[s] = dict(
            known_rows=int((rows & expected).sum()),
            known_positive_rows=int((rows & expected & positive).sum()),
            known_BMP_positive_pairs=int(bp.sum()),
            other_class_known_context_rows=int(
                (rows & expected & ~(bmp & ~raw["pool"].is_is.astype(bool))).sum()
            ),
            withheld=int((rows & withheld).sum()),
        )
    return result


def parent_support(data, allowed, excluded, independent, folds):
    prepared = runner.completion.prepare_training(
        data, training_groups=allowed, excluded_groups=excluded
    )
    result = {}
    for s, part in prepared.items():
        totals = Counter()
        perproject = Counter()
        positives = set()
        for d, raw in data.items():
            record = independent[(d, tuple(sorted(raw["excluded_folds"])))][s]
            for k in (
                "known_rows",
                "known_positive_rows",
                "known_BMP_positive_pairs",
                "other_class_known_context_rows",
            ):
                totals[k] += record[k]
            perproject[d[:7]] += record["known_rows"]
            if record["known_BMP_positive_pairs"]:
                positives.add(d[:7])
        support = part["support"]
        assert all(support[k] == int(v) for k, v in totals.items())
        assert support["BMP_positive_projects"] == sorted(positives)
        assert support["actual_projects"] == sorted(allowed) and support[
            "known_rows_by_project"
        ] == dict(perproject)
        assert part["supported"] == (
            totals["known_BMP_positive_pairs"] >= 20 and len(positives) >= 3
        )
        assert (
            part["metadata"]["training_pairs"] == totals["known_rows"]
            and part["metadata"]["positives"] == totals["known_positive_rows"]
        )
        assert len(part["metadata"]["columns"]) == 163
        for p in allowed:
            mask = np.array(part["projects"]) == p
            np.testing.assert_allclose(
                part["weights"][mask],
                len(part["weights"]) / len(allowed) / perproject[p],
                atol=0,
                rtol=0,
            )
            assert (
                abs(part["weights"][mask].sum() - len(part["weights"]) / len(allowed))
                < 1e-8
            )
        categories = {
            c: {v: i for i, v in enumerate(sorted(part["x"][c].astype(str).unique()))}
            for c in part["x"]
            if not pd.api.types.is_numeric_dtype(part["x"][c])
        }
        assert categories == part["metadata"]["categories"]
        result[s] = dict(
            supported=part["supported"],
            support=support,
            metadata=part["metadata"],
            **runner.completion.matrix_receipt(part),
        )
    return result


def disabled_export(d):
    raw = runner.load_predictors(d, {runner.inventory()[1][d]})
    previous = runner.previous_annotations(d)
    empty = raw["pool"].iloc[:0].copy()
    annotations, candidates, filled = runner.assemble(d, previous, empty)
    assert filled.empty
    pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
    pd.testing.assert_frame_equal(
        candidates,
        pd.read_pickle(runner.PREVIOUS / "runs" / d / "candidates.pkl"),
        check_exact=True,
    )


def main():
    configure_frozen_resources(REPO)
    from sklearn.ensemble import HistGradientBoostingClassifier

    datasets, folds = runner.inventory()
    hashes = runner.source_inputs()
    OUT.mkdir(exist_ok=False)
    runner.dump(OUT / "planned_inputs.json", hashes)
    unique = sorted(
        (d, tuple(sorted(mask)))
        for d in datasets
        for mask in runner.preparation.masks_for(folds[d])
        if len(mask) > 1
    )
    assert len(unique) == 330
    independent = {}
    for d, mask in unique:
        independent[(d, mask)] = independent_case(d, mask, datasets, folds)
        print("044 INDEPENDENT LABEL/SCOPE CHECK", d, mask, flush=True)
    configuration = json.loads((runner.CORE / "protocol.json").read_text())[
        "classifier"
    ]
    parameters = runner.original.inherited.common.estimator_parameters(configuration)
    constructor = HistGradientBoostingClassifier(**parameters)
    assert all(
        constructor.get_params()[k] == v for k, v in parameters.items()
    ) and not hasattr(constructor, "classes_")
    parents, uses = [], 0
    for outer in range(5):
        for inner in [None, *sorted(set(range(5)) - {outer})]:
            data, allowed, excluded = runner.training_data(
                outer, inner, datasets, folds
            )
            support = parent_support(data, allowed, excluded, independent, folds)
            uses += len(data)
            oldpath = runner.CORE / (
                f"fold_{outer}_nested_model.pkl"
                if inner is None
                else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
            )
            old = pd.read_pickle(oldpath)
            for s in runner.NP_MODES:
                assert all(
                    old["experts"][s]["estimator"].get_params()[k] == v
                    for k, v in parameters.items()
                )
            parents.append(
                dict(
                    outer=outer,
                    inner=inner,
                    training_query_count=len(data),
                    training_groups=sorted(allowed),
                    excluded_groups=sorted(excluded),
                    modes=support,
                )
            )
            print("044 NO-FIT PARENT PREPARED", outer, inner, len(data), flush=True)
    assert len(parents) == 25 and uses == 528
    for d in datasets:
        disabled_export(d)
        print("044 DISABLED ENTIRE041 EXPORT EXACT", d, flush=True)
    runner.dump(OUT / "parent_support.json", parents)
    assert hashes == runner.source_inputs()
    runner.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            unique_training_tables=330,
            parent_models_prepared=25,
            training_query_uses=528,
            disabled_whole_exports=33,
            classifier_fits=0,
            annotation_changes=False,
            parent_support_sha256=runner.sha(OUT / "parent_support.json"),
            source_input_sha256=hashes,
        ),
    )


if __name__ == "__main__":
    main()
