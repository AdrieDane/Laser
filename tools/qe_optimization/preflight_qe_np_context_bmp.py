"""040 no-fit gate: ALL330trainingtables/528uses,ALL33disabledfull exports.

Independent partial-label reconstruction,positive/predictor/source parity and
actual context support. No example classifier fits or annotation changes.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_np_context_bmp_experiment as runner

OUT = runner.QE / "np_context_BMP_preflight_v2"


def independent_case(dataset, mask, projects, folds):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.evidence_training import validate_feature_scope

    own = folds[dataset]
    parent = set(mask) - {own}
    assert own in mask and len(parent) in {1, 2}
    allowed = {d[:7] for d in projects if folds[d] not in parent}
    blocked = {d[:7] for d in projects if folds[d] in parent}
    raw = runner.load_predictors(dataset, mask)
    validate_feature_scope(raw, allowed, blocked, query_is_training=True)
    plain, manual, _ = runner.label_allowed_query(raw, allowed, blocked, partial=False)
    actual, _, audit = runner.label_allowed_query(raw, allowed, blocked, partial=True)
    bmp = plain["pool"]["class"].map(canonical_lipid_class).eq("BMP")
    present = set(
        manual.loc[manual.lipid_class.map(canonical_lipid_class).eq("BMP"), "stream"]
    ) & {"NP_Pos", "NP_Neg"}
    absent = ({"NP_Pos", "NP_Neg"} - present) if len(present) == 1 else set()
    expected = plain["pool"].training_label_known.copy()
    withheld = (
        bmp
        & plain["pool"].stream.isin(absent)
        & ~plain["pool"].is_is.astype(bool)
        & expected
    )
    assert not (withheld & plain["pool"].training_positive).any()
    expected.loc[withheld] = False
    pd.testing.assert_frame_equal(actual["evidence"], raw["evidence"], check_exact=True)
    pd.testing.assert_frame_equal(
        actual["pool"][list(raw["pool"])], raw["pool"], check_exact=True
    )
    pd.testing.assert_series_equal(
        actual["pool"].training_positive,
        plain["pool"].training_positive,
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        actual["pool"].training_label_known, expected, check_exact=True
    )
    assert audit["withheld_known_negatives"] == int(withheld.sum())
    assert (
        audit["positive_rows_before"]
        == audit["positive_rows_after"]
        == int(plain["pool"].training_positive.sum())
    )
    assert audit["known_rows_before"] == int(plain["pool"].training_label_known.sum())
    assert audit["known_rows_after"] == int(expected.sum())
    assert audit["curated_np_modes"] == sorted(present)
    assert audit["uncurated_opposite_np_modes"] == sorted(absent)
    assert audit["evaluation_labels_changed"] is False
    assert len(raw["evidence"].columns) == 108
    result = {}
    for mode in ("NP_Pos", "NP_Neg"):
        rows = plain["pool"].stream.eq(mode)
        positive = rows & plain["pool"].training_positive
        result[mode] = dict(
            original_known=int((rows & plain["pool"].training_label_known).sum()),
            known=int((rows & expected).sum()),
            positives=int(positive.sum()),
            BMP_positives=int(
                (positive & bmp & ~plain["pool"].is_is.astype(bool)).sum()
            ),
            other_class_known=int((rows & expected & ~bmp).sum()),
            withheld=int((rows & withheld).sum()),
        )
    return result


def main():
    configure_frozen_resources(REPO)
    from matlab_annotation.np_context_bmp_completion import (
        select_context_bmp_additions,
        NP_MODES,
    )

    projects = json.loads((runner.PREVIOUS / "run_manifest.json").read_text())[
        "datasets"
    ]
    folds_path = runner.QE / "rt_atlas_development/fold_assignment.json"
    folds = json.loads(folds_path.read_text())
    assert len(projects) == 33 and len({d[:7] for d in projects}) == 30
    from sklearn.ensemble import HistGradientBoostingClassifier

    configuration = json.loads((runner.PREVIOUS / "protocol.json").read_text())[
        "classifier"
    ]
    parameters = runner.estimator_parameters(configuration)
    constructor = HistGradientBoostingClassifier(**parameters)
    assert configuration["kind"] == "HistGradientBoostingClassifier"
    assert "kind" not in parameters
    for k, v in parameters.items():
        assert constructor.get_params()[k] == v
    if OUT.exists():
        raise RuntimeError(
            "Existing preflight artifacts must be preserved;use its complete verified receipt or explicitly version a new audit"
        )
    OUT.mkdir()
    sources = {
        str(p): runner.sha(p)
        for folder in ("app", "methods", "reference_data")
        for p in (REPO / folder).rglob("*")
        if p.is_file()
        and not {"__pycache__", ".DS_Store", ".pytest_cache", ".ruff_cache"}
        & set(p.parts)
        and p.suffix != ".pyc"
    }
    for p in (
        Path(__file__),
        Path(runner.__file__),
        Path(__file__).with_name("frozen_resources.py"),
        Path(__file__).with_name("run_qe_atlas_experiment.py"),
        REPO / "docs/validation/next_global_context_BMP_completion.txt",
    ):
        sources[str(p)] = runner.sha(p)
    files = {
        folds_path,
        runner.PREVIOUS / "protocol.json",
        runner.PREVIOUS / "integrity_receipt.json",
    }
    parents = []
    training_paths = set()
    for outer in range(5):
        for inner in [None, *sorted(set(folds.values()) - {outer})]:
            blocked = {outer} if inner is None else {outer, inner}
            model_path = runner.PREVIOUS / (
                f"fold_{outer}_nested_model.pkl"
                if inner is None
                else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
            )
            files.add(model_path)
            records = [
                (d, tuple(sorted(blocked | {folds[d]})))
                for d in projects
                if folds[d] not in blocked
            ]
            parents.append((outer, inner, model_path, records))
            training_paths.update(records)
    assert len(parents) == 25 and len(training_paths) == 330
    assert sum(len(rows) for _, _, _, rows in parents) == 528
    for d, mask in training_paths:
        files.add(runner.prepared_path(d, mask))
        files.add(runner.BASE / "evaluation_v2" / d / "manual_comparison.pkl")
    for d in projects:
        files.update(
            [
                runner.prepared_path(d, {folds[d]}),
                runner.PREVIOUS / "runs" / d / "automatic_annotations.csv",
                runner.PREVIOUS / "runs" / d / "candidates.pkl",
                runner.QE
                / "nested_standard_development_v1/automatic_inputs"
                / (d + ".pkl"),
                runner.PREPARED / "measurements" / f"exclude_{folds[d]}" / (d + ".pkl"),
                runner.QE
                / "nested_standard_development_v1"
                / f"exclude_{folds[d]}.json",
            ]
        )
    inputs = {str(p): runner.sha(p) for p in sorted(files)}
    runner.dump(
        OUT / "protocol.json",
        dict(
            classifier_fits=0,
            annotation_changes=False,
            source_sha256=sources,
            input_sha256=inputs,
        ),
    )
    cached = {}
    for d, mask in sorted(training_paths):
        cached[(d, mask)] = independent_case(d, mask, projects, folds)
        print("040 PREFLIGHT LABELS/PREDICTORS/SCOPE", d, mask, flush=True)
    parent_rows = []
    for outer, inner, model_path, rows in parents:
        model = pd.read_pickle(model_path)
        blocked = {outer} if inner is None else {outer, inner}
        allowed = {d[:7] for d in projects if folds[d] not in blocked}
        excluded = {d[:7] for d in projects if folds[d] in blocked}
        assert (
            set(model["training_cohorts"]) == allowed
            and set(model["excluded_groups"]) == excluded
        )
        for mode in NP_MODES:
            totals = Counter()
            contributors, bmp_projects = set(), set()
            for d, mask in rows:
                values = cached[(d, mask)][mode]
                totals.update(values)
                if values["known"]:
                    contributors.add(d[:7])
                if values["BMP_positives"]:
                    bmp_projects.add(d[:7])
            expert = model["experts"][mode]
            for k, v in parameters.items():
                assert expert["estimator"].get_params()[k] == v
            assert expert["training_pairs"] == totals["original_known"]
            assert expert["positives"] == totals["positives"]
            assert contributors == allowed
            assert totals["known"] == totals["original_known"] - totals["withheld"]
            assert totals["other_class_known"] > 0
            supported = totals["BMP_positives"] >= 20 and len(bmp_projects) >= 3
            parent_rows.append(
                dict(
                    outer=outer,
                    inner=inner,
                    mode=mode,
                    completion_supported=supported,
                    contributing_context_projects=sorted(contributors),
                    BMP_positive_projects=sorted(bmp_projects),
                    **dict(totals),
                )
            )
    runner.dump(OUT / "parent_training_counts.json", parent_rows)
    for d in projects:
        raw = runner.load_predictors(d, {folds[d]})
        previous = runner.read_previous(d)
        expected_candidate = pd.read_pickle(
            runner.PREVIOUS / "runs" / d / "candidates.pkl"
        )
        assert len(runner.accepted_keys(expected_candidate)) == len(expected_candidate)
        empty = []
        for mode in NP_MODES:
            pool = raw["pool"].loc[raw["pool"].stream.eq(mode)]
            proposal = select_context_bmp_additions(
                pool,
                pd.Series(np.nan, index=pool.index),
                previous.loc[previous.stream.eq(mode)],
                stream=mode,
                threshold=1.0,
                **runner.physical_protection(d, {folds[d]}, mode, raw),
            )
            assert proposal.empty
            empty.append(proposal)
        annotations, candidates = runner.assemble_exports(
            d, previous, pd.concat(empty, ignore_index=True)
        )
        pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
        pd.testing.assert_frame_equal(candidates, expected_candidate, check_exact=True)
        assert set(runner.accepted_keys(annotations)) == set(
            runner.accepted_keys(previous)
        )
        print("040 PREFLIGHT DISABLED ALL4MODES/ALLFIELDS EXACT029", d, flush=True)
    for p, h in {**sources, **inputs}.items():
        assert runner.sha(Path(p)) == h, p
    runner.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            classifier_fits=0,
            annotation_changes=False,
            unique_training_tables=330,
            training_query_uses=528,
            parents=25,
            maximum_supported_fresh_NP_classifiers=sum(
                r["completion_supported"] for r in parent_rows
            ),
            disabled_full_029_datasets=33,
            disabled_streams=132,
            predictor_columns=108,
            source_sha256=sources,
            input_sha256=inputs,
            parent_training_counts_sha256=runner.sha(
                OUT / "parent_training_counts.json"
            ),
            note="Engineering gate ONLY,not annotation accuracy. All permitted traininglabels/predictors/scopes audited;allpositivesunchanged,onlyexactpartialBMPnegativesunknown. All33disabled exports/candidateflags/measurementcells EXACT029. Zero example classifierfits/annotationchanges. No querytruth used for finalpredictions.",
        ),
    )


if __name__ == "__main__":
    main()
