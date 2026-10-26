"""All33 blind/scalar competition audit, then descriptive gold opportunity counts."""

import json
from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
from diagnose_qe_raw_counterpart_pilot import option_categories
from verify_qe_bmp_context_pilot import eligibility
import prepare_qe_formula_isotope_pilot as previous
import occupancy_competition_audit as measurement

common = previous.common
OUT = common.QE / "occupancy_competition_diagnostic_041_v1"
REFERENCE = common.QE / "experiment_041"
CORE = common.QE / "experiment_029"
PLAN = REPO / "docs/validation/next_occupancy_competition_diagnostic.txt"
MASKS = ("free_physical", "original_clear_identity_winner", "relaxed_opportunity")


def inputs(datasets, folds):
    paths = set(previous.inputs(datasets, folds))
    paths.update(
        str(p)
        for p in (
            Path(__file__),
            Path(measurement.__file__),
            PLAN,
            Path(__file__).with_name("test_occupancy_competition_audit.py"),
        )
    )
    for d in datasets:
        paths.update(
            str(p)
            for p in (
                CORE / f"fold_{folds[d]}_nested_model.pkl",
                CORE / "runs" / d / "scoring_audit.pkl",
                REFERENCE / "runs" / d / "automatic_annotations.csv",
            )
        )
    return {p: common.sha(Path(p)) for p in sorted(paths)}


def blind(d, fold):
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.evidence_training import validate_feature_scope

    raw = common.common.load_predictors(d, {fold})
    model = pd.read_pickle(CORE / f"fold_{fold}_nested_model.pkl")
    validate_feature_scope(
        raw,
        model["training_cohorts"],
        model["excluded_groups"],
        query_is_training=False,
    )
    scores = predict_mode_evidence_experts(raw, model)
    pd.testing.assert_series_equal(
        scores,
        pd.read_pickle(CORE / "runs" / d / "scoring_audit.pkl"),
        check_exact=True,
    )
    current = pd.read_csv(
        REFERENCE / "runs" / d / "automatic_annotations.csv",
        dtype=str,
        keep_default_na=False,
    )
    protections = {
        s: common.common.physical_protection(d, {fold}, s, raw) for s in common.MODES
    }
    physical = eligibility(raw["pool"], protections)
    before = raw["pool"].copy(deep=True)
    result = measurement.measure(raw["pool"], scores, current, physical)
    independent = measurement.independent_measure(
        raw["pool"], scores, current, physical
    )
    pd.testing.assert_frame_equal(result, independent, check_exact=True)
    pd.testing.assert_frame_equal(raw["pool"], before, check_exact=True)
    path = OUT / "private" / (d + ".pkl")
    pd.to_pickle(
        dict(
            query_dataset=d,
            excluded_folds=[fold],
            masks=result,
            scores=scores,
            source_prepared_sha256=common.sha(common.common.prepared_path(d, {fold})),
        ),
        path,
    )
    print(
        "BLIND SCALAR-CHECKED COMPETITION GRAPH",
        d,
        len(result),
        int(result.relaxed_opportunity.sum()),
        flush=True,
    )
    return str(path), common.sha(path)


def describe(datasets, folds, inputs_sha, artifacts):
    rows, categories = [], []
    total_manual = current_correct = 0
    for d in datasets:
        path = REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl"
        inputs_sha[str(path)] = common.sha(path)
        # First manual decoding, ONLY after ALL33 blind+scalar receipt.
        manual = pd.read_pickle(path)
        current = pd.read_csv(
            REFERENCE / "runs" / d / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        raw = common.common.load_predictors(d, {folds[d]})
        result = pd.read_pickle(OUT / "private" / (d + ".pkl"))["masks"]
        table = option_categories(raw["pool"], manual, current).join(result)
        table["free_physical"] = (
            table.physical & table.identity_free & table.feature_free
        )
        table["original_clear_identity_winner"] = (
            table.free_physical
            & table.identity_best
            & table.feature_top
            & table.feature_clear
        )
        keys = ["stream", "species", "adduct", "feature_id"]
        # Candidate construction duplicates cannot multiply hypotheses.
        table = table.sort_values(
            "identity_best", ascending=False, kind="stable"
        ).drop_duplicates(keys)
        primary = manual.loc[~manual.lipid_class.eq("X4")]
        total_manual += len(primary)
        current_correct += int(primary.same_manual_peak.sum())
        for mask in MASKS:
            available = table.loc[table[mask]]
            values = set(available[keys].itertuples(index=False, name=None))
            for (s, cls), part in primary.groupby(["stream", "lipid_class"]):
                missing = part.loc[
                    ~part.same_manual_peak & part.mapped & ~part.internal_standard
                ]
                rows.append(
                    dict(
                        dataset=d,
                        stream=s,
                        lipid_class=cls,
                        mask=mask,
                        manual_annotations=len(part),
                        current_correct=int(part.same_manual_peak.sum()),
                        candidate_options=int(
                            (
                                available.stream.eq(s) & available.lipid_class.eq(cls)
                            ).sum()
                        ),
                        missing_manual_rows_reached=sum(
                            k in values
                            for k in missing[keys].itertuples(index=False, name=None)
                        ),
                    )
                )
        for (s, c, category), part in table.loc[table.relaxed_opportunity].groupby(
            ["stream", "lipid_class", "diagnostic_category"]
        ):
            categories.append(
                dict(
                    dataset=d,
                    stream=s,
                    lipid_class=c,
                    category=category,
                    candidate_options=len(part),
                )
            )
    assert total_manual == 92100 and current_correct == 82740
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "opportunities_by_dataset_mode_class.csv", index=False)
    frame.groupby(["stream", "lipid_class", "mask"], as_index=False)[
        [
            "manual_annotations",
            "current_correct",
            "candidate_options",
            "missing_manual_rows_reached",
        ]
    ].sum().to_csv(OUT / "opportunities_by_mode_class.csv", index=False)
    frame.groupby(["stream", "mask"], as_index=False)[
        [
            "manual_annotations",
            "current_correct",
            "candidate_options",
            "missing_manual_rows_reached",
        ]
    ].sum().to_csv(OUT / "opportunities_by_mode.csv", index=False)
    pd.DataFrame(categories).to_csv(OUT / "relaxed_option_categories.csv", index=False)
    assert all(common.sha(Path(p)) == h for p, h in {**inputs_sha, **artifacts}.items())
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            primary_manual_annotations=total_manual,
            primary_current_correct=current_correct,
            all33_blind_and_scalar_checked_before_gold=True,
            source_input_sha256=inputs_sha,
            artifact_sha256=artifacts,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            note="Read-only optimistic candidate opportunity audit, NOT accepted gains, joint assignment performance or FDR. Original029 scores reproduced ALL33 bitexact; ALL041 calls fixed. Exposed development data, not external validation.",
        ),
    )
    print(
        frame.groupby(["stream", "mask"])[
            ["candidate_options", "missing_manual_rows_reached"]
        ]
        .sum()
        .to_string(),
        flush=True,
    )


def main():
    configure_frozen_resources(REPO)
    datasets, folds = common.inventory()
    assert json.loads((REFERENCE / "integrity_receipt.json").read_text())["all_passed"]
    hashes = inputs(datasets, folds)
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    common.dump(OUT / "planned_inputs.json", hashes)
    artifacts = dict(blind(d, folds[d]) for d in datasets)
    assert len(artifacts) == 33 and hashes == inputs(datasets, folds)
    common.dump(
        OUT / "blind_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            all33_tables_frozen=True,
            independent_scalar_checks=True,
            manual_files_opened=0,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            source_input_sha256=hashes,
            artifact_sha256=artifacts,
        ),
    )
    describe(datasets, folds, hashes, artifacts)


if __name__ == "__main__":
    main()
