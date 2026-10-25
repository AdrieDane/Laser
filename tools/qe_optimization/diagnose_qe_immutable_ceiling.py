"""Read-only structural bound: which classes need old-call changes for90%?"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import run_qe_add_only_class_experiment as common

QE = common.QE
REFERENCE = QE / "experiment_041"
OUT = QE / "immutable_call_ceiling_041_v2"


def main():
    configure_frozen_resources(REFERENCE / "source_snapshot")
    from standard_lipidomics.validation import _normalise
    from matlab_annotation.standard_rt_library import ion_key

    datasets, folds = common.inventory()
    paths = {
        Path(__file__),
        Path(common.__file__),
        Path(common.common.__file__),
        Path(__file__).with_name("run_qe_atlas_experiment.py"),
        REFERENCE / "integrity_receipt.json",
    }
    for d in datasets:
        paths.update(
            [
                REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl",
                REFERENCE / "runs" / d / "automatic_annotations.csv",
                common.common.prepared_path(d, {folds[d]}),
            ]
        )
    hashes = {str(p): common.sha(p) for p in sorted(paths)}
    counts = Counter()
    for d in datasets:
        manual = pd.read_pickle(
            REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        old = pd.read_csv(
            REFERENCE / "runs" / d / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        pool = pd.read_pickle(common.common.prepared_path(d, {folds[d]}))["pool"]
        a = _normalise(old.assign(adduct=old.adduct.map(ion_key)), manual=False)
        p = _normalise(pool.assign(adduct=pool.adduct.map(ion_key)), manual=False)
        occupied = set(a[["stream", "feature_id"]].itertuples(index=False, name=None))
        identities = set(
            a[["stream", "species", "adduct", "internal_standard"]].itertuples(
                index=False, name=None
            )
        )
        exact = set(
            p[
                ["stream", "species", "adduct", "internal_standard", "feature_id"]
            ].itertuples(index=False, name=None)
        )
        for r in manual.itertuples():
            if r.lipid_class == "X4":
                continue
            if r.same_manual_peak:
                reason = "already_correct"
            elif not r.mapped:
                reason = "unresolved_mapping"
            else:
                identity = (r.stream, r.species, r.adduct, bool(r.internal_standard))
                blocked_identity = identity in identities
                blocked_feature = (r.stream, str(r.feature_id)) in occupied
                if blocked_identity and blocked_feature:
                    reason = "identity_and_feature_blocked"
                elif blocked_identity:
                    reason = "identity_blocked"
                elif blocked_feature:
                    reason = "feature_blocked"
                else:
                    reason = (
                        "free_present_core"
                        if (*identity, str(r.feature_id)) in exact
                        else "free_absent_core"
                    )
            counts[(d, r.stream, r.lipid_class, reason)] += 1
        print("IMMUTABLE CALL STRUCTURAL BOUND", d, flush=True)
    table = pd.DataFrame(
        [
            dict(dataset=d, stream=s, lipid_class=c, reason=reason, count=n)
            for (d, s, c, reason), n in sorted(counts.items())
        ]
    )
    OUT.mkdir(exist_ok=False)
    table.to_csv(OUT / "by_dataset_mode_class_reason.csv", index=False)
    grouped = table.groupby(["stream", "lipid_class", "reason"], as_index=False)[
        "count"
    ].sum()
    grouped.to_csv(OUT / "by_mode_class_reason.csv", index=False)
    summary = (
        grouped.pivot(index=["stream", "lipid_class"], columns="reason", values="count")
        .fillna(0)
        .astype(int)
    )
    reasons = [
        "already_correct",
        "unresolved_mapping",
        "identity_and_feature_blocked",
        "identity_blocked",
        "feature_blocked",
        "free_present_core",
        "free_absent_core",
    ]
    for reason in reasons:
        if reason not in summary:
            summary[reason] = 0
    summary["manual_rows"] = summary[reasons].sum(axis=1)
    summary["current_correct_percent"] = (
        100 * summary.already_correct / summary.manual_rows
    )
    summary["immutable_core_ceiling_count"] = (
        summary.already_correct + summary.free_present_core
    )
    summary["immutable_ideal_discovery_ceiling_count"] = (
        summary.immutable_core_ceiling_count + summary.free_absent_core
    )
    summary["immutable_ideal_discovery_ceiling_percent"] = (
        100 * summary.immutable_ideal_discovery_ceiling_count / summary.manual_rows
    )
    summary["matches_requiring_old_call_changes_for90"] = np.maximum(
        0,
        np.ceil(0.9 * summary.manual_rows).astype(int)
        - summary.immutable_ideal_discovery_ceiling_count,
    )
    summary = summary.reset_index()
    priority = pd.read_csv(
        REFERENCE
        / "comparison_with_experiment_029/optimization_priority/by_mode_class_before_after.csv"
    )[["stream", "lipid_class", "optimization_priority"]]
    summary = summary.merge(
        priority, on=["stream", "lipid_class"], how="left", validate="one_to_one"
    )
    assert summary.manual_rows.sum() == 92100 and summary.already_correct.sum() == 82740
    assert summary.optimization_priority.notna().all()
    summary.to_csv(OUT / "by_mode_class_structural_bound.csv", index=False)
    assert all(common.sha(Path(p)) == h for p, h in hashes.items())
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            manual_rows=92100,
            correct=82740,
            classifier_fits=0,
            annotation_changes=False,
            source_input_sha256=hashes,
            note="Retrospective structural bound,not attainable performance/gains. Immutable idealdiscovery ceiling assumes ALLunblocked mapped manualfeatures couldbe found perfectly; unknownmappingexcludedfromceiling butremainsdenominator. Coreceiling additionallyrequiresexistingcorecandidate. Reasonsdisjoint;correctmatches/usermodelscoresunchanged. X4soleexcludedclass. Old-call-change requirement counts manualentries,not numberofswaps/actions.",
        ),
    )
    print(
        summary.loc[
            summary.optimization_priority.eq("prioritize")
            & summary.matches_requiring_old_call_changes_for90.gt(0)
        ]
        .sort_values("matches_requiring_old_call_changes_for90", ascending=False)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
