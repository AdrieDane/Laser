"""Prespecified descriptive relative-ion analysis AFTER both all33 receipts."""

import json
from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
from diagnose_qe_raw_counterpart_pilot import option_categories
import prepare_qe_competing_ion_family_pilot as prep

OUT = prep.common.QE / "competing_ion_family_diagnostic_041_v2"
REFERENCE = prep.common.QE / "experiment_041"
CONTRASTS = (
    "own_minus_alternative_best_raw_correlation",
    "own_minus_alternative_best_normalized_correlation",
    "own_minus_alternative_best_leave_one_out_correlation",
)


def main():
    configure_frozen_resources(REPO)
    from standard_lipidomics.validation import _lipid_class

    blind_path = prep.OUT / "blind_receipt.json"
    independent_path = (
        prep.common.QE / "competing_ion_family_independent_check_v2/receipt.json"
    )
    blind, checked = (json.loads(p.read_text()) for p in (blind_path, independent_path))
    assert blind["all_passed"] and checked["all_passed"] and checked["datasets"] == 33
    assert blind["manual_files_opened"] == checked["manual_files_opened"] == 0
    assert checked["blind_receipt_sha256"] == prep.common.sha(blind_path)
    datasets, folds = prep.common.inventory()
    hashes = {
        **checked["source_input_sha256"],
        str(blind_path): prep.common.sha(blind_path),
        str(independent_path): prep.common.sha(independent_path),
        str(Path(__file__)): prep.common.sha(Path(__file__)),
    }
    for d in datasets:
        for path in (
            REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl",
            REFERENCE / "runs" / d / "automatic_annotations.csv",
        ):
            hashes[str(path)] = prep.common.sha(path)
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    assert all(
        prep.common.sha(Path(p)) == h for p, h in blind["physical_sha256"].items()
    )
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    rows, pairs, distributions = [], [], []
    total_manual = current_correct = 0
    for d in datasets:
        raw = prep.load(d, folds[d])
        saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
        # First reference decoding in this pilot, after both all33 barriers.
        manual = pd.read_pickle(
            REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        accepted = pd.read_csv(
            REFERENCE / "runs" / d / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        primary = manual.loc[~manual.lipid_class.eq("X4")]
        total_manual += len(primary)
        current_correct += int(primary.same_manual_peak.sum())
        table = option_categories(raw["pool"], manual, accepted).join(saved["evidence"])
        table["source_physical"] = saved["physical"]
        table["dataset"] = d
        table["lipid_class"] = table.lipid_class.map(_lipid_class)
        keys = ["stream", "species", "adduct", "feature_id"]
        fields = [prep.measurement.PREFIX + f for f in prep.measurement.FIELDS]
        assert (
            table.groupby(keys, dropna=False)[fields]
            .nunique(dropna=False)
            .le(1)
            .all()
            .all()
        )
        table = table.drop_duplicates(keys)
        table.to_pickle(OUT / "private" / (d + ".pkl"))
        for (s, cls), part in primary.groupby(["stream", "lipid_class"]):
            exact = table.loc[
                table.stream.eq(s)
                & table.lipid_class.eq(cls)
                & table.diagnostic_category.isin(
                    ["manual_correct_current", "manual_correct_currently_missing"]
                )
            ]
            possible = set(exact[keys].itertuples(index=False, name=None))
            available = set(
                exact.loc[
                    exact[prep.measurement.PREFIX + CONTRASTS[0]].notna(), keys
                ].itertuples(index=False, name=None)
            )
            missing = part.loc[
                ~part.same_manual_peak & part.mapped & ~part.internal_standard
            ]
            rows.append(
                dict(
                    dataset=d,
                    stream=s,
                    lipid_class=cls,
                    manual_annotations=len(part),
                    current_correct=int(part.same_manual_peak.sum()),
                    manual_rows_in_pool=sum(
                        k in possible
                        for k in part[keys].itertuples(index=False, name=None)
                    ),
                    manual_rows_with_raw_contrast=sum(
                        k in available
                        for k in part[keys].itertuples(index=False, name=None)
                    ),
                    missing_manual_rows_with_raw_contrast=sum(
                        k in available
                        for k in missing[keys].itertuples(index=False, name=None)
                    ),
                )
            )
        known_counts = (
            manual.loc[manual.mapped & ~manual.internal_standard]
            .groupby(keys[:3])
            .feature_id.nunique()
        )
        physical = table.loc[table.source_physical & ~table.internal_standard]
        for identity, part in physical.groupby(keys[:3]):
            if known_counts.get(identity, 0) != 1:
                continue
            true = part.loc[
                part.diagnostic_category.isin(
                    ["manual_correct_current", "manual_correct_currently_missing"]
                )
            ]
            wrong = part.loc[
                part.diagnostic_category.eq(
                    "different_peak_for_manually_curated_identity"
                )
            ]
            if true.feature_id.nunique() != 1 or wrong.empty:
                continue
            for field in CONTRASTS:
                a, b = (
                    true[prep.measurement.PREFIX + field].dropna(),
                    wrong[prep.measurement.PREFIX + field].dropna(),
                )
                if a.empty or b.empty:
                    continue
                assert a.nunique() == 1
                margin = float(a.iloc[0] - b.max())
                relation = (
                    "manual_peak_better"
                    if margin > 1e-9
                    else "wrong_peak_better"
                    if margin < -1e-9
                    else "tie"
                )
                pairs.append(
                    dict(
                        dataset=d,
                        stream=identity[0],
                        lipid_class=true.lipid_class.iloc[0],
                        field=field,
                        currently_missing=bool(
                            true.diagnostic_category.eq(
                                "manual_correct_currently_missing"
                            ).any()
                        ),
                        relation=relation,
                    )
                )
        for (s, c, category), part in physical.groupby(
            ["stream", "lipid_class", "diagnostic_category"]
        ):
            for field in CONTRASTS:
                v = part[prep.measurement.PREFIX + field].dropna()
                distributions.append(
                    dict(
                        dataset=d,
                        stream=s,
                        lipid_class=c,
                        category=category,
                        field=field,
                        candidate_options=len(part),
                        measurable=len(v),
                        median=v.median() if len(v) else float("nan"),
                    )
                )
        print("DESCRIPTIVE COMPETING-ION ANALYSIS", d, len(table), flush=True)
    assert total_manual == 92100 and current_correct == 82740
    coverage = pd.DataFrame(rows)
    coverage.to_csv(OUT / "coverage_by_dataset_mode_class.csv", index=False)
    counts = [
        "manual_annotations",
        "current_correct",
        "manual_rows_in_pool",
        "manual_rows_with_raw_contrast",
        "missing_manual_rows_with_raw_contrast",
    ]
    coverage.groupby(["stream", "lipid_class"], as_index=False)[counts].sum().to_csv(
        OUT / "coverage_by_mode_class.csv", index=False
    )
    pairs = pd.DataFrame(
        pairs,
        columns=[
            "dataset",
            "stream",
            "lipid_class",
            "field",
            "currently_missing",
            "relation",
        ],
    )
    pairs.to_pickle(OUT / "private/paired_contrasts.pkl")
    aggregate = []
    for (s, c, f, missing), part in pairs.groupby(
        ["stream", "lipid_class", "field", "currently_missing"]
    ):
        values = part.relation.value_counts()
        aggregate.append(
            dict(
                stream=s,
                lipid_class=c,
                field=f,
                currently_missing=missing,
                comparable_identities=len(part),
                manual_peak_better=int(values.get("manual_peak_better", 0)),
                wrong_peak_better=int(values.get("wrong_peak_better", 0)),
                ties=int(values.get("tie", 0)),
                contributing_project_groups=part.dataset.str[:7].nunique(),
            )
        )
    pd.DataFrame(
        aggregate,
        columns=[
            "stream",
            "lipid_class",
            "field",
            "currently_missing",
            "comparable_identities",
            "manual_peak_better",
            "wrong_peak_better",
            "ties",
            "contributing_project_groups",
        ],
    ).to_csv(OUT / "paired_peak_contrasts.csv", index=False)
    pd.DataFrame(distributions).to_csv(
        OUT / "descriptive_distributions.csv", index=False
    )
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    prep.common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            primary_manual_annotations=total_manual,
            primary_current_correct=current_correct,
            source_input_sha256=hashes,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            note="Prespecified descriptive comparative-ion coverage/peak contrasts, NOT accepted gains or external validation/FDR. Disjoint native other-polarity families only; no MS/MS, fragmentation confirmation, hard absence gate, new searches or annotations.",
        ),
    )


if __name__ == "__main__":
    main()

