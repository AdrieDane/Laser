"""Gold descriptive analysis only AFTER all33 blind AND independent receipts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
from diagnose_qe_raw_counterpart_pilot import option_categories
import prepare_qe_formula_isotope_pilot as prep
from verify_qe_bmp_context_pilot import eligibility

OUT = prep.common.QE / "formula_isotope_diagnostic_041_v1"
REFERENCE = prep.common.QE / "experiment_041"
CONTRASTS = {
    "upper_carbon_error": -1,
    "upper_interval_error": -1,
    "lower_inverse_interval_error": 1,
    "lower2_inverse_interval_error": 1,
}


def main():
    configure_frozen_resources(REPO)
    from standard_lipidomics.validation import _lipid_class

    blind_path = prep.OUT / "blind_receipt.json"
    independent_path = (
        prep.common.QE / "formula_isotope_independent_check_v1/receipt.json"
    )
    blind, checked = (json.loads(p.read_text()) for p in (blind_path, independent_path))
    assert blind["all_passed"] and checked["all_passed"] and checked["datasets"] == 33
    assert checked["manual_files_opened"] == blind["manual_files_opened"] == 0
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
    cover, pairs, distributions = [], [], []
    manual_total = current_total = 0
    for d in datasets:
        raw = prep.common.common.load_predictors(d, {folds[d]})
        saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
        # FIRST gold decoding in this pilot; both completed all33 gates above.
        manual = pd.read_pickle(
            REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        accepted = pd.read_csv(
            REFERENCE / "runs" / d / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        primary = manual.loc[~manual.lipid_class.eq("X4")]
        manual_total += len(primary)
        current_total += int(primary.same_manual_peak.sum())
        table = option_categories(raw["pool"], manual, accepted).join(saved["evidence"])
        table["dataset"] = d
        table["lipid_class"] = table.lipid_class.map(_lipid_class)
        protections = {
            s: prep.common.common.physical_protection(d, {folds[d]}, s, raw)
            for s in prep.common.MODES
        }
        table["source_physical"] = eligibility(raw["pool"], protections)
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
        known_counts = (
            manual.loc[manual.mapped & ~manual.internal_standard]
            .groupby(keys[:3])
            .feature_id.nunique()
        )
        for (s, cls), m in primary.groupby(["stream", "lipid_class"]):
            p = table.loc[table.stream.eq(s) & table.lipid_class.eq(cls)]
            exact = p.loc[
                p.diagnostic_category.isin(
                    ["manual_correct_current", "manual_correct_currently_missing"]
                )
            ]
            formula = set(
                exact.loc[
                    exact[prep.measurement.PREFIX + "available"].eq(1), keys
                ].itertuples(index=False, name=None)
            )
            upper = set(
                exact.loc[
                    exact[prep.measurement.PREFIX + "upper_log_ratio"].notna(), keys
                ].itertuples(index=False, name=None)
            )
            allkeys = list(m[keys].itertuples(index=False, name=None))
            cover.append(
                dict(
                    dataset=d,
                    stream=s,
                    lipid_class=cls,
                    manual_annotations=len(m),
                    current_correct=int(m.same_manual_peak.sum()),
                    correct_manual_peak_in_pool=len(exact),
                    manual_rows_with_formula=sum(k in formula for k in allkeys),
                    manual_rows_with_upper_measurement=sum(k in upper for k in allkeys),
                )
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
            for field, direction in CONTRASTS.items():
                a = true[prep.measurement.PREFIX + field].dropna()
                b = wrong[prep.measurement.PREFIX + field].dropna()
                if a.empty or b.empty:
                    continue
                assert a.nunique() == 1
                margin = direction * float(a.iloc[0]) - (direction * b).max()
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
        for (s, cls, category), part in physical.groupby(
            ["stream", "lipid_class", "diagnostic_category"]
        ):
            for field in CONTRASTS:
                values = part[prep.measurement.PREFIX + field].dropna()
                distributions.append(
                    dict(
                        dataset=d,
                        stream=s,
                        lipid_class=cls,
                        category=category,
                        field=field,
                        candidate_options=len(part),
                        measurable=len(values),
                        median=values.median() if len(values) else np.nan,
                    )
                )
        print("DESCRIPTIVE FORMULA ISOTOPE ANALYSIS", d, len(table), flush=True)
    assert manual_total == 92100 and current_total == 82740
    coverage = pd.DataFrame(cover)
    coverage.to_csv(OUT / "coverage_by_dataset_mode_class.csv", index=False)
    coverage.groupby(["stream", "lipid_class"], as_index=False)[
        [
            "manual_annotations",
            "current_correct",
            "correct_manual_peak_in_pool",
            "manual_rows_with_formula",
            "manual_rows_with_upper_measurement",
        ]
    ].sum().to_csv(OUT / "coverage_by_mode_class.csv", index=False)
    pairs = pd.DataFrame(pairs)
    pairs.to_pickle(OUT / "private/paired_contrasts.pkl")
    aggregates = []
    for (s, c, f, missing), part in pairs.groupby(
        ["stream", "lipid_class", "field", "currently_missing"]
    ):
        counts = part.relation.value_counts()
        aggregates.append(
            dict(
                stream=s,
                lipid_class=c,
                field=f,
                currently_missing=missing,
                comparable_identities=len(part),
                manual_peak_better=int(counts.get("manual_peak_better", 0)),
                wrong_peak_better=int(counts.get("wrong_peak_better", 0)),
                ties=int(counts.get("tie", 0)),
                contributing_project_groups=part.dataset.str[:7].nunique(),
            )
        )
    pd.DataFrame(aggregates).to_csv(OUT / "paired_peak_contrasts.csv", index=False)
    pd.DataFrame(distributions).to_csv(
        OUT / "descriptive_distributions.csv", index=False
    )
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    prep.common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            primary_manual_annotations=manual_total,
            primary_current_correct=current_total,
            source_input_sha256=hashes,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            note="Prespecified descriptive contrasts on exposed DEVELOPMENT data; not an annotation benchmark, external validation, new matches or FDR. All original manual denominators retained. Native mass validity and isotope descriptors never changed calls.",
        ),
    )


if __name__ == "__main__":
    main()
