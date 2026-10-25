"""Retrospective descriptive distributions AFTER ALL33 independent checks.

No threshold selection, new fits, accepted calls or annotation-performance
experiment. Unconfirmed opposite-polarity calls are not proven false positives.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_bmp_context_pilot as prep
from bmp_context_measurements import PREFIX, FIELDS
from diagnose_qe_raw_counterpart_pilot import option_categories

OUT = prep.common.QE / "BMP_candidate_context_diagnostic_041_v1"


def main():
    configure_frozen_resources(REPO)
    receipt_path = prep.OUT / "blind_receipt.json"
    checked_path = (
        prep.common.QE / "BMP_candidate_context_independent_check_v1/receipt.json"
    )
    receipt, checked = (json.loads(p.read_text()) for p in (receipt_path, checked_path))
    assert receipt["all_passed"] and checked["all_passed"] and checked["datasets"] == 33
    assert checked["blind_receipt_sha256"] == prep.common.sha(receipt_path)
    assert receipt["manual_files_opened"] == checked["manual_files_opened"] == 0
    assert not receipt["annotation_changes"] and not checked["annotation_changes"]
    hashes = {
        **checked["source_input_sha256"],
        str(checked_path): prep.common.sha(checked_path),
        str(Path(__file__)): prep.common.sha(Path(__file__)),
    }
    datasets, folds = prep.common.inventory()
    for d in datasets:
        for p in (
            prep.common.OUT / "evaluation_v2" / d / "manual_comparison.pkl",
            prep.common.OUT / "runs" / d / "automatic_annotations.csv",
        ):
            hashes[str(p)] = prep.common.sha(p)
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    options, rows = [], []
    for d in datasets:
        pool = prep.common.common.load_predictors(d, {folds[d]})["pool"]
        saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
        # First reference-table decoding in THIS analysis stage. All33blind
        # physical files and independent checks already complete above.
        manual = pd.read_pickle(
            prep.common.OUT / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        accepted = pd.read_csv(
            prep.common.OUT / "runs" / d / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        table = option_categories(pool.loc[saved["evidence"].index], manual, accepted)
        table = table.join(saved["evidence"])
        assert table.index.equals(saved["evidence"].index)
        opposite = set(
            (r.stream, prep.common.common.species(r.species))
            for r in manual.loc[
                ~manual.internal_standard & manual.lipid_class.eq("BMP")
            ].itertuples()
        )
        table["opposite_NP_manual_composition_support"] = [
            (
                "NP_Neg" if r.stream == "NP_Pos" else "NP_Pos",
                prep.common.common.species(r.species),
            )
            in opposite
            for r in table.itertuples()
        ]
        table["dataset"] = d
        table.to_pickle(OUT / "private" / (d + ".pkl"))
        options.append(table)
        for s in ("NP_Pos", "NP_Neg"):
            sub = table.loc[table.stream.eq(s)]
            rows.append(
                dict(
                    dataset=d,
                    stream=s,
                    candidate_options=len(sub),
                    correct_current_options=int(
                        sub.diagnostic_category.eq("manual_correct_current").sum()
                    ),
                    missing_manual_correct_options=int(
                        sub.diagnostic_category.eq(
                            "manual_correct_currently_missing"
                        ).sum()
                    ),
                    wrong_peak_options=int(
                        sub.diagnostic_category.eq(
                            "different_peak_for_manually_curated_identity"
                        ).sum()
                    ),
                    unconfirmed_options=int(
                        sub.diagnostic_category.isin(
                            [
                                "manual_absent_identity_unconfirmed",
                                "alternative_label_on_manual_peak_unconfirmed",
                            ]
                        ).sum()
                    ),
                )
            )
    combined = pd.concat(options, ignore_index=True)
    summaries = []
    grouping = [
        "stream",
        "diagnostic_category",
        "opposite_NP_manual_composition_support",
    ]
    for key, group in combined.groupby(grouping):
        for field in FIELDS:
            values = pd.to_numeric(group[PREFIX + field], errors="coerce")
            finite = values.loc[np.isfinite(values)]
            q = finite.quantile([0.25, 0.5, 0.75])
            summaries.append(
                dict(
                    zip(grouping, key),
                    field=field,
                    candidate_options=len(group),
                    finite_options=len(finite),
                    q25=q.loc[0.25],
                    median=q.loc[0.5],
                    q75=q.loc[0.75],
                )
            )
    pd.DataFrame(summaries).to_csv(OUT / "descriptive_distributions.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "categories_by_dataset_mode.csv", index=False)
    assert len(rows) == 66 and all(
        prep.common.sha(Path(p)) == h for p, h in hashes.items()
    )
    prep.common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            source_input_sha256=hashes,
            candidate_options=len(combined),
            classifier_fits=0,
            annotation_changes=False,
            thresholds_selected=False,
            note="Descriptive candidate-option distributions, not accepted calls or annotation-performance gains. Labels first decoded only after completeALL33blind+independent receipts. Opposite NP curated composition is an evaluation category ONLY, never runtime input or proof of chemical identity. Whole041 retained; all unconfirmed extras remain counted as such in primary benchmarks.",
        ),
    )
    print(
        "ALL33 BMP CONTEXT DESCRIPTIVE DIAGNOSTIC COMPLETE", len(combined), flush=True
    )


if __name__ == "__main__":
    main()
