"""Descriptive labels ONLY AFTER all33 blind graphs AND independent checks.

No annotation/classifier/cutoff changes. Availability is not accepted gains.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
from diagnose_qe_raw_counterpart_pilot import option_categories
import prepare_qe_cross_phase_candidate_pilot as prep

OUT = prep.common.QE / "cross_phase_class_candidate_diagnostic_041_v1"
REFERENCE = prep.common.QE / "experiment_041"


def main():
    configure_frozen_resources(REPO)
    from standard_lipidomics.validation import _lipid_class

    blind_path = prep.OUT / "blind_receipt.json"
    checked_path = (
        prep.common.QE / "cross_phase_class_candidate_independent_check_v1/receipt.json"
    )
    blind, checked = (json.loads(p.read_text()) for p in (blind_path, checked_path))
    assert blind["all_passed"] and checked["all_passed"] and checked["datasets"] == 33
    assert blind["manual_files_opened"] == checked["manual_files_opened"] == 0
    assert checked["blind_receipt_sha256"] == prep.common.sha(blind_path)
    datasets, folds = prep.common.inventory()
    hashes = {
        **checked["source_input_sha256"],
        str(checked_path): prep.common.sha(checked_path),
        str(blind_path): prep.common.sha(blind_path),
        str(Path(__file__)): prep.common.sha(Path(__file__)),
    }
    for d in datasets:
        for path in (
            REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl",
            REFERENCE / "runs" / d / "automatic_annotations.csv",
        ):
            hashes[str(path)] = prep.common.sha(path)
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    tables, rows = [], []
    for d in datasets:
        raw = prep.common.common.load_predictors(d, {folds[d]})
        saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
        # First reference-label decoding in this pipeline; BOTHall33 gates above.
        manual = pd.read_pickle(
            REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        accepted = pd.read_csv(
            REFERENCE / "runs" / d / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        table = option_categories(
            raw["pool"].loc[saved["evidence"].index], manual, accepted
        )
        table = table.join(saved["evidence"])
        table["dataset"] = d
        table["lipid_class"] = table.lipid_class.map(_lipid_class)
        old = set(prep.common.common.accepted_keys(accepted))
        ids = {k[:4] for k in old}
        occupied = {(k[0], k[4]) for k in old}
        table["source_is_free"] = [
            (r.stream, r.species, r.adduct, False) not in ids
            and (r.stream, str(r.feature_id)) not in occupied
            for r in table.itertuples()
        ]
        table.to_pickle(OUT / "private" / (d + ".pkl"))
        tables.append(table)
        for s in prep.measurement.MODES:
            for cls in ("FA", "FA-OH"):
                part = table.loc[table.stream.eq(s) & table.lipid_class.eq(cls)]
                sub = manual.loc[manual.stream.eq(s) & manual.lipid_class.eq(cls)]
                eligible = part.source_is_free & part[
                    prep.measurement.PREFIX + "source_physical"
                ].eq(1)
                keys = ["stream", "species", "adduct", "feature_id"]
                physical = set(
                    part.loc[eligible, keys].itertuples(index=False, name=None)
                )
                profile = set(
                    part.loc[
                        eligible
                        & part[prep.measurement.PREFIX + "counterpart_profiles"].gt(0),
                        keys,
                    ].itertuples(index=False, name=None)
                )
                missing = [
                    r
                    for r in sub.itertuples()
                    if not r.same_manual_peak and r.mapped and not r.internal_standard
                ]
                rows.append(
                    dict(
                        dataset=d,
                        stream=s,
                        lipid_class=cls,
                        manual_rows=len(sub),
                        current_correct=int(sub.same_manual_peak.sum()),
                        candidate_options=len(part),
                        missing_free_physical_manual_rows=sum(
                            (r.stream, r.species, r.adduct, str(r.feature_id))
                            in physical
                            for r in missing
                        ),
                        missing_free_profile_measurable_manual_rows=sum(
                            (r.stream, r.species, r.adduct, str(r.feature_id))
                            in profile
                            for r in missing
                        ),
                    )
                )
    combined = pd.concat(tables, ignore_index=True)
    summaries = []
    grouping = ["stream", "lipid_class", "diagnostic_category"]
    for key, part in combined.groupby(grouping):
        for field in prep.measurement.FIELDS:
            values = pd.to_numeric(
                part[prep.measurement.PREFIX + field], errors="coerce"
            )
            finite = values.loc[np.isfinite(values)]
            q = finite.quantile([0.25, 0.5, 0.75])
            summaries.append(
                dict(
                    zip(grouping, key),
                    field=field,
                    candidate_options=len(part),
                    finite_options=len(finite),
                    q25=q.loc[0.25],
                    median=q.loc[0.5],
                    q75=q.loc[0.75],
                )
            )
    pd.DataFrame(summaries).to_csv(OUT / "descriptive_distributions.csv", index=False)
    context = pd.DataFrame(rows)
    assert len(context) == 132 and context.dataset.nunique() == 33
    context.to_csv(OUT / "availability_by_dataset_mode_class.csv", index=False)
    totals = context.groupby(["stream", "lipid_class"], as_index=False)[
        [
            "manual_rows",
            "current_correct",
            "candidate_options",
            "missing_free_physical_manual_rows",
            "missing_free_profile_measurable_manual_rows",
        ]
    ].sum()
    totals.to_csv(OUT / "availability_by_mode_class.csv", index=False)
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    prep.common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            source_input_sha256=hashes,
            note="Descriptive availability and candidate-option distributions,NOT accepted annotations/gains. Bothall33blind+independentgates beforequerylabels. Fullmanualclassdenominators remaininclspikes/unresolved. Onlytwo negative modes/twoclasses;132contextrows isNOT132four-modebenchmark. Extensively exposed developmentdata,notuntouchedvalidation/FDR.",
        ),
    )
    print(totals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
