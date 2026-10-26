"""Descriptive comparison only AFTER all33 blind and independent gates."""

import json

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import matrix_peer_pilot_data as data
import matrix_peer_composition as peer
from diagnose_qe_raw_counterpart_pilot import option_categories


def main():
    configure_frozen_resources(data.OUT / "source_snapshot")
    blind_path = data.OUT / "blind_receipt.json"
    check_path = data.CHECK / "receipt.json"
    blind, check = (json.loads(p.read_text()) for p in (blind_path, check_path))
    assert blind["all_passed"] and check["all_passed"] and check["datasets"] == 33
    assert check["blind_receipt_sha256"] == data.sha(blind_path)
    assert (
        blind["query_gold_validation_tables_opened"]
        == check["query_gold_validation_tables_opened"]
        == 0
    )
    assert check["source_input_sha256"] == data.inputs()
    data.DIAG.mkdir(exist_ok=False)
    (data.DIAG / "private").mkdir()
    from annotator.composition_ranges import canonical_lipid_class

    datasets, folds = data.inventory()
    summaries, cases = [], []
    for dataset in datasets:
        original = data.load_pool(dataset, folds[dataset])
        pool = original["pool"]
        measured = pd.read_pickle(data.OUT / "private" / (dataset + ".pkl"))
        # FIRST actual heldout query validation labels, after BOTH all33 gates.
        manual = pd.read_pickle(
            data.REFERENCE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        accepted = pd.read_csv(
            data.REFERENCE / "runs" / dataset / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        labels = option_categories(pool, manual, accepted)
        labels["lipid_class"] = pool["class"].map(canonical_lipid_class)
        table = labels.join(measured)
        keys = ["stream", "species", "adduct", "feature_id"]
        assert (
            table.groupby(keys, dropna=False)[list(measured)]
            .nunique(dropna=False)
            .le(1)
            .all()
            .all()
        )
        table = table.drop_duplicates(keys)
        table.to_pickle(data.DIAG / "private" / (dataset + ".pkl"))
        for stream in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"):
            part = table.loc[table.stream.eq(stream)]
            cases.append(
                dict(
                    dataset=dataset,
                    stream=stream,
                    candidate_options=len(part),
                    supported_options=int(
                        part[peer.PREFIX + peer.FIELDS[0]].notna().sum()
                    ),
                    missing_correct_options=int(
                        part.diagnostic_category.eq(
                            "manual_correct_currently_missing"
                        ).sum()
                    ),
                    supported_missing_correct_options=int(
                        (
                            part.diagnostic_category.eq(
                                "manual_correct_currently_missing"
                            )
                            & part[peer.PREFIX + peer.FIELDS[0]].notna()
                        ).sum()
                    ),
                )
            )
        for (stream, cls, category), group in table.groupby(
            ["stream", "lipid_class", "diagnostic_category"]
        ):
            for field in measured:
                numeric = group[field].loc[np.isfinite(group[field])]
                summaries.append(
                    dict(
                        dataset=dataset,
                        stream=stream,
                        lipid_class=cls,
                        category=category,
                        field=field,
                        candidate_options=len(group),
                        measured_options=len(numeric),
                        p10=numeric.quantile(0.1),
                        median=numeric.median(),
                        p90=numeric.quantile(0.9),
                    )
                )
        print(
            "MATRIX PEER DESCRIPTIVE QUERY DIAGNOSTIC ONLY",
            dataset,
            len(table),
            flush=True,
        )
    assert len(cases) == 132 and check["source_input_sha256"] == data.inputs()
    pd.DataFrame(cases).to_csv(
        data.DIAG / "availability_by_dataset_mode.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        data.DIAG / "descriptive_by_dataset_mode_class.csv", index=False
    )
    data.dump(
        data.DIAG / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            query_gold_validation_tables_opened=33,
            independent_receipt_sha256=data.sha(check_path),
            source_input_sha256=check["source_input_sha256"],
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            note="Conditional composition descriptors,NOTaccepted gains/extras/FDR. Wrongoriginalpeaksnecessarilyreceiveidenticalcompositionevidence. Same041retained;nohardlimits. AuthorizedOTHERPROJECTmanualpresenceusedforreferencefitting;ownqueryvalidationonlyafterALL33gates.",
        ),
    )
    print("ALL33 MATRIX PEER DESCRIPTIVE DIAGNOSTIC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
