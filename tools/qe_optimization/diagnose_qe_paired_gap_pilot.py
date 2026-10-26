"""Query curation ONLY after ALL33 blind paired-gap outputs and verification.

Candidate coverage/RT contrasts are not accepted gains, extras or chemical FDR.
No rule/threshold/classifier/annotation changes are authorized by this report.
"""

import json

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import paired_gap_pilot_data as data
import paired_retention_gap as gap
import prepare_qe_paired_gap_pilot as prep

OUT = data.QE / "paired_isomer_gap_diagnostic_041_v1"
CHECK = data.QE / "paired_isomer_gap_independent_check_v1/receipt.json"


def evaluate_query(dataset, predictions, graph, manual, baseline):
    from standard_lipidomics.validation import _species, canonical_feature_id
    from matlab_annotation.standard_rt_library import ion_key
    from annotator.composition_ranges import canonical_lipid_class

    m = manual.loc[~manual.internal_standard].copy()
    m["lipid_class"] = m.lipid_class.map(canonical_lipid_class)
    m["adduct"] = m.adduct.map(ion_key)
    mapped = m.loc[m.mapped & ~m.duplicate_manual_entry]
    exact = {
        (r.stream, r.species, r.adduct, canonical_feature_id(r.feature_id))
        for r in mapped.itertuples()
    }
    identities = set(
        m[["stream", "species", "adduct"]].itertuples(index=False, name=None)
    )
    originals = {}
    for s, p in baseline["streams"].items():
        features = p["features"]
        originals[s] = features.set_index(features.feature_id.map(canonical_feature_id))
    annotated = graph.copy()
    if len(annotated):
        annotated["species"] = annotated["name"].map(_species)
        annotated["adduct"] = annotated.adduct.map(ion_key)
        annotated["lipid_class"] = annotated["class"].map(canonical_lipid_class)
        annotated["same_manual_peak"] = [
            (r.stream, r.species, r.adduct, canonical_feature_id(r.feat_feature_id))
            in exact
            for r in annotated.itertuples()
        ]
        annotated["identity_in_manual"] = [
            (r.stream, r.species, r.adduct) in identities
            for r in annotated.itertuples()
        ]
    else:
        for column, dtype in (
            ("species", str),
            ("lipid_class", str),
            ("same_manual_peak", bool),
            ("identity_in_manual", bool),
        ):
            annotated[column] = pd.Series(dtype=dtype)
    rows = []
    for index, r in predictions.iterrows():
        species, ion = _species(r["name"]), ion_key(r.adduct)
        target = m.loc[m.stream.eq(r.stream) & m.species.eq(species) & m.adduct.eq(ion)]
        part = annotated.loc[
            annotated.prediction_index.eq(index) & annotated.physical_partner_eligible
        ]
        partner_key = (
            r.stream,
            _species(r.partner_name),
            ion,
            canonical_feature_id(r.partner_feature_id),
        )
        row = dict(
            r,
            dataset=dataset,
            species=species,
            lipid_class=canonical_lipid_class(r["class"]),
            partner_matches_manual=partner_key in exact,
            target_manual_rows=len(target),
            target_is_unique_mapped=False,
            target_was_correct=False,
            target_is_missing_mapped=False,
            target_RT_error_seconds=np.nan,
            target_raw_ppm=np.nan,
            correct_target_in_graph=False,
            correct_target_free=False,
            correct_target_new_to_core=False,
            has_wrong_physical_alternative=False,
            true_gap_error_better=False,
            wrong_gap_error_better=False,
            gap_error_tied=False,
        )
        unique = target.loc[target.mapped & ~target.duplicate_manual_entry]
        if len(unique) == 1:
            chosen = unique.iloc[0]
            fid = canonical_feature_id(chosen.feature_id)
            raw = originals[r.stream].loc[fid]
            correct = part.loc[part.feat_feature_id.map(canonical_feature_id).eq(fid)]
            wrong = part.loc[~part.feat_feature_id.map(canonical_feature_id).eq(fid)]
            error = float(raw.rt) * 60 - r.predicted_rt_seconds
            row.update(
                target_is_unique_mapped=True,
                target_was_correct=bool(chosen.same_manual_peak),
                target_is_missing_mapped=not bool(chosen.same_manual_peak),
                target_RT_error_seconds=error,
                target_raw_ppm=(float(raw.mz) / r.ref_mz - 1) * 1e6,
                correct_target_in_graph=bool(len(correct)),
                correct_target_free=bool(
                    len(correct)
                    and (correct.identity_is_free & correct.feature_is_free).any()
                ),
                correct_target_new_to_core=bool(
                    len(correct) and correct.novel_to_old029_core_pool.any()
                ),
                has_wrong_physical_alternative=bool(len(wrong)),
            )
            if len(correct) and len(wrong):
                true_error = float(correct.paired_rt_error_seconds.abs().min())
                wrong_error = float(wrong.paired_rt_error_seconds.abs().min())
                row.update(
                    true_gap_error_better=true_error < wrong_error - 1e-10,
                    wrong_gap_error_better=wrong_error < true_error - 1e-10,
                    gap_error_tied=abs(true_error - wrong_error) <= 1e-10,
                )
        rows.append(row)
    return pd.DataFrame(rows), annotated, m


def main():
    configure_frozen_resources(data.OUT / "source_snapshot")
    blind_path = data.OUT / "blind_receipt.json"
    blind, checked = (json.loads(p.read_text()) for p in (blind_path, CHECK))
    assert blind["all_passed"] and checked["all_passed"] and checked["datasets"] == 33
    assert (
        blind["query_gold_validation_tables_opened"]
        == checked["query_gold_validation_tables_opened"]
        == 0
    )
    assert checked["blind_receipt_sha256"] == data.sha(blind_path)
    assert checked["source_input_sha256"] == prep.source_inputs()
    hashes = dict(checked["source_input_sha256"])
    hashes.update({str(CHECK): data.sha(CHECK), str(blind_path): data.sha(blind_path)})
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    all_predictions, case_rows, class_rows, contrasts = [], [], [], []
    datasets, _ = data.inventory()
    from annotator.composition_ranges import canonical_lipid_class

    for d in datasets:
        folder = data.OUT / "private" / d
        predictions = pd.read_pickle(folder / "predictions.pkl")
        graph = pd.read_pickle(folder / "candidate_graph.pkl")
        # FIRST heldout-query curation validation, both complete numeric gates above.
        manual = pd.read_pickle(
            data.REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        baseline = pd.read_pickle(data.BASE / "runs" / d / "full_annotation.pkl")
        evaluated, annotated, m = evaluate_query(
            d, predictions, graph, manual, baseline
        )
        evaluated.to_pickle(OUT / "private" / (d + "_prediction_labels.pkl"))
        annotated.to_pickle(OUT / "private" / (d + "_graph_labels.pkl"))
        if len(evaluated):
            all_predictions.append(evaluated)
        for stream in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"):
            q = (
                evaluated.loc[evaluated.stream.eq(stream)]
                if len(evaluated)
                else evaluated
            )
            h = annotated.loc[annotated.stream.eq(stream)]
            case_rows.append(
                dict(
                    dataset=d,
                    stream=stream,
                    applicable=stream.startswith("RP_"),
                    numeric_predictions=len(q),
                    raw_candidate_rows=len(h),
                    distinct_ordered_candidates=int(h.physical_partner_eligible.sum()),
                    missing_correct_targets_in_graph=int(
                        (q.target_is_missing_mapped & q.correct_target_in_graph).sum()
                    )
                    if len(q)
                    else 0,
                    missing_correct_targets_free=int(
                        (q.target_is_missing_mapped & q.correct_target_free).sum()
                    )
                    if len(q)
                    else 0,
                )
            )
        for family, (stream, ion, a, b, _, _) in gap.FAMILIES.items():
            for cls in (a, b):
                cls = canonical_lipid_class(cls)
                q = (
                    evaluated.loc[
                        evaluated.stream.eq(stream) & evaluated.lipid_class.eq(cls)
                    ]
                    if len(evaluated)
                    else evaluated
                )
                sub = m.loc[m.stream.eq(stream) & m.lipid_class.eq(cls)]
                primary = sub.loc[sub.adduct.eq(ion)]
                class_rows.append(
                    dict(
                        dataset=d,
                        stream=stream,
                        family=family,
                        lipid_class=cls,
                        all_channel_manual_endogenous_rows=len(sub),
                        primary_channel_manual_rows=len(primary),
                        current_primary_correct=int(primary.same_manual_peak.sum()),
                        numeric_predictions=len(q),
                        unique_mapped_target_predictions=int(
                            q.target_is_unique_mapped.sum()
                        )
                        if len(q)
                        else 0,
                        manual_matching_source_partners=int(
                            q.partner_matches_manual.sum()
                        )
                        if len(q)
                        else 0,
                        missing_correct_targets_in_graph=int(
                            (
                                q.target_is_missing_mapped & q.correct_target_in_graph
                            ).sum()
                        )
                        if len(q)
                        else 0,
                        missing_correct_targets_free=int(
                            (q.target_is_missing_mapped & q.correct_target_free).sum()
                        )
                        if len(q)
                        else 0,
                        missing_correct_targets_free_and_new_to_core=int(
                            (
                                q.target_is_missing_mapped
                                & q.correct_target_free
                                & q.correct_target_new_to_core
                            ).sum()
                        )
                        if len(q)
                        else 0,
                    )
                )
        if len(evaluated):
            paired = evaluated.loc[
                evaluated.target_is_unique_mapped
                & evaluated.correct_target_in_graph
                & evaluated.has_wrong_physical_alternative
            ]
            for (stream, family, before), part in paired.groupby(
                ["stream", "family", "target_was_correct"]
            ):
                contrasts.append(
                    dict(
                        dataset=d,
                        stream=stream,
                        family=family,
                        target_previously_correct=bool(before),
                        paired_cases=len(part),
                        true_gap_closer=int(part.true_gap_error_better.sum()),
                        wrong_gap_closer=int(part.wrong_gap_error_better.sum()),
                        gap_tied=int(part.gap_error_tied.sum()),
                        source_partner_matches_manual=int(
                            part.partner_matches_manual.sum()
                        ),
                    )
                )
        print("PAIRED-GAP QUERY DIAGNOSTIC ONLY", d, len(evaluated), flush=True)
    cases, classes = pd.DataFrame(case_rows), pd.DataFrame(class_rows)
    assert len(cases) == 132 and len(classes) == 264
    cases.to_csv(OUT / "by_dataset_mode_availability.csv", index=False)
    classes.to_csv(OUT / "by_dataset_mode_class_availability.csv", index=False)
    numeric = [
        c for c in classes if c not in {"dataset", "stream", "family", "lipid_class"}
    ]
    classes.groupby(["stream", "family", "lipid_class"], as_index=False)[
        numeric
    ].sum().to_csv(OUT / "by_mode_class_availability.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "paired_gap_peak_contrasts.csv", index=False)
    combined = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    combined.to_pickle(OUT / "private/all_prediction_labels.pkl")
    assert checked["source_input_sha256"] == prep.source_inputs()
    summary = dict(
        all_passed=True,
        datasets=33,
        dataset_modes=132,
        query_gold_validation_tables_opened=33,
        annotation_changes=False,
        annotation_outputs=0,
        annotation_classifier_fits=0,
        acceptance_cutoffs_selected=False,
        numeric_predictions=len(combined),
        raw_candidate_rows=int(cases.raw_candidate_rows.sum()),
        missing_correct_targets_covered=int(
            classes.missing_correct_targets_in_graph.sum()
        ),
        missing_correct_targets_free=int(classes.missing_correct_targets_free.sum()),
        missing_correct_targets_free_and_new_to_core=int(
            classes.missing_correct_targets_free_and_new_to_core.sum()
        ),
        independent_receipt_sha256=data.sha(CHECK),
        source_input_sha256=hashes,
        note="Candidateopportunities andconditional RTcontrasts,NOTaccepted gains/extras/chemicalFDR. Same041retainedannotationsunchanged. Primary-channel endogenous diagnostic,not a replacement92100denominator. NPnotapplicable. CuratedOTHERPROJECTtrainingusedbeforeblindgate;ownquerygoldvalidationonlyafterALL33independentchecks. No new actual049method authorized.",
    )
    data.dump(OUT / "receipt.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "source_input_sha256"}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
