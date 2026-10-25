"""Retrospective measurement separation ONLY after ALL33 blind tables freeze.

NO training/selection/annotations. Counts are available hypotheses, NOT new
correct annotations. Manual-absent identities and alternative labels on a
curated peak are unconfirmed, not proven false positives.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_raw_counterpart_pilot as prep
import run_qe_add_only_class_experiment as common

OUT = common.QE / "raw_counterpart_diagnostic_041_v1"


def option_categories(pool, manual, accepted):
    """Disjoint categories, original FID+ion+normalized identity comparison."""
    from standard_lipidomics.validation import _normalise
    from matlab_annotation.standard_rt_library import ion_key

    p = _normalise(pool.assign(adduct=pool.adduct.map(ion_key)), manual=False)
    p.index = pool.index
    a = _normalise(accepted.assign(adduct=accepted.adduct.map(ion_key)), manual=False)
    manual = manual.loc[~manual.internal_standard & ~manual.lipid_class.eq("X4")]
    a = a.loc[~a.internal_standard]
    known = manual.loc[manual.mapped]
    keys = ["stream", "species", "adduct", "feature_id"]
    exact = set(known[keys].itertuples(index=False, name=None))
    accepted_exact = set(a[keys].itertuples(index=False, name=None))
    unresolved = set(
        manual.loc[~manual.mapped, keys[:3]].itertuples(index=False, name=None)
    )
    identities = set(known[keys[:3]].itertuples(index=False, name=None))
    curated_features = set(
        known[["stream", "feature_id"]].itertuples(index=False, name=None)
    )
    status = []
    for key, standard in zip(
        p[keys].itertuples(index=False, name=None), p.internal_standard
    ):
        if standard:
            category = "internal_standard_excluded"
        elif key in exact:
            category = (
                "manual_correct_current"
                if key in accepted_exact
                else "manual_correct_currently_missing"
            )
        elif key[:3] in unresolved:
            category = "unresolved_manual_identity_unknown"
        elif key[:3] in identities:
            category = "different_peak_for_manually_curated_identity"
        elif (key[0], key[3]) in curated_features:
            category = "alternative_label_on_manual_peak_unconfirmed"
        else:
            category = "manual_absent_identity_unconfirmed"
        status.append(category)
    p["diagnostic_category"] = status
    return p


def main():
    configure_frozen_resources(REPO)
    from matlab_annotation.raw_counterpart_evidence import PREFIX, FIELDS
    from annotator.composition_ranges import canonical_lipid_class

    receipt_path = prep.OUT / "blind_physical_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    independent_path = (
        common.QE / "raw_counterpart_feature_independent_check_v1/receipt.json"
    )
    independent = json.loads(independent_path.read_text())
    assert independent["all_passed"] and independent["datasets"] == 33
    assert independent["manual_files_opened"] == 0
    assert independent["blind_receipt_sha256"] == common.sha(receipt_path)
    assert receipt["all_passed"] and receipt["all33_physical_tables_frozen"]
    assert receipt["manual_files_opened"] == receipt["classifier_fits"] == 0
    assert all(
        common.sha(Path(p)) == h for p, h in receipt["source_input_sha256"].items()
    )
    assert all(common.sha(Path(p)) == h for p, h in receipt["physical_sha256"].items())
    datasets, folds = common.inventory()
    paths = {
        Path(__file__),
        Path(prep.__file__),
        receipt_path,
        independent_path,
        common.OUT / "integrity_receipt.json",
    }
    for d in datasets:
        paths.update(
            [
                common.OUT / "evaluation_v2" / d / "manual_comparison.pkl",
                common.OUT / "runs" / d / "automatic_annotations.csv",
            ]
        )
    hashes = {str(p): common.sha(p) for p in paths}
    assert json.loads((common.OUT / "integrity_receipt.json").read_text())["all_passed"]
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    metrics, ranks, target_counts = [], [], Counter()
    fields = [PREFIX + x for x in FIELDS]
    rank_fields = [
        "best_raw_correlation",
        "best_normalized_correlation",
        "best_within_group_correlation",
        "best_leave_one_out_correlation",
        "best_RT_supported_correlation",
    ]
    total_manual = total_current = 0
    for d in datasets:
        pool = common.common.load_predictors(d, {folds[d]})["pool"]
        saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
        evidence = saved["evidence"]
        assert evidence.index.equals(pool.index) and list(evidence) == fields
        # Labels first opened here, AFTER the all33 physical receipt checks.
        manual = pd.read_pickle(
            common.OUT / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        accepted = pd.read_csv(
            common.OUT / "runs" / d / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        relevant = manual.loc[~manual.internal_standard & ~manual.lipid_class.eq("X4")]
        total_manual += len(relevant)
        total_current += int(relevant.same_manual_peak.sum())
        p = option_categories(pool, manual, accepted)
        p["lipid_class"] = pool["class"].map(canonical_lipid_class)
        keys = ["stream", "species", "adduct", "feature_id"]
        # Candidate-construction versions cannot inflate counts. Measurements
        # must agree for the same exact identity/ion/original feature tuple.
        combined = pd.concat(
            [p[keys + ["lipid_class", "diagnostic_category"]], evidence], axis=1
        )
        assert (
            combined.groupby(keys, dropna=False)[fields]
            .nunique(dropna=False)
            .le(1)
            .all()
            .all()
        )
        combined = combined.drop_duplicates(keys).loc[
            ~combined.diagnostic_category.eq("internal_standard_excluded")
        ]
        combined.to_pickle(OUT / "private" / (d + ".pkl"))
        for (stream, cls, category), group in combined.groupby(
            ["stream", "lipid_class", "diagnostic_category"]
        ):
            target_counts[(d, stream, cls, category)] += len(group)
            for f in fields:
                values = group[f].dropna()
                metrics.append(
                    dict(
                        dataset=d,
                        stream=stream,
                        lipid_class=cls,
                        category=category,
                        measurement=f.removeprefix(PREFIX),
                        unique_options=len(group),
                        measured_options=len(values),
                        median=values.median() if len(values) else np.nan,
                        p10=values.quantile(0.1) if len(values) else np.nan,
                        p90=values.quantile(0.9) if len(values) else np.nan,
                    )
                )
        # Pure identity-local raw-evidence comparison, NOT a proposed global
        # assignment. Ignores occupancy/spike/final selection and is not gain.
        for identity, group in combined.groupby(keys[:3]):
            truth = group.diagnostic_category.isin(
                ["manual_correct_current", "manual_correct_currently_missing"]
            )
            wrong = group.diagnostic_category.eq(
                "different_peak_for_manually_curated_identity"
            )
            if not truth.any() or not wrong.any():
                continue
            for field in rank_fields:
                t = group.loc[truth, PREFIX + field].dropna()
                w = group.loc[wrong, PREFIX + field].dropna()
                if t.empty or w.empty:
                    relation, gap = "unsupported_comparison", np.nan
                else:
                    gap = float(t.max() - w.max())
                    relation = (
                        "manual_peak_higher"
                        if gap > 1e-9
                        else "different_peak_higher"
                        if gap < -1e-9
                        else "tie"
                    )
                ranks.append(
                    dict(
                        dataset=d,
                        stream=identity[0],
                        lipid_class=group.lipid_class.iloc[0],
                        measurement=field,
                        relation=relation,
                        gap=gap,
                        manual_peak_currently_missing=bool(
                            group.loc[truth, "diagnostic_category"]
                            .eq("manual_correct_currently_missing")
                            .any()
                        ),
                    )
                )
        print("RETROSPECTIVE RAW MEASUREMENT DIAGNOSTIC", d, len(combined), flush=True)
    assert total_manual == 92100 and total_current == 82740
    table = pd.DataFrame(metrics)
    table.to_csv(OUT / "by_dataset_mode_class_category_measurement.csv", index=False)
    counts = pd.DataFrame(
        [
            dict(dataset=d, stream=s, lipid_class=c, category=k, unique_options=n)
            for (d, s, c, k), n in sorted(target_counts.items())
        ]
    )
    counts.to_csv(OUT / "option_counts.csv", index=False)
    ranks = pd.DataFrame(ranks)
    ranks.to_pickle(OUT / "private/peak_comparisons.pkl")
    ranks.groupby(
        [
            "stream",
            "lipid_class",
            "measurement",
            "manual_peak_currently_missing",
            "relation",
        ],
        as_index=False,
    ).size().to_csv(OUT / "peak_comparison_counts.csv", index=False)
    # Pooled distributions computed from individual options, not means of
    # dataset medians. Private IDs never appear in public summaries.
    all_options = pd.concat(
        [pd.read_pickle(OUT / "private" / (d + ".pkl")) for d in datasets],
        ignore_index=True,
    )
    aggregate = []
    for (stream, cls, category), g in all_options.groupby(
        ["stream", "lipid_class", "diagnostic_category"]
    ):
        row = dict(
            stream=stream, lipid_class=cls, category=category, unique_options=len(g)
        )
        for f in rank_fields:
            v = g[PREFIX + f].dropna()
            row[f + "_available"] = len(v)
            row[f + "_median"] = v.median() if len(v) else np.nan
            row[f + "_ge08"] = int(v.ge(0.8).sum())
        aggregate.append(row)
    pd.DataFrame(aggregate).to_csv(OUT / "by_mode_class_category.csv", index=False)
    assert all(common.sha(Path(p)) == h for p, h in hashes.items())
    assert all(
        common.sha(Path(p)) == h for p, h in receipt["source_input_sha256"].items()
    )
    artifact_hashes = {str(p): common.sha(p) for p in OUT.rglob("*") if p.is_file()}
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            modes=132,
            annotation_changes=False,
            classifier_fits=0,
            manual_entries_checked=total_manual,
            retained_current_correct=total_current,
            blind_physical_receipt_sha256=common.sha(receipt_path),
            source_input_sha256=hashes,
            artifact_sha256=artifact_hashes,
            note="Retrospective diagnostic ONLY, not selection validation/gains/precision/FDR. Unique original identity+adduct+feature options, no candidate-version inflation. Identity-local comparisons ignore occupancy/final gates. Manual-absent and alternative-label categories remain unconfirmed. No thresholds/calls/defaults changed; all33 measurements frozen before this diagnostic opens curation.",
        ),
    )
    print("ALL33 RETROSPECTIVE MEASUREMENT CHECKS PASS", flush=True)


if __name__ == "__main__":
    main()
