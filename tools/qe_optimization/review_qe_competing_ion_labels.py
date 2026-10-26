"""Supplementary exploratory same-FEATURE label contrast, not original protocol.

Added after the blind pilot started; not a prespecified inference or annotation
experiment. Does not change frozen measurements, candidates, models or calls.
Manual alternative labels are unconfirmed, not chemically proven false.
"""

import json
from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import diagnose_qe_competing_ion_family_pilot_v2 as diagnostic

prep = diagnostic.prep
OUT = prep.common.QE / "competing_ion_label_review_041_v1"


def main():
    configure_frozen_resources(REPO)
    from standard_lipidomics.validation import _normalise
    from matlab_annotation.standard_rt_library import ion_key

    receipt_path = diagnostic.OUT / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["all_passed"] and receipt["datasets"] == 33
    datasets, folds = prep.common.inventory()
    hashes = {
        **receipt["source_input_sha256"],
        str(Path(__file__)): prep.common.sha(Path(__file__)),
        str(receipt_path): prep.common.sha(receipt_path),
    }
    for d in datasets:
        for path in (
            diagnostic.OUT / "private" / (d + ".pkl"),
            prep.OUT / "private" / (d + ".pkl"),
            diagnostic.REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl",
            diagnostic.REFERENCE / "runs" / d / "automatic_annotations.csv",
        ):
            hashes[str(path)] = prep.common.sha(path)
    OUT.mkdir(exist_ok=False)
    rows, comparisons = [], []
    for d in datasets:
        table = pd.read_pickle(diagnostic.OUT / "private" / (d + ".pkl"))
        saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
        manual = pd.read_pickle(
            diagnostic.REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        current = pd.read_csv(
            diagnostic.REFERENCE / "runs" / d / "automatic_annotations.csv",
            dtype=str,
            keep_default_na=False,
        )
        called = _normalise(
            current.assign(adduct=current.adduct.map(ion_key)), manual=False
        )
        oldids = set(
            called[["stream", "species", "adduct"]].itertuples(index=False, name=None)
        )
        old_features = set(
            called[["stream", "feature_id"]].itertuples(index=False, name=None)
        )
        truth = manual.loc[
            manual.mapped & ~manual.internal_standard & ~manual.lipid_class.eq("X4")
        ]
        counts = truth.groupby(["stream", "feature_id"]).apply(
            lambda x: len(x[["species", "adduct"]].drop_duplicates()),
            include_groups=False,
        )
        links = {
            i: set(g.alternative_index)
            for i, g in saved["links"].groupby("source_index")
        }
        source = prep.load(d, folds[d])["pool"]
        normalized = _normalise(
            source.assign(adduct=source.adduct.map(ion_key)), manual=False
        )
        normalized.index = source.index
        key_columns = ["stream", "species", "adduct", "feature_id"]
        version_indices = {
            k: set(part.index) for k, part in normalized.groupby(key_columns)
        }
        kept = {tuple(r[c] for c in key_columns): i for i, r in table.iterrows()}
        equivalent_index = {
            i: kept.get(tuple(r[c] for c in key_columns))
            for i, r in normalized.iterrows()
        }
        old_exact = set(called[key_columns].itertuples(index=False, name=None))
        opportunities = {}
        for _idx, row in table.loc[
            table.source_physical
            & table.diagnostic_category.isin(
                ["manual_correct_current", "manual_correct_currently_missing"]
            )
        ].iterrows():
            if counts.get((row.stream, row.feature_id), 0) != 1:
                continue
            # Diagnostic dedup kept the first construction version. The blind
            # graph contains original versions; equality is resolved explicitly.
            key = (row.stream, row.species, row.adduct, row.feature_id)
            potential_indices = set()
            for source in version_indices[key]:
                potential_indices.update(links.get(source, set()))
            deduplicated_indices = {
                equivalent_index[j]
                for j in potential_indices
                if equivalent_index.get(j) is not None
            }
            available = table.loc[table.index.intersection(deduplicated_indices)]
            if available.empty:
                continue
            alternatives = available.loc[
                ~available.diagnostic_category.isin(
                    ["manual_correct_current", "manual_correct_currently_missing"]
                )
            ]
            if alternatives.empty:
                continue
            is_missing = row.diagnostic_category == "manual_correct_currently_missing"
            already_other = (
                row.stream,
                row.feature_id,
            ) in old_features and key not in old_exact
            identity_free = key[:3] not in oldids
            for field in diagnostic.CONTRASTS:
                a = row[prep.measurement.PREFIX + field]
                b = alternatives[prep.measurement.PREFIX + field].dropna()
                if pd.isna(a) or b.empty:
                    continue
                gap = float(a - b.max())
                relation = (
                    "curated_label_better"
                    if gap > 1e-9
                    else "alternative_label_better"
                    if gap < -1e-9
                    else "tie"
                )
                comparisons.append(
                    dict(
                        dataset=d,
                        stream=row.stream,
                        lipid_class=row.lipid_class,
                        field=field,
                        currently_missing=is_missing,
                        current_other_label_on_feature=already_other,
                        relation=relation,
                    )
                )
            if is_missing:
                # Count identities/physical features once, without thresholding
                # the relative evidence or choosing a new annotation.
                record = opportunities.setdefault(
                    (row.stream, row.lipid_class),
                    dict(
                        candidate_rows=0,
                        missing_rows_with_competing_family=0,
                        current_other_label_and_identity_free=0,
                    ),
                )
                record["candidate_rows"] += 1
                record["missing_rows_with_competing_family"] += 1
                record["current_other_label_and_identity_free"] += int(
                    already_other and identity_free
                )
        for (s, c), values in opportunities.items():
            rows.append(dict(dataset=d, stream=s, lipid_class=c, **values))
    pairs = pd.DataFrame(
        comparisons,
        columns=[
            "dataset",
            "stream",
            "lipid_class",
            "field",
            "currently_missing",
            "current_other_label_on_feature",
            "relation",
        ],
    )
    aggregate = []
    for (s, c, f, missing, other), part in pairs.groupby(
        [
            "stream",
            "lipid_class",
            "field",
            "currently_missing",
            "current_other_label_on_feature",
        ]
    ):
        count = part.relation.value_counts()
        aggregate.append(
            dict(
                stream=s,
                lipid_class=c,
                field=f,
                currently_missing=missing,
                current_other_label_on_feature=other,
                comparable_features=len(part),
                curated_label_better=int(count.get("curated_label_better", 0)),
                alternative_label_better=int(count.get("alternative_label_better", 0)),
                ties=int(count.get("tie", 0)),
                contributing_project_groups=part.dataset.str[:7].nunique(),
            )
        )
    pd.DataFrame(aggregate).to_csv(
        OUT / "same_feature_label_contrasts.csv", index=False
    )
    pd.DataFrame(rows).to_csv(OUT / "missing_label_opportunities.csv", index=False)
    pairs.to_pickle(OUT / "private_label_comparisons.pkl")
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    prep.common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            posthoc_descriptive_analysis=True,
            source_input_sha256=hashes,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            note="Supplementary same-feature curated-vs-alternative-label comparison; added after blind pilot began. Not original prospective inference, accepted annotations/gains, chemical false-positive proof or external validation. Current occupancy used for descriptive free-identity counts only.",
        ),
    )


if __name__ == "__main__":
    main()
