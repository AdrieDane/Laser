"""Read-only structural audit: SAME original feature, competing isomer labels.

No fitting, preferences, annotation changes or selection of class winners.
Reference labels are retrospective diagnostics ONLY, not runtime predictors.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_qe_conditional_peak_experiment as runner
from frozen_resources import configure_frozen_resources


def main():
    configure_frozen_resources(runner.OLD / "source_snapshot")
    projects = json.loads((runner.BASE / "run_manifest.json").read_text())["datasets"]
    folds_path = runner.QE / "rt_atlas_development/fold_assignment.json"
    folds = json.loads(folds_path.read_text())
    out = runner.QE / "isomer_label_competition_diagnostic_029_v1"
    out.mkdir(exist_ok=False)
    inputs = {str(folds_path): runner.sha(folds_path)}
    sources = {str(Path(__file__)): runner.sha(Path(__file__))}
    counts, support = Counter(), defaultdict(set)
    for dataset in projects:
        mask = {folds[dataset]}
        raw_path = (
            runner.PREPARED
            / "prepared"
            / ("exclude_" + str(folds[dataset]))
            / (dataset + ".pkl")
        )
        manual_path = runner.OLD / "evaluation_v2" / dataset / "manual_comparison.pkl"
        for p in (raw_path, manual_path):
            inputs[str(p)] = runner.sha(p)
        raw = runner.load_predictors(dataset, mask)
        pool = raw["pool"].drop_duplicates(
            ["stream", "name", "adduct", "feat_feature_id"]
        )
        groups = {
            key: p for key, p in pool.groupby(["stream", "feat_feature_id"], sort=False)
        }
        manual = pd.read_pickle(manual_path)
        manual = manual[manual.mapped & ~manual.internal_standard]
        for row in manual.itertuples():
            part = groups.get((row.stream, str(row.feature_id)))
            if part is None:
                continue
            positive = part[
                part.name.map(runner.legacy.species).eq(row.species)
                & part.adduct.map(runner.legacy.adduct).eq(row.adduct)
            ]
            if positive.empty:
                continue
            gold = positive.iloc[0]
            negative = part[
                part.adduct.eq(gold.adduct)
                & np.isclose(
                    part.ref_mz.to_numpy(float), float(gold.ref_mz), rtol=0, atol=1e-8
                )
                & ~part.name.eq(gold["name"])
            ]
            if negative.empty:
                continue
            cell = (dataset, row.stream, row.lipid_class)
            counts[(*cell, "curated_features_with_competing_isomer_label")] += 1
            counts[(*cell, "competing_distinct_labels")] += len(negative)
            counts[(*cell, "currently_missing_same_original_peak")] += int(
                not row.same_manual_peak
            )
            support[(row.stream, row.lipid_class)].add(dataset[:7])
        print("READONLY SAME-FEATURE ISOMER COMPETITION", dataset, flush=True)
    table = pd.DataFrame(
        [
            dict(dataset=d, stream=s, lipid_class=c, measure=k, count=n)
            for (d, s, c, k), n in sorted(counts.items())
        ]
    )
    table.to_csv(out / "by_dataset_mode_class_measure.csv", index=False)
    summary = table.groupby(["stream", "lipid_class", "measure"], as_index=False)[
        "count"
    ].sum()
    summary["independent_project_groups"] = [
        len(support[(s, c)])
        for s, c in summary[["stream", "lipid_class"]].itertuples(
            index=False, name=None
        )
    ]
    summary.to_csv(out / "by_mode_class_measure.csv", index=False)
    assert len(projects) == 33
    assert all(runner.sha(Path(p)) == h for p, h in {**sources, **inputs}.items())
    runner.dump(
        out / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            classifier_fits=0,
            annotation_changes=False,
            source_sha256=sources,
            input_sha256=inputs,
            note="Retrospective structural opportunities ONLY,not gains or independent biological observations. Same original corefeature,adduct and theoreticalmass1e-8Da;different label. No querytruth entersannotation. Mappedendogenous-only diagnostic does NOT change fullmanualbenchmarkN.",
        ),
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
