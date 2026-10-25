"""Read-only: freeze ALL33 physical near-mass rivals BEFORE manual comparison.

No models, annotation decisions or widened candidate mass/RT windows. Numeric
near-equivalence is NOT proof of identical formulas or correct identification.
"""

from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import run_qe_add_only_class_experiment as common

QE, REFERENCE = common.QE, common.OUT
OUT = QE / "near_mass_label_diagnostic_041_v1"
EXACT_DA = 1e-8
NEAR_PPM = 0.05


def physical_pairs(pool):
    required = {
        "stream",
        "feat_feature_id",
        "name",
        "adduct",
        "ref_mz",
        "class",
        "is_is",
    }
    assert required.issubset(pool) and pool.index.is_unique
    p = pool.copy()
    p["_identity"] = p.name.map(common.common.species)
    p["_ion"] = p.adduct.map(common.common.adduct)
    assert p.is_is.isin([True, False]).all()
    p = p.loc[~p.is_is & ~p.name.str.startswith("IS: ") & ~p.name.str.endswith(" (IS)")]
    p = p.drop_duplicates(["stream", "feat_feature_id", "_identity", "_ion"])
    rows = []
    for (stream, fid, ion), group in p.groupby(
        ["stream", "feat_feature_id", "_ion"], sort=False
    ):
        if len(group) < 2:
            continue
        for (i, left), (j, right) in combinations(group.iterrows(), 2):
            a, b = float(left.ref_mz), float(right.ref_mz)
            assert np.isfinite([a, b]).all() and min(a, b) > 0
            delta = abs(a - b)
            if delta > max(EXACT_DA, NEAR_PPM * min(a, b) / 1e6):
                continue
            rows.append(
                dict(
                    stream=stream,
                    feat_feature_id=str(fid),
                    adduct=ion,
                    left_index=i,
                    right_index=j,
                    left_species=left._identity,
                    right_species=right._identity,
                    left_class=left["class"],
                    right_class=right["class"],
                    left_mass=a,
                    right_mass=b,
                    absolute_delta_Da=delta,
                    absolute_delta_ppm=1e6 * delta / min(a, b),
                    numerical_scope="exact" if delta <= EXACT_DA else "new_near",
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "stream",
            "feat_feature_id",
            "adduct",
            "left_index",
            "right_index",
            "left_species",
            "right_species",
            "left_class",
            "right_class",
            "left_mass",
            "right_mass",
            "absolute_delta_Da",
            "absolute_delta_ppm",
            "numerical_scope",
        ],
    )


def main():
    configure_frozen_resources(REFERENCE / "source_snapshot")
    datasets, folds = common.inventory()
    assert json.loads((REFERENCE / "integrity_receipt.json").read_text())["all_passed"]
    source = {str(Path(__file__)): common.sha(Path(__file__))}
    inputs = {
        str(REFERENCE / "integrity_receipt.json"): common.sha(
            REFERENCE / "integrity_receipt.json"
        )
    }
    paths = {}
    for d in datasets:
        pool = common.common.prepared_path(d, {folds[d]})
        manual = REFERENCE / "evaluation_v2" / d / "manual_comparison.pkl"
        paths[d] = (pool, manual)
        inputs.update({str(p): common.sha(p) for p in (pool, manual)})
    OUT.mkdir(exist_ok=False)
    physical = OUT / "physical"
    physical.mkdir()
    files = {}
    # No manual file is read in this phase. Freeze all cohorts first.
    for d in datasets:
        pool_path, _ = paths[d]
        raw = pd.read_pickle(pool_path)
        assert raw["query_dataset"] == d and set(raw["excluded_folds"]) == {folds[d]}
        assert not {
            "training_positive",
            "training_label_known",
            "same_manual_peak",
        } & set(raw["pool"])
        pairs = physical_pairs(raw["pool"])
        path = physical / (d + ".pkl")
        pairs.to_pickle(path)
        files[str(path)] = common.sha(path)
        print("FROZEN NEAR-MASS PHYSICAL LABEL PAIRS", d, len(pairs), flush=True)
    common.dump(
        OUT / "blind_physical_receipt.json",
        dict(
            all33_physical_pools_frozen=True,
            manual_files_opened=0,
            datasets=33,
            modes=132,
            exact_Da=EXACT_DA,
            near_ppm=NEAR_PPM,
            source_sha256=source,
            input_sha256=inputs,
            physical_sha256=files,
        ),
    )
    # Retrospective manual comparison ONLY AFTER the ALL33 blind freeze.
    counts, pair_counts = Counter(), Counter()
    for d in datasets:
        pairs = pd.read_pickle(physical / (d + ".pkl"))
        manual = pd.read_pickle(paths[d][1])
        manual = manual.loc[manual.mapped & ~manual.internal_standard]
        groups = {
            k: v
            for k, v in pairs.groupby(
                ["stream", "feat_feature_id", "adduct"], sort=False
            )
        }
        for r in manual.itertuples():
            part = groups.get((r.stream, str(r.feature_id), r.adduct))
            if part is None:
                continue
            rivals = part.loc[
                part.left_species.eq(r.species) | part.right_species.eq(r.species)
            ]
            for scope in ("exact", "new_near"):
                selected = rivals.loc[rivals.numerical_scope.eq(scope)]
                if selected.empty:
                    continue
                counts[
                    (d, r.stream, r.lipid_class, scope, "manual_features_with_rival")
                ] += 1
                counts[
                    (
                        d,
                        r.stream,
                        r.lipid_class,
                        scope,
                        "missing_correct_manual_feature",
                    )
                ] += int(not r.same_manual_peak)
                for row in selected.itertuples():
                    other = (
                        row.right_class
                        if row.left_species == r.species
                        else row.left_class
                    )
                    pair_counts[
                        (
                            r.stream,
                            r.lipid_class,
                            other,
                            scope,
                            "manual_features_with_rival",
                        )
                    ] += 1
                    pair_counts[
                        (
                            r.stream,
                            r.lipid_class,
                            other,
                            scope,
                            "missing_correct_manual_feature",
                        )
                    ] += int(not r.same_manual_peak)
    detail = pd.DataFrame(
        [
            dict(
                dataset=d,
                stream=s,
                lipid_class=c,
                numerical_scope=scope,
                measure=measure,
                count=n,
            )
            for (d, s, c, scope, measure), n in sorted(counts.items())
        ]
    )
    detail.to_csv(OUT / "by_dataset_mode_class_measure.csv", index=False)
    summary = detail.groupby(
        ["stream", "lipid_class", "numerical_scope", "measure"], as_index=False
    )["count"].sum()
    summary.to_csv(OUT / "by_mode_class_measure.csv", index=False)
    pd.DataFrame(
        [
            dict(
                stream=s,
                lipid_class=c,
                competing_class=other,
                numerical_scope=scope,
                measure=measure,
                count=n,
            )
            for (s, c, other, scope, measure), n in sorted(pair_counts.items())
        ]
    ).to_csv(OUT / "by_mode_competing_class_measure.csv", index=False)
    assert all(
        common.sha(Path(p)) == h for p, h in {**source, **inputs, **files}.items()
    )
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            classifier_fits=0,
            annotation_changes=False,
            exact_Da=EXACT_DA,
            near_ppm=NEAR_PPM,
            source_sha256=source,
            input_sha256=inputs,
            physical_sha256=files,
            note="All33 physical rivals frozen BEFORE manual evaluation. Mapped endogenous diagnostic opportunities only, not gains/chemical isomer identity or a changed benchmark denominator. No candidate window widened or annotation changed. A manual feature can appear in both scopes; do not sum overlapping feature counts as independent gains.",
        ),
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
