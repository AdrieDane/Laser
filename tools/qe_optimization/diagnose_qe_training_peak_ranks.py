"""Training-OOF-only ranking diagnosis; no outertruth/annotations/thresholdfit.

An identity-wise best feature is only a diagnostic upperbound. It ignores
cross-identity occupancy and MUST NOT be exported as an annotation decision.
"""

from collections import Counter
from pathlib import Path

import pandas as pd

from frozen_resources import configure_frozen_resources
import run_qe_feature_label_experiment as runner

OUT = runner.QE / "feature_label_training_peak_rank_audit_v1"
GAPS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def main():
    configure_frozen_resources(runner.OUT / "source_snapshot")
    datasets, folds = runner.inventory()
    sources = {
        Path(__file__),
        Path(runner.__file__),
        Path(runner.inherited.__file__),
        Path(runner.inherited.common.__file__),
        Path(__file__).with_name("run_qe_atlas_experiment.py"),
    }
    files = set(sources)
    for outer in range(5):
        files.update(
            [
                runner.CORE / "calibration" / f"fold_{outer}/operating_points.json",
                runner.PREVIOUS / "calibration" / f"fold_{outer}/operating_points.json",
                runner.OUT / "calibration" / f"fold_{outer}/operating_points.json",
            ]
        )
        for d in datasets:
            if folds[d] == outer:
                continue
            mask = {outer, folds[d]}
            name = "exclude_" + "_".join(map(str, sorted(mask)))
            files.update(
                [
                    runner.inherited.common.prepared_path(d, mask),
                    runner.OUT
                    / "calibration"
                    / f"fold_{outer}"
                    / (d + "_blind_scores.pkl"),
                    runner.CORE
                    / "calibration"
                    / f"fold_{outer}"
                    / (d + "_blind_scores.pkl"),
                    runner.BASE / "evaluation_v2" / d / "manual_comparison.pkl",
                    runner.BASE / "runs" / d / "full_annotation.pkl",
                    runner.QE
                    / "nested_standard_development_v1/automatic_inputs"
                    / (d + ".pkl"),
                    runner.QE / "nested_standard_development_v1" / (name + ".json"),
                    runner.inherited.common.PREPARED
                    / "measurements"
                    / name
                    / (d + ".pkl"),
                ]
            )
    hashes = {str(p): runner.sha(p) for p in sorted(files)}
    counts = Counter()
    cases = 0
    for outer in range(5):
        training = {d[:7] for d in datasets if folds[d] != outer}
        excluded = {d[:7] for d in datasets if folds[d] == outer}
        for d in datasets:
            inner = folds[d]
            if inner == outer:
                continue
            raw = runner.load_predictors(d, {outer, inner})
            ranks = pd.read_pickle(
                runner.OUT / "calibration" / f"fold_{outer}" / (d + "_blind_scores.pkl")
            )
            labeled, manual, _ = runner.authorize_labels(raw, training, excluded)
            scores = pd.read_pickle(
                runner.CORE
                / "calibration"
                / f"fold_{outer}"
                / (d + "_blind_scores.pkl")
            )
            _, current = runner.inner_previous(d, outer, inner, labeled, scores)
            pool = labeled["pool"].copy()
            pool["species"] = pool.name.map(runner.inherited.common.species)
            pool["ion"] = pool.adduct.map(runner.inherited.common.adduct)
            pool["joint_rank"] = ranks.joint_agreement_rank
            pool["supported"] = ranks.supported_cluster
            groups = {
                k: g for k, g in pool.groupby(["stream", "species", "ion"], sort=False)
            }
            gold = {
                (r.stream, r.species, r.adduct): str(r.feature_id)
                for r in manual.loc[manual.mapped].itertuples()
            }
            current = current.assign(lipid_class=current["class"])
            for r in current.itertuples():
                key = (
                    r.stream,
                    runner.inherited.common.species(r.name),
                    runner.inherited.common.adduct(r.adduct),
                )
                if key not in gold or key not in groups:
                    continue
                part = groups[key]
                oldfid = str(r.feat_feature_id)
                original = part.loc[part.feat_feature_id.astype(str).eq(oldfid)]
                if original.empty or not original.supported.any():
                    continue
                eligible = (
                    part.loc[part.supported]
                    .sort_values(
                        ["joint_rank", "feat_intensity"], ascending=False, kind="stable"
                    )
                    .drop_duplicates("feat_feature_id")
                )
                oldrank = float(original.joint_rank.max())
                best = eligible.iloc[0]
                gain = float(best.joint_rank) - oldrank
                before = oldfid == gold[key]
                after = str(best.feat_feature_id) == gold[key]
                if str(best.feat_feature_id) == oldfid:
                    continue
                for gap in GAPS:
                    if gain <= gap:
                        continue
                    kind = (
                        "wrong_to_correct"
                        if not before and after
                        else "correct_to_wrong"
                        if before and not after
                        else "wrong_to_other_wrong"
                        if not before
                        else "no_correctness_change"
                    )
                    counts[
                        (
                            outer,
                            d,
                            r.stream,
                            r.lipid_class,
                            gap,
                            kind,
                        )
                    ] += 1
            cases += 1
            print("TRAINING-ONLY PEAK RANK DIAGNOSIS", outer, d, flush=True)
    assert cases == 132
    OUT.mkdir(exist_ok=False)
    table = pd.DataFrame(
        [
            dict(
                outer=o,
                dataset=d,
                stream=s,
                lipid_class=c,
                minimum_rank_gain=g,
                change=kind,
                count=n,
            )
            for (o, d, s, c, g, kind), n in sorted(counts.items())
        ]
    )
    table.to_csv(OUT / "by_outer_dataset_mode_class.csv", index=False)
    table.groupby(
        ["stream", "lipid_class", "minimum_rank_gain", "change"], as_index=False
    )["count"].sum().to_csv(OUT / "by_mode_class.csv", index=False)
    assert all(runner.sha(Path(p)) == h for p, h in hashes.items())
    runner.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            training_query_cases=132,
            classifier_fits=0,
            annotation_changes=False,
            outer_query_truth_opened=False,
            source_input_sha256=hashes,
            note="Matching INNER-OOF training queries ONLY; same dataset repeats underfouroutertrainingcontexts. Identity-wise localrank diagnostic,not globallycompatible annotation moves/acceptedgains or an externalperformancebenchmark;ignoresglobaloccupancy/finalspikegates. Originaloldcalls/counts/extrasunchanged. BOTHold/newfeaturesupportrequired;nofallbackfromunknownoldscore.",
        ),
    )
    print(
        table.groupby(["minimum_rank_gain", "change"])["count"].sum().to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
