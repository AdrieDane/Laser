"""Read-only029 selection diagnosis; no fits, thresholds or annotations changed."""

import hashlib
import json

import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_atlas_experiment import species, adduct

QE = WORKSPACE / "outputs/qe_optimization_20261022"
CORE = QE / "experiment_029"
OUT = QE / "common_selection_diagnosis_029_v1"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    configure_frozen_resources(CORE / "source_snapshot")
    from matlab_annotation.operating_point import select_operating_candidates

    assert json.loads((CORE / "integrity_receipt.json").read_text())["all_passed"]
    projects = json.loads((CORE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    manual = pd.concat(
        [
            pd.read_pickle(CORE / "evaluation_v2" / d / "manual_comparison.pkl").assign(
                dataset=d, project=d[:7]
            )
            for d in projects
        ],
        ignore_index=True,
    )
    support = manual.groupby(["stream", "lipid_class"]).agg(
        manual=("dataset", "size"), projects=("project", "nunique")
    )
    common = set(support[support.manual.ge(100) & support.projects.ge(3)].index)
    assert len(common) == 86
    OUT.mkdir(exist_ok=False)
    rows, inputs = [], {}
    for dataset in projects:
        fold = folds[dataset]
        pool_path = (
            QE
            / "standard_context_preparation_v1/prepared"
            / f"exclude_{fold}"
            / (dataset + ".pkl")
        )
        score_path = CORE / "runs" / dataset / "scoring_audit.pkl"
        threshold_path = CORE / "calibration" / f"fold_{fold}/operating_points.json"
        manual_path = CORE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        for path in (pool_path, score_path, threshold_path, manual_path):
            inputs[str(path)] = sha(path)
        pool = pd.read_pickle(pool_path)["pool"]
        scores = pd.read_pickle(score_path)
        assert pool.index.equals(scores.index)
        thresholds = json.loads(threshold_path.read_text())["thresholds"]
        ranked, selected = [], []
        for stream, part in pool.groupby("stream", sort=False):
            chosen = select_operating_candidates(
                part,
                scores.loc[part.index],
                threshold=thresholds[stream],
                ambiguity_margin=0.05,
            )
            selected.append(chosen)
            r = part.assign(score=scores.loc[part.index]).sort_values(
                ["score", "feat_intensity"], ascending=False, kind="stable"
            )
            r = r.drop_duplicates(["feat_feature_id", "name", "adduct"])
            r["wins_feature"] = r.groupby("feat_feature_id").cumcount().eq(0)
            second = r.groupby("feat_feature_id").score.shift(-1)
            r["clear"] = second.isna() | (r.score - second).ge(0.05)
            r["above_cutoff"] = r.score.ge(thresholds[stream])
            ranked.append(r)
        ranked, selected = pd.concat(ranked), pd.concat(selected)
        def keys(frame):
            return [
                (s, species(n), adduct(a), str(fid))
                for s, n, a, fid in frame[
                    ["stream", "name", "adduct", "feat_feature_id"]
                ].itertuples(index=False, name=None)
            ]
        selected_keys = set(keys(selected))
        groups = {}
        for index, key in zip(ranked.index, keys(ranked)):
            groups.setdefault(key, []).append(index)
        missing = manual[manual.dataset.eq(dataset) & ~manual.same_manual_peak]
        for entry in missing.itertuples():
            if (entry.stream, entry.lipid_class) not in common:
                continue
            key = (
                entry.stream,
                entry.species,
                adduct(entry.adduct),
                str(entry.feature_id),
            )
            if not entry.mapped:
                reason = "manual_original_peak_unresolved"
            elif entry.internal_standard:
                reason = "standard_specialist_not_core_selector"
            elif key not in groups:
                reason = "correct_original_peak_absent_from_core_candidates"
            elif key in selected_keys:
                reason = "selected_by_core_but_not_final_after_standard_cleanup"
            else:
                candidates = ranked.loc[groups[key]]
                winners = candidates[candidates.wins_feature]
                clear = winners[winners.clear]
                if clear.above_cutoff.any():
                    reason = "different_original_peak_won_same_lipid_identity"
                elif len(clear):
                    reason = "correct_original_peak_below_mode_score_cutoff"
                elif len(winners):
                    reason = "correct_original_peak_withheld_for_close_label_scores"
                else:
                    reason = "other_lipid_label_ranked_higher_on_original_peak"
            rows.append(
                dict(
                    dataset=dataset,
                    stream=entry.stream,
                    lipid_class=entry.lipid_class,
                    reason=reason,
                )
            )
        print("READ-ONLY SELECTION DIAGNOSIS", dataset, flush=True)
    rows = pd.DataFrame(rows)
    assert len(rows) == 8975
    rows.groupby(["dataset", "stream", "lipid_class", "reason"]).size().rename(
        "missing_manual_annotations"
    ).reset_index().to_csv(OUT / "by_dataset_mode_class_reason.csv", index=False)
    rows.groupby(["stream", "lipid_class", "reason"]).size().rename(
        "missing_manual_annotations"
    ).reset_index().to_csv(OUT / "by_mode_class_reason.csv", index=False)
    result = dict(
        annotation_changes=False,
        fits_or_new_cutoffs=False,
        reference="029",
        datasets=33,
        priority_class_modes=86,
        priority_missing_manual_annotations=len(rows),
        by_reason=rows.reason.value_counts().to_dict(),
        inputs_sha256=inputs,
        note="Read-only diagnostic opportunities,NOT guaranteed recoverable matches or annotation gains. Uses known curation ONLYfor diagnosis,never feature preparation. Core candidates exclude separate005STD/fixed027massstage; available-category boundaries therefore differ from the full029candidate audit. Prior development exposure,not external validation.",
    )
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in result.items() if k != "inputs_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
