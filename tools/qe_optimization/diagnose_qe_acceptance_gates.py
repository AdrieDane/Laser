"""Read-only009/022 recorded-score/gate diagnosis, NOT a new annotation run.

No counterfactual selector settings, prediction changes, model fits or cutoff
choices. The audit reproduces each existing selector at its recorded settings.
Curated identities remain diagnostic labels, never new inference predictors.
"""

from collections import Counter
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import REPO, WORKSPACE
from run_qe_atlas_experiment import species, adduct

sys.path.insert(0, str(REPO / "app"))
from matlab_annotation.selection_diagnostics import audit_operating_candidates

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "acceptance_gate_diagnostic_v2"
RUNS = {"before": QE / "experiment_009", "after": QE / "experiment_022"}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def key(stream, name, ion, fid):
    return stream, species(name), adduct(ion), str(fid)


def selector_signature(path):
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "select_operating_candidates"
    )
    return ast.dump(function, include_attributes=False)


def main():
    source = REPO / "app/matlab_annotation/operating_point.py"
    for run in RUNS.values():
        # Only the audited selector is relevant. The later balanced-cutoff
        # fitter elsewhere in this module was formatted after009; its byte
        # differences must not be mistaken for a selector behavior change.
        assert selector_signature(source) == selector_signature(
            run / "source_snapshot/app/matlab_annotation/operating_point.py"
        )
        assert json.loads((run / "reports_v2/summary.json").read_text())["complete"]
    OUT.mkdir(exist_ok=False)
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    projects = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
    sources = [
        Path(__file__),
        source,
        REPO / "app/matlab_annotation/selection_diagnostics.py",
    ]
    for p in sources:
        shutil.copy2(p, OUT / p.name)
    inputs, rows, parity_checks, normalized_duplicate_pairs = {}, [], 0, 0
    for dataset in projects:
        path = QE / "baseline/evaluation_v2" / dataset / "manual_comparison.pkl"
        inputs[str(path)] = sha(path)
        manual = pd.read_pickle(path)
        manual = manual[~manual.internal_standard].copy()
        path = (
            QE
            / "experiment_007/prepared"
            / f"fold_{folds[dataset]}"
            / (dataset + ".pkl")
        )
        inputs[str(path)] = sha(path)
        pool = pd.read_pickle(path)["pool"]
        pair_keys = [
            key(*v)
            for v in pool[["stream", "name", "adduct", "feat_feature_id"]].itertuples(
                index=False, name=None
            )
        ]
        normalized_duplicate_pairs += len(pair_keys) - len(set(pair_keys))
        pairs = pd.Series(pair_keys, index=pool.index)
        state = {}
        for side, root in RUNS.items():
            cut_path = (
                root / "calibration" / f"fold_{folds[dataset]}/operating_points.json"
            )
            score_path = root / "runs" / dataset / "scoring_audit.pkl"
            auto_path = root / "runs" / dataset / "automatic_annotations.csv"
            for path in (cut_path, score_path, auto_path):
                inputs[str(path)] = sha(path)
            thresholds = json.loads(cut_path.read_text())["thresholds"]
            scores = pd.read_pickle(score_path)
            assert scores.index.equals(pool.index), (dataset, side)
            audit_parts = []
            for stream, part in pool.groupby("stream", sort=False):
                audit_parts.append(
                    audit_operating_candidates(
                        part,
                        scores.loc[part.index],
                        threshold=thresholds[stream],
                        ambiguity_margin=0.05,
                    )
                )
                parity_checks += 1
            audit = pd.concat(audit_parts).reindex(pool.index)
            ranked = pool.assign(_score=scores).sort_values(
                ["_score", "feat_intensity"], ascending=False, kind="stable"
            )
            by_key = {}
            for i in ranked.index:
                by_key.setdefault(pairs.loc[i], i)
            automatic = pd.read_csv(
                auto_path,
                usecols=["stream", "name", "adduct", "feat_feature_id"],
                dtype={"feat_feature_id": str},
            )
            accepted = {
                key(*v)
                for v in automatic[
                    ["stream", "name", "adduct", "feat_feature_id"]
                ].itertuples(index=False, name=None)
            }
            state[side] = dict(
                audit=audit, by_key=by_key, accepted=accepted, thresholds=thresholds
            )
        for row in manual.to_dict("records"):
            k = key(row["stream"], row["species"], row["adduct"], row["feature_id"])
            record = dict(
                dataset=dataset,
                stream=row["stream"],
                lipid_class=row["lipid_class"],
                feature_id=row["feature_id"],
                species=row["species"],
                adduct=row["adduct"],
                mapped=bool(row["mapped"]),
            )
            for side, data in state.items():
                correct = bool(row["mapped"] and k in data["accepted"])
                index = data["by_key"].get(k) if row["mapped"] else None
                gate = (
                    "unresolved_manual_feature"
                    if not row["mapped"]
                    else "correct_feature_absent_from_pool"
                )
                score, rank, gap, clear = np.nan, np.nan, np.nan, False
                if index is not None:
                    a = data["audit"].loc[index]
                    score, rank, gap = (
                        float(a.score),
                        float(a.feature_rank),
                        float(a.feature_score_gap),
                    )
                    clear = bool(a.clear_feature_margin)
                    gate = (
                        a.primary_gate
                        if not a.selector_accepts
                        else "post_selector_or_reporting_change"
                    )
                if correct:
                    gate = "correct_original_feature"
                record.update(
                    {
                        "correct_" + side: correct,
                        "gate_" + side: gate,
                        "score_" + side: score,
                        "feature_rank_" + side: rank,
                        "feature_score_gap_" + side: gap,
                        "clear_margin_" + side: clear,
                        "threshold_" + side: float(data["thresholds"][row["stream"]]),
                    }
                )
            record["gained_correct"] = (
                record["correct_after"] and not record["correct_before"]
            )
            record["lost_correct"] = (
                record["correct_before"] and not record["correct_after"]
            )
            record["score_change"] = record["score_after"] - record["score_before"]
            # Inequalities on recorded values ONLY, not a new selector replay at
            # another setting. These conditions can overlap; not causal sums.
            record["new_score_meets_old_cutoff_but_not_new"] = (
                record["score_after"] >= record["threshold_before"]
                and record["score_after"] < record["threshold_after"]
            )
            record["score_dropped_across_new_cutoff"] = (
                record["score_before"] >= record["threshold_after"]
                and record["score_after"] < record["threshold_after"]
            )
            rows.append(record)
        print("AUDITED RECORDED GATES", dataset, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_pickle(OUT / "private_manual_gate_rows.pkl")
    grouped = frame.groupby(["dataset", "stream", "lipid_class"])
    stats = grouped.agg(
        manual=("mapped", "size"),
        mapped=("mapped", "sum"),
        correct_before=("correct_before", "sum"),
        correct_after=("correct_after", "sum"),
        gained_correct=("gained_correct", "sum"),
        lost_correct=("lost_correct", "sum"),
        paired_scores=("score_change", "count"),
        median_score_before=("score_before", "median"),
        median_score_after=("score_after", "median"),
        median_paired_score_change=("score_change", "median"),
    )
    stats.to_csv(OUT / "by_dataset_mode_class.csv")
    keys = [
        "stream",
        "lipid_class",
        "gate_before",
        "gate_after",
        "correct_before",
        "correct_after",
    ]
    frame.groupby(keys).size().rename("manual_rows").to_csv(
        OUT / "by_class_gate_transition.csv"
    )
    summaries = []
    for (stream, cls), part in frame.groupby(["stream", "lipid_class"]):
        lost = part[part.lost_correct]
        summaries.append(
            dict(
                stream=stream,
                lipid_class=cls,
                manual=len(part),
                correct_before=int(part.correct_before.sum()),
                correct_after=int(part.correct_after.sum()),
                gained_correct=int(part.gained_correct.sum()),
                lost_correct=int(part.lost_correct.sum()),
                score_pairs=int(part.score_change.notna().sum()),
                median_score_before=float(part.score_before.median()),
                median_score_after=float(part.score_after.median()),
                median_paired_score_change=float(part.score_change.median()),
                lost_newscore_meets_old_cutoff_not_new=int(
                    lost.new_score_meets_old_cutoff_but_not_new.sum()
                ),
                lost_score_crosses_new_cutoff=int(
                    lost.score_dropped_across_new_cutoff.sum()
                ),
                **{
                    "lost_gate_" + k: int(v)
                    for k, v in Counter(lost.gate_after).items()
                },
            )
        )
    pd.DataFrame(summaries).fillna(
        {c: 0 for r in summaries for c in r if c.startswith("lost_gate_")}
    ).to_csv(OUT / "by_mode_class.csv", index=False)
    summary = dict(
        datasets=len(projects),
        manual_endogenous=len(frame),
        selector_parity_checks=parity_checks,
        normalized_duplicate_candidate_pairs=normalized_duplicate_pairs,
        correct_before=int(frame.correct_before.sum()),
        correct_after=int(frame.correct_after.sum()),
        gained_correct=int(frame.gained_correct.sum()),
        lost_correct=int(frame.lost_correct.sum()),
        annotation_changes=False,
        source_sha256={str(p): sha(p) for p in sources},
        input_sha256=inputs,
        note="READ-ONLY recorded009/022 scores and SAME recorded settings. No changed annotations, counterfactual cutoff/margin choices, model fit or promised gains. Gate reasons ordered, not independent causes; score/cutoff inequalities can overlap. Endogenous diagnostic denominator, ALL unresolved mappings retained; normal performance reports also include standards. Private row-level IDs/scores remain outside Git. Prior development exposure precludes untouched external/FDR claims.",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"source_sha256", "input_sha256"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
