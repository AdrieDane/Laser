"""Read-only diagnostics of experiment 001, no alternative selections."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from context import WORKSPACE as ROOT

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
EXP = ROOT / "outputs/qe_optimization_20261022/experiment_001"
sys.path[:0] = [str(EXP / "source_snapshot/app"), str(Path(__file__).resolve().parent)]
from matlab_annotation.class_candidates import score_linear_evidence
from run_qe_atlas_experiment import species, adduct


def main():
    global EXP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXP)
    args = parser.parse_args()
    EXP = args.experiment.resolve()
    protocol = json.loads((EXP / "protocol.json").read_text())
    folds = json.loads(
        (
            ROOT
            / "outputs/qe_optimization_20261022/rt_atlas_development/fold_assignment.json"
        ).read_text()
    )
    before = pd.read_pickle(BASE / "reports_v2/all_manual_comparison.pkl")
    rows = []
    for dataset, fold in folds.items():
        manual = pd.read_pickle(
            EXP / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        old = before[before.dataset.eq(dataset)].set_index("source_row")
        data = pd.read_pickle(EXP / "prepared" / f"fold_{fold}" / (dataset + ".pkl"))
        pool = data["pool"].copy()
        if (EXP / f"fold_{fold}_selection_model.json").exists():
            model = json.loads((EXP / f"fold_{fold}_selection_model.json").read_text())
            pool["score"] = score_linear_evidence(data["evidence"], model)
        else:
            from run_qe_dual_experiment import model_matrix

            model = pd.read_pickle(EXP / f"fold_{fold}_nonlinear_model.pkl")
            pool["score"] = model["estimator"].predict_proba(
                model_matrix(data["evidence"], model)
            )[:, 1]
        pool["species"] = pool.name.map(species)
        pool["adduct_key"] = pool.adduct.map(adduct)
        pool["fid"] = pool.feat_feature_id.astype(str)
        groups = {
            k: g
            for k, g in pool.groupby(
                ["stream", "species", "adduct_key", "fid"], sort=False
            )
        }
        for r in manual.itertuples():
            b = bool(old.loc[r.source_row, "same_manual_peak"])
            g = groups.get((r.stream, r.species, r.adduct, r.feature_id))
            reason = (
                "correct"
                if r.same_manual_peak
                else "unresolved"
                if not r.mapped
                else "protected_standard_not_selected"
                if r.internal_standard
                else "outside_candidate_envelope"
                if g is None
                else "score_below_threshold"
                if g.score.max() < protocol["threshold"]
                else "competition_or_margin"
            )
            rows.append(
                dict(
                    dataset=dataset,
                    stream=r.stream,
                    lipid_class=r.lipid_class,
                    species=r.species,
                    adduct=r.adduct,
                    feature_id=r.feature_id,
                    internal_standard=r.internal_standard,
                    before_correct=b,
                    after_correct=r.same_manual_peak,
                    reason=reason,
                    exact_candidate_available=g is not None,
                    exact_score=float(g.score.max()) if g is not None else None,
                )
            )
        print("DIAGNOSED", dataset, flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(EXP / "full_comparison/missing_reason_diagnostics.csv", index=False)
    print(detail.reason.value_counts().to_string())
    print(
        "LOSSES",
        detail[detail.before_correct & ~detail.after_correct]
        .reason.value_counts()
        .to_string(),
    )
    print("GAINS", int((~detail.before_correct & detail.after_correct).sum()))
    print(
        "LARGEST GAPS\n"
        + detail[~detail.after_correct]
        .groupby(["stream", "lipid_class", "reason"])
        .size()
        .sort_values(ascending=False)
        .head(45)
        .to_string()
    )
    print("OPTIMISTIC CANDIDATE AVAILABILITY, not annotation accuracy")
    for stream, g in detail.groupby("stream"):
        n = int((g.exact_candidate_available | g.after_correct).sum())
        print(stream, n, "/", len(g), f"{100 * n / len(g):.1f}%")


if __name__ == "__main__":
    main()
