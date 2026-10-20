"""Fit the portable PE scorer from runtime candidates and curated crosswalks.

Curation is calibration/evaluation-only. Each input pool must be generated
independently by prepare_pe_rebuild_candidates, before labels are attached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from matlab_annotation.pe_rebuild import pe_rebuild_evidence, score_pe_rebuild


def fit_pe_model(data: pd.DataFrame) -> tuple[dict, GradientBoostingClassifier]:
    evidence = pd.concat(
        [pe_rebuild_evidence(group) for _, group in data.groupby("dataset", sort=False)]
    ).loc[data.index]
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.04,
        min_samples_leaf=12,
        random_state=19,
    )
    model.fit(evidence, data.correct)
    prior = float(np.average(data.correct))
    payload = dict(
        format="laser-pe-rebuild-v1",
        features=evidence.columns.tolist(),
        intercept=float(np.log(prior / (1 - prior))),
        learning_rate=model.learning_rate,
        training_datasets=sorted(data.dataset.unique().tolist()),
        training_candidates=len(data),
        same_peak_training_positives=int(data.correct.sum()),
        default_score_threshold=0.10,
        default_feature_margin=0.03,
        target="same curated PE species, adduct, stream and original XCMS feature",
        caveat="agreement with curation, not analytical identification probability or FDR; unlisted identities are unconfirmed negatives",
        trees=[
            dict(
                left=t.tree_.children_left.tolist(),
                right=t.tree_.children_right.tolist(),
                feature=t.tree_.feature.tolist(),
                threshold=t.tree_.threshold.tolist(),
                value=t.tree_.value[:, 0, 0].tolist(),
            )
            for t in model.estimators_.ravel()
        ],
    )
    np.testing.assert_allclose(
        score_pe_rebuild(evidence, payload),
        model.predict_proba(evidence)[:, 1],
        atol=1e-12,
        rtol=1e-12,
    )
    return payload, model


def labeled_candidates(
    candidate_root: Path, curation_root: Path, dataset: str, *, stream: str = "RP_Pos"
) -> pd.DataFrame:
    pool = pd.read_pickle(candidate_root / dataset / "raw_candidates.pkl")
    cross = pd.read_csv(curation_root / dataset / "curated_feature_crosswalk.csv")
    cross = cross[cross.stream.eq(stream) & cross.matlab_class.eq("PE")]
    mapped = cross[cross.crosswalk_status.eq("profile_match")]
    truth = set(
        zip(
            mapped.matlab_species,
            mapped.matlab_adduct,
            mapped.xcms_feature_id.astype(int).astype(str),
        )
    )
    pool["correct"] = [
        (n, a, str(f)) in truth
        for n, a, f in zip(pool["name"], pool.adduct, pool.feat_feature_id)
    ]
    pool["dataset"] = dataset
    return pool


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--curation-root", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--stream", default="RP_Pos", choices=["RP_Pos", "RP_Neg", "NP_Pos"])
    args = p.parse_args()
    data = pd.concat(
        [
            labeled_candidates(
                args.candidate_root, args.curation_root, d, stream=args.stream
            )
            for d in args.datasets
        ],
        ignore_index=True,
    )
    payload, _ = fit_pe_model(data)
    payload["stream"] = args.stream
    args.model.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print({k: v for k, v in payload.items() if k not in {"features", "trees"}})


if __name__ == "__main__":
    main()
