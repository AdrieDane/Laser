"""Fit PE evidence on development cohorts only; never include test cohorts.

Inputs are candidate_decisions.csv and a profile-matched curated crosswalk for
each development dataset. Paths are CLI arguments, not runtime dependencies.
Unlisted identities are treated as unconfirmed negatives for this agreement
model. Its score is not an analytical false-discovery or structural certainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matlab_annotation.composition_plausibility import fit_composition_priors
from matlab_annotation.pe_completion import pe_evidence_features


def fit_payload(data: pd.DataFrame) -> dict:
    positives = data[data.correct][["class", "carbon", "db"]].rename(
        columns={"class": "lipid_class"}
    )
    priors = fit_composition_priors(positives)
    x = pe_evidence_features(data, priors)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=2000),
    )
    pipeline.fit(x, data.correct)
    imputer = pipeline.named_steps["simpleimputer"]
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    return dict(
        format="laser-pe-confidence-v1",
        decision_threshold=0.9,
        intercept=float(classifier.intercept_[0]),
        numeric={
            name: dict(
                mean=float(scaler.mean_[i]),
                scale=float(scaler.scale_[i]),
                impute=float(imputer.statistics_[i]),
                coefficient=float(classifier.coef_[0, i]),
            )
            for i, name in enumerate(x.columns)
        },
        categorical={},
        composition_priors=priors,
        provenance=dict(
            training_datasets=sorted(data.dataset.unique().tolist()),
            training_candidates=len(data),
            same_feature_positive=int(data.correct.sum()),
            target="same curated species, adduct, stream and original XCMS feature",
            unknown_identities="unconfirmed negatives, not proven false assignments",
            inputs="local PE mass/RT residuals, coherence, abundance, independently aligned negative mode and broad training-only composition evidence",
            excludes="old globally fitted class confidence; curated workbook access at annotation runtime",
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--curation-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for dataset in args.datasets:
        x = pd.read_csv(args.candidate_root / dataset / "candidate_decisions.csv")
        x = x[x.pe_decision.eq("supported")].copy()
        c = pd.read_csv(args.curation_root / dataset / "curated_feature_crosswalk.csv")
        c = c[
            c.stream.eq("RP_Pos")
            & c.matlab_class.eq("PE")
            & c.crosswalk_status.eq("profile_match")
        ]
        truth = set(
            zip(
                c.matlab_species,
                c.matlab_adduct,
                c.xcms_feature_id.astype(int).astype(str),
            )
        )
        x["correct"] = [
            (n, a, str(int(f))) in truth
            for n, a, f in zip(x["name"], x.adduct, x.feat_feature_id)
        ]
        x["dataset"] = dataset
        frames.append(x)
    data = pd.concat(frames, ignore_index=True)
    from matlab_annotation.confidence import _score_features

    folds = []
    for dataset in args.datasets:
        train = data[~data.dataset.eq(dataset)]
        test = data[data.dataset.eq(dataset)]
        model = fit_payload(train)
        score = _score_features(
            pe_evidence_features(test, model["composition_priors"]), model
        )
        selected = score.ge(0.9)
        folds.append(
            dict(
                dataset=dataset,
                candidates=len(test),
                selected=int(selected.sum()),
                same_peak=int(test.loc[selected, "correct"].sum()),
                not_same_peak=int((~test.loc[selected, "correct"]).sum()),
            )
        )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(
        json.dumps(fit_payload(data), indent=2, allow_nan=False) + "\n"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(args.report, index=False)
    print(pd.DataFrame(folds).to_string(index=False))


if __name__ == "__main__":
    main()
