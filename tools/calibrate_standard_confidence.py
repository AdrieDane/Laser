"""Fit/evaluate standard confidence using curated benchmark exports.

Run with PYTHONPATH=app and the optional ``calibration`` dependencies installed.
Each evaluation fold derives its coefficients and composition priors only from
the other datasets. The final model is fitted on all supplied training cohorts;
independent validation cohorts must be kept outside ``--benchmark-root``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from matlab_annotation.automatic import matlab_annotation_adducts
from matlab_annotation.composition_plausibility import (
    COMPOSITION_FEATURES, composition_features, fit_composition_priors,
)
from matlab_annotation.confidence import (
    PostRegressionCompletionSettings, _candidate_features, _preserve_count,
    _score_features, _select_confident,
)
from matlab_annotation.fitted_grid_completion import filter_uncertain_new_features


BASE_NUMERIC = [
    "abs_ppm", "abs_rt_seconds", "coherence", "stream_intensity_z",
    "class_intensity_z", "feature_candidates", "identity_features",
]
BASE_CATEGORICAL = ["stream", "lipid_class", "adduct"]
NUMERIC = BASE_NUMERIC + list(COMPOSITION_FEATURES)
CATEGORICAL = BASE_CATEGORICAL + ["stream_class"]


def load_training(root: Path) -> pd.DataFrame:
    datasets = []
    for directory in sorted(root.glob("*/biological_abundance")):
        candidates = pd.read_csv(directory / "laser_annotations.csv")
        crosswalk = pd.read_csv(directory / "feature_crosswalk.csv")
        truth = crosswalk[
            crosswalk["crosswalk_status"].eq("profile_match")
            & crosswalk["xcms_feature_id"].notna()
        ]
        truth_keys = set(zip(
            truth["stream"].astype(str), truth["identity_key"].astype(str),
            truth["xcms_feature_id"].astype(int).astype(str), strict=True,
        ))
        truth_identities = set(zip(
            truth["stream"].astype(str), truth["identity_key"].astype(str), strict=True,
        ))
        allowed = [
            str(adduct) in matlab_annotation_adducts(str(stream).split("_")[0])
            for stream, adduct in zip(
                candidates["laser_stream"], candidates["laser_adduct"], strict=True,
            )
        ]
        candidates = candidates.loc[allowed].copy()
        parsed = candidates["laser_species"].astype(str).str.extract(
            r"\((?:O\+P-|O-|P-)?[dtmq]?(\d+):(\d+)\)"
        ).apply(pd.to_numeric, errors="coerce")
        candidates["carbon"] = parsed[0]
        candidates["db"] = parsed[1]
        candidates["name"] = candidates["laser_species"].astype(str).map(
            lambda value: re.sub(r" \(IS\)$", "", value)
        )
        candidates["correct"] = [
            (str(stream), str(identity), str(int(feature))) in truth_keys
            for stream, identity, feature in zip(
                candidates["laser_stream"], candidates["identity_key"],
                candidates["laser_feature_id"], strict=True,
            )
        ]
        candidates["identity_confirmed_in_reference"] = [
            (str(stream), str(identity)) in truth_identities
            for stream, identity in zip(
                candidates["laser_stream"], candidates["identity_key"], strict=True,
            )
        ]
        candidates = candidates.rename(columns={
            "laser_stream": "stream", "laser_feature_id": "feat_feature_id",
            "laser_class": "class", "laser_adduct": "adduct",
            "laser_intensity": "feat_intensity",
            "laser_grid_coherence": "grid_coherence_score",
            "laser_ppm_error": "ppm_error",
        })
        candidates["feat_feature_id"] = candidates["feat_feature_id"].astype(str)
        candidates["rt_error"] = candidates["laser_rt_error"] / 60.0
        candidates["dataset"] = directory.parent.name
        datasets.append(candidates)
    if not datasets:
        raise ValueError(f"No curated benchmark exports under {root}")
    return pd.concat(datasets, ignore_index=True)


def features_for(data: pd.DataFrame, priors: dict) -> pd.DataFrame:
    frames = []
    for (_dataset, stream), group in data.groupby(["dataset", "stream"], sort=False):
        frames.append(_candidate_features(group, str(stream)))
    features = pd.concat(frames).reindex(data.index)
    composition = pd.DataFrame({
        "lipid_class": features["lipid_class"],
        "carbon": data["carbon"], "db": data["db"],
    })
    for column, values in composition_features(composition, priors).items():
        features[column] = values
    return features


def fit_payload(
    data: pd.DataFrame, *, baseline: bool = False, fitted_grid: bool = False,
) -> dict:
    correct = data.loc[data["correct"], ["class", "carbon", "db"]].rename(
        columns={"class": "lipid_class"}
    )
    priors = fit_composition_priors(correct)
    features = features_for(data, priors)
    numeric = BASE_NUMERIC if baseline else NUMERIC
    categorical = BASE_CATEGORICAL if baseline else CATEGORICAL
    if fitted_grid:
        categorical = categorical + ["class_carbon", "class_db"]
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
        ), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = make_pipeline(
        preprocessing, LogisticRegression(max_iter=2000, C=0.3),
    )
    model.fit(features, data["correct"])
    preprocessing = model.named_steps["columntransformer"]
    scaler = preprocessing.named_transformers_["numeric"].named_steps["standardscaler"]
    imputer = preprocessing.named_transformers_["numeric"].named_steps["simpleimputer"]
    encoder = preprocessing.named_transformers_["categorical"]
    classifier = model.named_steps["logisticregression"]
    coefficients = classifier.coef_[0]
    offset = len(numeric)
    categories = {}
    for column, values in zip(categorical, encoder.categories_, strict=True):
        categories[column] = dict(zip(
            map(str, values), map(float, coefficients[offset:offset + len(values)]),
            strict=True,
        ))
        offset += len(values)
    payload = {
        "format": (
            "laser-fitted-grid-confidence-v1" if fitted_grid else
            f"laser-post-regression-confidence-v{1 if baseline else 2}"
        ),
        "decision_threshold": 0.5,
        "candidate_limits": {
            "ppm": 3.0, "rt_seconds": 10.0,
            "require_grid_coherence_pass": True, "monoisotopic_only": True,
        },
        "intercept": float(classifier.intercept_[0]),
        "numeric": {
            column: {
                "mean": float(mean), "scale": float(scale),
                "impute": float(impute), "coefficient": float(coefficient),
            }
            for column, mean, scale, impute, coefficient in zip(
                numeric, scaler.mean_, scaler.scale_, imputer.statistics_,
                coefficients[:len(numeric)], strict=True,
            )
        },
        "categorical": categories,
        "training": {
            "datasets": sorted(data["dataset"].unique().tolist()),
            "candidates": int(len(data)),
            "curated_matches": int(data["correct"].sum()),
            "label_interpretation": "Exact curated feature/identity agreement; unmatched is unconfirmed.",
        },
    }
    if not baseline:
        payload["composition_priors"] = priors
    # Validate the dependency-free runtime implementation against sklearn.
    np.testing.assert_allclose(
        _score_features(features, payload), model.predict_proba(features)[:, 1],
        rtol=1e-10, atol=1e-10,
    )
    return payload


def evaluate_fold(
    training: pd.DataFrame, validation: pd.DataFrame, *, fitted_grid: bool = False,
) -> list[dict]:
    baseline = fit_payload(training, baseline=True)
    model = fit_payload(training, fitted_grid=fitted_grid)
    reference_v2 = fit_payload(training) if fitted_grid else None
    features = features_for(validation, model["composition_priors"])
    settings = PostRegressionCompletionSettings()
    scores = _score_features(features, model)
    baseline_scores = _score_features(features, baseline)
    rows = []
    for stream, group in validation.groupby("stream", sort=True):
        group_features = features.loc[group.index]
        original, _passing, _ambiguous = _select_confident(
            group, group_features, baseline_scores.loc[group.index], settings,
        )
        if reference_v2:
            reference_scores = _score_features(group_features, reference_v2)
            reference_calls, _passing, _ambiguous = _select_confident(
                group, group_features, reference_scores, settings,
            )
            original, _retained = _preserve_count(
                reference_calls, original,
                group.assign(post_regression_confidence=reference_scores),
                group_features,
            )
        selected, _passing, _ambiguous = _select_confident(
            group, group_features, scores.loc[group.index], settings,
        )
        group = group.assign(post_regression_confidence=scores.loc[group.index])
        if fitted_grid:
            selected = filter_uncertain_new_features(
                selected, group, group_features, original, margin=0.10,
            )
        selected, retained = _preserve_count(
            selected, original, group, group_features,
        )
        rows.append({
            "dataset": str(validation["dataset"].iloc[0]), "stream": stream,
            "baseline_accepted": len(original),
            "baseline_exact": int(original["correct"].sum()),
            "updated_accepted": len(selected),
            "updated_exact": int(selected["correct"].sum()),
            "baseline_wrong_feature": int((
                ~original["correct"] & original["identity_confirmed_in_reference"]
            ).sum()),
            "updated_wrong_feature": int((
                ~selected["correct"] & selected["identity_confirmed_in_reference"]
            ).sum()),
            "baseline_calls_retained": retained,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True,
                        help="Deployed v1 model, or v2 model containing its frozen v1 reference")
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fitted-grid", action="store_true",
                        help="Fit the count-preserving fitted-grid refinement model")
    args = parser.parse_args()
    data = load_training(args.benchmark_root)
    rows = []
    for dataset in sorted(data["dataset"].unique()):
        validation = data[data["dataset"].eq(dataset)]
        training = data[~data["dataset"].eq(dataset)]
        rows.extend(evaluate_fold(training, validation, fitted_grid=args.fitted_grid))
    report = pd.DataFrame(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)
    payload = fit_payload(data, fitted_grid=args.fitted_grid)
    reference = json.loads(args.reference_model.read_text(encoding="utf-8"))
    reference = reference.get("count_reference_model", reference)
    if reference["format"] != "laser-post-regression-confidence-v1":
        raise ValueError("--reference-model must be the frozen deployed v1 model")
    if args.fitted_grid:
        payload["selection_policy"] = "preserve_v2_count_per_stream_after_grid_completion"
        payload["feature_ambiguity_margin"] = 0.10
    else:
        payload["count_reference_model"] = reference
        payload["selection_policy"] = "preserve_reference_count_per_stream"
    payload["validation"] = {
        "method": "leave-one-dataset-out; model and composition priors refitted per fold",
        "scope": "stored strict candidate pools; not full-pipeline reruns",
        "baseline_accepted": int(report["baseline_accepted"].sum()),
        "baseline_exact": int(report["baseline_exact"].sum()),
        "updated_accepted": int(report["updated_accepted"].sum()),
        "updated_exact": int(report["updated_exact"].sum()),
        "baseline_wrong_feature": int(report["baseline_wrong_feature"].sum()),
        "updated_wrong_feature": int(report["updated_wrong_feature"].sum()),
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(report.groupby("dataset")[[
        "baseline_accepted", "baseline_exact", "updated_accepted", "updated_exact",
    ]].sum().to_string())


if __name__ == "__main__":
    main()
