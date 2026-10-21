"""Fit portable PE peak-preference models on explicitly chosen development data.

Only reliably mapped ordinary PE identities with a correct peak in the frozen
candidate pool supply supervised pairs. Unlisted identities are not negative
training labels. Annotation runtime never reads these manual crosswalks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from matlab_annotation.experimental_pe import (
    MODEL_FILENAMES,
    ExperimentalPESelectionSettings,
    experimental_pe_evidence,
    select_experimental_pe,
)
from matlab_annotation.pe_rebuild import score_pe_rebuild
from output_io import atomic_write_csv, atomic_write_json, sha256_file
from standard_lipidomics.validation import (
    compare_manual_annotations,
    summarize_comparison,
)


def training_pairs(pool: pd.DataFrame, crosswalk: pd.DataFrame, stream: str) -> tuple:
    c = crosswalk[
        crosswalk.stream.eq(stream)
        & crosswalk.matlab_class.eq("PE")
        & crosswalk.crosswalk_status.eq("profile_match")
    ]
    if "matlab_is_internal_standard" in c:
        c = c[
            ~c.matlab_is_internal_standard.fillna(False)
            .astype(str)
            .str.casefold()
            .isin({"true", "1", "1.0"})
        ]
    from standard_lipidomics.validation import canonical_feature_id

    truth = set(zip(c.matlab_species, c.xcms_feature_id.map(canonical_feature_id)))
    labelled = pool.copy()
    labelled["_correct"] = [
        (n, str(f)) in truth for n, f in zip(pool["name"], pool.feat_feature_id)
    ]
    x = experimental_pe_evidence(pool)
    data, labels, weights = [], [], []
    groups = 0
    for name, group in labelled.groupby("name", sort=True):
        if name not in set(c.matlab_species):
            continue
        positive, alternatives = group[group._correct], group[~group._correct]
        if positive.empty or alternatives.empty:
            continue
        groups += 1
        weight = 1 / (len(positive) * len(alternatives))
        for a in positive.index:
            for b in alternatives.index:
                delta = x.loc[a].to_numpy() - x.loc[b].to_numpy()
                data.extend([delta, -delta])
                labels.extend([1, 0])
                weights.extend([weight, weight])
    return np.asarray(data), np.asarray(labels), np.asarray(weights), list(x), groups


def fit_pair_model(
    candidates: list[tuple[pd.DataFrame, pd.DataFrame]], stream: str
) -> dict:
    parts = [training_pairs(p, c, stream) for p, c in candidates]
    usable = [p for p in parts if len(p[0])]
    if not usable or any(p[3] != usable[0][3] for p in usable):
        raise ValueError("Missing or inconsistent labelled peak-preference data")
    x, y, w = [np.concatenate([p[i] for p in usable]) for i in range(3)]
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.04,
        min_samples_leaf=8,
        random_state=19,
    )
    model.fit(x, y, sample_weight=w)
    prior = float(np.average(y, weights=w))
    portable = dict(
        format="laser-pe-rebuild-v1",
        features=usable[0][3],
        intercept=float(np.log(prior / (1 - prior))),
        learning_rate=model.learning_rate,
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
        score_pe_rebuild(pd.DataFrame(x, columns=portable["features"]), portable),
        model.predict_proba(x)[:, 1],
        atol=1e-12,
        rtol=1e-12,
    )
    return dict(
        format="laser-experimental-pe-selector-v1",
        stream=stream,
        pair_model=portable,
        training_pair_rows=len(y),
        training_identity_groups=sum(p[4] for p in usable),
        label_policy="mapped same-identity preferred peaks versus competing peaks; unlisted identities excluded",
        score_caveat="antisymmetrized peak preference, not identity probability or calibrated accuracy",
        policy=dict(
            minimum_pair_score=0.9,
            minimum_rank_score=0.65,
            mass_worsening_max_ppm=0.2,
            local_RT_worsening_max_seconds=1,
            minimum_mass_gain_ppm=0.3,
            minimum_local_RT_gain_seconds=2,
            RP_positive_replacement="both minimum mass and local RT gain required",
            NP_negative_addition_mass_ppm=2.5,
            NP_negative_addition_local_RT_seconds=5,
            NP_negative_addition_support="isotope envelope OR accepted opposite-mode RT corroboration",
        ),
    )


def _metrics(ann, crosswalk, stream):
    selected = (
        ann[
            [
                c
                for c in ("name", "class", "adduct", "feat_feature_id", "is_is")
                if c in ann
            ]
        ]
        .copy()
        .assign(stream=stream)
    )
    result = compare_manual_annotations(crosswalk, selected)
    s = summarize_comparison(result, group_by=("stream", "lipid_class"))
    row = s[s.stream.eq(stream) & s.lipid_class.eq("PE")].iloc[0]
    return {
        k: int(row[k])
        for k in (
            "manual_same_peak",
            "manual_same_lipid_different_peak",
            "laser_not_in_manual_list",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--curation-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--stream", choices=sorted(MODEL_FILENAMES), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cross-validation-output", type=Path)
    parser.add_argument("--trust-local-pickles", action="store_true")
    args = parser.parse_args()
    if not args.trust_local_pickles:
        parser.error(
            "Use --trust-local-pickles only for trusted, locally generated candidate artifacts"
        )
    data, provenance = {}, []
    for name in args.datasets:
        p = args.candidate_root / args.stream / name / "raw_candidates.pkl"
        c = args.curation_root / name / "curated_feature_crosswalk.csv"
        data[name] = (pd.read_pickle(p), pd.read_csv(c))
        provenance.append(
            dict(
                dataset=name,
                candidate_sha256=sha256_file(p),
                crosswalk_sha256=sha256_file(c),
            )
        )
    payload = fit_pair_model(list(data.values()), args.stream)
    payload.update(
        training_datasets=args.datasets,
        training_artifacts=provenance,
        validation_caveat="development data; preceding annotation models are not refitted within selector-only folds",
    )
    target = args.output_directory / MODEL_FILENAMES[args.stream]
    atomic_write_json(target, payload, allow_nan=False)
    print(
        f"Fitted {args.stream}: {payload['training_pair_rows']} pair rows in {payload['training_identity_groups']} lipid groups"
    )
    if args.cross_validation_output:
        records = []
        for held_out, (pool, crosswalk) in data.items():
            fold = fit_pair_model(
                [v for n, v in data.items() if n != held_out], args.stream
            )
            folder = args.candidate_root / args.stream / held_out
            baseline = pd.read_csv(
                folder / "baseline_annotations.csv",
                dtype={"feat_feature_id": str},
                low_memory=False,
            )
            protected = pd.read_pickle(folder / "protected_candidates.pkl")
            pairs = pd.concat([baseline, protected], ignore_index=True, sort=False)
            after, _, _, changes, _ = select_experimental_pe(
                baseline,
                pairs,
                pool,
                stream=args.stream,
                model=fold["pair_model"],
                settings=ExperimentalPESelectionSettings(enabled=True),
            )
            before = _metrics(baseline, crosswalk, args.stream)
            now = _metrics(after, crosswalk, args.stream)
            records.append(
                dict(
                    dataset=held_out,
                    stream=args.stream,
                    changes=len(changes),
                    **{f"before_{k}": v for k, v in before.items()},
                    **now,
                )
            )
        table = pd.DataFrame(records)
        atomic_write_csv(args.cross_validation_output, table)
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
