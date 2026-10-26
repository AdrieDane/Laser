"""048 whole new core + unchanged005standards/027mass; no fits or gold access.

Intentionally does NOT append old041 class-completion calls. Reference replay
with original029scores/cuts must reproduce the whole029 scientific output.
"""

import json

import numpy as np
import pandas as pd

import catboost_experiment_data as data
from catboost_operating_points import select_candidates


def disabled_replay(dataset):
    """Explicit whole-method off switch, never per-class outer winner choices."""
    before = pd.read_csv(
        data.PREVIOUS / "runs" / dataset / "automatic_annotations.csv",
        dtype=str,
        keep_default_na=False,
    )
    candidates = pd.read_pickle(data.PREVIOUS / "runs" / dataset / "candidates.pkl")
    return before.copy(deep=True), candidates.copy(deep=True)


def check_original_coordinates(annotations, baseline):
    from standard_lipidomics.validation import canonical_feature_id

    for stream, part in annotations.groupby("stream", sort=False):
        original = baseline["streams"][stream]["features"]
        ids = original.feature_id.map(canonical_feature_id)
        assert not ids.duplicated().any()
        current_ids = part.feat_feature_id.map(canonical_feature_id)
        assert not current_ids.duplicated().any()
        reference = original.set_index(ids, drop=False).loc[current_ids]
        for actual, raw in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            np.testing.assert_allclose(
                pd.to_numeric(part[actual], errors="coerce").to_numpy(float),
                pd.to_numeric(reference[raw], errors="coerce").to_numpy(float),
                rtol=8 * np.finfo(float).eps,
                atol=0,
                equal_nan=True,
            )
    return True


def assemble(
    dataset,
    outer,
    raw,
    scores,
    mode_thresholds,
    class_thresholds,
    snapshot,
    *,
    status="catboost_class_calibrated_selected",
):
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

    datasets, folds = data.inventory()
    assert dataset in datasets and folds[dataset] == outer
    assert raw["query_dataset"] == dataset and set(raw["excluded_folds"]) == {outer}
    assert raw["pool"].index.equals(scores.index)
    assert set(mode_thresholds) == set(data.MODES)
    assert set(class_thresholds) == set(data.MODES)
    before_pool = raw["pool"].copy(deep=True)
    before_evidence = raw["evidence"].copy(deep=True)
    before_scores = scores.copy(deep=True)
    baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
    standards = json.loads(
        (data.STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    assert not set(standards["training_groups"]) & excluded
    catalogue = declared_standard_targets(
        snapshot / "reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    mass_pool = pd.read_pickle(
        data.MASS / "runs" / dataset / "private_mass_candidates.pkl"
    )
    before_mass = mass_pool.copy(deep=True)
    assert not {
        "training_positive",
        "training_label_known",
        "same_manual_peak",
        "calibration_identity_in_manual",
    } & set(mass_pool)
    automatic = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    parts, candidates, audits, mass_audits = [], [raw["pool"], mass_pool], [], []
    for stream, payload in baseline["streams"].items():
        pool = raw["pool"].loc[raw["pool"].stream.eq(stream)]
        selected = select_candidates(
            pool,
            scores.loc[pool.index],
            baseline_threshold=mode_thresholds[stream],
            class_thresholds=class_thresholds[stream],
            ambiguity_margin=0.05,
        )
        selected["auto_annotation_status"] = status
        protected = (
            payload["annotations"]
            .loc[payload["annotations"].is_is.fillna(False).astype(bool)]
            .copy()
            .assign(stream=stream)
        )
        combined = pd.concat([selected, protected], ignore_index=True)
        assert not combined.feat_feature_id.astype(str).duplicated().any()
        corrected, extra, audit = complete_declared_standards(
            combined,
            payload["features"],
            catalogue.loc[catalogue.stream.eq(stream)],
            standards["library"],
            standards["atlas"],
            standards["transfer"],
            standards["channels"],
            stream=stream,
            alignment_annotations=payload["annotations"],
        )
        local_mass = (
            mass_pool.loc[mass_pool.stream.eq(stream)] if len(mass_pool) else mass_pool
        )
        added = select_calibrated_additions(
            local_mass, automatic, corrected, stream=stream
        )
        mass_audits.append(
            dict(
                dataset=dataset,
                stream=stream,
                added_annotations=len(added),
                current_core_annotations=len(corrected),
                fixed_027_rule=True,
            )
        )
        combined = pd.concat([corrected, added], ignore_index=True)
        assert not combined.feat_feature_id.astype(str).duplicated().any()
        parts.append(combined.assign(stream=stream))
        audits.append(audit.assign(dataset=dataset, stream=stream))
        candidates.extend(
            [
                protected,
                extra.assign(stream=stream),
                corrected.loc[corrected.is_is.fillna(False).astype(bool)].assign(
                    stream=stream
                ),
            ]
        )
    annotations = pd.concat(parts, ignore_index=True)
    candidate = pd.concat(candidates, ignore_index=True)
    keys = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    candidate = candidate.drop_duplicates(keys, keep="last").reset_index(drop=True)
    accepted = set(annotations[keys].itertuples(index=False, name=None))
    candidate["automatic_accepted"] = [
        k in accepted for k in candidate[keys].itertuples(index=False, name=None)
    ]
    assert accepted == set(
        candidate.loc[candidate.automatic_accepted, keys].itertuples(
            index=False, name=None
        )
    )
    check_original_coordinates(annotations, baseline)
    pd.testing.assert_frame_equal(raw["pool"], before_pool, check_exact=True)
    pd.testing.assert_frame_equal(raw["evidence"], before_evidence, check_exact=True)
    pd.testing.assert_frame_equal(mass_pool, before_mass, check_exact=True)
    pd.testing.assert_series_equal(scores, before_scores, check_exact=True)
    return dict(
        annotations=annotations,
        candidates=candidate,
        standard_audit=pd.concat(audits, ignore_index=True),
        mass_audit=pd.DataFrame(mass_audits),
    )
