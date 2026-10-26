"""Prospective049 ENTIRE041 algorithm with new scores, no query curation.

Same native029 global selector/005STD/027mass stage. Completion is recomputed
with current scores/occupancy and new INNER-calibrated041-policy cutoffs, not
copied old outer outputs. No model fitting or operating-point calibration here.
"""

import json

import numpy as np
import pandas as pd

import matrix_experiment_data as data
import run_qe_np_context_bmp_experiment as protection
from catboost_pipeline_assembly import check_original_coordinates, disabled_replay


def check_original_sample_cells(annotations, baseline):
    """Every declared raw sample column, including QC/blank/standards, exact."""
    from annotator.peak_evidence import sample_roles
    from standard_lipidomics.validation import canonical_feature_id

    cells = 0
    for stream, part in annotations.groupby("stream", sort=False):
        raw = baseline["streams"][stream]["features"]
        columns, _, _ = sample_roles(raw)
        assert raw.feature_id.map(canonical_feature_id).is_unique
        lookup = raw.set_index(raw.feature_id.map(canonical_feature_id)).loc[
            part.feat_feature_id.map(canonical_feature_id)
        ]
        for column in columns:
            destination = "feat_" + column
            assert destination in part, (stream, destination)
            actual = pd.to_numeric(part[destination], errors="coerce").to_numpy(float)
            expected = pd.to_numeric(lookup[column], errors="coerce").to_numpy(float)
            np.testing.assert_array_equal(actual, expected)
            cells += len(part)
    return cells


def restore_original_cells(annotations, baseline):
    """Output-only copy of ORIGINAL cells after ID/MZ/RT/intensity validation.

    Old saved core candidate tables can lack per-sample values. Do not overwrite
    those frozen sources or infer/rescale data: copy the chosen raw feature.
    """
    from standard_lipidomics.validation import canonical_feature_id

    check_original_coordinates(annotations, baseline)
    output = annotations.copy(deep=True)
    for stream, indices in annotations.groupby("stream", sort=False).groups.items():
        raw = baseline["streams"][stream]["features"]
        source = raw.set_index(raw.feature_id.map(canonical_feature_id)).loc[
            annotations.loc[indices, "feat_feature_id"].map(canonical_feature_id)
        ]
        for column in raw:
            if column == "feature_id":
                # Already validated; preserve exact annotation ID representation.
                continue
            dest = "feat_" + column
            if dest not in output:
                output[dest] = pd.Series(index=output.index, dtype=object)
            else:
                output[dest] = output[dest].astype(object)
            output.loc[indices, dest] = pd.Series(
                source[column].to_numpy(), index=indices
            )
    check_original_coordinates(output, baseline)
    check_original_sample_cells(output, baseline)
    return output


def assemble_core(dataset, outer, raw, scores, thresholds, snapshot):
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.declared_standards import declared_standard_targets
    from matlab_annotation.standard_completion import complete_declared_standards
    from matlab_annotation.calibrated_mass_discovery import select_calibrated_additions

    datasets, folds = data.inventory()
    assert dataset in datasets and folds[dataset] == outer
    assert raw["query_dataset"] == dataset and set(raw["excluded_folds"]) == {outer}
    assert not data.peer.FORBIDDEN & set(raw["pool"])
    assert set(thresholds) == set(data.MODES) and scores.index.equals(raw["pool"].index)
    original_pool, original_evidence, original_scores = (
        raw["pool"].copy(deep=True),
        raw["evidence"].copy(deep=True),
        scores.copy(deep=True),
    )
    baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
    standards = json.loads(
        (data.old.STD / "models" / f"fold_{outer}_standard_models.json").read_text()
    )
    blocked = {d[:7] for d in datasets if folds[d] == outer}
    assert not set(standards["training_groups"]) & blocked
    catalogue = declared_standard_targets(
        snapshot / "reference_data/matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    mass_pool = pd.read_pickle(
        data.old.MASS / "runs" / dataset / "private_mass_candidates.pkl"
    )
    original_mass = mass_pool.copy(deep=True)
    assert not data.peer.FORBIDDEN & set(mass_pool)
    automatic = pd.concat(
        [
            p["annotations"].copy().assign(stream=s)
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    parts, candidates, standard_audits, mass_audits = (
        [],
        [raw["pool"], mass_pool],
        [],
        [],
    )
    for stream, payload in baseline["streams"].items():
        pool = raw["pool"].loc[raw["pool"].stream.eq(stream)]
        selected = select_operating_candidates(
            pool,
            scores.loc[pool.index],
            threshold=thresholds[stream],
            ambiguity_margin=0.05,
        )
        selected["auto_annotation_status"] = "matrix_context_mode_expert_selected"
        selected["operating_threshold"] = thresholds[stream]
        standard = (
            payload["annotations"]
            .loc[payload["annotations"].is_is.fillna(False).astype(bool)]
            .copy()
            .assign(stream=stream)
        )
        combined = pd.concat([selected, standard], ignore_index=True)
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
        combined = pd.concat([corrected, added], ignore_index=True)
        assert not combined.feat_feature_id.astype(str).duplicated().any()
        parts.append(combined.assign(stream=stream))
        candidates.extend(
            [
                standard,
                extra.assign(stream=stream),
                corrected.loc[corrected.is_is.fillna(False).astype(bool)].assign(
                    stream=stream
                ),
            ]
        )
        standard_audits.append(audit.assign(dataset=dataset, stream=stream))
        mass_audits.append(
            dict(
                dataset=dataset,
                stream=stream,
                added_annotations=len(added),
                current_core_annotations=len(corrected),
                fixed_027_rule=True,
            )
        )
    annotations = pd.concat(parts, ignore_index=True)
    annotations["automatic_accepted"], annotations["automatic_candidate"] = True, False
    annotations = restore_original_cells(annotations, baseline)
    candidate = pd.concat(candidates, ignore_index=True)
    columns = ["stream", "name", "adduct", "feat_feature_id", "is_is"]
    candidate = candidate.drop_duplicates(columns, keep="last").reset_index(drop=True)
    accepted = set(annotations[columns].itertuples(index=False, name=None))
    candidate["automatic_accepted"] = [
        k in accepted for k in candidate[columns].itertuples(index=False, name=None)
    ]
    candidate["automatic_candidate"] = ~candidate.automatic_accepted
    assert accepted == set(
        candidate.loc[candidate.automatic_accepted, columns].itertuples(
            index=False, name=None
        )
    )
    # Replace only accepted identical hypotheses with complete raw-cell copies;
    # unaccepted competing identities/features remain unchanged in the graph.
    candidate = pd.concat(
        [candidate.loc[~candidate.automatic_accepted], annotations], ignore_index=True
    )
    check_original_coordinates(annotations, baseline)
    check_original_sample_cells(annotations, baseline)
    pd.testing.assert_frame_equal(raw["pool"], original_pool, check_exact=True)
    pd.testing.assert_frame_equal(raw["evidence"], original_evidence, check_exact=True)
    pd.testing.assert_frame_equal(mass_pool, original_mass, check_exact=True)
    pd.testing.assert_series_equal(scores, original_scores, check_exact=True)
    return dict(
        annotations=annotations,
        candidates=candidate,
        standard_audit=pd.concat(standard_audits, ignore_index=True),
        mass_audit=pd.DataFrame(mass_audits),
    )


def proposals(dataset, mask, raw, scores, previous):
    from matlab_annotation.add_only_class_completion import free_completion_winners

    parts = []
    for stream in data.MODES:
        pool = raw["pool"].loc[raw["pool"].stream.eq(stream)]
        parts.append(
            free_completion_winners(
                pool,
                scores.loc[pool.index],
                previous.loc[previous.stream.eq(stream)],
                stream=stream,
                **protection.physical_protection(dataset, mask, stream, raw),
            )
        )
    return pd.concat(parts, ignore_index=True)


def append_completion(previous, candidates, filled):
    """Current core candidate graph, NEVER the fixed old029 candidate file."""
    old_keys = set(protection.accepted_keys(previous))
    if filled.empty:
        return previous.copy(deep=True), candidates.copy(deep=True)
    new_keys = set(protection.accepted_keys(filled))
    assert len(new_keys) == len(filled) and not old_keys & new_keys
    assert not {k[:4] for k in old_keys} & {k[:4] for k in new_keys}
    assert not {(k[0], k[4]) for k in old_keys} & {(k[0], k[4]) for k in new_keys}
    annotations = pd.concat([previous, filled], ignore_index=True)
    # Replace ONLY the identical accepted hypothesis with its full original
    # raw-cell copy; preserve all competing original-feature/identity tuples.
    current_keys = protection.accepted_keys(candidates)
    retained = candidates.loc[[k not in new_keys for k in current_keys]].copy()
    candidate = pd.concat([retained, filled], ignore_index=True)
    all_keys = protection.accepted_keys(candidate)
    accepted = old_keys | new_keys
    candidate["automatic_accepted"] = [k in accepted for k in all_keys]
    candidate["automatic_candidate"] = ~candidate.automatic_accepted
    assert accepted == set(
        protection.accepted_keys(candidate.loc[candidate.automatic_accepted])
    )
    assert set(current_keys) <= set(all_keys)
    assert not annotations.duplicated(["stream", "feat_feature_id"]).any()
    pd.testing.assert_frame_equal(
        annotations.iloc[: len(previous)][previous.columns].reset_index(drop=True),
        previous.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    return annotations, candidate


def complete(dataset, outer, raw, scores, core, thresholds):
    from matlab_annotation.add_only_class_completion import select_completion

    free = proposals(dataset, {outer}, raw, scores, core["annotations"])
    chosen = select_completion(free, thresholds)
    filled = protection.fill_original_features(dataset, chosen)
    if len(filled):
        filled["auto_annotation_status"] = "conservative_add_only_class_completion"
        filled.rename(
            columns={"context_completion_score": "class_completion_score"}, inplace=True
        )
    annotations, candidates = append_completion(
        core["annotations"], core["candidates"], filled
    )
    baseline = pd.read_pickle(data.BASE / "runs" / dataset / "full_annotation.pkl")
    check_original_coordinates(annotations, baseline)
    check_original_sample_cells(annotations, baseline)
    return dict(
        core,
        annotations=annotations,
        candidates=candidates,
        completion_additions=filled,
    )


def whole_disabled(dataset):
    """Explicit complete041 off switch, not selective old/new class reuse."""
    return disabled_replay(dataset)
