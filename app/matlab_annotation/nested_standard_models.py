"""Unused research: query-excluded models for declared-standard CONTEXT.

Fits chemical standard RT/channel metadata only. No endogenous choices, query
curation, original feature identifiers or scoring labels enter the fitter.
Caller must generate raw standard measurements independently of endo selection.
"""

from __future__ import annotations

import pandas as pd

from .nested_feature_folds import plan_nested_feature_folds
from .retention_atlas import predict_aligned_retention_atlas
from .standard_rt_library import fit_standard_retention_library
from .standard_rt_transfer import (
    standard_class_targets,
    exclude_declared_standard_masses,
    fit_standard_class_transfer,
)
from .standard_completion import fit_standard_channel_prior


def _validate_observations(frame, folds, groups, required):
    if not required.issubset(frame):
        raise ValueError("Complete standard-reference observations required")
    if not set(frame.dataset).issubset(folds):
        raise ValueError("Unassigned standard-reference dataset")
    if not frame.cohort.eq(frame.dataset.map(groups)).all():
        raise ValueError("Standard-reference cohort metadata disagrees with groups")
    if not frame.mapped.isin([True, False]).all():
        raise ValueError("Standard mapping validity must be explicit")


def fit_nested_standard_bundle(
    observations,
    channel_observations,
    automatic_by_dataset,
    declared_by_dataset,
    reference_bundle,
    fold_assignment,
    project_groups,
    excluded_folds,
):
    """Filter BOTH standard fits and transfer labels BEFORE numerical modeling.

    Reference atlas has the SAME exclusion mask. Transfer calibration loops
    only allowed reference projects and excludes every declared spike mass from
    its endogenous alignment. Dataset/cohort are splitting metadata, not model
    predictor columns. Baseline automatic anchors have known prior exposure.
    """
    plan = plan_nested_feature_folds(fold_assignment, project_groups)
    mask = tuple(sorted(set(excluded_folds)))
    if mask not in plan["models"] or len(mask) != len(excluded_folds):
        raise ValueError("Unknown/duplicate nested standard exclusion mask")
    scope = plan["models"][mask]
    if (
        tuple(reference_bundle["excluded_folds"]) != mask
        or set(reference_bundle["training_groups"]) != set(scope["training_groups"])
        or set(reference_bundle["excluded_groups"]) != set(scope["excluded_groups"])
    ):
        raise ValueError("Standard and endogenous reference scopes must agree")
    meta = {"dataset", "cohort", "stream", "mapped"}
    _validate_observations(
        observations,
        fold_assignment,
        project_groups,
        meta
        | {
            "standard_compound_id",
            "adduct",
            "rt_seconds",
            "reporting_name",
            "reporting_class",
        },
    )
    _validate_observations(
        channel_observations,
        fold_assignment,
        project_groups,
        meta | {"name", "adduct"},
    )
    allowed = set(scope["training_datasets"])
    standard_columns = [
        "dataset",
        "cohort",
        "stream",
        "standard_compound_id",
        "adduct",
        "rt_seconds",
        "reporting_name",
        "reporting_class",
    ]
    train = observations.loc[
        observations.dataset.isin(allowed) & observations.mapped.astype(bool),
        standard_columns,
    ].copy()
    channel_train = channel_observations.loc[
        channel_observations.dataset.isin(allowed)
        & channel_observations.mapped.astype(bool),
        ["cohort", "stream", "name", "adduct"],
    ].copy()
    if train.empty or channel_train.empty:
        raise ValueError("No mapped standard references in permitted groups")
    if not set(train.cohort).issubset(scope["training_groups"]):
        raise ValueError("Excluded project entered standard-reference fitting")
    library = fit_standard_retention_library(train)
    channels = fit_standard_channel_prior(channel_train)
    residuals = []
    atlas = reference_bundle["atlas"]
    for dataset in sorted(allowed):
        manual = train[train.dataset.eq(dataset)]
        if manual.empty:
            continue
        if dataset not in automatic_by_dataset or dataset not in declared_by_dataset:
            raise ValueError(
                "Allowed transfer project lacks automatic/declaration inputs"
            )
        catalogue = declared_by_dataset[dataset]
        for stream, declared in catalogue.groupby("stream"):
            targets = standard_class_targets(declared, library, stream=stream)
            clean = exclude_declared_standard_masses(
                automatic_by_dataset[dataset][stream], declared
            )
            predicted = predict_aligned_retention_atlas(
                targets, clean, atlas, stream=stream
            )
            targets = targets.join(predicted)
            lookup = {s: g.iloc[0] for s, g in targets.groupby("standard_compound_id")}
            for row in manual[manual.stream.eq(stream)].itertuples():
                target = lookup.get(row.standard_compound_id)
                if target is not None:
                    residuals.append(
                        dict(
                            cohort=project_groups[dataset],
                            stream=stream,
                            standard_compound_id=row.standard_compound_id,
                            **{"class": target["class"]},
                            residual_seconds=row.rt_seconds
                            - float(target.atlas_rt_seconds),
                        )
                    )
    residual_columns = [
        "cohort",
        "stream",
        "standard_compound_id",
        "class",
        "residual_seconds",
    ]
    residual_frame = pd.DataFrame(residuals, columns=residual_columns)
    transfer = fit_standard_class_transfer(residual_frame)
    return dict(
        format="laser-nested-standard-context-models-v1",
        excluded_folds=list(mask),
        training_groups=scope["training_groups"],
        excluded_groups=scope["excluded_groups"],
        training_datasets=scope["training_datasets"],
        excluded_datasets=scope["excluded_datasets"],
        standard_observation_rows=len(train),
        channel_observation_rows=len(channel_train),
        effective_standard_groups=sorted(train.cohort.unique()),
        transfer_reference_rows=len(residual_frame),
        library=library,
        channels=channels,
        atlas=atlas,
        transfer=transfer,
        note="Reference fitting ONLY, not annotations or standard-correctness/accuracy proof. Query/outer/inner projects excluded before fits. No original feature IDs/scoring labels exported. Inherited baseline anchor exposure remains.",
    )
