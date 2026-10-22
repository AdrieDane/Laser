"""Unused research: fit numerical reference models from explicit allowed groups.

This fitter creates RT/prior/range parameters only, never annotations. Query
roles/exclusion masks come from nested_feature_folds and must be checked again
when predictors are generated and scoring models are fitted.
"""

from __future__ import annotations

from .nested_feature_folds import plan_nested_feature_folds
from .retention_atlas import fit_retention_atlas
from .class_candidates import fit_class_composition_priors
from .chemical_targets import fit_composition_envelopes


def fit_nested_reference_bundle(
    observations, fold_assignment, project_groups, excluded_folds
):
    """Filter BEFORE any learned RT, composition prevalence or envelope fit."""
    required = {
        "dataset",
        "cohort",
        "stream",
        "class",
        "carbon",
        "db",
        "rt_seconds",
        "mapped",
    }
    if not required.issubset(observations):
        raise ValueError("Complete reference observation metadata required")
    plan = plan_nested_feature_folds(fold_assignment, project_groups)
    mask = tuple(sorted(set(excluded_folds)))
    if mask not in plan["models"] or len(mask) != len(excluded_folds):
        raise ValueError("Unknown/duplicate nested exclusion mask")
    if not set(observations.dataset).issubset(fold_assignment):
        raise ValueError("Reference observations contain an unassigned dataset")
    if not observations.cohort.eq(observations.dataset.map(project_groups)).all():
        raise ValueError(
            "Reference cohort metadata disagrees with explicit project groups"
        )
    if not observations.mapped.isin([True, False]).all():
        raise ValueError("Reference mapping validity must be explicit")
    scope = plan["models"][mask]
    allowed = set(scope["training_datasets"])
    train = observations.loc[
        observations.dataset.isin(allowed) & observations.mapped.astype(bool),
        ["cohort", "stream", "class", "carbon", "db", "rt_seconds"],
    ].copy()
    groups = set(scope["training_groups"])
    assert set(train.cohort) == groups
    assert not groups & set(scope["excluded_groups"])
    return dict(
        format="laser-nested-reference-bundle-v1",
        excluded_folds=list(mask),
        training_groups=scope["training_groups"],
        excluded_groups=scope["excluded_groups"],
        training_datasets=scope["training_datasets"],
        excluded_datasets=scope["excluded_datasets"],
        training_rows=len(train),
        atlas=fit_retention_atlas(train),
        composition_priors=fit_class_composition_priors(train),
        composition_envelope=fit_composition_envelopes(train),
        note="Numerical reference preparation only, NOT annotation accuracy. Original-feature identifiers are not fitter inputs or exported parameters.",
    )
