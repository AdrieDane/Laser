"""Unused research: grouped nested feature-generation exclusion plans.

Splitting metadata only, NEVER numerical scoring features. No curation reads,
model fitting, annotation calls or production hooks. All related acquisitions
must have the SAME explicit project-group key before calling this helper.
"""

from __future__ import annotations

from itertools import combinations


def plan_nested_feature_folds(fold_assignment, project_groups):
    """Plan true query-excluded features using existing project-group blocks.

    For outer f: final training query in block k uses RT/prior/envelope models
    excluding {f,k}; held-out queries use {f}. For inner validation block j:
    fitting query in k uses {f,j,k}; validation queries use {f,j}. Thus neither
    evaluated nor calibration projects can leak through training predictors.
    With five original blocks this requires only25 distinct exclusion masks
    (5 singles+10 pairs+10 triples), not a model per original feature/dataset.
    """
    if not fold_assignment or set(fold_assignment) != set(project_groups):
        raise ValueError("Every dataset needs an explicit fold and project group")
    if any(
        isinstance(f, bool) or not isinstance(f, int) or f < 0
        for f in fold_assignment.values()
    ):
        raise ValueError("Fold IDs must be nonnegative integers")
    if any(not str(g).strip() for g in project_groups.values()):
        raise ValueError("Project groups must be explicit and nonempty")
    folds = sorted(set(fold_assignment.values()))
    if len(folds) < 4:
        raise ValueError("Nested block feature exclusion needs at least4 folds")
    seen = {}
    for dataset, group in project_groups.items():
        fold = fold_assignment[dataset]
        if group in seen and seen[group] != fold:
            raise ValueError("Related acquisitions cannot cross project-group folds")
        seen[group] = fold
    datasets = sorted(fold_assignment)
    masks = sorted({tuple(c) for n in (1, 2, 3) for c in combinations(folds, n)})
    models = {}
    for mask in masks:
        training = [d for d in datasets if fold_assignment[d] not in mask]
        excluded = [d for d in datasets if fold_assignment[d] in mask]
        assert training and excluded
        models[mask] = dict(
            training_datasets=training,
            excluded_datasets=excluded,
            training_groups=sorted({project_groups[d] for d in training}),
            excluded_groups=sorted({project_groups[d] for d in excluded}),
        )
    outer = {}
    for fold in folds:
        train = [d for d in datasets if fold_assignment[d] != fold]
        test = [d for d in datasets if fold_assignment[d] == fold]
        inner = {}
        for validation_fold in folds:
            if validation_fold == fold:
                continue
            fitting = [d for d in train if fold_assignment[d] != validation_fold]
            validation = [d for d in train if fold_assignment[d] == validation_fold]
            feature_masks = {
                d: tuple(sorted({fold, validation_fold, fold_assignment[d]}))
                for d in fitting
            }
            validation_mask = tuple(sorted({fold, validation_fold}))
            assert all(
                d not in models[m]["training_datasets"]
                for d, m in feature_masks.items()
            )
            assert all(
                not set(models[m]["training_datasets"]) & set(test + validation)
                for m in feature_masks.values()
            )
            inner[validation_fold] = dict(
                score_training_datasets=fitting,
                validation_datasets=validation,
                training_feature_masks=feature_masks,
                validation_feature_mask=validation_mask,
            )
        outer[fold] = dict(
            score_training_datasets=train,
            heldout_datasets=test,
            training_feature_masks={
                d: tuple(sorted({fold, fold_assignment[d]})) for d in train
            },
            heldout_feature_mask=(fold,),
            inner=inner,
        )
    return dict(
        format="laser-nested-feature-exclusion-plan-v1",
        models=models,
        outer=outer,
        note="Plan only. No predictors/models/annotations have been generated; no performance claim.",
    )
