"""Unused research: permitted TRAINING isomer-label pairs on ONE raw feature.

No fitting, annotation decisions, file reads or production changes. Names/FIDs
only group physical hypotheses; original label-free numerical evidence enters
the pair matrix. Manual-absent features never become fabricated negatives.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class

from .conditional_peak_pairs import pair_predictors
from .evidence_training import validate_feature_scope


def isomer_training_pairs(data_by_dataset, *, training_groups, excluded_groups):
    matrices, labels, projects, initial_weights = [], [], [], []
    contribution, features = Counter(), Counter()
    class_projects = defaultdict(set)
    for data in data_by_dataset.values():
        validate_feature_scope(
            data, training_groups, excluded_groups, query_is_training=True
        )
        pool, evidence = data["pool"], data["evidence"]
        required = {
            "stream",
            "name",
            "adduct",
            "class",
            "ref_mz",
            "is_is",
            "feat_feature_id",
            "training_positive",
            "training_label_known",
        }
        if not required.issubset(pool) or not pool.index.is_unique:
            raise ValueError("Explicit unique physical isomer hypotheses required")
        if (
            not pool[["training_positive", "training_label_known", "is_is"]]
            .isin([True, False])
            .all()
            .all()
        ):
            raise ValueError("Explicit boolean supervision and standard flags required")
        if (pool.training_positive & ~pool.training_label_known).any():
            raise ValueError("Positive isomer training labels cannot be unknown")
        keys = ["stream", "feat_feature_id", "name", "adduct"]
        flags = pool.groupby(keys)[
            ["training_positive", "training_label_known", "is_is"]
        ].nunique()
        if flags.gt(1).any().any():
            raise ValueError("Contradictory duplicate physical isomer labels")
        unique = pool.drop_duplicates(keys, keep="first")
        if (
            unique[unique.training_positive]
            .groupby(["stream", "feat_feature_id"])
            .size()
            .gt(1)
            .any()
        ):
            raise ValueError("Multiple positive identities on one original feature")
        named_standard = unique.name.astype(str).str.startswith(
            "IS: "
        ) | unique.name.astype(str).str.endswith(" (IS)")
        unique = unique[~(unique.is_is | named_standard)]
        group_keys = ["stream", "feat_feature_id", "adduct"]
        flags = (
            unique.assign(
                _known_negative=unique.training_label_known & ~unique.training_positive
            )
            .groupby(group_keys)[["training_positive", "_known_negative"]]
            .transform("any")
        )
        usable = flags.training_positive & flags._known_negative
        unique = unique[usable]
        left, right = [], []
        project = data["query_group"]
        for _, group in unique.groupby(group_keys, sort=False):
            positive = group[group.training_positive]
            assert len(positive) == 1
            gold = positive.iloc[0]
            mass = float(gold.ref_mz)
            if not np.isfinite(mass) or mass <= 0:
                continue
            negative = group[
                group.training_label_known
                & ~group.training_positive
                & ~group.name.eq(gold["name"])
                & np.isclose(group.ref_mz.to_numpy(float), mass, rtol=0, atol=1e-8)
            ]
            if negative.empty:
                continue
            pi, ni = positive.index[0], negative.index.tolist()
            left.extend([pi] * len(ni) + ni)
            right.extend(ni + [pi] * len(ni))
            labels.extend([1] * len(ni) + [0] * len(ni))
            projects.extend([project] * (2 * len(ni)))
            initial_weights.extend([1 / (2 * len(ni))] * (2 * len(ni)))
            contribution[project] += 2 * len(ni)
            features[project] += 1
            for cls in negative["class"]:
                pair = tuple(
                    sorted(
                        (
                            canonical_lipid_class(gold["class"]),
                            canonical_lipid_class(cls),
                        )
                    )
                )
                class_projects[pair].add(project)
        if left:
            matrices.append(
                pair_predictors(
                    evidence.loc[left].reset_index(drop=True),
                    evidence.loc[right].reset_index(drop=True),
                )
            )
    if not matrices:
        raise ValueError("No permitted same-feature isomer-label pairs")
    totals = Counter()
    for project, weight in zip(projects, initial_weights):
        totals[project] += weight
    target = len(projects) / len(totals)
    weights = np.asarray(
        [w * target / totals[g] for g, w in zip(projects, initial_weights)]
    )
    return dict(
        evidence=pd.concat(matrices, ignore_index=True),
        positive=np.asarray(labels, int),
        groups=np.asarray(projects),
        sample_weight=weights,
        training_reference_permissions=sorted(training_groups),
        excluded_groups=sorted(excluded_groups),
        supervised_contributing_projects=sorted(contribution),
        pair_rows_by_project=dict(contribution),
        features_by_project=dict(features),
        class_pair_projects={p: sorted(g) for p, g in sorted(class_projects.items())},
        note="Conditional isomer-label pair construction ONLY;no classifier/annotation. Same originalfeature/adduct/theoreticalmass1e-8Da. No fabricated labels for entirely manual-absent features.",
    )
