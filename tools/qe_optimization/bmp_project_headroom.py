"""Pure operatingpoint policy:25%INNERheadroom and leave-one-PROJECT safety.

No file/model/curation access, new annotations, physical-window changes or
final acceptance-limit relaxation. Project labels are training grouping only.
"""

import numpy as np
import pandas as pd

from bmp_context_completion import MODES, LEVELS, choose_joint_grid

COUNTS = (
    "manual",
    "endogenous_manual",
    "previous_correct",
    "added_correct",
    "extras",
    "extras_without_opposite_composition",
    "different_peak",
    "unresolved_calls",
)


def pooled_curves(project_table, *, training_groups, excluded_groups):
    allowed, excluded = set(training_groups), set(excluded_groups)
    required = {"project", "stream", "threshold", *COUNTS}
    if (
        not allowed
        or not excluded
        or allowed & excluded
        or not required.issubset(project_table)
    ):
        raise ValueError(
            "Explicit complete disjoint authorized project curves required"
        )
    if set(project_table.project) != allowed or set(project_table.project) & excluded:
        raise ValueError("Held-out/missing project in training curves")
    if set(project_table.stream) != set(MODES):
        raise ValueError("Exactly the two NP modes are required")
    if project_table.duplicated(["project", "stream", "threshold"]).any():
        raise ValueError("Related folders must be combined per PROJECT first")
    values = project_table.loc[:, list(COUNTS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values).all().all() or values.lt(0).any().any():
        raise ValueError("Finite nonnegative project counts required")
    if not np.equal(values.to_numpy(), np.floor(values.to_numpy())).all():
        raise ValueError("Project counts must be integers")
    for project in sorted(allowed):
        for s in MODES:
            part = project_table.loc[
                project_table.project.eq(project) & project_table.stream.eq(s)
            ]
            if sorted(part.threshold.tolist()) != sorted(LEVELS):
                raise ValueError("Each PROJECT/mode needs every declared threshold")
            if (
                not part[["manual", "endogenous_manual", "previous_correct"]]
                .nunique()
                .eq(1)
                .all()
            ):
                raise ValueError(
                    "Project denominators and previous calls must be fixed"
                )
    rows = []
    for t in LEVELS:
        for s in MODES:
            part = project_table.loc[
                project_table.threshold.eq(t) & project_table.stream.eq(s)
            ]
            rows.append(
                dict(
                    stream=s,
                    threshold=t,
                    **{c: int(part[c].sum()) for c in COUNTS},
                    positive_projects=int(part.added_correct.gt(0).sum()),
                )
            )
    return pd.DataFrame(rows)


def combine_project_folders(dataset_rows):
    """Distinct folders remain benchmark rows but NOT independent projects."""
    required = {"project", "stream", "threshold", *COUNTS}
    if not required.issubset(dataset_rows):
        raise ValueError("Explicit folder-derived project counts required")
    return dataset_rows.groupby(
        ["project", "stream", "threshold"], as_index=False, sort=True
    )[list(COUNTS)].sum()


def choose_project_headroom(project_table, *, training_groups, excluded_groups):
    """Complete169jointpoints; no cut is selected by its held-out outcome."""
    pooled = pooled_curves(
        project_table, training_groups=training_groups, excluded_groups=excluded_groups
    )
    _, grid = choose_joint_grid(pooled)
    grid.rename(columns={"eligible": "original_safety_eligible"}, inplace=True)
    lookup = {
        (s, t): part.set_index("project").reindex(sorted(training_groups))
        for (s, t), part in project_table.groupby(["stream", "threshold"])
    }
    min_surplus, robust, witnesses = [], [], []
    for row in grid.itertuples():
        parts = [lookup[("NP_Pos", row.NP_Pos)], lookup[("NP_Neg", row.NP_Neg)]]
        projects = pd.DataFrame(index=sorted(training_groups))
        projects["added_correct"] = sum(p.added_correct for p in parts)
        projects["extras"] = sum(p.extras for p in parts)
        surpluses = []
        for project, part in projects.iterrows():
            rest_correct = int(row.added_correct - part.added_correct)
            rest_extras = int(row.extras - part.extras)
            surplus = rest_correct - rest_extras
            surpluses.append(surplus)
            witnesses.append(
                dict(
                    NP_Pos=row.NP_Pos,
                    NP_Neg=row.NP_Neg,
                    removed_project=project,
                    remaining_new_correct=rest_correct,
                    remaining_extras=rest_extras,
                    remaining_correct_surplus=surplus,
                    passed=bool(surplus >= 0),
                )
            )
        min_surplus.append(min(surpluses))
        robust.append(all(v >= 0 for v in surpluses))
    grid["training_headroom_pass"] = grid.extras.le(0.75 * grid.added_correct)
    grid["minimum_leave_one_project_correct_surplus"] = min_surplus
    grid["leave_one_project_pass"] = robust
    grid["projects_checked"] = len(set(training_groups))
    grid["eligible"] = (
        grid.original_safety_eligible
        & grid.training_headroom_pass
        & grid.leave_one_project_pass
    )
    eligible = grid.loc[grid.eligible]
    if eligible.empty:
        thresholds = dict.fromkeys(MODES, 1.0)
    else:
        chosen = eligible.sort_values(
            [
                "capped90_sum",
                "added_correct",
                "extras",
                "different_peak",
                "NP_Pos",
                "NP_Neg",
            ],
            ascending=[False, False, True, True, False, False],
            kind="stable",
        ).iloc[0]
        thresholds = {s: float(chosen[s]) for s in MODES}
    return thresholds, pooled, grid, pd.DataFrame(witnesses)
