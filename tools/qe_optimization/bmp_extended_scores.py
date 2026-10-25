"""047 extends ONLY admissible calibration-score domain to native .05..1.

No new classifier fits, score transforms, physical-window changes or fallback.
Native selector body must AST-match old code after restoring .5 validation.
"""

from itertools import product

import numpy as np
import pandas as pd

from bmp_context_measurements import PREFIX
from bmp_context_completion import (
    MODES,
    FORMAT as FORMAT,
    predict_context as predict_context,
)
from bmp_project_headroom import (
    COUNTS,
    combine_project_folders as combine_project_folders,
)

NP_MODES = MODES
LEVELS = tuple([round(0.05 * i, 2) for i in range(1, 20)] + [0.975, 0.99, 1.0])


def select_native_additions(
    pool, scores, previous, *, stream, threshold, protected_features, declared_masses
):
    """GLOBAL all-label winners first,then BMP,then original-call occupancy.

    No weaker alternative promotion,no old call replacement,no mutation. Raw
    strict3ppm/10s and independently protected spike IDs/masses rechecked.
    Training flags may remain attached for calibration counts but are NOT read.
    """
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.automatic import matlab_annotation_adducts
    from matlab_annotation.coherent_retention import _protected_mass
    from matlab_annotation.operating_point import select_operating_candidates
    from matlab_annotation.standard_rt_library import ion_key

    if (
        stream not in NP_MODES
        or not np.isfinite(threshold)
        or not 0.05 <= threshold <= 1
    ):
        raise ValueError("Explicit NP mode and declared operating grid required")
    required = {
        "stream",
        "name",
        "class",
        "adduct",
        "is_is",
        "feat_feature_id",
        "feat_intensity",
        "ref_mz",
        "feat_mz",
        "feat_rt",
    }
    if (
        not required.issubset(pool)
        or not pool.index.is_unique
        or not scores.index.equals(pool.index)
    ):
        raise ValueError("Aligned full physical all-class NP competition required")
    if not pool.stream.eq(stream).all() or (
        len(previous) and not previous.stream.eq(stream).all()
    ):
        raise ValueError("Competition and occupancy must use the same mode")
    if threshold == 1.0 or pool.empty:
        return pool.iloc[:0].copy()
    if not np.isfinite(scores.to_numpy(float)).all():
        raise ValueError("Unsupported context scores require disabled completion")
    winners = (
        pool.assign(_context_score=scores)
        .sort_values(
            ["_context_score", "feat_intensity"], ascending=False, kind="stable"
        )
        .drop_duplicates(["name", "adduct"])
    )
    winner_keys = set(
        winners[["name", "adduct", "feat_feature_id"]].itertuples(
            index=False, name=None
        )
    )
    selected = select_operating_candidates(
        pool, scores, threshold=threshold, ambiguity_margin=0.05
    )
    selected = selected.loc[
        selected["class"].map(canonical_lipid_class).eq("BMP")
        & ~selected.is_is.astype(bool)
    ].copy()
    selected = selected.loc[
        [
            k in winner_keys
            for k in selected[["name", "adduct", "feat_feature_id"]].itertuples(
                index=False, name=None
            )
        ]
    ].copy()
    if selected.empty:
        return selected
    previous_ids = {
        (str(n), ion_key(a))
        for n, a in previous[["name", "adduct"]].itertuples(index=False, name=None)
    }
    occupied = set(previous.feat_feature_id.astype(str)) | set(
        map(str, protected_features)
    )
    keep = ~selected.feat_feature_id.astype(str).isin(occupied)
    keep &= pd.Series(
        [
            (str(n), ion_key(a)) not in previous_ids
            for n, a in selected[["name", "adduct"]].itertuples(index=False, name=None)
        ],
        index=selected.index,
    )
    keep &= ~selected.name.astype(str).str.endswith(" (IS)") & ~selected.name.astype(
        str
    ).str.startswith("IS: ")
    allowed_ions = {ion_key(i) for i in matlab_annotation_adducts("NP")}
    keep &= selected.adduct.map(ion_key).isin(allowed_ions)
    keep &= selected.adduct.map(ion_key).str.endswith(
        "+" if stream.endswith("Pos") else "-"
    )
    keep &= (
        selected.get("isotopologue", pd.Series("", index=selected.index))
        .fillna("")
        .astype(str)
        .eq("")
    )
    masses = np.asarray(list(declared_masses), float)
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("Finite positive declared spike masses required")
    keep &= ~_protected_mass(selected.feat_mz.to_numpy(float), masses)
    keep &= ~_protected_mass(selected.ref_mz.to_numpy(float), masses)
    keep &= np.isfinite(
        selected[["ref_mz", "feat_mz", "feat_rt", "feat_intensity"]].to_numpy(float)
    ).all(axis=1)
    keep &= (
        selected.ref_mz.gt(0)
        & selected.feat_mz.gt(0)
        & selected.feat_rt.gt(0)
        & selected.feat_intensity.gt(0)
    )
    keep &= ((selected.feat_mz / selected.ref_mz - 1) * 1e6).abs().lt(3)
    expected = []
    for c, scale in (("ref_rt", 60), ("rt_expected", 60), ("atlas_rt_seconds", 1)):
        if c in selected:
            expected.append(
                (
                    selected.feat_rt * 60
                    - pd.to_numeric(selected[c], errors="coerce") * scale
                ).abs()
            )
    if not expected:
        raise ValueError("Permitted fitted/atlas numerical RT reference required")
    keep &= pd.concat(expected, axis=1).min(axis=1).le(10 + 1e-9)
    result = selected.loc[keep].copy()
    result["auto_annotation_status"] = "global_context_BMP_completion"
    return result


def select_additions(
    pool,
    scores,
    evidence,
    previous,
    *,
    stream,
    threshold,
    protected_features,
    declared_masses,
):
    if not pool.index.equals(evidence.index):
        raise ValueError("Aligned original pool/measurements required")
    if PREFIX + "source_physical" not in evidence:
        raise ValueError("Masked native physical eligibility required")
    selected = select_native_additions(
        pool,
        scores,
        previous,
        stream=stream,
        threshold=threshold,
        protected_features=protected_features,
        declared_masses=declared_masses,
    )
    # Strict native/global/identity-best decisions precede the independent
    # physical mask. No counterpart existence/correlation/metadata hard veto.
    return selected.loc[
        evidence[PREFIX + "source_physical"].eq(1).reindex(selected.index)
    ].copy()


def proposals(d, mask, raw, scores, previous, thresholds, supported):
    import run_qe_bmp_context_experiment as source

    pieces = []
    for s in source.NP_MODES:
        part = raw["pool"].loc[raw["pool"].stream.eq(s)]
        if not supported[s]:
            pieces.append(part.iloc[:0].copy())
            continue
        protection = source.original.inherited.common.physical_protection(
            d, mask, s, raw
        )
        pieces.append(
            select_additions(
                part,
                scores.loc[part.index],
                raw["evidence"].loc[part.index],
                previous.loc[previous.stream.eq(s)],
                stream=s,
                threshold=thresholds[s],
                **protection,
            )
        )
    return pd.concat(pieces, axis=0)


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


def choose_joint_grid(table):
    """Complete NP Cartesian grid under PROSPECTIVE budgets, explicit disable.

    Table rows summarize each mode/threshold over authorized INNER queries.
    No truth-reading or manual-presence filtering is performed by this helper.
    """
    required = {
        "stream",
        "threshold",
        "manual",
        "endogenous_manual",
        "previous_correct",
        "added_correct",
        "extras",
        "extras_without_opposite_composition",
        "different_peak",
        "positive_projects",
    }
    if not required.issubset(table) or set(table.stream) != set(MODES):
        raise ValueError("Complete two-mode training grid required")
    numeric = table.loc[:, sorted(required - {"stream"})].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric).all().all() or numeric.lt(0).any().any():
        raise ValueError("Finite nonnegative complete training counts required")
    if (
        table.endogenous_manual.gt(table.manual).any()
        or (table.previous_correct + table.added_correct).gt(table.manual).any()
        or table.extras_without_opposite_composition.gt(table.extras).any()
    ):
        raise ValueError("Inconsistent manual/extra counts")
    rows = {}
    for s in MODES:
        part = table.loc[table.stream.eq(s)]
        if sorted(part.threshold.tolist()) != sorted(LEVELS):
            raise ValueError("Every declared threshold including disable required")
        if (
            not part[["manual", "endogenous_manual", "previous_correct"]]
            .nunique()
            .eq(1)
            .all()
        ):
            raise ValueError("Fixed manual denominator and prior checkpoint required")
        rows[s] = list(part.itertuples(index=False))
    grid = []
    for pos, neg in product(rows[MODES[0]], rows[MODES[1]]):
        pair = [pos, neg]
        correct, extras, outside, wrong = (
            sum(getattr(r, c) for r in pair)
            for c in (
                "added_correct",
                "extras",
                "extras_without_opposite_composition",
                "different_peak",
            )
        )
        manual, endo = (
            sum(r.manual for r in pair),
            sum(r.endogenous_manual for r in pair),
        )
        supported = all(
            (r.threshold == 1 and r.added_correct == r.extras == r.different_peak == 0)
            or (r.threshold < 1 and r.added_correct >= 20 and r.positive_projects >= 3)
            for r in pair
        )
        eligible = bool(
            supported
            and correct > 0
            and extras <= correct
            and extras <= 0.5 * manual
            and outside <= 0.1 * endo
            and wrong <= 0.005 * endo
            and all(r.different_peak <= 0.005 * r.endogenous_manual for r in pair)
        )
        coverage = sum(
            min((r.previous_correct + r.added_correct) / r.manual, 0.9)
            if r.manual
            else 0
            for r in pair
        )
        grid.append(
            dict(
                NP_Pos=pos.threshold,
                NP_Neg=neg.threshold,
                added_correct=correct,
                extras=extras,
                extras_without_opposite_composition=outside,
                different_peak=wrong,
                capped90_sum=coverage,
                eligible=eligible,
            )
        )
    full = pd.DataFrame(grid)
    eligible = full.loc[full.eligible]
    if eligible.empty:
        return {s: 1.0 for s in MODES}, full
    selected = eligible.sort_values(
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
    return {s: float(selected[s]) for s in MODES}, full


def threshold_subset(potential, threshold):
    if threshold not in LEVELS:
        raise ValueError("Declared extended score cutoff required")
    if potential.empty or threshold == 1:
        return potential.iloc[:0].copy()
    return potential.loc[potential.class_selection_score.ge(threshold)].copy()


def assert_native_body_parity():
    """Actual helper must differ from native ONLY in declared score-floor check."""
    import ast
    import inspect
    from matlab_annotation.np_context_bmp_completion import select_context_bmp_additions

    old = ast.parse(inspect.getsource(select_context_bmp_additions)).body[0]
    new = ast.parse(inspect.getsource(select_native_additions)).body[0]
    new.name = old.name
    new.body = [n for n in new.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    changed = 0
    for node in ast.walk(new):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Constant)
            and node.left.value == 0.05
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == "threshold"
        ):
            node.left.value = 0.5
            changed += 1
    assert changed == 1 and ast.dump(old) == ast.dump(new)
