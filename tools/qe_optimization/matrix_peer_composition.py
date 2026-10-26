"""Read-only OTHER-family sample-context composition descriptors, not limits.

Only authorized other-project curation enters fitting. Query context is blind
native automatic output; same-target families are withheld from similarities.
These descriptors are identical for every original peak of one composition.
"""

import math
import re

import numpy as np
import pandas as pd

FORMAT = "laser-other-family-peer-composition-v1"
FIELDS = (
    "peer_exact_frequency",
    "peer_near_frequency",
    "peer_vs_global_logodds",
    "peer_mean_similarity",
    "peer_effective_projects",
    "peer_projects",
    "peer_carbon_outside_fraction",
    "peer_db_outside_fraction",
)
PREFIX = "q_matrix_"
FORBIDDEN = {
    "same_manual_peak",
    "manual_feature_id",
    "training_positive",
    "training_label_known",
    "calibration_identity_in_manual",
}


def root(value):
    from annotator.composition_ranges import canonical_lipid_class

    cls = canonical_lipid_class(value).split("[")[0]
    cls = re.sub(r"^[12]-acyl\s*", "", cls)
    return {
        "LPC": "PC",
        "LPE": "PE",
        "LPI": "PI",
        "LPS": "PS",
        "LPG": "PG",
        "BMP": "PG",
        "LPA": "PA",
    }.get(cls, cls)


def signature(automatic):
    """Normalized fixed C/DB bins, no learned scaling or manual observations."""
    from annotator.composition_ranges import canonical_lipid_class

    if FORBIDDEN & set(automatic):
        raise ValueError("No query supervision in automatic context")
    required = {
        "stream",
        "class",
        "carbon",
        "db",
        "is_is",
        "ref_mz",
        "feat_mz",
        "feat_intensity",
        "post_regression_confidence",
    }
    if not required <= set(automatic) or not automatic.is_is.isin([True, False]).all():
        raise ValueError(
            "Explicit native automatic coordinates and standard flags required"
        )
    x = automatic.copy()
    numeric = x[
        [
            "carbon",
            "db",
            "ref_mz",
            "feat_mz",
            "feat_intensity",
            "post_regression_confidence",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric).all(axis=1) & ~x.is_is.astype(bool)
    valid &= numeric.carbon.gt(0) & numeric.db.ge(0)
    valid &= (
        numeric[["carbon", "db"]].eq(np.floor(numeric[["carbon", "db"]])).all(axis=1)
    )
    valid &= numeric.ref_mz.gt(0) & numeric.feat_mz.gt(0) & numeric.feat_intensity.gt(0)
    valid &= numeric.post_regression_confidence.between(0.9, 1)
    valid &= ((numeric.feat_mz / numeric.ref_mz - 1) * 1e6).abs().lt(1.5)
    x = x.loc[valid].assign(**numeric.loc[valid].to_dict("series"))
    x["canonical"] = x["class"].map(canonical_lipid_class)
    x = x.drop_duplicates(["stream", "canonical", "carbon", "db"])
    x["root"] = x["class"].map(root)
    groups = {}
    for (stream, family), part in x.groupby(["stream", "root"], sort=True):
        if len(part) < 5:
            continue
        bins = (
            part.assign(cbin=np.floor(part.carbon / 2).astype(int))
            .groupby(["cbin", "db"], sort=True)
            .size()
        )
        groups[f"{stream}|{family}"] = [
            [int(c), int(d), float(n / bins.sum())] for (c, d), n in bins.items()
        ]
    return dict(groups=groups)


def combine_signatures(signatures):
    """A related-tissue project supplies ONE histogram, not multiple votes."""
    groups = {}
    for sig in signatures:
        for group, bins in sig["groups"].items():
            part = groups.setdefault(group, {})
            for c, d, value in bins:
                part[(c, d)] = part.get((c, d), 0.0) + value
    result = {}
    for group, bins in sorted(groups.items()):
        total = sum(bins[k] for k in sorted(bins))
        result[group] = [[c, d, bins[(c, d)] / total] for c, d in sorted(bins)]
    return dict(groups=result)


def similarity(query, reference, *, excluded_root):
    shared = sorted(set(query["groups"]) & set(reference["groups"]))
    shared = [g for g in shared if g.split("|", 1)[1] != excluded_root]
    if len({g.split("|", 1)[1] for g in shared}) < 3:
        return None
    values = []
    for group in shared:
        a = {(c, d): p for c, d, p in query["groups"][group]}
        b = {(c, d): p for c, d, p in reference["groups"][group]}
        affinity = sum(math.sqrt(a[k] * b[k]) for k in sorted(set(a) & set(b)))
        values.append(min(1.0, max(0.0, affinity)))
    return sum(values) / len(values)


def fit_reference(curated, signatures, *, training_groups, excluded_groups):
    """Curated composition PRESENCE only, not feature labels or own-query truth."""
    from annotator.composition_ranges import canonical_lipid_class

    allowed, excluded = set(training_groups), set(excluded_groups)
    if not allowed or not excluded or allowed & excluded:
        raise ValueError("Explicit disjoint reference training/excluded scope required")
    if FORBIDDEN & set(curated):
        raise ValueError("Only authorized composition presence enters this model")
    if not {"dataset", "cohort", "stream", "lipid_class", "carbon", "db"} <= set(
        curated
    ):
        raise ValueError("Authorized grouped class/composition rows required")
    if set(curated.cohort) != allowed or {d[:7] for d in signatures} != allowed:
        raise ValueError("Excluded or unknown project in peer reference")
    if any(str(r.dataset)[:7] != r.cohort for r in curated.itertuples()):
        raise ValueError("Project provenance and dataset disagree")
    x = curated.copy()
    x["lipid_class"] = x.lipid_class.map(canonical_lipid_class)
    x[["carbon", "db"]] = x[["carbon", "db"]].apply(pd.to_numeric, errors="coerce")
    x = x.loc[
        np.isfinite(x[["carbon", "db"]]).all(axis=1) & x.carbon.gt(0) & x.db.ge(0)
    ]
    x = x.loc[x[["carbon", "db"]].eq(np.floor(x[["carbon", "db"]])).all(axis=1)]
    project_signatures = {
        g: combine_signatures([signatures[d] for d in sorted(signatures) if d[:7] == g])
        for g in sorted(allowed)
    }
    classes = {}
    for (stream, cls), part in x.groupby(["stream", "lipid_class"], sort=True):
        if len(part) < 100 or part.cohort.nunique() < 3:
            continue
        projects = {}
        for g, sub in part.groupby("cohort", sort=True):
            projects[g] = sorted(
                {
                    (int(c), int(d))
                    for c, d in sub[["carbon", "db"]].itertuples(index=False, name=None)
                }
            )
        classes[f"{stream}|{cls}"] = dict(manual_rows=len(part), projects=projects)
    return dict(
        format=FORMAT,
        training_groups=sorted(allowed),
        excluded_groups=sorted(excluded),
        project_signatures=project_signatures,
        classes=classes,
    )


def predict(pool, query_signature, model, *, query_group):
    """Eight numeric descriptors only; no calls, cutoffs or feature selection."""
    from annotator.composition_ranges import canonical_lipid_class

    if (
        model.get("format") != FORMAT
        or query_group in model["training_groups"]
        or query_group not in model["excluded_groups"]
    ):
        raise ValueError("Query must be explicitly excluded from peer reference")
    if set(model["training_groups"]) & set(model["excluded_groups"]):
        raise ValueError("Overlapping reference groups")
    if set(model["project_signatures"]) != set(model["training_groups"]):
        raise ValueError("Reference signatures do not match permitted projects")
    if FORBIDDEN & set(pool):
        raise ValueError("No query outcomes in peer descriptors")
    if (
        not {"stream", "class", "carbon", "db", "is_is"} <= set(pool)
        or not pool.is_is.isin([True, False]).all()
    ):
        raise ValueError("Explicit chemical queries and standard flags required")
    output = pd.DataFrame(
        np.nan, index=pool.index, columns=[PREFIX + f for f in FIELDS]
    )
    targets = pool.loc[
        ~pool.is_is.astype(bool), ["stream", "class", "carbon", "db"]
    ].copy()
    targets["class"] = targets["class"].map(canonical_lipid_class)
    target_groups = targets.groupby(
        ["stream", "class", "carbon", "db"], sort=True
    ).groups
    cached = {}
    for (stream, cls, c, d), indices in target_groups.items():
        reference = model["classes"].get(f"{stream}|{cls}")
        if reference is None or not np.isfinite([c, d]).all():
            continue
        projects = reference["projects"]
        if set(projects) - set(model["training_groups"]):
            raise ValueError("Unpermitted class reference project")
        family = root(cls)
        if family not in cached:
            cached[family] = {
                g: similarity(query_signature, sig, excluded_root=family)
                for g, sig in model["project_signatures"].items()
            }
        peers = sorted(
            ((cached[family][g], g) for g in projects if cached[family][g] is not None),
            key=lambda pair: (-pair[0], pair[1]),
        )
        if len(peers) < 3:
            continue
        boundary = peers[min(5, len(peers)) - 1][0]
        peers = [(s, g) for s, g in peers if s >= boundary]
        exact, near, carbon_out, db_out = {}, {}, {}, {}
        for g, compositions in projects.items():
            points = np.asarray(compositions, float)
            distance = ((points[:, 0] - c) / 2) ** 2 + (points[:, 1] - d) ** 2
            exact[g] = float(np.any((points[:, 0] == c) & (points[:, 1] == d)))
            near[g] = float(np.exp(-0.5 * distance).max())
            carbon_out[g] = float(not points[:, 0].min() <= c <= points[:, 0].max())
            db_out[g] = float(not points[:, 1].min() <= d <= points[:, 1].max())
        n = len(projects)
        global_exact = (sum(exact.values()) + 0.5) / (n + 1)
        global_near = sum(near.values()) / n
        weights = np.asarray([math.exp(5 * (s - 1)) for s, _ in peers])
        total = float(weights.sum())
        exact_p = (
            sum(w * exact[g] for w, (_, g) in zip(weights, peers)) + 2 * global_exact
        ) / (total + 2)
        near_p = (
            sum(w * near[g] for w, (_, g) in zip(weights, peers)) + 2 * global_near
        ) / (total + 2)
        logodds = math.log(exact_p / (1 - exact_p)) - math.log(
            global_exact / (1 - global_exact)
        )
        values = (
            exact_p,
            near_p,
            logodds,
            sum(w * s for w, (s, _) in zip(weights, peers)) / total,
            total * total / float(np.square(weights).sum()),
            len(peers),
            sum(w * carbon_out[g] for w, (_, g) in zip(weights, peers)) / total,
            sum(w * db_out[g] for w, (_, g) in zip(weights, peers)) / total,
        )
        output.loc[list(indices)] = values
    return output
