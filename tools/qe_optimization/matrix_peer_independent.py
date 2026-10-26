"""Separate scalar oracle for sample-context composition pilot, no selections."""

import math

import pandas as pd

from matrix_peer_composition import FORMAT, FIELDS, PREFIX, FORBIDDEN, root


def signature_scalar(table):
    from annotator.composition_ranges import canonical_lipid_class

    assert not FORBIDDEN & set(table)
    assert table.is_is.isin([True, False]).all()
    observed = {}
    for row in table.to_dict("records"):
        if row["is_is"]:
            continue
        try:
            c, d, mz, theory, intensity, score = (
                float(row[k])
                for k in (
                    "carbon",
                    "db",
                    "feat_mz",
                    "ref_mz",
                    "feat_intensity",
                    "post_regression_confidence",
                )
            )
        except (ValueError, TypeError):
            continue
        if not all(math.isfinite(v) for v in (c, d, mz, theory, intensity, score)):
            continue
        if (
            c <= 0
            or d < 0
            or c != int(c)
            or d != int(d)
            or min(mz, theory, intensity) <= 0
        ):
            continue
        if not 0.9 <= score <= 1 or abs((mz / theory - 1) * 1e6) >= 1.5:
            continue
        key = (str(row["stream"]), root(row["class"]))
        observed.setdefault(key, set()).add(
            (canonical_lipid_class(row["class"]), int(c), int(d))
        )
    output = {}
    for (stream, family), values in sorted(observed.items()):
        if len(values) < 5:
            continue
        bins = {}
        for _, c, d in sorted(values):
            pair = (c // 2, d)
            bins[pair] = bins.get(pair, 0) + 1
        output[stream + "|" + family] = [
            [c, d, n / len(values)] for (c, d), n in sorted(bins.items())
        ]
    return dict(groups=output)


def fit_scalar(rows, signatures, *, training_groups, excluded_groups):
    from annotator.composition_ranges import canonical_lipid_class

    allowed, excluded = set(training_groups), set(excluded_groups)
    assert allowed and excluded and not allowed & excluded
    assert not FORBIDDEN & set(rows)
    assert set(rows.cohort) == allowed and {d[:7] for d in signatures} == allowed
    project_signatures = {}
    for project in sorted(allowed):
        accumulated = {}
        for dataset in sorted(signatures):
            if dataset[:7] != project:
                continue
            for family, bins in signatures[dataset]["groups"].items():
                destination = accumulated.setdefault(family, {})
                for c, d, value in bins:
                    destination[(c, d)] = destination.get((c, d), 0) + value
        merged = {}
        for family, values in sorted(accumulated.items()):
            denominator = sum(values[key] for key in sorted(values))
            merged[family] = [
                [c, d, value / denominator] for (c, d), value in sorted(values.items())
            ]
        project_signatures[project] = dict(groups=merged)
    collected = {}
    for r in rows.to_dict("records"):
        assert str(r["dataset"])[:7] == r["cohort"]
        try:
            c, d = float(r["carbon"]), float(r["db"])
        except (ValueError, TypeError):
            continue
        if (
            not math.isfinite(c)
            or not math.isfinite(d)
            or c <= 0
            or d < 0
            or c != int(c)
            or d != int(d)
        ):
            continue
        cls = canonical_lipid_class(r["lipid_class"])
        key = (str(r["stream"]), cls)
        collected.setdefault(key, []).append((r["cohort"], int(c), int(d)))
    classes = {}
    for (stream, cls), observations in sorted(collected.items()):
        projects = {}
        for g, c, d in observations:
            projects.setdefault(g, set()).add((c, d))
        if len(observations) >= 100 and len(projects) >= 3:
            classes[stream + "|" + cls] = dict(
                manual_rows=len(observations),
                projects={g: sorted(points) for g, points in sorted(projects.items())},
            )
    return dict(
        format=FORMAT,
        training_groups=sorted(allowed),
        excluded_groups=sorted(excluded),
        project_signatures=project_signatures,
        classes=classes,
    )


def affinity_scalar(a, b, target_root):
    values, families = [], set()
    for group in sorted(a["groups"]):
        family = group.split("|", 1)[1]
        if family == target_root or group not in b["groups"]:
            continue
        families.add(family)
        right = {(c, d): p for c, d, p in b["groups"][group]}
        value = 0.0
        for c, d, p in a["groups"][group]:
            value += math.sqrt(p * right.get((c, d), 0.0))
        values.append(min(1.0, max(0.0, value)))
    if len(families) < 3:
        return None
    return sum(values) / len(values)


def predict_scalar(pool, query, model, *, query_group):
    from annotator.composition_ranges import canonical_lipid_class

    assert model["format"] == FORMAT
    assert (
        query_group not in model["training_groups"]
        and query_group in model["excluded_groups"]
    )
    assert not set(model["training_groups"]) & set(model["excluded_groups"])
    assert set(model["project_signatures"]) == set(model["training_groups"])
    assert not FORBIDDEN & set(pool) and pool.is_is.isin([True, False]).all()
    result = pd.DataFrame(
        float("nan"), index=pool.index, columns=[PREFIX + f for f in FIELDS]
    )
    for index, row in pool.iterrows():
        if row.is_is:
            continue
        cls = canonical_lipid_class(row["class"])
        reference = model["classes"].get(str(row.stream) + "|" + cls)
        c, d = float(row.carbon), float(row.db)
        if reference is None or not math.isfinite(c) or not math.isfinite(d):
            continue
        projects = reference["projects"]
        assert set(projects) <= set(model["training_groups"])
        peers = []
        for group in sorted(projects):
            score = affinity_scalar(
                query, model["project_signatures"][group], root(cls)
            )
            if score is not None:
                peers.append((score, group))
        peers.sort(key=lambda x: (-x[0], x[1]))
        if len(peers) < 3:
            continue
        cutoff = peers[min(4, len(peers) - 1)][0]
        peers = [(s, g) for s, g in peers if s >= cutoff]
        exact, nearby, cout, dout = {}, {}, {}, {}
        for group, points in projects.items():
            exact[group] = float(any(x == c and y == d for x, y in points))
            nearby[group] = max(
                math.exp(-0.5 * (((x - c) / 2) ** 2 + (y - d) ** 2)) for x, y in points
            )
            cout[group] = float(
                c < min(x for x, _ in points) or c > max(x for x, _ in points)
            )
            dout[group] = float(
                d < min(y for _, y in points) or d > max(y for _, y in points)
            )
        global_exact = (sum(exact.values()) + 0.5) / (len(projects) + 1)
        global_near = sum(nearby.values()) / len(projects)
        weight = [(math.exp(5 * (s - 1)), s, g) for s, g in peers]
        total = sum(w for w, _, _ in weight)
        ep = (sum(w * exact[g] for w, _, g in weight) + 2 * global_exact) / (total + 2)
        np = (sum(w * nearby[g] for w, _, g in weight) + 2 * global_near) / (total + 2)
        result.loc[index] = (
            ep,
            np,
            math.log(ep / (1 - ep)) - math.log(global_exact / (1 - global_exact)),
            sum(w * s for w, s, _ in weight) / total,
            total**2 / sum(w * w for w, _, _ in weight),
            len(peers),
            sum(w * cout[g] for w, _, g in weight) / total,
            sum(w * dout[g] for w, _, g in weight) / total,
        )
    return result
