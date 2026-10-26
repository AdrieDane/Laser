"""Exploratory conditional-versus-global composition-presence comparison.

NOT an annotation experiment or same-peak coverage measurement. This analysis
was added after pilot distributions, not prospectively selected acceptance
guards. No parameter/cutoff selection. A composition absent from manual curation
is unconfirmed, not a chemically false identification. Query classes with no
curated entries are unavailable for this conditional comparison, NEVER removed
from the primary annotation benchmark or filtered at runtime.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import matrix_peer_pilot_data as data
import matrix_peer_composition as peer

OUT = data.QE / "matrix_peer_presence_diagnostic_v1"


def losses(labels, conditional, global_prior):
    y, c, g = (np.asarray(v, float) for v in (labels, conditional, global_prior))
    assert np.isfinite(np.column_stack([y, c, g])).all()
    assert np.isin(y, [0, 1]).all() and ((c > 0) & (c < 1) & (g > 0) & (g < 1)).all()
    a, b = (
        -(y * np.log(c) + (1 - y) * np.log1p(-c)),
        -(y * np.log(g) + (1 - y) * np.log1p(-g)),
    )
    # Independent standard-library arithmetic; no clipping or modified losses.
    for truth, cp, gp, ca, ga in zip(y, c, g, a, b):
        expected_c = -math.log(cp if truth else 1 - cp)
        expected_g = -math.log(gp if truth else 1 - gp)
        assert math.isclose(ca, expected_c, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(ga, expected_g, rel_tol=0, abs_tol=1e-12)
    return a, b


def main():
    configure_frozen_resources(data.OUT / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class

    paths = [
        data.OUT / "blind_receipt.json",
        data.CHECK / "receipt.json",
        data.DIAG / "receipt.json",
    ]
    blind, checked, diagnosed = [json.loads(p.read_text()) for p in paths]
    assert blind["all_passed"] and checked["all_passed"] and diagnosed["all_passed"]
    assert checked["datasets"] == 33 and checked["blind_receipt_sha256"] == data.sha(
        paths[0]
    )
    assert checked["source_input_sha256"] == data.inputs()
    source_hashes = {
        **data.inputs(),
        str(Path(__file__)): data.sha(Path(__file__)),
        **{str(p): data.sha(p) for p in paths},
    }
    OUT.mkdir(exist_ok=False)
    datasets, folds = data.inventory()
    rows = []
    for dataset in datasets:
        model = json.loads(
            (data.OUT / "models" / f"fold_{folds[dataset]}" / "model.json").read_text()
        )
        assert (
            dataset[:7] not in model["training_groups"]
            and dataset[:7] in model["excluded_groups"]
        )
        pool = data.load_pool(dataset, folds[dataset])["pool"]
        measured = pd.read_pickle(data.OUT / "private" / (dataset + ".pkl"))
        q = pool.loc[~pool.is_is, ["stream", "class", "carbon", "db"]].copy()
        q["lipid_class"] = q["class"].map(canonical_lipid_class)
        q["conditional_prior"] = measured[peer.PREFIX + "peer_exact_frequency"]
        keys = ["stream", "lipid_class", "carbon", "db"]
        assert q.groupby(keys).conditional_prior.nunique(dropna=False).le(1).all()
        q = q.drop_duplicates(keys)
        # Evaluation ONLY, after all33blind+independent gates and first diagnostic.
        manual = pd.read_pickle(
            data.REFERENCE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        manual = manual.loc[
            ~manual.internal_standard & ~manual.lipid_class.eq("X4")
        ].copy()
        manual["lipid_class"] = manual.lipid_class.map(canonical_lipid_class)
        reviewed = set(
            manual[["stream", "lipid_class"]].itertuples(index=False, name=None)
        )
        positive = set()
        for r in manual.itertuples():
            composition = data.parse_composition(r.species)
            if composition is not None:
                positive.add((r.stream, r.lipid_class, *composition))
        for (stream, cls), sub in q.groupby(["stream", "lipid_class"], sort=True):
            if (stream, cls) not in reviewed:
                continue
            sub = sub.loc[sub.conditional_prior.notna()].copy()
            if sub.empty:
                continue
            projects = model["classes"][stream + "|" + cls]["projects"]
            y, global_prior = [], []
            for r in sub.itertuples():
                y.append(float((stream, cls, int(r.carbon), int(r.db)) in positive))
                frequency = sum(
                    any(c == r.carbon and d == r.db for c, d in points)
                    for points in projects.values()
                )
                global_prior.append((frequency + 0.5) / (len(projects) + 1))
            c_loss, g_loss = losses(y, sub.conditional_prior, global_prior)
            rows.append(
                dict(
                    dataset=dataset,
                    cohort=dataset[:7],
                    stream=stream,
                    lipid_class=cls,
                    evaluated_compositions=len(sub),
                    compositions_in_manual=int(sum(y)),
                    global_loss=float(g_loss.mean()),
                    conditional_loss=float(c_loss.mean()),
                    conditional_minus_global=float((c_loss - g_loss).mean()),
                )
            )
        print("COMPOSITION PRESENCE DIAGNOSTIC ONLY", dataset, flush=True)
    classes = pd.DataFrame(rows)
    classes.to_csv(OUT / "by_dataset_mode_class.csv", index=False)
    project = classes.groupby(["cohort", "stream"], as_index=False)[
        ["global_loss", "conditional_loss", "conditional_minus_global"]
    ].mean()
    project.to_csv(OUT / "by_project_mode.csv", index=False)
    modes = project.groupby("stream", as_index=False).agg(
        projects=("cohort", "size"),
        global_loss=("global_loss", "mean"),
        conditional_loss=("conditional_loss", "mean"),
        conditional_minus_global=("conditional_minus_global", "mean"),
        projects_better=("conditional_minus_global", lambda s: int(s.lt(-1e-12).sum())),
        projects_worse=("conditional_minus_global", lambda s: int(s.gt(1e-12).sum())),
    )
    modes.to_csv(OUT / "by_mode_project_macro.csv", index=False)
    assert checked["source_input_sha256"] == data.inputs()
    data.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            explored_class_mode_cases=len(classes),
            scalar_loss_arithmetic_checked=True,
            source_input_sha256=source_hashes,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            note="EXPLORATORY after pilot distributions; not prospective acceptance guards. Composition-presence logloss,not correctsamepeakcoverage or chemicalFDR. Classes with noquerycuration unavailable for this CONDITIONAL diagnostic only,not runtime/exclusions. Macro average withinproject thenequalprojects; candidate availability conditioning. Whole041unchanged.",
        ),
    )
    print(modes.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
