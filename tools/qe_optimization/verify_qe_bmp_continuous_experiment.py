"""046 full reused-model/project-curve/PHYSICAL/allouter integrity, no newfits."""

from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import hashlib
import json

import pandas as pd

from frozen_resources import configure_frozen_resources
from verify_immutable_exports import check_export
import run_qe_bmp_continuous_experiment as runner
import verify_qe_bmp_context_experiment as unchanged


def check_parent(job):
    # Freshly redo ALL528 training-use/matrix/target/weight/support checks for
    # unchanged source parents; no use of early stdout or partial receipts.
    result = unchanged.check_parent(job)
    return dict(
        parents=result["parents"],
        training_query_uses=result["training_query_uses"],
        reused_experts=result["fresh_classifiers"],
    )


def independent_proposals(case, thresholds):
    from matlab_annotation.np_context_bmp_completion import select_context_bmp_additions
    from bmp_context_measurements import PREFIX

    parts = []
    for mode in runner.NP_MODES:
        pool = case["raw"]["pool"]
        part = pool.loc[pool.stream.eq(mode)]
        if not case["supported"][mode]:
            parts.append(part.iloc[:0].copy())
            continue
        protection = runner.original.inherited.common.physical_protection(
            case["dataset"], case["mask"], mode, case["raw"]
        )
        base = select_context_bmp_additions(
            part,
            case["scores"].loc[part.index],
            case["occupied"].loc[case["occupied"].stream.eq(mode)],
            stream=mode,
            threshold=thresholds[mode],
            **protection,
        )
        wanted = [
            idx
            for idx in base.index
            if case["raw"]["evidence"].at[idx, PREFIX + "source_physical"] == 1
        ]
        parts.append(base.loc[wanted].copy())
    return pd.concat(parts)


def independent_grid(projects, grid, allowed):
    """Scalar recomputation of all169 original044 joint points/project support."""
    assert set(projects.project) == set(allowed)
    eligible_rows = []
    for row in grid.itertuples():
        pair = projects.loc[
            (projects.stream.eq("NP_Pos") & projects.threshold.eq(row.NP_Pos))
            | (projects.stream.eq("NP_Neg") & projects.threshold.eq(row.NP_Neg))
        ]
        mode = {s: pair.loc[pair.stream.eq(s)] for s in runner.NP_MODES}
        correct, extras, outside, wrong, manual, endo = (
            int(pair[c].sum())
            for c in (
                "added_correct",
                "extras",
                "extras_without_opposite_composition",
                "different_peak",
                "manual",
                "endogenous_manual",
            )
        )
        supported = True
        capped = 0.0
        for s, part in mode.items():
            if getattr(row, s) == 1:
                supported &= (
                    int(part.added_correct.sum())
                    == int(part.extras.sum())
                    == int(part.different_peak.sum())
                    == 0
                )
            else:
                supported &= (
                    int(part.added_correct.sum()) >= 20
                    and int(part.added_correct.gt(0).sum()) >= 3
                )
            n = int(part.manual.sum())
            capped += (
                min(
                    (int(part.previous_correct.sum()) + int(part.added_correct.sum()))
                    / n,
                    0.9,
                )
                if n
                else 0
            )
        original = bool(
            supported
            and correct > 0
            and extras <= correct
            and extras <= 0.5 * manual
            and outside <= 0.1 * endo
            and wrong <= 0.005 * endo
            and all(
                int(v.different_peak.sum()) <= 0.005 * int(v.endogenous_manual.sum())
                for v in mode.values()
            )
        )
        eligible = original
        assert (
            row.added_correct == correct
            and row.extras == extras
            and row.extras_without_opposite_composition == outside
            and row.different_peak == wrong
            and row.capped90_sum == capped
        )
        assert bool(row.eligible) == bool(eligible)
        if eligible:
            eligible_rows.append(
                (capped, correct, -extras, -wrong, row.NP_Pos, row.NP_Neg)
            )
    assert len(grid) == 169
    if not eligible_rows:
        return dict.fromkeys(runner.NP_MODES, 1.0)
    chosen = max(eligible_rows)
    return {"NP_Pos": chosen[-2], "NP_Neg": chosen[-1]}


def check_calibration(outer):
    configure_frozen_resources(runner.OUT / "source_snapshot")
    cases, folders, projects, pooled, allowed, excluded = runner.cases_and_curves(outer)
    path = runner.OUT / "calibration" / f"fold_{outer}"
    for case in cases:
        pd.testing.assert_series_equal(
            case["scores"],
            pd.read_pickle(path / (case["dataset"] + "_blind_scores.pkl")),
            check_exact=True,
        )
        # Native/global selection is unchanged, ONLY profile veto removed.
        # Independently scalar-check new physical admission at EVERY threshold.
        lowest = independent_proposals(case, dict.fromkeys(runner.NP_MODES, 0.5))
        for t in runner.completion.LEVELS:
            thresholds = dict.fromkeys(runner.NP_MODES, t)
            selected = runner.proposals(
                case["dataset"],
                case["mask"],
                case["raw"],
                case["scores"],
                case["occupied"],
                thresholds,
                case["supported"],
            )
            expected = independent_proposals(case, thresholds)
            pd.testing.assert_frame_equal(selected, expected, check_exact=True)
            unchanged.check_cached_threshold(
                selected,
                runner.source.threshold_subset(lowest, t),
                case["raw"]["pool"].columns,
            )
            stored = folders.loc[
                folders.dataset.eq(case["dataset"]) & folders.threshold.eq(t)
            ]
            assert len(stored) == 2
            for s in runner.NP_MODES:
                part = selected.loc[selected.stream.eq(s)]
                counts = unchanged.independent_counts(part, case["manual"])
                row = stored.loc[stored.stream.eq(s)].iloc[0]
                assert row.project == case["raw"]["query_group"]
                assert all(row[k] == v for k, v in counts.items())
    cuts, grid = runner.completion.choose_joint_grid(pooled)
    for name, expected in (
        ("by_folder_mode_threshold", folders),
        ("by_project_mode_threshold", projects),
        ("training_operating_points", pooled),
        ("complete_joint_grid", grid),
    ):
        pd.testing.assert_frame_equal(
            expected,
            pd.read_csv(path / (name + ".csv"), float_precision="round_trip"),
            check_dtype=False,
            check_exact=True,
        )
    assert cuts == independent_grid(projects, grid, allowed)
    choice = json.loads((path / "operating_points.json").read_text())
    assert (
        cuts == choice["thresholds"]
        and choice["training_groups"] == sorted(allowed)
        and choice["outer_excluded"] == sorted(excluded)
    )
    assert choice["project_curves_sha256"] == runner.sha(
        path / "by_project_mode_threshold.csv"
    ) and choice["joint_grid_sha256"] == runner.sha(path / "complete_joint_grid.csv")
    print(
        "046 FULL INNER/PROJECT/169GRID/PHYSICAL CHECK", outer, len(cases), flush=True
    )
    return len(cases)


def check_outer(item):
    d, own = item
    configure_frozen_resources(runner.OUT / "source_snapshot")
    path = runner.OUT / "runs" / d
    raw = runner.load_predictors(d, {own})
    model = pd.read_pickle(runner.parent_path(own, None))
    scores = runner.completion.predict_context(raw, model)
    pd.testing.assert_series_equal(
        scores, pd.read_pickle(path / "scoring_audit.pkl"), check_exact=True
    )
    pd.testing.assert_series_equal(
        scores,
        pd.read_pickle(runner.source.OUT / "runs" / d / "scoring_audit.pkl"),
        check_exact=True,
    )
    previous = runner.previous_annotations(d)
    cuts = json.loads(
        (runner.OUT / "calibration" / f"fold_{own}/operating_points.json").read_text()
    )["thresholds"]
    supported = {s: model["experts"][s] is not None for s in runner.NP_MODES}
    selected = runner.proposals(d, {own}, raw, scores, previous, cuts, supported)
    expected = independent_proposals(
        dict(
            dataset=d,
            mask={own},
            raw=raw,
            scores=scores,
            occupied=previous,
            supported=supported,
        ),
        cuts,
    )
    pd.testing.assert_frame_equal(selected, expected, check_exact=True)
    annotations, candidates, filled = runner.assemble(d, previous, selected)
    pd.testing.assert_frame_equal(
        candidates, pd.read_pickle(path / "candidates.pkl"), check_exact=True
    )
    pd.testing.assert_frame_equal(
        filled, pd.read_pickle(path / "private_additions.pkl"), check_exact=True
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(
            StringIO(annotations.to_csv(index=False)), dtype=str, keep_default_na=False
        ),
        pd.read_csv(
            path / "automatic_annotations.csv", dtype=str, keep_default_na=False
        ),
        check_exact=True,
    )
    baseline = pd.read_pickle(runner.BASE / "runs" / d / "full_annotation.pkl")
    quant = check_export(
        path / "automatic_annotations.csv",
        previous,
        candidates,
        pd.read_pickle(runner.PREVIOUS / "runs" / d / "candidates.pkl"),
        baseline,
    )
    from standard_lipidomics.validation import compare_manual_annotations
    from evaluate_qe_baseline import adduct

    reference = pd.read_pickle(
        runner.QE / "manual_reference_v2" / d / "curated_feature_crosswalk.pkl"
    )
    csv = pd.read_csv(
        path / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
    )
    csv["adduct"] = csv.adduct.map(adduct)
    candidate_eval = candidates.copy()
    candidate_eval["adduct"] = candidate_eval.adduct.map(adduct)
    fresh = compare_manual_annotations(
        reference, csv, candidates=candidate_eval, dataset=d
    )
    for field in ("manual", "laser"):
        pd.testing.assert_frame_equal(
            getattr(fresh, field),
            pd.read_pickle(
                runner.OUT / "evaluation_v2" / d / (field + "_comparison.pkl")
            ),
            check_exact=True,
        )
    digest = hashlib.sha256(reference.to_csv(index=False).encode()).hexdigest()
    for endpoint in (runner.OUT, runner.PREVIOUS, runner.BASE):
        assert (
            json.loads(
                (
                    endpoint / "evaluation_v2" / d / "evaluation_identity.json"
                ).read_text()
            )["manual_sha256"]
            == digest
        )
    print("046 FULL OUTER/ORIGINAL QUANT/GOLD CHECK", d, len(filled), flush=True)
    return quant


def main():
    protocol, datasets = runner.initialize()
    _, folds = runner.inventory()
    assert all((runner.OUT / "runs" / d / "complete.json").exists() for d in datasets)
    source_protocol = json.loads((runner.source.OUT / "protocol.json").read_text())
    jobs = [
        (o, i, datasets, folds, source_protocol)
        for o in range(5)
        for i in [None, *sorted(set(range(5)) - {o})]
    ]
    with ProcessPoolExecutor(max_workers=3) as executor:
        parents = list(executor.map(check_parent, jobs))
    assert (
        sum(p["parents"] for p in parents) == 25
        and sum(p["training_query_uses"] for p in parents) == 528
        and sum(p["reused_experts"] for p in parents) == 50
    )
    with ProcessPoolExecutor(max_workers=3) as executor:
        counts = list(executor.map(check_calibration, range(5)))
    assert sum(counts) == 132
    with ProcessPoolExecutor(max_workers=3) as executor:
        quantitative = list(
            executor.map(check_outer, [(d, folds[d]) for d in datasets])
        )
    assert (
        len(quantitative) == 33
        and protocol["source_input_sha256"] == runner.source_inputs()
    )
    artifacts = {
        str(p): runner.sha(p)
        for p in runner.OUT.rglob("*")
        if p.is_file()
        and p.name != "integrity_receipt.json"
        and "__pycache__" not in p.parts
    }
    runner.dump(
        runner.OUT / "integrity_receipt.json",
        dict(
            all_passed=True,
            reused_parents=25,
            reused_experts=50,
            training_query_uses=528,
            inner_query_cases=132,
            datasets=33,
            dataset_mode_cases=132,
            new_classifier_fits=0,
            planned_verifier_sha256=protocol["planned_verifier_sha256"],
            source_input_sha256=protocol["source_input_sha256"],
            artifact_sha256=artifacts,
            quantitative_checks=quantitative,
            note="ALL25frozenparents/50experts explicitlyreused;full528target/matrix/weight/scopechecks,132INNERscore/per-projectcounts/all5complete169jointgrids/scalaradmissionallcuts,33outerpredictions/rawquant/oldcalls/freshmanualcomparisons/samegold. Notnewfits/deployment/untouchedexternalvalidation/FDR.",
        ),
    )
    print("046 FULL MASTER COMPLETE ALL PASS", flush=True)


if __name__ == "__main__":
    main()
