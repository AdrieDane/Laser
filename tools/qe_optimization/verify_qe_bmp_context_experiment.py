"""044 full25parent/528training/132INNER/fivejointgrids/33rawexports audit."""

from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
from verify_immutable_exports import check_export
import run_qe_bmp_context_experiment as runner
import preflight_qe_bmp_context_models as preflight


def check_parent(job):
    outer, inner, datasets, folds, protocol = job
    configure_frozen_resources(runner.OUT / "source_snapshot")
    data, allowed, excluded = runner.training_data(outer, inner, datasets, folds)
    independent = {
        (d, tuple(sorted(raw["excluded_folds"]))): preflight.independent_case(
            d, set(raw["excluded_folds"]), datasets, folds
        )
        for d, raw in data.items()
    }
    support = preflight.parent_support(data, allowed, excluded, independent, folds)
    gate = json.loads((runner.PREFLIGHT / "parent_support.json").read_text())
    expected = next(r for r in gate if r["outer"] == outer and r["inner"] == inner)
    assert (
        support == expected["modes"]
        and sorted(allowed) == expected["training_groups"]
        and sorted(excluded) == expected["excluded_groups"]
    )
    path = runner.parent_path(outer, inner)
    model = pd.read_pickle(path)
    complete = json.loads(path.with_suffix(".complete.json").read_text())
    assert complete["model_sha256"] == runner.sha(path)
    assert model["format"] == runner.completion.FORMAT and len(model["columns"]) == 163
    assert model["training_cohorts"] == sorted(allowed) and model[
        "excluded_groups"
    ] == sorted(excluded)
    assert model["training_feature_masks"] == {
        d: raw["excluded_folds"] for d, raw in data.items()
    }
    parameters = runner.original.inherited.common.estimator_parameters(
        protocol["classifier"]
    )
    fresh = 0
    for s in runner.NP_MODES:
        part = support[s]
        assert model["support"][s] == part["support"]
        assert model["training_matrix_receipts"][s] == {
            k: v for k, v in part.items() if k.endswith("_sha256")
        }
        expert = model["experts"][s]
        if not part["supported"]:
            assert expert is None
            continue
        fresh += 1
        assert expert is not None and all(
            expert[k] == v for k, v in part["metadata"].items()
        )
        estimator = expert["estimator"]
        assert all(estimator.get_params()[k] == v for k, v in parameters.items())
        assert list(estimator.get_params()["categorical_features"]) == [
            c in expert["categories"] for c in expert["columns"]
        ]
        assert (
            estimator.n_features_in_ == 163
            and estimator.n_iter_ == parameters["max_iter"]
        )
    assert fresh == complete["fresh_classifiers"] == model["fresh_mode_classifier_fits"]
    print(
        "044 FULL PARENT/INDEPENDENT TRAINING CHECK",
        outer,
        inner,
        len(data),
        fresh,
        flush=True,
    )
    return dict(parents=1, training_query_uses=len(data), fresh_classifiers=fresh)


def independent_proposals(case, thresholds):
    from matlab_annotation.np_context_bmp_completion import select_context_bmp_additions

    parts = []
    for s in runner.NP_MODES:
        part = case["raw"]["pool"].loc[case["raw"]["pool"].stream.eq(s)]
        if not case["supported"][s]:
            parts.append(part.iloc[:0])
            continue
        protection = runner.original.inherited.common.physical_protection(
            case["dataset"], case["mask"], s, case["raw"]
        )
        base = select_context_bmp_additions(
            part,
            case["scores"].loc[part.index],
            case["occupied"].loc[case["occupied"].stream.eq(s)],
            stream=s,
            threshold=thresholds[s],
            **protection,
        )
        wanted = []
        for idx in base.index:
            e = case["raw"]["evidence"].loc[idx]
            values = [
                float(e[runner.completion.PREFIX + c])
                for c in (
                    "source_physical",
                    "candidate_counterparts",
                    "counterpart_profiles",
                    "counterpart_n",
                    "counterpart_raw",
                    "counterpart_normalized",
                    "counterpart_leave_one_out",
                )
            ]
            wanted.append(
                bool(
                    np.isfinite(values).all()
                    and values[0] == 1
                    and values[1] >= 1
                    and values[2] >= 1
                    and values[3] >= 8
                    and values[4] >= 0.8
                    and values[5] >= 0.8
                    and values[6] >= 0.6
                )
            )
        parts.append(base.loc[np.array(wanted, dtype=bool)])
    return pd.concat(parts)


def independent_counts(chosen, manual):
    common = runner.original.inherited.common
    endogenous = manual.loc[~manual.internal_standard]
    identities = {
        (r.stream, common.species(r.species), common.adduct(r.adduct))
        for r in endogenous.itertuples()
    }
    unresolved = {
        (r.stream, common.species(r.species), common.adduct(r.adduct))
        for r in endogenous.loc[~endogenous.mapped].itertuples()
    }
    exact = {
        (
            r.stream,
            common.species(r.species),
            common.adduct(r.adduct),
            str(r.feature_id),
        )
        for r in endogenous.loc[endogenous.mapped].itertuples()
    }
    opposite = {
        (r.stream, common.species(r.species))
        for r in endogenous.loc[endogenous.lipid_class.eq("BMP")].itertuples()
    }
    counts = dict(
        added_correct=0,
        extras=0,
        extras_without_opposite_composition=0,
        different_peak=0,
        unresolved_calls=0,
    )
    for r in chosen.itertuples():
        key = (
            r.stream,
            common.species(r.name),
            common.adduct(r.adduct),
            str(r.feat_feature_id),
        )
        counts["added_correct"] += int(key in exact)
        counts["extras"] += int(key[:3] not in identities)
        other = "NP_Neg" if r.stream == "NP_Pos" else "NP_Pos"
        counts["extras_without_opposite_composition"] += int(
            key[:3] not in identities and (other, key[1]) not in opposite
        )
        counts["different_peak"] += int(
            key[:3] in identities and key[:3] not in unresolved and key not in exact
        )
        counts["unresolved_calls"] += int(key[:3] in unresolved and key not in exact)
    return counts


def check_cached_threshold(actual, cached, original_columns):
    if actual.empty or cached.empty:
        assert actual.empty and cached.empty
        # Explicit disable returns the original empty pool, while an empty
        # slice of ranked winners also carries unused score/status columns.
        # Both contain ZERO decisions and must retain all original columns.
        assert set(original_columns).issubset(actual) and set(
            original_columns
        ).issubset(cached)
    else:
        pd.testing.assert_frame_equal(actual, cached, check_exact=True)


def check_calibration(outer):
    configure_frozen_resources(runner.OUT / "source_snapshot")
    protocol = json.loads((runner.OUT / "protocol.json").read_text())
    datasets, folds = runner.inventory()
    cases = runner.calibration_cases(outer, protocol, datasets, folds, fit=False)
    folder = runner.OUT / "calibration" / f"fold_{outer}"
    assert len(cases) == sum(folds[d] != outer for d in datasets)
    for case in cases:
        pd.testing.assert_series_equal(
            case["scores"],
            pd.read_pickle(folder / (case["dataset"] + "_blind_scores.pkl")),
            check_exact=True,
        )
        lowest = runner.proposals(
            case["dataset"],
            case["mask"],
            case["raw"],
            case["scores"],
            case["occupied"],
            dict.fromkeys(runner.NP_MODES, min(runner.completion.LEVELS)),
            case["supported"],
        )
        for t in runner.completion.LEVELS:
            thresholds = dict.fromkeys(runner.NP_MODES, t)
            actual = runner.proposals(
                case["dataset"],
                case["mask"],
                case["raw"],
                case["scores"],
                case["occupied"],
                thresholds,
                case["supported"],
            )
            expected = independent_proposals(case, thresholds)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            check_cached_threshold(
                actual, runner.threshold_subset(lowest, t), case["raw"]["pool"].columns
            )
            for s in runner.NP_MODES:
                selected = actual.loc[actual.stream.eq(s)]
                assert runner.addition_counts(
                    selected, case["manual"]
                ) == independent_counts(selected, case["manual"])
    thresholds, table, joint = runner.calibration_tables(cases)
    pd.testing.assert_frame_equal(
        table,
        pd.read_csv(
            folder / "training_operating_points.csv", float_precision="round_trip"
        ),
        check_dtype=False,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        joint,
        pd.read_csv(folder / "complete_joint_grid.csv", float_precision="round_trip"),
        check_dtype=False,
        check_exact=True,
    )
    choice = json.loads((folder / "operating_points.json").read_text())
    assert thresholds == choice["thresholds"]
    assert choice["training_table_sha256"] == runner.sha(
        folder / "training_operating_points.csv"
    ) and choice["complete_grid_sha256"] == runner.sha(
        folder / "complete_joint_grid.csv"
    )
    assert choice["training_groups"] == sorted(
        {d[:7] for d in datasets if folds[d] != outer}
    ) and choice["outer_excluded"] == sorted(
        {d[:7] for d in datasets if folds[d] == outer}
    )
    print(
        "044 FULL INNER SCORES/COUNTS/CARTESIAN GRID CHECK",
        outer,
        len(cases),
        flush=True,
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
    previous = runner.previous_annotations(d)
    cuts = json.loads(
        (runner.OUT / "calibration" / f"fold_{own}/operating_points.json").read_text()
    )["thresholds"]
    supported = {s: model["experts"][s] is not None for s in runner.NP_MODES}
    selected = runner.proposals(d, {own}, raw, scores, previous, cuts, supported)
    pd.testing.assert_frame_equal(
        selected,
        independent_proposals(
            dict(
                dataset=d,
                mask={own},
                raw=raw,
                scores=scores,
                occupied=previous,
                supported=supported,
            ),
            cuts,
        ),
        check_exact=True,
    )
    annotations, candidates, filled = runner.assemble(d, previous, selected)
    pd.testing.assert_frame_equal(
        candidates, pd.read_pickle(path / "candidates.pkl"), check_exact=True
    )
    pd.testing.assert_frame_equal(
        filled, pd.read_pickle(path / "private_additions.pkl"), check_exact=True
    )
    expected_text = pd.read_csv(
        StringIO(annotations.to_csv(index=False)), dtype=str, keep_default_na=False
    )
    actual_text = pd.read_csv(
        path / "automatic_annotations.csv", dtype=str, keep_default_na=False
    )
    pd.testing.assert_frame_equal(actual_text, expected_text, check_exact=True)
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
    # Same reference hash is separately bound in source_inputs. Manual is used
    # ONLY for this post-output evaluation, not scoring/physical selection.
    evaluated = pd.read_pickle(
        runner.OUT / "evaluation_v2" / d / "manual_comparison.pkl"
    )
    previous_eval = pd.read_pickle(
        runner.PREVIOUS / "evaluation_v2" / d / "manual_comparison.pkl"
    )
    assert len(evaluated) == len(previous_eval)
    assert set(runner.original.inherited.common.accepted_keys(previous)).issubset(
        set(runner.original.inherited.common.accepted_keys(annotations))
    )
    selected_csv = pd.read_csv(
        path / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        low_memory=False,
    )
    selected_csv["adduct"] = selected_csv.adduct.map(adduct)
    candidate_eval = candidates.copy()
    candidate_eval["adduct"] = candidate_eval.adduct.map(adduct)
    fresh = compare_manual_annotations(
        reference, selected_csv, candidates=candidate_eval, dataset=d
    )
    pd.testing.assert_frame_equal(fresh.manual, evaluated, check_exact=True)
    pd.testing.assert_frame_equal(
        fresh.laser,
        pd.read_pickle(runner.OUT / "evaluation_v2" / d / "laser_comparison.pkl"),
        check_exact=True,
    )
    digest = hashlib.sha256(reference.to_csv(index=False).encode()).hexdigest()
    for endpoint in (runner.OUT, runner.PREVIOUS, runner.BASE):
        identity = json.loads(
            (endpoint / "evaluation_v2" / d / "evaluation_identity.json").read_text()
        )
        assert identity["manual_sha256"] == digest
    print("044 FULL OUTER/ORIGINAL RAW EXPORT CHECK", d, len(filled), flush=True)
    return quant


def main():
    protocol, datasets = runner.initialize()
    folds = runner.inventory()[1]
    assert all((runner.OUT / "runs" / d / "complete.json").exists() for d in datasets)
    construct = json.loads((runner.CONSTRUCTION / "receipt.json").read_text())
    assert construct["all_passed"] and construct["unique_nested_tables"] == 363
    assert runner.sha(Path(__file__)) == protocol["planned_verifier_sha256"]
    jobs = [
        (o, i, datasets, folds, protocol)
        for o in range(5)
        for i in [None, *sorted(set(range(5)) - {o})]
    ]
    with ProcessPoolExecutor(max_workers=3) as executor:
        parents = list(executor.map(check_parent, jobs))
    assert (
        sum(p["parents"] for p in parents) == 25
        and sum(p["training_query_uses"] for p in parents) == 528
    )
    with ProcessPoolExecutor(max_workers=3) as executor:
        cases = list(executor.map(check_calibration, range(5)))
    assert sum(cases) == 132
    with ProcessPoolExecutor(max_workers=3) as executor:
        quant = list(executor.map(check_outer, [(d, folds[d]) for d in datasets]))
    assert (
        len(quant) == 33 and runner.source_inputs() == protocol["source_input_sha256"]
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
            parents=25,
            training_query_uses=528,
            inner_query_cases=132,
            datasets=33,
            dataset_mode_cases=132,
            fresh_classifiers=sum(p["fresh_classifiers"] for p in parents),
            planned_verifier_sha256=protocol["planned_verifier_sha256"],
            source_input_sha256=protocol["source_input_sha256"],
            artifact_sha256=artifacts,
            quantitative_checks=quant,
            note="Complete363independent construction,25parent matrices/labels/scopes/weights/support/parameters,132INNERscores/independent proposal/count checks/fivefulljointgrids,33outer predictions/fulloldcells/neworiginalrawquant. Previously exposed development;notdeployment/externalvalidation/FDR.",
        ),
    )
    print("044 FULL MASTER INTEGRITY COMPLETE ALL PASS", flush=True)


if __name__ == "__main__":
    main()
