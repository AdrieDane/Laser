"""038: ALL training support/scopes, INNER grids, held-out moves and raw outputs.

Does not refit models or select algorithms from outer results. Comparison truth
is read only by verification, never by blind peak-refinement runtime.
"""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_qe_conditional_peak_experiment as runner
from frozen_resources import configure_frozen_resources
from preflight_qe_conditional_peaks import verify_pairs

QE, OUT, OLD, BASE = runner.QE, runner.OUT, runner.OLD, runner.BASE
sha = runner.sha


def inventory():
    return (
        json.loads((OUT / "run_manifest.json").read_text())["datasets"],
        json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text()),
    )


def independent_counts(manual, accepted, moves):
    """Use validation-normalized full keys, independently of runner truth_case."""
    from standard_lipidomics.validation import _normalise

    calls = accepted.copy()
    for r in moves.itertuples():
        calls.loc[r.annotation_index, "feat_feature_id"] = str(r.new_feature_id)
    calls["adduct"] = calls.adduct.map(runner.legacy.adduct)
    calls = _normalise(calls, manual=False)
    keys = ["stream", "species", "adduct", "internal_standard"]
    called = set(calls[keys].itertuples(index=False, name=None))
    exact = set(calls[keys + ["feature_id"]].itertuples(index=False, name=None))
    manual_ids = set(manual[keys].itertuples(index=False, name=None))
    correct, wrong = Counter(), Counter()
    for row in manual.itertuples():
        identity = (row.stream, row.species, row.adduct, bool(row.internal_standard))
        cell = (row.stream, row.lipid_class)
        if row.mapped and (*identity, str(row.feature_id)) in exact:
            correct[cell] += 1
        elif row.mapped and identity in called:
            wrong[cell] += 1
    extras = Counter(
        k[0]
        for k in calls[keys].itertuples(index=False, name=None)
        if k not in manual_ids
    )
    return correct, wrong, extras


def verify_model(path):
    from matlab_annotation.evidence_training import validate_feature_scope

    model, old = pd.read_pickle(path), pd.read_pickle(OLD / path.relative_to(OUT))
    training, excluded = set(model["training_cohorts"]), set(model["excluded_groups"])
    assert training == set(old["training_cohorts"]) and excluded == set(
        old["excluded_groups"]
    )
    assert model["training_reference_permissions"] == sorted(training)
    assert model["base_identity_model_sha256"] == sha(OLD / path.relative_to(OUT))
    assert model["original_columns"] == old["columns"]
    assert model["training_feature_masks"] == old["training_feature_masks"]
    assert (
        model["standard_training_feature_scopes"]
        == old["standard_training_feature_scopes"]
    )
    records = {}
    for dataset, mask in model["training_feature_masks"].items():
        raw = runner.load_predictors(dataset, set(mask))
        validate_feature_scope(raw, training, excluded, query_is_training=True)
        records[dataset], _, _ = runner.with_training_labels(
            raw, permitted_groups=training, excluded_groups=excluded
        )
        pd.testing.assert_frame_equal(
            raw["evidence"], records[dataset]["evidence"], check_exact=True
        )
        scope = model["standard_training_feature_scopes"][dataset]
        assert set(scope["training"]).issubset(training)
        assert not set(scope["training"]) & excluded
        assert dataset[:7] not in scope["training"] and dataset[:7] in scope["excluded"]
        reference = (
            QE
            / "nested_standard_development_v1"
            / ("exclude_" + "_".join(map(str, sorted(mask))) + ".json")
        )
        assert sha(reference) == scope["reference_sha256"]
    expected = verify_pairs(records, training, excluded)
    protocol = json.loads((OUT / "protocol.json").read_text())
    supported = 0
    for mode, expert in model["experts"].items():
        obs = expected[mode]
        assert expert["pair_rows"] == obs["pair_rows"]
        assert (
            expert["supervised_contributing_projects"]
            == obs["supervised_contributing_projects"]
        )
        enough = (
            obs["pair_rows"] >= protocol["minimum_pair_rows"]
            and len(obs["supervised_contributing_projects"])
            >= protocol["minimum_contributing_projects"]
        )
        assert expert["supported"] == enough
        if obs["pair_rows"]:
            for key in (
                "pair_rows_by_project",
                "identities_by_project",
                "conditional_class_projects",
            ):
                assert expert[key] == obs[key]
        if not enough:
            continue
        supported += 1
        assert set(expert["training_reference_permissions"]) == training
        assert set(expert["excluded_groups"]) == excluded
        assert expert["estimator"].n_iter_ == 180
        for key, value in protocol["classifier"].items():
            if key != "kind":
                assert expert["estimator"].get_params()[key] == value
        assert set(expert["categories"]["left__stream"]) == {mode}
        assert set(expert["categories"]["right__stream"]) == {mode}
    print(
        "038 ALL TRAINING PAIRS/SUPPORT/SCOPES PASS", path.relative_to(OUT), flush=True
    )
    return dict(
        model=path.relative_to(OUT).as_posix(),
        training_queries=len(records),
        fresh_experts=supported,
    )


def verify_calibration(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.conditional_peak_refinement import (
        peak_refinement_preferences,
        select_peak_refinements,
    )

    protocol = json.loads((OUT / "protocol.json").read_text())
    projects, folds = inventory()
    folder = OUT / "calibration" / f"fold_{outer}"
    allowed = {d[:7] for d in projects if folds[d] != outer}
    excluded = {d[:7] for d in projects if folds[d] == outer}
    cases, totals, projects_by_cell = {}, Counter(), defaultdict(set)
    for inner in range(5):
        if inner == outer:
            continue
        model = pd.read_pickle(folder / f"inner_block_{inner}_model.pkl")
        assert set(model["training_cohorts"]) == {
            d[:7] for d in projects if folds[d] not in {outer, inner}
        }
        for dataset in projects:
            if folds[dataset] != inner:
                continue
            raw = runner.load_predictors(dataset, {outer, inner})
            accepted = runner.base_core(dataset, outer, raw)
            pd.testing.assert_frame_equal(
                accepted,
                pd.read_pickle(folder / (dataset + "_base_core.pkl")),
                check_exact=True,
            )
            protected, masses, _ = runner.physical_context(dataset, {outer, inner})
            preferences = peak_refinement_preferences(
                raw,
                accepted,
                model,
                protected_features_by_stream=protected,
                protected_masses_by_stream=masses,
            )
            pd.testing.assert_frame_equal(
                preferences,
                pd.read_pickle(folder / (dataset + "_blind_preferences.pkl")),
                check_exact=True,
            )
            _, manual, _ = runner.with_training_labels(
                raw, permitted_groups=allowed, excluded_groups=excluded
            )
            cases[dataset] = (accepted, preferences, manual)
            for r in manual.itertuples():
                cell = (r.stream, r.lipid_class)
                totals[cell] += 1
                projects_by_cell[cell].add(dataset[:7])
    assert len(cases) == sum(folds[d] != outer for d in projects)
    common = {
        c for c, n in totals.items() if n >= 100 and len(projects_by_cell[c]) >= 3
    }
    empty = pd.DataFrame(columns=["annotation_index", "new_feature_id"])
    before_c, before_w, before_e = Counter(), Counter(), Counter()
    for accepted, _, manual in cases.values():
        c, w, e = independent_counts(manual, accepted, empty)
        assert (c, w, e) == runner.case_counts(
            runner.truth_case(manual, accepted), empty
        )
        before_c.update(c)
        before_w.update(w)
        before_e.update(e)
    class_rows, mode_rows = [], []
    for threshold in protocol["confidence_grid"]:
        correct, wrong, extras, moved = Counter(), Counter(), Counter(), Counter()
        for accepted, preferences, manual in cases.values():
            moves, _ = select_peak_refinements(
                preferences,
                {s: threshold for s in runner.MODES},
                ambiguity_margin=protocol["ambiguity_margin"],
            )
            c, w, e = independent_counts(manual, accepted, moves)
            assert (c, w, e) == runner.case_counts(
                runner.truth_case(manual, accepted), moves
            )
            correct.update(c)
            wrong.update(w)
            extras.update(e)
            moved.update(moves.stream.value_counts().to_dict())
        assert extras == before_e
        class_rows.extend(
            dict(
                stream=s,
                lipid_class=cls,
                threshold=threshold,
                manual=n,
                training_project_groups=len(projects_by_cell[(s, cls)]),
                eligible_priority=(s, cls) in common,
                correct_before=before_c[(s, cls)],
                correct_after=correct[(s, cls)],
                wrong_before=before_w[(s, cls)],
                wrong_after=wrong[(s, cls)],
            )
            for (s, cls), n in totals.items()
        )
        for mode in runner.MODES:
            cells = [c for c in sorted(common) if c[0] == mode]
            cb = sum(n for c, n in before_c.items() if c[0] == mode)
            wb = sum(n for c, n in before_w.items() if c[0] == mode)
            ca = sum(n for c, n in correct.items() if c[0] == mode)
            wa = sum(n for c, n in wrong.items() if c[0] == mode)
            objective = (
                sum(min(0.9, correct[c] / totals[c]) for c in cells) / len(cells)
                if cells
                else 0.0
            )
            feasible = (
                all(correct[c] >= before_c[c] for c in cells) and ca >= cb and wa <= wb
            )
            mode_rows.append(
                dict(
                    stream=mode,
                    threshold=threshold,
                    manual=sum(n for c, n in totals.items() if c[0] == mode),
                    correct_before=cb,
                    correct_after=ca,
                    wrong_before=wb,
                    wrong_after=wa,
                    extras_before=before_e[mode],
                    extras_after=extras[mode],
                    moves=moved[mode],
                    priority_class_modes=len(cells),
                    capped90_class_mean=objective,
                    feasible=feasible,
                )
            )
    classes, modes = pd.DataFrame(class_rows), pd.DataFrame(mode_rows)
    for table, name in (
        (classes, "class_move_calibration.csv"),
        (modes, "mode_move_calibration.csv"),
    ):
        pd.testing.assert_frame_equal(
            table,
            pd.read_csv(folder / name, float_precision="round_trip"),
            check_exact=True,
            check_dtype=False,
        )
    chosen = {}
    for mode in runner.MODES:
        valid = modes[modes.stream.eq(mode) & modes.feasible]
        assert valid.threshold.eq(1.0).any()
        chosen[mode] = float(
            valid.sort_values(
                ["capped90_class_mean", "correct_after", "threshold"],
                ascending=False,
                kind="stable",
            )
            .iloc[0]
            .threshold
        )
    recorded = json.loads((folder / "operating_points.json").read_text())
    assert chosen == recorded["thresholds"]
    assert recorded["training_groups"] == sorted(allowed) and recorded[
        "outer_excluded"
    ] == sorted(excluded)
    assert recorded["class_table_sha256"] == sha(folder / "class_move_calibration.csv")
    assert recorded["mode_table_sha256"] == sha(folder / "mode_move_calibration.csv")
    print("038 ALL INNER PREFERENCES/GRIDS/CUTS PASS", outer, len(cases), flush=True)
    return len(cases)


def verify_dataset(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.conditional_peak_refinement import (
        peak_refinement_preferences,
        select_peak_refinements,
        apply_peak_refinements,
    )

    _, folds = inventory()
    outer = folds[dataset]
    folder = OUT / "runs" / dataset
    model = pd.read_pickle(OUT / f"fold_{outer}_nested_model.pkl")
    thresholds = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    before = pd.read_csv(
        OLD / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    actual = pd.read_csv(
        folder / "automatic_annotations.csv",
        dtype={"feat_feature_id": str, "peak_refinement_old_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    raw = runner.load_predictors(dataset, {outer})
    protected, masses, features = runner.physical_context(dataset, {outer})
    preferences = peak_refinement_preferences(
        raw,
        before,
        model,
        protected_features_by_stream=protected,
        protected_masses_by_stream=masses,
    )
    pd.testing.assert_frame_equal(
        preferences, pd.read_pickle(folder / "blind_preferences.pkl"), check_exact=True
    )
    moves, audit = select_peak_refinements(preferences, thresholds)
    pd.testing.assert_frame_equal(
        audit,
        pd.read_pickle(folder / "private_peak_refinement_audit.pkl"),
        check_exact=True,
    )
    expected = apply_peak_refinements(
        before, raw["pool"], moves, features_by_stream=features
    )
    raw_csv_text = (
        pd.read_csv(
            folder / "automatic_annotations.csv", dtype=str, keep_default_na=False
        )
        if len(moves)
        else None
    )
    assert runner.call_keys(actual) == runner.call_keys(before)
    assert list(actual) == list(expected)
    rounding = 0
    for col in actual:
        if pd.api.types.is_numeric_dtype(
            expected[col]
        ) and pd.api.types.is_numeric_dtype(actual[col]):
            a, b = actual[col].to_numpy(float), expected[col].to_numpy(float)
            np.testing.assert_allclose(
                a, b, rtol=8 * np.finfo(float).eps, atol=0, equal_nan=True, err_msg=col
            )
        else:
            assert (
                actual[col].fillna("").astype(str).tolist()
                == expected[col].fillna("").astype(str).tolist()
            ), col
    for mode in runner.MODES:
        selected = actual[actual.stream.eq(mode)]
        assert not selected.feat_feature_id.duplicated().any()
        originals = features[mode].set_index(features[mode].feature_id.astype(str))
        raw_rows = originals.loc[selected.feat_feature_id]
        for out, source in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            a, b = selected[out].to_numpy(float), raw_rows[source].to_numpy(float)
            np.testing.assert_allclose(
                a, b, rtol=8 * np.finfo(float).eps, atol=0, equal_nan=True
            )
            rounding += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        # ALL quantification/boundary fields on EVERY moved row, not merely m/z/RT.
        for r in moves[moves.stream.eq(mode)].itertuples():
            original = originals.loc[str(r.new_feature_id)]
            for col, value in original.items():
                observed = actual.loc[r.annotation_index, "feat_" + str(col)]
                if col == "feature_id":
                    assert str(observed) == str(value)
                elif pd.isna(value):
                    assert pd.isna(observed)
                elif isinstance(value, (int, float, np.number)):
                    np.testing.assert_allclose(
                        float(observed),
                        float(value),
                        rtol=8 * np.finfo(float).eps,
                        atol=0,
                    )
                else:
                    assert raw_csv_text.loc[
                        r.annotation_index, "feat_" + str(col)
                    ] == str(value), col
    candidate = pd.read_pickle(folder / "candidates.pkl")
    pd.testing.assert_frame_equal(
        candidate,
        runner.updated_candidates(
            pd.read_pickle(OLD / "runs" / dataset / "candidates.pkl"), before, expected
        ),
        check_exact=True,
    )
    assert runner.feature_keys(
        candidate[candidate.automatic_accepted]
    ) == runner.feature_keys(actual)
    hashes = [
        json.loads(
            (
                QE / v / "evaluation_v2" / dataset / "evaluation_identity.json"
            ).read_text()
        )["manual_sha256"]
        for v in ("baseline", "experiment_029", "experiment_038")
    ]
    assert len(set(hashes)) == 1
    old_metrics = pd.read_pickle(
        OLD / "evaluation_v2" / dataset / "manual_comparison.pkl"
    )
    new_metrics = pd.read_pickle(
        OUT / "evaluation_v2" / dataset / "manual_comparison.pkl"
    )
    assert len(old_metrics) == len(new_metrics)
    recorded = json.loads((folder / "complete.json").read_text())
    assert recorded["moved_peaks"] == len(moves)
    print(
        "038 OUTER PREFERENCES/IDENTITIES/RAWQUANT/EXPORT PASS",
        dataset,
        len(moves),
        flush=True,
    )
    return dict(
        dataset=dataset, moved_peaks=len(moves), raw_stream_checks=4, roundoff=rounding
    )


def verification_binding():
    files = {
        OUT / "protocol.json",
        OUT / "runner_source.py",
        Path(__file__),
        Path(__file__).with_name("preflight_qe_conditional_peaks.py"),
    }
    files.update(OUT.glob("fold_*_nested_model.pkl"))
    for name in ("calibration", "runs", "evaluation_v2", "reports_v2"):
        files.update(p for p in (OUT / name).rglob("*") if p.is_file())
    hashes = {str(p): sha(p) for p in sorted(files)}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return identity, hashes


def completed_component(path, kind, name, binding):
    """No stdout/partial checks accepted; only complete hash-bound receipts."""
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not (
        record.get("all_passed") is True
        and record.get("kind") == kind
        and record.get("name") == name
        and record.get("verification_identity_sha256") == binding
    ):
        return None
    result = record.get("result")
    if kind == "model":
        return (
            result
            if isinstance(result, dict)
            and result.get("model") == name
            and result.get("training_queries", 0) > 0
            and result.get("fresh_experts", 0) > 0
            else None
        )
    if kind == "inner":
        projects, folds = inventory()
        return (
            result
            if isinstance(result, int)
            and result == sum(folds[d] != int(name) for d in projects)
            else None
        )
    if kind == "outer":
        return (
            result
            if isinstance(result, dict)
            and result.get("dataset") == name
            and result.get("raw_stream_checks") == 4
            else None
        )
    return None


def unchanged_components(previous_source, current_source):
    """Exact AST proof: only dataset CSV/check dispatch may change, not science."""
    import ast

    a, b = ast.parse(previous_source), ast.parse(current_source)
    dump_ast = lambda node: ast.dump(node, include_attributes=False)
    globals_a = [
        dump_ast(n)
        for n in a.body
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    globals_b = [
        dump_ast(n)
        for n in b.body
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert globals_a == globals_b, "Module imports/global bindings changed"
    old = {n.name: n for n in a.body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in b.body if isinstance(n, ast.FunctionDef)}
    assert set(new) - set(old) <= {"unchanged_components", "migration_proof"}
    assert not set(old) - set(new)
    proof = {}
    for name in set(old) - {"main", "verify_component", "verify_dataset"}:
        x, y = dump_ast(old[name]), dump_ast(new[name])
        assert x == y, (name, "changed: cannot reuse completed component")
        proof[name] = hashlib.sha256(x.encode()).hexdigest()
    return dict(
        unchanged_function_ast_sha256=proof,
        imports_globals_ast_sha256=hashlib.sha256(
            json.dumps(globals_a).encode()
        ).hexdigest(),
    )


def migration_proof(binding, artifacts, paths):
    previous = OUT / "verification_sources/verify_v2.py"
    proof = unchanged_components(previous.read_text(), Path(__file__).read_text())
    old_artifacts = dict(artifacts)
    old_artifacts[str(Path(__file__))] = sha(previous)
    old_binding = hashlib.sha256(
        json.dumps(old_artifacts, sort_keys=True).encode()
    ).hexdigest()
    names = [("model", p.relative_to(OUT).as_posix()) for p in paths] + [
        ("inner", str(f)) for f in range(5)
    ]
    for kind, name in names:
        path = (
            OUT
            / "verification_components"
            / (hashlib.sha256((kind + ":" + name).encode()).hexdigest() + ".json")
        )
        assert completed_component(path, kind, name, old_binding) is not None, (
            kind,
            name,
        )
    proof.update(
        previous_verifier_sha256=sha(previous),
        current_verifier_sha256=sha(Path(__file__)),
        previous_verification_identity_sha256=old_binding,
        current_verification_identity_sha256=binding,
        permitted_reuse_kinds=["model", "inner"],
        eligible_complete_components=30,
        note="ONLYcompletedhash-boundV2receipts,ALLartifact/inputhashesunchanged,EXACTsamecheckerfunctions/imports/globals/dependencies. NOstdout/partialreuse. ALL33outerchecks rerun withrawCSVtextpreservation;V2outerreceipts NOTreused.",
    )
    path = OUT / "verification_migration_v2_to_v3.json"
    runner.dump(path, proof)
    return dict(
        previous_binding=old_binding,
        previous_source_sha256=sha(previous),
        proof_sha256=sha(path),
    )


def verify_component(task):
    kind, name, binding, migration = task
    configure_frozen_resources(OUT / "source_snapshot")
    filename = hashlib.sha256((kind + ":" + name).encode()).hexdigest() + ".json"
    receipt = OUT / "verification_components" / binding / filename
    previous = completed_component(receipt, kind, name, binding)
    if previous is not None:
        print("HASH-BOUND COMPLETED CHECK REUSED", kind, name, flush=True)
        return kind, name, previous
    result, reused = None, False
    if kind in {"model", "inner"}:
        assert (
            sha(OUT / "verification_migration_v2_to_v3.json")
            == migration["proof_sha256"]
        )
        result = completed_component(
            OUT / "verification_components" / filename,
            kind,
            name,
            migration["previous_binding"],
        )
        reused = result is not None
    if reused:
        print(
            "EXACT-AST/UNCHANGED-ARTIFACT COMPLETED V2 CHECK REUSED",
            kind,
            name,
            flush=True,
        )
    elif kind == "model":
        result = verify_model(OUT / name)
    elif kind == "inner":
        result = verify_calibration(int(name))
    elif kind == "outer":
        result = verify_dataset(name)
    else:
        raise ValueError(kind)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    runner.dump(
        receipt,
        dict(
            all_passed=True,
            kind=kind,
            name=name,
            verification_identity_sha256=binding,
            result=result,
            reused_unchanged_v2_component=reused,
            source_ast_migration_proof_sha256=migration["proof_sha256"]
            if reused
            else None,
        ),
    )
    return kind, name, result


def main():
    protocol, projects = runner.initialize()
    assert (
        len(projects) == 33
        and json.loads((OUT / "reports_v2/summary.json").read_text())["complete"]
    )
    configure_frozen_resources(OUT / "source_snapshot")
    paths = sorted(
        list(OUT.glob("fold_*_nested_model.pkl"))
        + list(OUT.glob("calibration/fold_*/inner_block_*_model.pkl"))
    )
    assert len(paths) == 25
    binding, artifacts = verification_binding()
    migration = migration_proof(binding, artifacts, paths)
    tasks = [
        ("model", p.relative_to(OUT).as_posix(), binding, migration) for p in paths
    ]
    tasks += [("inner", str(f), binding, migration) for f in range(5)]
    tasks += [("outer", d, binding, migration) for d in projects]
    with ProcessPoolExecutor(max_workers=4) as executor:
        checked = list(executor.map(verify_component, tasks))
    models = [v for kind, _, v in checked if kind == "model"]
    inner = [v for kind, _, v in checked if kind == "inner"]
    outer = [v for kind, _, v in checked if kind == "outer"]
    assert len(models) == 25 and len(inner) == 5 and len(outer) == 33
    assert sum(m["training_queries"] for m in models) == 528
    assert sum(inner) == 132 and sum(d["raw_stream_checks"] for d in outer) == 132
    # Aggregated extras/annotation totals invariant, independently from runtime.
    for name, keys in (
        ("by_dataset_mode.csv", ["dataset", "stream"]),
        ("by_dataset_mode_class.csv", ["dataset", "stream", "lipid_class"]),
        ("by_mode_class.csv", ["stream", "lipid_class"]),
    ):
        old_path, new_path = OLD / "reports_v2" / name, OUT / "reports_v2" / name
        old, new = (
            pd.read_csv(old_path).set_index(keys),
            pd.read_csv(new_path).set_index(keys),
        )
        pd.testing.assert_frame_equal(
            old[
                ["manual_rows", "laser_not_in_manual_list", "unique_laser_annotations"]
            ].sort_index(),
            new[
                ["manual_rows", "laser_not_in_manual_list", "unique_laser_annotations"]
            ].sort_index(),
            check_exact=True,
        )
    runner.initialize()  # Recheck EVERY bound input/source after the full replay.
    assert verification_binding() == (binding, artifacts)
    assert (
        sha(OUT / "verification_sources/verify_v2.py")
        == migration["previous_source_sha256"]
    )
    assert (
        sha(OUT / "verification_migration_v2_to_v3.json") == migration["proof_sha256"]
    )
    result = dict(
        all_passed=True,
        annotation_experiment="038",
        integrity_tool_sha256=sha(Path(__file__)),
        verification_identity_sha256=binding,
        bound_verification_artifacts=len(artifacts),
        completed_component_receipts=63,
        exact_ast_unchanged_v2_training_calibration_components=30,
        freshly_replayed_outer_components=33,
        source_ast_migration_proof_sha256=migration["proof_sha256"],
        source_file_hash_checks=len(protocol["source_snapshot_sha256"]),
        input_file_hash_checks=len(protocol["input_sha256"]),
        new_parent_fits=25,
        new_mode_classifiers=sum(m["fresh_experts"] for m in models),
        all_training_query_scope_pair_weight_reconstructions=528,
        inner_preference_reproductions=132,
        full_training_grid_cutoff_reproductions=5,
        heldout_preference_reproductions=33,
        raw_coordinate_stream_checks=132,
        all_moved_raw_quantification_boundary_checks=sum(
            d["moved_peaks"] for d in outer
        ),
        accepted_candidate_export_parity_checks=33,
        identical_manual_reference_hash_checks=33,
        preexisting_roundoff_cells=sum(d["roundoff"] for d in outer),
        production_enabled=False,
        note="Saved-input WHOLE029identity-preserving refinement;no newidentities/standards/extras. IntegrityNOTexternalvalidation/FDR/deployment. IDsreadasTEXT;rawCSVtextpreservesblankmobilityunits. Only30completedV2training/INNERreceipts reusedwithEXACTcheckerfunction/import/globalASTandunchangedALLartifact/inputhashproof. ALL33outer/rawquant checks rerun. NOstdout/partialreuse. All63completedcomponentreceipts hashbound,currentFULLsource/input/artifactsrecheckedbefore/aftermaster.",
    )
    pd.DataFrame(models).to_csv(OUT / "conditional_training_audit.csv", index=False)
    pd.DataFrame(outer).to_csv(OUT / "peak_refinement_output_audit.csv", index=False)
    runner.dump(OUT / "integrity_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
