"""Full041 replay: reused models, all INNER curves/outer rows, raw exports."""

from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import run_qe_add_only_class_experiment as runner
from verify_qe_np_context_bmp_experiment import verify_candidate_flags

OUT, OLD, QE = runner.OUT, runner.OLD, runner.QE


def verify_models():
    datasets, folds = runner.inventory()
    experts = 0
    for outer in range(5):
        for inner in [None, *sorted(set(folds.values()) - {outer})]:
            mask = {outer} if inner is None else {outer, inner}
            path = OLD / (
                f"fold_{outer}_nested_model.pkl"
                if inner is None
                else f"calibration/fold_{outer}/inner_block_{inner}_model.pkl"
            )
            model = pd.read_pickle(path)
            allowed = {d[:7] for d in datasets if folds[d] not in mask}
            excluded = {d[:7] for d in datasets if folds[d] in mask}
            assert set(model["training_cohorts"]) == allowed
            assert set(model["excluded_groups"]) == excluded
            assert set(model["experts"]) == set(runner.MODES)
            for expert in model["experts"].values():
                assert set(expert["training_cohorts"]) == allowed
                assert set(expert["excluded_groups"]) == excluded
                experts += 1
    assert experts == 100
    return experts


def verify_calibration(outer):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.add_only_class_completion import (
        class_completion_curves,
        choose_completion_thresholds,
    )

    p, m, counts, allowed, excluded, audits = runner.calibration_data(outer)
    table = class_completion_curves(
        p, m, counts, training_groups=allowed, excluded_groups=excluded
    )
    thresholds, chosen = choose_completion_thresholds(table)
    folder = OUT / "calibration" / f"fold_{outer}"
    for name, expected in (
        ("training_class_operating_points.csv", table),
        ("selected_training_points.csv", chosen),
        ("scope_reproduction_audit.csv", pd.DataFrame(audits)),
    ):
        actual = pd.read_csv(folder / name, float_precision="round_trip")
        normalized = pd.read_csv(
            StringIO(expected.to_csv(index=False)), float_precision="round_trip"
        )
        pd.testing.assert_frame_equal(actual, normalized, check_exact=True)
    selected = chosen.loc[chosen.threshold.lt(1)]
    assert selected.feasible.all() and selected.different_peak.eq(0).all()
    assert (selected.extras <= 0.2 * selected.added_correct).all()
    assert (selected.extras <= 0.02 * selected.manual).all()
    assert selected.minimum_leave_one_project_agreement.ge(0.8).all()
    assert (
        selected.positive_projects.ge(3).all() and selected.added_correct.ge(10).all()
    )
    saved = json.loads((folder / "operating_points.json").read_text())
    assert saved == dict(
        thresholds=thresholds,
        training_groups=sorted(allowed),
        outer_excluded=sorted(excluded),
    )
    print(
        "041 ALLINNER SCORES/CLASScurves/cutoffs PASS", outer, len(audits), flush=True
    )
    return len(audits)


def verify_dataset(dataset):
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.add_only_class_completion import select_completion
    from standard_lipidomics.validation import _boolean

    _, folds = runner.inventory()
    outer = folds[dataset]
    folder = OUT / "runs" / dataset
    model = pd.read_pickle(OLD / f"fold_{outer}_nested_model.pkl")
    raw, scores = runner.reproduce(
        dataset, {outer}, model, OLD / "runs" / dataset / "scoring_audit.pkl"
    )
    pd.testing.assert_series_equal(
        scores, pd.read_pickle(folder / "scoring_audit.pkl"), check_exact=True
    )
    cuts = json.loads(
        (OUT / "calibration" / f"fold_{outer}/operating_points.json").read_text()
    )["thresholds"]
    previous = runner.common.read_previous(dataset)
    free = runner.proposals_for(dataset, {outer}, raw, scores, previous)
    added = select_completion(free, cuts)
    expected, candidates, filled = runner.assemble(dataset, previous, added)
    pd.testing.assert_frame_equal(
        filled, pd.read_pickle(folder / "private_additions.pkl"), check_exact=True
    )
    pd.testing.assert_frame_equal(
        candidates, pd.read_pickle(folder / "candidates.pkl"), check_exact=True
    )
    actual = pd.read_csv(
        folder / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    normalized = pd.read_csv(
        StringIO(expected.to_csv(index=False)),
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    pd.testing.assert_frame_equal(actual, normalized, check_exact=True)
    text = pd.read_csv(
        folder / "automatic_annotations.csv", dtype=str, keep_default_na=False
    )
    pd.testing.assert_frame_equal(
        text.iloc[: len(previous)][list(previous)].reset_index(drop=True),
        previous.reset_index(drop=True),
        check_exact=True,
    )
    keys = set(runner.common.accepted_keys(actual))
    old_keys = set(runner.common.accepted_keys(previous))
    assert old_keys.issubset(keys) and len(keys) == len(actual)
    accepted = candidates.automatic_accepted.map(_boolean).fillna(False).astype(bool)
    assert set(runner.common.accepted_keys(candidates.loc[accepted])) == keys
    assert int(accepted.sum()) == len(actual)
    legacy = verify_candidate_flags(
        candidates, pd.read_pickle(OLD / "runs" / dataset / "candidates.pkl")
    )
    # Every original available measurement checked; old missing cache fields are
    # explicitly counted, never fabricated or allowed on newly accepted rows.
    baseline = pd.read_pickle(runner.BASE / "runs" / dataset / "full_annotation.pkl")
    populated, inherited, roundoff = 0, 0, 0
    for stream, payload in baseline["streams"].items():
        rows = actual.loc[actual.stream.eq(stream)]
        assert not rows.feat_feature_id.astype(str).duplicated().any()
        original = payload["features"].copy()
        original.index = original.feature_id.astype(str)
        source = original.loc[rows.feat_feature_id.astype(str)]
        new = rows.index.to_numpy() >= len(previous)
        for dest, origin in (
            ("feat_mz", "mz"),
            ("feat_rt", "rt"),
            ("feat_intensity", "intensity"),
        ):
            a, b = rows[dest].to_numpy(float), source[origin].to_numpy(float)
            np.testing.assert_allclose(
                a, b, rtol=8 * np.finfo(float).eps, atol=0, equal_nan=True
            )
            roundoff += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        for col in original:
            dest = "feat_" + col
            if dest not in text:
                assert not new.any(), (dataset, stream, dest)
                continue
            values = text.loc[rows.index, dest]
            measured = source[col].reset_index(drop=True)
            present = values.ne("").to_numpy()
            if pd.api.types.is_numeric_dtype(original[col]):
                a = pd.to_numeric(values, errors="coerce").to_numpy(float)
                b = measured.to_numpy(float)
                np.testing.assert_allclose(
                    a[present | new],
                    b[present | new],
                    rtol=8 * np.finfo(float).eps,
                    atol=0,
                    equal_nan=True,
                    err_msg=dest,
                )
                inherited += int((~present & np.isfinite(b) & ~new).sum())
            else:
                b = measured.fillna("").astype(str).to_numpy()
                a = values.astype(str).to_numpy()
                assert (a[present | new] == b[present | new]).all(), dest
                inherited += int((~present & (b != "") & ~new).sum())
            populated += int(present.sum())
    identities = [
        json.loads(
            (
                QE / version / "evaluation_v2" / dataset / "evaluation_identity.json"
            ).read_text()
        )
        for version in ("baseline", "experiment_029", OUT.name)
    ]
    assert len({i["manual_sha256"] for i in identities}) == 1
    assert identities[-1]["selected_sha256"] == runner.sha(
        folder / "automatic_annotations.csv"
    )
    assert identities[-1]["candidates_sha256"] == runner.sha(folder / "candidates.pkl")
    complete = json.loads((folder / "complete.json").read_text())
    assert complete["previous_calls_immutable"] and complete[
        "added_annotations"
    ] == len(filled)
    assert complete["counts"] == actual.groupby("stream").size().to_dict()
    print(
        "041 ALL4MODES/OLDALLCELLS/NEWRAWQUANT/GOLD PASS",
        dataset,
        len(filled),
        flush=True,
    )
    return dict(
        dataset=dataset,
        added=len(filled),
        legacy_flags=legacy,
        available_measurement_cells=populated,
        inherited_empty_cache_cells=inherited,
        roundoff=roundoff,
    )


def main():
    protocol = runner.initialize()
    datasets, _ = runner.inventory()
    files = {
        Path(__file__),
        Path(__file__).with_name("verify_qe_np_context_bmp_experiment.py"),
        OUT / "protocol.json",
        OUT / "runner_source.py",
    }
    for name in ("calibration", "runs", "evaluation_v2", "reports_v2"):
        files.update(p for p in (OUT / name).rglob("*") if p.is_file())
    bound = {str(p): runner.sha(p) for p in sorted(files)}
    experts = verify_models()
    with ProcessPoolExecutor(max_workers=3) as ex:
        inner = list(ex.map(verify_calibration, range(5)))
        outer = list(ex.map(verify_dataset, datasets))
    assert sum(inner) == 132 and len(outer) == 33
    assert runner.binding_files() == protocol["source_input_sha256"]
    assert all(runner.sha(Path(p)) == h for p, h in bound.items())
    assert runner.sha(runner.PREFLIGHT / "receipt.json") == protocol["preflight_sha256"]
    receipt = dict(
        all_passed=True,
        source_sha256=runner.sha(Path(__file__)),
        planned_verifier_sha256=protocol["planned_verifier_sha256"],
        source_input_hash_checks=len(protocol["source_input_sha256"]),
        source_snapshot_hash_checks=len(protocol["source_snapshot_sha256"]),
        bound_verification_artifacts=len(bound),
        fresh_classifier_fits=0,
        deliberately_reused_029_parents=25,
        deliberately_reused_029_mode_classifiers=experts,
        inner_score_reproductions=132,
        complete_class_grid_replays=5,
        outer_datasets=33,
        original_coordinate_stream_checks=132,
        accepted_candidate_export_checks=33,
        old_allcell_preservation_checks=33,
        identical_manual_reference_hash_checks=33,
        added_annotations=sum(r["added"] for r in outer),
        inherited_accepted_candidate_flags=sum(r["legacy_flags"] for r in outer),
        available_measurement_cells_checked=sum(
            r["available_measurement_cells"] for r in outer
        ),
        inherited_empty_old_cache_measurement_cells=sum(
            r["inherited_empty_cache_cells"] for r in outer
        ),
        preexisting_roundoff_cells=sum(r["roundoff"] for r in outer),
        note="Complete engineering replay, not chemical identity/FDR or deployment approval. All old calls immutable, all new original measurements checked; same33 previously exposed datasets, INNER-only thresholds, frozen029models explicitly reused, not newly fitted.",
    )
    runner.dump(OUT / "integrity_receipt.json", receipt)
    print("041 MASTER ALLPASS", flush=True)


if __name__ == "__main__":
    main()
