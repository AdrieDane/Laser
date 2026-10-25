"""047 no-new-cut preflight: ALL25/50reuse,132score replays,33disabledexports."""

import json

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_bmp_extended_experiment as runner


def verify_reused_models(datasets, folds):
    protocol = json.loads((runner.source.OUT / "protocol.json").read_text())
    parameters = runner.source.original.inherited.common.estimator_parameters(
        protocol["classifier"]
    )
    gate = json.loads((runner.source.PREFLIGHT / "parent_support.json").read_text())
    parent_count = expert_count = 0
    for outer in range(5):
        for inner in [None, *sorted(set(range(5)) - {outer})]:
            blocked = {outer} if inner is None else {outer, inner}
            allowed = {d[:7] for d in datasets if folds[d] not in blocked}
            excluded = {d[:7] for d in datasets if folds[d] in blocked}
            path = runner.parent_path(outer, inner)
            model = pd.read_pickle(path)
            assert (
                runner.sha(path)
                == json.loads(path.with_suffix(".complete.json").read_text())[
                    "model_sha256"
                ]
            )
            assert (
                model["format"] == runner.source.completion.FORMAT
                and len(model["columns"]) == 163
            )
            assert model["training_cohorts"] == sorted(allowed) and model[
                "excluded_groups"
            ] == sorted(excluded)
            assert model["training_feature_masks"] == {
                d: sorted(blocked | {folds[d]})
                for d in datasets
                if folds[d] not in blocked
            }
            expected = next(
                p for p in gate if p["outer"] == outer and p["inner"] == inner
            )
            assert set(model["experts"]) == set(runner.NP_MODES)
            for s in runner.NP_MODES:
                expert = model["experts"][s]
                assert expert is not None and expected["modes"][s]["supported"]
                assert all(
                    expert[k] == v for k, v in expected["modes"][s]["metadata"].items()
                )
                assert model["support"][s] == expected["modes"][s]["support"]
                assert set(model["support"][s]["actual_projects"]) == allowed and set(
                    model["support"][s]["BMP_positive_projects"]
                ).issubset(allowed)
                assert model["training_matrix_receipts"][s] == {
                    k: v
                    for k, v in expected["modes"][s].items()
                    if k.endswith("_sha256")
                }
                assert all(
                    expert["estimator"].get_params()[k] == v
                    for k, v in parameters.items()
                )
                assert (
                    expert["estimator"].n_features_in_ == 163
                    and expert["estimator"].n_iter_ == 180
                )
                expert_count += 1
            parent_count += 1
    assert parent_count == 25 and expert_count == 50
    return parent_count, expert_count


def curve_fold(outer):
    cases, folders, projects, pooled, allowed, excluded = runner.cases_and_curves(outer)
    assert set(projects.project) == allowed and not set(projects.project) & excluded
    path = runner.PREFLIGHT / f"fold_{outer}"
    path.mkdir()
    paths = []
    for name, table in (
        ("folder_curves", folders),
        ("project_curves", projects),
        ("pooled_curves", pooled),
    ):
        p = path / (name + ".pkl")
        table.to_pickle(p)
        paths.append(p)
    print(
        "047 NO-CUT FULL INNER SCORE/PROJECT CURVES CHECK",
        outer,
        len(cases),
        flush=True,
    )
    return len(cases), {str(p): runner.sha(p) for p in paths}


def main():
    configure_frozen_resources(REPO)
    runner.completion.assert_native_body_parity()
    hashes = runner.source_inputs()
    runner.PREFLIGHT.mkdir(exist_ok=False)
    runner.dump(runner.PREFLIGHT / "planned_inputs.json", hashes)
    datasets, folds = runner.inventory()
    parents, experts = verify_reused_models(datasets, folds)
    cases, artifacts = 0, {}
    for o in range(5):
        count, paths = curve_fold(o)
        cases += count
        artifacts.update(paths)
    assert cases == 132
    for d in datasets:
        raw = runner.load_predictors(d, {folds[d]})
        previous = runner.previous_annotations(d)
        annotations, candidates, added = runner.assemble(
            d, previous, raw["pool"].iloc[:0].copy()
        )
        assert added.empty
        pd.testing.assert_frame_equal(annotations, previous, check_exact=True)
        pd.testing.assert_frame_equal(
            candidates,
            pd.read_pickle(runner.PREVIOUS / "runs" / d / "candidates.pkl"),
            check_exact=True,
        )
    assert hashes == runner.source_inputs()
    runner.dump(
        runner.PREFLIGHT / "receipt.json",
        dict(
            all_passed=True,
            reused_parents=parents,
            reused_experts=experts,
            inner_query_cases=cases,
            disabled_whole_exports=33,
            new_classifier_fits=0,
            new_operating_points_selected=False,
            annotation_changes=False,
            source_input_sha256=hashes,
            curve_sha256=artifacts,
        ),
    )


if __name__ == "__main__":
    main()
