"""All33 paired-gap scope/reference/prediction/physical-multiset verification.

Authorized training curations are reconstructed separately. No heldout query
curation is decoded for validation before the complete blind numeric gate.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import re

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import paired_gap_pilot_data as data
import paired_gap_independent as independent
import prepare_qe_paired_gap_pilot as prep

OUT = data.QE / "paired_isomer_gap_independent_check_v1"


def independent_training(outer):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key
    from standard_lipidomics.validation import _species, canonical_feature_id

    datasets, folds = data.inventory()
    allowed = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    output, support = [], []
    reads = 0
    for d in datasets:
        if folds[d] == outer:
            continue
        assert d[:7] in allowed and d[:7] not in excluded
        manual = pd.read_pickle(
            data.BASE / "evaluation_v2" / d / "manual_comparison.pkl"
        )
        reads += 1
        baseline = pd.read_pickle(data.BASE / "runs" / d / "full_annotation.pkl")
        theory, raw = {}, {}
        for stream, payload in baseline["streams"].items():
            for name, ion, mz in payload["reference"][
                ["name", "adduct", "ref_mz"]
            ].itertuples(index=False, name=None):
                if str(name).endswith(" (IS)") or str(name).startswith("IS: "):
                    continue
                theory.setdefault((stream, _species(name), ion_key(ion)), []).append(
                    float(mz)
                )
            features = payload["features"]
            for fid, rt, mz in features[["feature_id", "rt", "mz"]].itertuples(
                index=False, name=None
            ):
                key = (stream, canonical_feature_id(fid))
                assert key not in raw
                raw[key] = (float(rt), float(mz))
        counts = {}
        for r in manual.to_dict("records"):
            if bool(r["internal_standard"]):
                continue
            stream = str(r["stream"])
            cls = canonical_lipid_class(r["lipid_class"])
            ion = ion_key(r["adduct"])
            counts[(stream, cls)] = counts.get((stream, cls), 0) + 1
            if (
                not bool(r["mapped"])
                or bool(r["duplicate_manual_entry"])
                or bool(r["conflicting_manual_feature"])
            ):
                continue
            if independent.member(stream, ion, cls) is None:
                continue
            match = re.search(r"\((?:(?:O|P)-)?([0-9]+):([0-9]+)\)", r["species"])
            masses = theory.get((stream, r["species"], ion), [])
            if (
                match is None
                or not masses
                or not np.isfinite(masses).all()
                or max(masses) - min(masses) > 1e-8
            ):
                continue
            c, db = (int(v) for v in match.groups())
            rt, mz = raw[(stream, canonical_feature_id(r["feature_id"]))]
            output.append(
                dict(
                    dataset=d,
                    cohort=d[:7],
                    stream=stream,
                    lipid_class=cls,
                    carbon=c,
                    db=db,
                    adduct=ion,
                    feature_id=canonical_feature_id(r["feature_id"]),
                    rt_seconds=rt * 60,
                    reference_mz=masses[0],
                    observed_mz=mz,
                )
            )
        for (stream, cls), n in sorted(counts.items()):
            support.append(
                dict(cohort=d[:7], stream=stream, lipid_class=cls, manual_rows=n)
            )
    return (
        pd.DataFrame(output, columns=data.OBSERVATION),
        pd.DataFrame(support),
        allowed,
        excluded,
        reads,
    )


def same_model(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], dict):
            same_model(a[key], b[key])
        elif isinstance(a[key], float):
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-10)
        else:
            assert a[key] == b[key], key


def check_model(outer):
    configure_frozen_resources(data.OUT / "source_snapshot")
    rows, support, training, excluded, reads = independent_training(outer)
    folder = data.OUT / "models" / f"fold_{outer}"
    pd.testing.assert_frame_equal(
        rows, pd.read_pickle(folder / "authorized_training_rows.pkl"), check_exact=True
    )
    pd.testing.assert_frame_equal(
        support,
        pd.read_pickle(folder / "authorized_training_support.pkl"),
        check_exact=True,
    )
    model, pairs = independent.fit_reference(
        rows, support, training_groups=training, excluded_groups=excluded
    )
    same_model(model, json.loads((folder / "model.json").read_text()))
    pd.testing.assert_frame_equal(
        pairs, pd.read_pickle(folder / "training_pairs.pkl"), check_exact=True
    )
    audit = pd.read_csv(folder / "training_role_audit.csv")
    assert len(audit) == reads and audit.authorized_training.all()
    assert set(audit.cohort) <= training and not set(audit.cohort) & excluded
    print(
        "INDEPENDENT OTHER-PROJECT GAP REFERENCE",
        outer,
        reads,
        len(model["models"]),
        flush=True,
    )
    return reads


def check_query(item):
    configure_frozen_resources(data.OUT / "source_snapshot")
    from standard_lipidomics.validation import _species, canonical_feature_id
    from matlab_annotation.standard_rt_library import ion_key

    dataset, outer = item
    model_path = data.OUT / "models" / f"fold_{outer}" / "model.json"
    model = json.loads(model_path.read_text())
    anchors, chemical, masses, baseline, accepted = data.query_inputs(dataset, outer)
    predictions = independent.predict_reference(
        anchors, chemical, model, query_group=dataset[:7], declared_masses=masses
    )
    folder = data.OUT / "private" / dataset
    receipt = json.loads((folder / "receipt.json").read_text())
    assert (
        receipt["model_sha256"] == data.sha(model_path)
        and receipt["query_gold_used"] is False
    )
    for name, h in receipt["artifact_sha256"].items():
        assert data.sha(folder / name) == h
    actual_prediction = pd.read_pickle(folder / "predictions.pkl")
    pd.testing.assert_frame_equal(
        predictions, actual_prediction, check_exact=False, rtol=0, atol=1e-10
    )
    graph = independent.raw_reference(
        predictions, {s: p["features"] for s, p in baseline["streams"].items()}
    )
    original_path = (
        data.QE
        / "standard_context_preparation_v1/prepared"
        / f"exclude_{outer}"
        / (dataset + ".pkl")
    )
    original = pd.read_pickle(original_path)
    assert original["query_dataset"] == dataset and set(original["excluded_folds"]) == {
        outer
    }
    assert dataset[:7] not in original["reference_training_groups"]
    assert receipt["old_pool_sha256"] == data.sha(original_path)
    previous_pairs = set()
    for stream, name, ion, std, fid in original["pool"][
        ["stream", "name", "adduct", "is_is", "feat_feature_id"]
    ].itertuples(index=False, name=None):
        previous_pairs.add(
            (
                str(stream),
                _species(name),
                ion_key(ion),
                bool(std),
                canonical_feature_id(fid),
            )
        )
    occupied, identities = set(), set()
    for stream, name, ion, std, fid in accepted[
        ["stream", "name", "adduct", "is_is", "feat_feature_id"]
    ].itertuples(index=False, name=None):
        occupied.add((str(stream), canonical_feature_id(fid)))
        identities.add((str(stream), _species(name), ion_key(ion), bool(std)))
    if len(graph):
        keys = [
            (
                r.stream,
                _species(r["name"]),
                ion_key(r.adduct),
                False,
                canonical_feature_id(r.feat_feature_id),
            )
            for _, r in graph.iterrows()
        ]
        graph["novel_to_old029_core_pool"] = [k not in previous_pairs for k in keys]
        graph["identity_is_free"] = [k[:4] not in identities for k in keys]
        graph["feature_is_free"] = [(k[0], k[4]) not in occupied for k in keys]
    else:
        for field in (
            "novel_to_old029_core_pool",
            "identity_is_free",
            "feature_is_free",
        ):
            graph[field] = pd.Series(dtype=bool)
    actual_graph = pd.read_pickle(folder / "candidate_graph.pkl")
    keys = ["prediction_index", "feat_feature_id"]
    expected = graph.sort_values(keys, kind="stable").reset_index(drop=True)
    actual = actual_graph.sort_values(keys, kind="stable").reset_index(drop=True)
    assert len(actual) == len(expected) and list(actual) == list(expected)
    # Full tables, not sets: retain duplicate multiplicity and all metadata.
    pd.testing.assert_frame_equal(
        actual, expected, check_exact=False, rtol=0, atol=1e-10
    )
    for c in ("feat_mz", "feat_rt", "feat_intensity", "feat_feature_id"):
        pd.testing.assert_series_equal(actual[c], expected[c], check_exact=True)
    assert receipt["predictions"] == len(predictions) and receipt["graph_rows"] == len(
        graph
    )
    assert graph.stream.str.startswith("RP_").all()
    print(
        "INDEPENDENT BLIND PREDICTIONS/FULL GRAPH",
        dataset,
        len(predictions),
        len(graph),
        flush=True,
    )
    return dict(dataset=dataset, predictions=len(predictions), graph_rows=len(graph))


def main():
    configure_frozen_resources(data.OUT / "source_snapshot")
    blind_path = data.OUT / "blind_receipt.json"
    blind = json.loads(blind_path.read_text())
    assert (
        blind["all_passed"]
        and blind["datasets"] == 33
        and blind["query_gold_validation_tables_opened"] == 0
    )
    assert blind["annotation_outputs"] == 0 and blind["annotation_changes"] is False
    assert blind["source_input_sha256"] == prep.source_inputs()
    assert (
        json.loads((data.OUT / "protocol.json").read_text())["numeric_runtime"]
        == data.environment()
    )
    for p, h in blind["artifact_sha256"].items():
        assert data.sha(p) == h
    OUT.mkdir(exist_ok=False)
    datasets, folds = data.inventory()
    with ProcessPoolExecutor(max_workers=2) as executor:
        roles = list(executor.map(check_model, range(5)))
    assert len(roles) == 5 and sum(roles) == 132
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(check_query, [(d, folds[d]) for d in datasets]))
    assert len(results) == 33
    pd.DataFrame(results).to_csv(OUT / "all33_prediction_graph_checks.csv", index=False)
    assert blind["source_input_sha256"] == prep.source_inputs()
    data.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            reference_models=5,
            authorized_other_project_training_table_uses=132,
            query_gold_validation_tables_opened=0,
            annotation_changes=False,
            blind_receipt_sha256=data.sha(blind_path),
            source_input_sha256=blind["source_input_sha256"],
            predictions=sum(r["predictions"] for r in results),
            raw_graph_rows=sum(r["graph_rows"] for r in results),
            note="Separate training row extraction/raw original RT+mass joins,scalar paired-gap medians/scopes/quality,partner predictions and unsorted full raw-graph MULTISETS. All132original coordinate checks and zero heldout-query-gold validation before complete numeric gate. Candidateopportunities NOTaccepted gains.",
        ),
    )
    print(
        "ALL33 PAIRED-GAP INDEPENDENT CHECKS PASS; NOW QUERY-GOLD DIAGNOSIS ALLOWED",
        flush=True,
    )


if __name__ == "__main__":
    main()
