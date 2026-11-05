"""Frozen all-feature hard/soft comparison; reference labels stay outside workers."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pandas as pd

from freeze_untargeted_baseline import (
    ROOT,
    MODES,
    digest,
    fingerprint,
    project_files,
    write_json,
    score_representatives,
    ratio,
)
from evaluate_untargeted_method import scored_ledger, class_mode_metrics
from audit_untargeted_checkpoint import metric_rows


def reference_pair_metrics(reference, predictions):
    """Reference identity agreement, not analytical false discovery rate."""
    gold = {
        (r.stream, str(r.feature_id)): r
        for r in reference[reference.mapped].itertuples()
    }
    assigned = {
        (r.stream, str(r.feature_id)): r
        for r in predictions.itertuples()
        if r.adduct and getattr(r, "isotope_index", 0) == 0
    }
    groups = defaultdict(list)
    for key, row in assigned.items():
        if key in gold:
            groups[row.compound_id].append(key)
    expected = set()
    for _, frame in reference.groupby("name"):
        rows = list(frame[frame.mapped].itertuples())
        for a, b in combinations(rows, 2):
            if a.stream == b.stream:
                continue  # Isomer ambiguity within one mode is not resolved by a name.
            if (
                a.stream[:2] == b.stream[:2]
                and abs(float(a.manual_rt_seconds) - float(b.manual_rt_seconds)) > 12
            ):
                continue
            expected.add(
                tuple(
                    sorted(
                        ((a.stream, str(a.feature_id)), (b.stream, str(b.feature_id)))
                    )
                )
            )
    predicted, correct = set(), set()
    for keys in groups.values():
        for a, b in combinations(keys, 2):
            if a[0] == b[0]:
                continue  # Same cross-mode scope as the expected-pair denominator.
            pair = tuple(sorted((a, b)))
            predicted.add(pair)
            if (
                pair in expected
                and assigned[a].adduct == gold[a].adduct
                and assigned[b].adduct == gold[b].adduct
            ):
                correct.add(pair)
    return {
        "reference_pair_agreement": ratio(len(correct), len(predicted)),
        "compatible_mapped_reference_pair_recovery": ratio(len(correct), len(expected)),
        "discordant_reference_pairs": len(predicted - correct),
        "denominator_limitation": "Mapped exact-name cross-mode reference pairs; same-phase RT separation <=12 s. Unknown extras not false positives.",
    }


def run(args):
    cp, out = args.checkpoint.resolve(), args.output.resolve()
    if (
        out.is_relative_to(ROOT)
        or out.is_relative_to(cp)
        or cp.is_relative_to(out)
        or out.exists()
    ):
        raise ValueError(
            "Use a new private output directory outside checkout/checkpoint."
        )
    node = Path(args.node or shutil.which("node") or "")
    if not node.is_file():
        raise ValueError("Node executable required.")
    inputs = json.loads((cp / "input_fingerprints.json").read_text())
    assert fingerprint(list(project_files(cp / "project")), cp / "project") == inputs
    old = json.loads((cp / "provenance.json").read_text())
    assert digest(cp / "reference/manual_mapping.csv") == old["manual_mapping_sha256"]
    for file, value in old["replay_hashes"][0].items():
        assert digest(cp / "replay_1" / file) == value
    revision = subprocess.check_output(
        ["git", "rev-parse", args.revision + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    out.mkdir(parents=True)
    archive = out / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(
            ["git", "archive", revision], cwd=ROOT, stdout=handle, check=True
        )
    source = out / "source"
    source.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter="data")
    method = json.loads(args.method.read_text())
    method["adduct_scoring"] = "legacy"
    write_json(out / "method.json", method)
    queries = json.loads((cp / "reference/regression_queries.json").read_text())
    write_json(
        out / "feature_keys.json",
        [{"stream": q["stream"], "feature_id": str(q["feature_id"])} for q in queries],
    )
    reference = pd.read_csv(
        cp / "reference/query_ledger.csv", dtype={"feature_id": str}
    ).fillna("")
    protocol = {
        "source_revision": revision,
        "source_archive_sha256": digest(archive),
        "scope": "one frozen ALD development acquisition, all 63 labels and four modes",
        "reference_sha256": old["manual_mapping_sha256"],
        "checkpoint": str(cp),
        "arms": ["current", "full_feature_hard", "full_feature_soft"],
        "recipe_scoring": "legacy in every arm, isolating grouping from Stage 2 priors",
        "statistics_formula_isotope_outputs": "Original saved outputs held fixed; separate full-list isotope candidates are unverified",
        "soft_rule": {
            "rho_min": 0.2,
            "within_group_min": 0.05,
            "score_min": 0.45,
            "score_weights": [0.30, 0.40, 0.15, 0.15],
        },
        "hard_rule": {"rho_min": 0.6, "within_group_min": 0.2},
        "co_detected_biological_specimens_min": 6,
        "hard_to_soft_cache": "Same verified mass/profile candidate pool, no reference labels or scores reused",
        "extra_ion_scope": "Historical primary: displayed detections in significant families; full-ledger unmatched ions are not annotation calls",
        "rank_weights": {"catalog": 0.01, "chemistry": 0.05, "family_member": 0.025, "family_member_cap": 3},
        "default_promotion": False,
        "Binner": "Four-mode input export; native comparison pending trusted executable",
    }
    write_json(out / "protocol.json", protocol)
    baseline = pd.read_csv(
        cp / "replay_1/compound_features.csv", dtype={"feature_id": str}
    ).fillna("")
    base_scored = scored_ledger(reference, baseline)
    base_metrics = metric_rows(base_scored, baseline)
    base_classes = {(r["scope"], r["mode"]): r for r in class_mode_metrics(base_scored)}
    comparisons = []
    classes = []
    summaries = {}
    pairs = {"baseline": reference_pair_metrics(reference, baseline)}
    for arm in protocol["arms"]:
        directory = out / arm
        directory.mkdir()
        command = [
            sys.executable,
            str(source / "tools/freeze_untargeted_baseline.py"),
            "--worker",
            str(source),
            str(cp / "project"),
            str(out / "feature_keys.json"),
            str(directory),
            str(node.resolve()),
            str(out / "method.json"),
            arm,
        ]
        if arm == "full_feature_soft":
            command.append(str(out / "full_feature_hard"))
            write_json(
                directory / "candidate_cache_provenance.json",
                {
                    name: digest(out / "full_feature_hard" / (name + ".parquet"))
                    for name in ("ion_relationships", "ion_isotope_candidates")
                },
            )
        print("RUNNING " + arm, flush=True)
        subprocess.run(command, cwd=source, check=True)
        path = directory / (
            "ion_feature_ledger.parquet"
            if arm != "current"
            else "compound_features.csv"
        )
        predicted = (
            pd.read_parquet(path)
            if path.suffix == ".parquet"
            else pd.read_csv(path, dtype={"feature_id": str})
        ).fillna("")
        displayed = pd.read_csv(
            directory / "compound_features.csv", dtype={"feature_id": str}
        ).fillna("")
        scored = scored_ledger(reference, predicted)
        scored.to_csv(directory / "all_reference_predictions.csv", index=False)
        before = base_scored.correct_adduct
        transitions = reference.copy()
        transitions["correct_before"] = before
        transitions["correct_after"] = scored.correct_adduct
        transitions["adduct_before"] = base_scored.assigned_adduct
        transitions["adduct_after"] = scored.assigned_adduct
        transitions["gained"] = ~before & scored.correct_adduct
        transitions["lost"] = before & ~scored.correct_adduct
        transitions.to_csv(directory / "all_reference_transitions.csv", index=False)
        after = {(r["scope"], r["mode"]): r for r in metric_rows(scored, displayed)}
        for row in base_metrics:
            key = (row["scope"], row["mode"])
            r = after[key]
            comparisons.append(
                {
                    "arm": arm,
                    "scope": row["scope"],
                    "mode": row["mode"],
                    **{
                        k + "_before": v
                        for k, v in row.items()
                        if k not in ("scope", "mode")
                    },
                    **{
                        k + "_after": v
                        for k, v in r.items()
                        if k not in ("scope", "mode")
                    },
                }
            )
        for row in class_mode_metrics(scored):
            b = base_classes[(row["scope"], row["mode"])]
            classes.append(
                {
                    "arm": arm,
                    "class": row["scope"],
                    "mode": row["mode"],
                    **{
                        k + "_before": v
                        for k, v in b.items()
                        if k not in ("scope", "mode")
                    },
                    **{
                        k + "_after": v
                        for k, v in row.items()
                        if k not in ("scope", "mode")
                    },
                }
            )
        pc_rows, pc_metric = score_representatives(
            queries,
            json.loads((directory / "representatives.json").read_text()),
            reference,
        )
        pd.DataFrame(pc_rows).to_csv(
            directory / "all_60_PC_mode_slots.csv", index=False
        )
        pairs[arm] = reference_pair_metrics(reference, predicted)
        summaries[arm] = {
            "ledger_rows": len(predicted),
            "displayed_family_rows": len(displayed),
            "compound_count": predicted.compound_id.nunique(),
            "assigned_monoisotopic_ions": int(
                (
                    predicted.adduct.ne("")
                    & (
                        predicted.get(
                            "isotope_index", pd.Series(0, index=predicted.index)
                        )
                        == 0
                    )
                ).sum()
            ),
            "exact_positive_PC_counterparts": pc_metric,
            "gained_lost": [
                {
                    **{"mode": mode},
                    "gained": int(
                        transitions[transitions.stream.eq(mode)].gained.sum()
                    ),
                    "lost": int(transitions[transitions.stream.eq(mode)].lost.sum()),
                }
                for mode in MODES
            ],
        }
        if arm == "current":
            cols = [
                "stream",
                "feature_id",
                "compound_id",
                "adduct",
                "neutral_mass",
                "mode_representative",
                "cluster_id",
            ]
            pd.testing.assert_frame_equal(
                baseline[cols]
                .sort_values(["stream", "feature_id"])
                .reset_index(drop=True),
                predicted[cols]
                .sort_values(["stream", "feature_id"])
                .reset_index(drop=True),
            )
        else:
            assert len(predicted) == sum(
                len(pd.read_parquet(p))
                for p in (cp / "project/Untargeted/tables/streams").glob(
                    "*/features.csv.parquet"
                )
            )
    pd.DataFrame(comparisons).to_csv(out / "all_mode_comparison.csv", index=False)
    pd.DataFrame(classes).to_csv(
        out / "all_252_class_mode_comparisons.csv", index=False
    )
    write_json(out / "reference_pair_metrics.json", pairs)
    assert fingerprint(list(project_files(cp / "project")), cp / "project") == inputs
    write_json(
        out / "summary.json",
        {
            "source_revision": revision,
            "arms": summaries,
            "checkpoint_unchanged": True,
            "default_changed": False,
            "reference_pairs": pairs,
            "independent_validation": False,
            "Binner_native_benchmark_performed": False,
        },
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--node")
    run(parser.parse_args())
