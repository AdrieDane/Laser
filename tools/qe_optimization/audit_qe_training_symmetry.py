"""Read-only duplicate inner-fit audit; never substitutes models or annotations."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pickle

import pandas as pd

from frozen_resources import configure_frozen_resources


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment.resolve()
    configure_frozen_resources(root / "source_snapshot")
    rows, inputs = [], {}
    for first, second in itertools.combinations(range(5), 2):
        paths = [
            root / "calibration" / f"fold_{first}" / f"inner_block_{second}_model.pkl",
            root / "calibration" / f"fold_{second}" / f"inner_block_{first}_model.pkl",
        ]
        for path in paths:
            assert path.exists(), "All20innerfits must be complete"
            inputs[str(path)] = sha(path)
        a, b = (pd.read_pickle(path) for path in paths)
        assert set(a["training_cohorts"]) == set(b["training_cohorts"])
        assert set(a["excluded_groups"]) == set(b["excluded_groups"])
        assert set(a["training_cohorts"]).isdisjoint(a["excluded_groups"])
        assert a["training_feature_masks"] == b["training_feature_masks"]
        assert (
            a["standard_training_feature_scopes"]
            == b["standard_training_feature_scopes"]
        )
        assert a["columns"] == b["columns"]
        assert a["training_pairs"] == b["training_pairs"]
        assert a["positives"] == b["positives"]
        assert (
            set(a["experts"])
            == set(b["experts"])
            == {"RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"}
        )
        identical = {
            mode: pickle.dumps(a["experts"][mode], protocol=5)
            == pickle.dumps(b["experts"][mode], protocol=5)
            for mode in sorted(a["experts"])
        }
        rows.append(
            dict(
                excluded_fold_first=first,
                excluded_fold_second=second,
                training_pairs=a["training_pairs"],
                positives=a["positives"],
                training_groups=len(a["training_cohorts"]),
                excluded_groups=len(a["excluded_groups"]),
                identical_scopes_masks_and_columns=True,
                **{
                    mode + "_entire_expert_pickle_identical": equal
                    for mode, equal in identical.items()
                },
            )
        )
    for path, digest in inputs.items():
        assert sha(Path(path)) == digest
    out = root / "training_compute_symmetry_v1"
    out.mkdir(exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / "by_inner_pair.csv", index=False)
    cols = [c for c in table if c.endswith("_entire_expert_pickle_identical")]
    summary = dict(
        annotation_changes=False,
        models_fitted=0,
        paired_exclusion_scopes=10,
        compared_inner_model_files=20,
        compared_mode_expert_pairs=40,
        entire_expert_pickle_identical_checks=int(table[cols].sum().sum()),
        all_expert_pairs_identical=bool(table[cols].all().all()),
        verified_input_hashes=len(inputs),
        source_sha256=sha(Path(__file__)),
        note="READ-ONLY numerical serialization/scope symmetry audit,not an annotation experiment, "
        "performance gain or cache implementation. Entire expert dictionaries include estimator "
        "AND predictor transforms/metadata; parent core-reference hashes remain distinct provenance. "
        "Even if all40match,current25fit runner/output remains unchanged. Future caching must key "
        "complete input/label/source/parameter/reference scopes,record reuse honestly,and verify all "
        "heldout/calibration scores and final132outputs before claiming annotation parity.",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
