"""Read-only audit of029negative supervision, not annotation or label changes.

One outer-excluded physical table per dataset. Manual labels are read ONLY for
this retrospective diagnostic, never used to generate or score candidates.
Absent curation does not establish chemical absence or prove a wrong label.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd

from context import WORKSPACE
from frozen_resources import configure_frozen_resources
from run_qe_atlas_experiment import species, adduct

QE = WORKSPACE / "outputs/qe_optimization_20261022"
CORE = QE / "experiment_029"
OUT = QE / "negative_supervision_diagnostic_029_v2"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def categorize(pool, manual, canonical):
    """Mutually exclusive audit categories; no inferred positive labels."""
    manual = manual[~manual.internal_standard]
    exact = set(
        (s, species(n), adduct(a), str(f))
        for s, n, a, f in manual.loc[
            manual.mapped, ["stream", "species", "adduct", "feature_id"]
        ].itertuples(index=False, name=None)
    )
    identities = set(k[:3] for k in exact)
    unresolved = set(
        (s, species(n), adduct(a))
        for s, n, a in manual.loc[
            ~manual.mapped, ["stream", "species", "adduct"]
        ].itertuples(index=False, name=None)
    )
    occupied = {(k[0], k[3]) for k in exact}
    reviewed = {
        (s, canonical(c))
        for s, c in manual[["stream", "lipid_class"]].itertuples(index=False, name=None)
    }
    classes = {c for _, c in reviewed}
    records = []
    for s, n, a, f, c in pool[
        ["stream", "name", "adduct", "feat_feature_id", "class"]
    ].itertuples(index=False, name=None):
        c = canonical(c)
        k = (s, species(n), adduct(a), str(f))
        if k in exact:
            category = "manual_correct_original_peak_positive"
        elif k[:3] in unresolved:
            category = "already_unknown_unresolved_manual_identity"
        elif k[:3] in identities:
            category = "negative_other_peak_for_curated_identity"
        elif (s, str(f)) in occupied:
            category = "negative_other_label_on_curated_peak"
        elif (s, c) in reviewed:
            category = "negative_absent_identity_in_curated_class_mode"
        elif c in classes:
            category = "negative_class_absent_in_mode_present_in_other_mode"
        else:
            category = "negative_class_absent_from_all_four_manual_modes"
        records.append((s, c, category))
    counts = Counter(records)
    assert sum(counts.values()) == len(pool)
    return [
        dict(stream=s, lipid_class=c, category=r, candidate_rows=n)
        for (s, c, r), n in sorted(counts.items())
    ]


def main():
    configure_frozen_resources(CORE / "source_snapshot")
    from annotator.composition_ranges import canonical_lipid_class

    assert json.loads((CORE / "integrity_receipt.json").read_text())["all_passed"]
    projects = json.loads((CORE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    assert len(projects) == 33
    OUT.mkdir(exist_ok=False)
    inputs, results = {}, []
    for dataset in projects:
        pp = (
            QE
            / "standard_context_preparation_v1/prepared"
            / f"exclude_{folds[dataset]}"
            / (dataset + ".pkl")
        )
        mp = CORE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        for path in (pp, mp):
            inputs[str(path)] = sha(path)
        data = pd.read_pickle(pp)
        assert data["query_dataset"] == dataset
        assert dataset[:7] not in data["reference_training_groups"]
        assert not {"training_positive", "training_label_known"} & set(data["pool"])
        results.extend(
            dict(dataset=dataset, **r)
            for r in categorize(data["pool"], pd.read_pickle(mp), canonical_lipid_class)
        )
        print("READ-ONLY NEGATIVE SUPERVISION AUDIT", dataset, flush=True)
    rows = pd.DataFrame(results)
    rows.to_csv(OUT / "by_dataset_mode_class_category.csv", index=False)
    rows.groupby(
        ["stream", "lipid_class", "category"], as_index=False
    ).candidate_rows.sum().to_csv(OUT / "by_mode_class_category.csv", index=False)
    totals = rows.groupby("category").candidate_rows.sum().astype(int).to_dict()
    assert (
        rows.dataset.nunique() == 33
        and rows.groupby(["dataset", "stream"]).ngroups == 132
    )
    assert all(sha(Path(p)) == h for p, h in inputs.items())
    receipt = dict(
        all_passed=True,
        annotation_changes=False,
        label_changes=False,
        fits_or_new_cutoffs=False,
        datasets=33,
        dataset_modes=132,
        candidate_rows=int(rows.candidate_rows.sum()),
        by_category=totals,
        source_sha256=sha(Path(__file__)),
        inputs_sha256=inputs,
        note="Read-only029outer-query audit,NOT an actual fit's pooled row counts,not independent biological replicates,not accuracy/FDR. Categories have explicit precedence and sum to physical rows. A class with no manual entry may truly be absent OR unreviewed; this audit cannot distinguish these. No inferred positives or evaluation relabeling. All manual classes incl X4 remain visible. No runtime query-curation input. Related acquisitions stay grouped in actual fits.",
    )
    (OUT / "summary.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k != "inputs_sha256"}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
