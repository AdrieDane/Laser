"""Freeze all33 score-free eligibility masks, THEN decode curation for ceilings.

No models, thresholds, candidate generation, app changes or accepted calls.
Upper bounds are optimistic availability, not annotation performance gains.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
from bmp_context_measurements import PREFIX
from bmp_context_completion import profile_gate
import run_qe_bmp_context_experiment as source

OUT = source.QE / "BMP_admission_ceiling_041_v1"
PLAN = Path(__file__).parents[2] / "docs/validation/next_BMP_admission_diagnostic.txt"
MASKS = ("physical", "counterpart", "measurable_profile", "strong_profile")


def freeze_masks(raw, previous):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.raw_counterpart_rt import SUPERVISION

    p, e = raw["pool"], raw["evidence"]
    if SUPERVISION & set(p) or not p.index.equals(e.index):
        raise ValueError("Aligned unsupervised original predictors required")
    common = source.original.inherited.common
    old = set(common.accepted_keys(previous))
    ids = {k[:4] for k in old}
    occupied = {k[0:1] + k[4:] for k in old}
    keys = list(common.accepted_keys(p))
    if len(keys) != len(p):
        raise ValueError("Canonical keys must preserve original pool rows")
    available = pd.Series(
        [k[:4] not in ids and (k[0], k[4]) not in occupied for k in keys],
        index=p.index,
    )
    physical = (
        p.stream.isin(source.NP_MODES)
        & p["class"].map(canonical_lipid_class).eq("BMP")
        & ~p.is_is.astype(bool)
        & e[PREFIX + "source_physical"].eq(1)
        & available
    )
    counterpart = physical & e[PREFIX + "candidate_counterparts"].ge(1)
    measurable = (
        counterpart
        & e[PREFIX + "counterpart_profiles"].ge(1)
        & e[PREFIX + "counterpart_n"].ge(8)
    )
    strong = physical & profile_gate(e)
    # Independently scalar-check the hard gate (unknown/inf never true).
    gate_columns = [
        PREFIX + s
        for s in (
            "source_physical", "candidate_counterparts", "counterpart_profiles",
            "counterpart_n", "counterpart_raw", "counterpart_normalized",
            "counterpart_leave_one_out",
        )
    ]
    values = e[gate_columns].apply(pd.to_numeric, errors="coerce")
    scalar = pd.Series(
        [
            all(np.isfinite(v) for v in x)
            and x[0] == 1 and x[1] >= 1 and x[2] >= 1 and x[3] >= 8
            and x[4] >= .8 and x[5] >= .8 and x[6] >= .6
            for x in values.itertuples(index=False, name=None)
        ], index=p.index,
    )
    assert profile_gate(e).equals(scalar)
    result = pd.DataFrame(dict(zip(MASKS, (physical, counterpart, measurable, strong))))
    assert not (strong & ~measurable).any()
    assert not (measurable & ~counterpart).any()
    assert not (counterpart & ~physical).any()
    return dict(keys=keys, masks=result)


def main():
    configure_frozen_resources(source.OUT / "source_snapshot")
    master_path = source.OUT / "integrity_receipt.json"
    master = json.loads(master_path.read_text())
    assert master["all_passed"] and master["datasets"] == 33
    datasets, folds = source.inventory()
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    inputs = {str(p): source.sha(p) for p in (Path(__file__), PLAN, master_path)}
    artifacts = {}
    for d in datasets:
        raw = source.load_predictors(d, {folds[d]})
        allowed = {x[:7] for x in datasets if folds[x] != folds[d]}
        excluded = {x[:7] for x in datasets if folds[x] == folds[d]}
        source.preparation.validate_scope(raw, allowed, excluded, query_is_training=False)
        for path in (
            source.preparation.original.preparation.prepared_path(d, {folds[d]}),
            source.preparation.sidecar_path(d, {folds[d]}),
            source.PREVIOUS / "runs" / d / "automatic_annotations.csv",
        ):
            assert master["source_input_sha256"][str(path)] == source.sha(path)
            inputs[str(path)] = source.sha(path)
        saved = freeze_masks(raw, source.previous_annotations(d))
        destination = OUT / "private" / (d + ".pkl")
        pd.to_pickle(saved, destination)
        artifacts[str(destination)] = source.sha(destination)
        print("BLIND ADMISSION MASKS", d, flush=True)
    source.dump(OUT / "blind_receipt.json", dict(
        all_passed=True, datasets=33, manual_tables_opened=0,
        source_input_sha256=inputs, artifact_sha256=artifacts,
        annotation_changes=False, classifier_fits=0, thresholds_selected=False,
    ))
    # No query curation was decoded above this ALL33 barrier.
    rows = []
    for d in datasets:
        saved = pd.read_pickle(OUT / "private" / (d + ".pkl"))
        mp = source.PREVIOUS / "evaluation_v2" / d / "manual_comparison.pkl"
        inputs[str(mp)] = source.sha(mp)
        manual = pd.read_pickle(mp)
        sets = {
            m: {k for k, yes in zip(saved["keys"], saved["masks"][m]) if yes}
            for m in MASKS
        }
        for s in source.NP_MODES:
            sub = manual.loc[manual.stream.eq(s) & manual.lipid_class.eq("BMP")]
            correct = int(sub.same_manual_peak.sum())
            for mask in MASKS:
                missing = sum(
                    not r.same_manual_peak and bool(r.mapped)
                    and (r.stream, r.species, r.adduct, bool(r.internal_standard),
                         str(r.feature_id)) in sets[mask]
                    for r in sub.itertuples()
                )
                rows.append(dict(dataset=d, stream=s, gate=mask,
                    manual_rows=len(sub), current_correct=correct,
                    eligible_missing_manual_rows=int(missing),
                    optimistic_ceiling_count=correct+int(missing)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "by_dataset_mode_gate.csv", index=False)
    total = table.groupby(["stream", "gate"], as_index=False)[[
        "manual_rows", "current_correct", "eligible_missing_manual_rows",
        "optimistic_ceiling_count",
    ]].sum()
    total["optimistic_ceiling_percent"] = 100 * total.optimistic_ceiling_count / total.manual_rows
    total.to_csv(OUT / "by_mode_gate.csv", index=False)
    assert all(source.sha(Path(p)) == h for p,h in {**inputs, **artifacts}.items())
    source.dump(OUT / "receipt.json", dict(
        all_passed=True, datasets=33, dataset_modes=66,
        source_input_sha256=inputs, artifact_sha256=artifacts,
        annotation_changes=False, classifier_fits=0, thresholds_selected=False,
        note="Optimistic candidate availability,NOT annotation performance or attainable gains. ALL33 masks frozen before query curation read in this diagnostic; historical exposure acknowledged. All BMP manual rows remain denominators; old calls unchanged. No rank/conflict/extra constraints included. Never a query-curation runtime filter.",
    ))
    print(total.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
