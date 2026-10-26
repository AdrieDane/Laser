"""All33 read-only known-spike diagnostic, no endogenous gold or annotations.

Separate from048; does not touch its sources, models, cutoffs or outputs.
Cached ORIGINAL feature measurements, not a fresh raw-file processing run.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources
import standard_mass_trend_pilot as probe

QE = WORKSPACE / "outputs/qe_optimization_20261022"
BASE, MASS = QE / "baseline", QE / "experiment_027"
OUT = QE / "standard_mass_trend_diagnostic_v1"
PLAN = REPO / "docs/validation/next_standard_mass_trend_diagnostic.txt"
MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def input_hashes(datasets):
    files = {
        Path(__file__),
        Path(probe.__file__),
        PLAN,
        Path(__file__).with_name("test_standard_mass_trend_pilot.py"),
        Path(__file__).with_name("context.py"),
        Path(__file__).with_name("frozen_resources.py"),
        BASE / "run_manifest.json",
        MASS / "protocol.json",
        MASS / "integrity_receipt.json",
    }
    for d in datasets:
        files.add(MASS / "runs" / d / "private_mass_anchors.pkl")
        files.add(BASE / "runs" / d / "full_annotation.pkl")
    for root in ("app", "methods", "reference_data"):
        files.update(
            p
            for p in (REPO / root).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
        )
    return {str(p): sha(p) for p in sorted(files)}


def verify_original_anchors(anchors, baseline):
    from matlab_annotation.declared_standards import declared_standard_targets

    declared = declared_standard_targets(
        REPO / "reference_data/matlab_pipeline", baseline["internal_standards_expected"]
    )
    for stream, part in anchors.groupby("stream", sort=False):
        features = baseline["streams"][stream]["features"]
        assert features.feature_id.astype(str).is_unique
        lookup = features.set_index(features.feature_id.astype(str))
        mz = lookup.loc[part.feature_id.astype(str), "mz"].to_numpy(float)
        calculated = (mz / part.reference_mz.to_numpy(float) - 1) * 1e6
        np.testing.assert_allclose(
            calculated, part.ppm.to_numpy(float), rtol=0, atol=5e-9
        )
        local = declared.loc[declared.stream.eq(stream)].copy()
        local["neutral_key"] = (
            pd.to_numeric(local.internal_standard_neutral_mass).round(6).astype(str)
        )
        for row in part.itertuples():
            actual = local.loc[
                local.neutral_key.eq(str(row.neutral_key))
                & local.internal_standard_quantity.gt(0)
            ]
            assert (
                len(actual)
                and np.isclose(
                    actual.ref_mz.to_numpy(float), row.reference_mz, rtol=0, atol=1e-9
                ).any()
            )
    return len(anchors)


def describe(case, rows, grouped):
    record = dict(case)
    record["neutral_standards"] = len(grouped)
    record["admitted_physical_ions"] = int(grouped.physical_ions.sum())
    record["minimum_mass_Da"] = (
        float(grouped.reference_mz.min()) if len(grouped) else None
    )
    record["maximum_mass_Da"] = (
        float(grouped.reference_mz.max()) if len(grouped) else None
    )
    for name, interior in (("all_LOO", False), ("interior_LOO", True)):
        selected = (
            rows.loc[rows.trend_LOO_absolute_error_ppm.notna()] if len(rows) else rows
        )
        if interior and len(selected):
            selected = selected.loc[selected.inside_training_mass_span]
        record[name + "_neutrals"] = len(selected)
        for model in ("constant", "trend"):
            record[name + "_" + model + "_median_absolute_ppm"] = (
                float(selected[model + "_LOO_absolute_error_ppm"].median())
                if len(selected)
                else None
            )
            record[name + "_" + model + "_p90_absolute_ppm"] = (
                float(selected[model + "_LOO_absolute_error_ppm"].quantile(0.9))
                if len(selected)
                else None
            )
    return record


def main():
    configure_frozen_resources(REPO)
    datasets = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    assert len(datasets) == 33 and len({d[:7] for d in datasets}) == 30
    assert json.loads((MASS / "integrity_receipt.json").read_text())["all_passed"]
    hashes = input_hashes(datasets)
    OUT.mkdir(exist_ok=False)
    dump(OUT / "planned_inputs.json", hashes)
    # ALL33 cached physical inputs frozen before any diagnostic regression.
    frozen = {}
    for d in datasets:
        anchors = pd.read_pickle(MASS / "runs" / d / "private_mass_anchors.pkl")
        baseline = pd.read_pickle(BASE / "runs" / d / "full_annotation.pkl")
        verify_original_anchors(anchors, baseline)
        frozen[d] = probe.grouped_anchors(anchors)
    assert hashes == input_hashes(datasets)
    dump(
        OUT / "all33_physical_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            manual_comparison_tables_opened=0,
            annotation_changes=False,
            source_input_sha256=hashes,
        ),
    )
    cases, results = [], []
    for d in datasets:
        grouped = frozen[d]
        rows = probe.leave_neutral_out(grouped)
        if len(rows):
            results.append(rows.assign(dataset=d))
        for mode in MODES:
            part = grouped.loc[grouped.stream.eq(mode)]
            tested = rows.loc[rows.stream.eq(mode)] if len(rows) else rows
            cases.append(
                describe(
                    dict(dataset=d, project_group=d[:7], stream=mode), tested, part
                )
            )
        print("STANDARD-ONLY SCALAR/VECTOR LOO CHECK", d, len(rows), flush=True)
    table = pd.DataFrame(cases)
    assert len(table) == 132
    table.to_csv(OUT / "by_dataset_mode_standard_diagnostic.csv", index=False)
    private = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    private.to_csv(OUT / "private_leave_neutral_out.csv", index=False)
    aggregate = []
    for mode, part in table.groupby("stream", sort=False):
        record = dict(
            stream=mode,
            dataset_mode_cases=len(part),
            independent_project_groups=part.project_group.nunique(),
            qualified_neutral_standards=int(part.neutral_standards.sum()),
        )
        for scope in ("all_LOO", "interior_LOO"):
            constant = part[scope + "_constant_median_absolute_ppm"]
            trend = part[scope + "_trend_median_absolute_ppm"]
            valid = constant.notna() & trend.notna()
            a, b = constant.loc[valid], trend.loc[valid]
            record[scope + "_cases"] = int(valid.sum())
            record[scope + "_trend_better_cases"] = int((b < a - 1e-12).sum())
            record[scope + "_constant_better_cases"] = int((a < b - 1e-12).sum())
            record[scope + "_ties"] = int((abs(a - b) <= 1e-12).sum())
            record[scope + "_median_case_improvement_ppm"] = (
                float((a - b).median()) if valid.any() else None
            )
        aggregate.append(record)
    pd.DataFrame(aggregate).to_csv(OUT / "by_mode_standard_diagnostic.csv", index=False)
    assert hashes == input_hashes(datasets)
    receipt = dict(
        all_passed=True,
        datasets=33,
        dataset_modes=132,
        known_spike_LOO_predictions=len(private),
        manual_comparison_tables_opened=0,
        endogenous_annotation_outputs=0,
        annotation_classifier_fits=0,
        annotation_changes=False,
        thresholds_selected=False,
        sources_unchanged=True,
        source_input_sha256=hashes,
        note="Read-only known-spike mass trend diagnostic, separate from048. Scalar/vector coefficient/prediction parity plus original feature ppm/declared actual mass validation. Censored<3ppm anchors; grouped-neutral median approximates channels; RT/adduct confounding possible. LOO is on other known spikes,not endogenous identification accuracy; no mass window/candidate/call altered. Outer-only cache cannot enter future INNER predictions.",
    )
    dump(OUT / "receipt.json", receipt)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
