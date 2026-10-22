"""Grouped standard-RT library development; NO annotation selections."""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO, SOURCE

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/standard_rt_development"
REF = ROOT / "outputs/qe_optimization_20261022/manual_reference_v2"
sys.path[:0] = [str(REPO), str(REPO / "app"), str(Path(__file__).resolve().parent)]
from tools.dataset_scanner.core import curated_rows
from evaluate_qe_baseline import verified_reference
from matlab_annotation.declared_standards import declared_standard_targets
from matlab_annotation.standard_rt_library import (
    fit_standard_retention_library,
    predict_standard_retention,
    ion_key,
    nominal_standard_name,
)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    snapshot = OUT / "sources"
    snapshot.mkdir()
    hashes = {}
    for name in [
        "declared_standards.py",
        "standard_rt_library.py",
        "retention_atlas.py",
    ]:
        source = REPO / "app/matlab_annotation" / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, snapshot / name)
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    observations = []
    manual_rows = []
    catalogs = {}
    automatic = {}
    for dataset in sorted(folds):
        cross = verified_reference(dataset, REF)
        rows, _ = curated_rows(SOURCE / dataset / "MS_data.xlsx")
        lookup = {(r["sheet"], r["excel_row"]): r for r in rows}
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        targets = declared_standard_targets(
            BASE / "source_snapshot/reference_data/matlab_pipeline",
            result["internal_standards_expected"],
        )
        targets["adduct_key"] = targets.adduct.map(ion_key)
        catalog = OUT / "declared_targets"
        catalog.mkdir(exist_ok=True)
        targets.to_pickle(catalog / (dataset + ".pkl"))
        catalogs[dataset] = targets
        automatic[dataset] = {s: p["annotations"] for s, p in result["streams"].items()}
        bykey = {
            k: g
            for k, g in targets.groupby(["stream", "name", "adduct_key"], sort=False)
        }
        comp = pd.read_pickle(
            BASE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        ).set_index("source_row")
        for index, r in cross[cross.matlab_is_internal_standard].iterrows():
            original = lookup[(r.source_sheet, r.source_excel_row)]
            label = r.matlab_species
            available = bykey.get(
                (r.stream, nominal_standard_name(label), ion_key(r.matlab_adduct))
            )
            selected = None
            if available is not None:
                delta = (available.ref_mz - original["mz"]).abs()
                best = delta.idxmin()
                if (
                    len(available) == 1
                    or int((delta / available.ref_mz * 1e6 < 3).sum()) == 1
                ):
                    selected = available.loc[best]
            record = dict(
                dataset=dataset,
                cohort=dataset[:7],
                stream=r.stream,
                name=label,
                lipid_class=r.matlab_class,
                adduct=ion_key(r.matlab_adduct),
                rt_seconds=original["rt_seconds"],
                mapped=r.crosswalk_status == "profile_match",
                before_same_peak=bool(comp.loc[index + 1, "same_manual_peak"]),
                standard_compound_id=selected.standard_compound_id
                if selected is not None
                else None,
                reporting_name=label,
                reporting_class=r.matlab_class,
            )
            manual_rows.append(record)
            if selected is not None and record["mapped"]:
                observations.append(record)
        print("STANDARD TRAIN/EVAL REFERENCE", dataset, flush=True)
    obs = pd.DataFrame(observations)
    obs.to_pickle(OUT / "observations.pkl")
    manual = pd.DataFrame(manual_rows)
    manual.to_pickle(OUT / "manual_standard_entries.pkl")
    libraries = {}
    for fold in range(5):
        train = obs[obs.dataset.map(folds).ne(fold)]
        test_groups = {d[:7] for d, f in folds.items() if f == fold}
        assert not set(train.cohort) & test_groups
        libraries[fold] = fit_standard_retention_library(train)
        payload = dict(
            fold=fold,
            training_groups=sorted(train.cohort.unique()),
            held_out_groups=sorted(test_groups),
            library=libraries[fold],
        )
        (OUT / f"fold_{fold}_standard_library.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    diagnostics = []
    for dataset, fold in folds.items():
        targets = catalogs[dataset]
        predicted = []
        for stream, group in targets.groupby("stream"):
            prediction = predict_standard_retention(
                group, automatic[dataset][stream], libraries[fold], stream=stream
            )
            predicted.append(group.join(prediction))
        predicted = pd.concat(predicted, ignore_index=True)
        bykey = {
            k: g
            for k, g in predicted.groupby(["stream", "name", "adduct_key"], sort=False)
        }
        for row in manual[manual.dataset.eq(dataset)].to_dict("records"):
            g = bykey.get(
                (row["stream"], nominal_standard_name(row["name"]), row["adduct"])
            )
            if g is not None and row["standard_compound_id"] is not None:
                g = g[g.standard_compound_id.eq(row["standard_compound_id"])]
            p = g.iloc[0] if g is not None and len(g) == 1 else None
            rt = float(p.standard_rt_seconds) if p is not None else np.nan
            diagnostics.append(
                {
                    **row,
                    "predicted_rt_seconds": rt,
                    "RT_error_seconds": rt - row["rt_seconds"],
                    "adduct_supported": bool(p.standard_adduct_supported)
                    if p is not None
                    else False,
                }
            )
        print("STANDARD RT DIAGNOSTIC ONLY", dataset, flush=True)
    details = pd.DataFrame(diagnostics)
    details.to_csv(OUT / "held_out_standard_RT_diagnostics.csv", index=False)
    summaries = []
    for (stream, correct), g in details.groupby(["stream", "before_same_peak"]):
        summaries.append(
            dict(
                stream=stream,
                already_correct=correct,
                manual=len(g),
                available=int(g.predicted_rt_seconds.notna().sum()),
                supported_within10=int(
                    (g.RT_error_seconds.abs().le(10) & g.adduct_supported).sum()
                ),
            )
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "diagnostic_by_mode.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    (OUT / "provenance.json").write_text(
        json.dumps(
            dict(
                sources_sha256=hashes,
                datasets=33,
                annotation_outputs_generated=False,
                note="RT-only diagnosis. Each group is excluded from standard RT/adduct-prior fits. Runtime prediction uses declared chemical masses and OTHER automatic standards; not curated test RTs. Inherited baseline has prior development exposure.",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
