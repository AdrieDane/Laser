"""Fixed standard class-transfer RT diagnosis; no annotation selections."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
STD = ROOT / "outputs/qe_optimization_20261022/standard_rt_development_v3"
OUT = ROOT / "outputs/qe_optimization_20261022/standard_transfer_development"
sys.path.insert(0, str(REPO / "app"))
from matlab_annotation.retention_atlas import predict_aligned_retention_atlas
from matlab_annotation.standard_rt_library import (
    predict_standard_retention,
    ion_key,
    nominal_standard_name,
)
from matlab_annotation.standard_rt_transfer import (
    exclude_declared_standard_masses,
    standard_class_targets,
    fit_standard_class_transfer,
    predict_standard_class_transfer,
    combine_standard_rt_sources,
)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    source = OUT / "sources"
    source.mkdir()
    hashes = {}
    for name in [
        "standard_rt_transfer.py",
        "standard_rt_library.py",
        "retention_atlas.py",
    ]:
        p = REPO / "app/matlab_annotation" / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        shutil.copy2(p, source / name)
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    libraries = {
        f: json.loads((STD / f"fold_{f}_standard_library.json").read_text())["library"]
        for f in range(5)
    }
    atlases = {
        f: json.loads((ATLAS / f"fold_{f}_atlas.json").read_text())["atlas"]
        for f in range(5)
    }
    obs = pd.read_pickle(STD / "observations.pkl")
    records = {f: [] for f in range(5)}
    catalogs = {}
    automatic = {}
    for dataset in sorted(folds):
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        catalog = pd.read_pickle(STD / "declared_targets" / (dataset + ".pkl"))
        catalogs[dataset] = catalog
        automatic[dataset] = {s: p["annotations"] for s, p in result["streams"].items()}
        manual = obs[obs.dataset.eq(dataset)]
        for fold in range(5):
            for stream, group in catalog.groupby("stream"):
                targets = standard_class_targets(group, libraries[fold], stream=stream)
                clean = exclude_declared_standard_masses(
                    automatic[dataset][stream], group
                )
                prediction = predict_aligned_retention_atlas(
                    targets, clean, atlases[fold], stream=stream
                )
                targets = targets.join(prediction)
                lookup = {
                    s: g.iloc[0] for s, g in targets.groupby("standard_compound_id")
                }
                for row in manual[manual.stream.eq(stream)].itertuples():
                    target = lookup.get(row.standard_compound_id)
                    if target is None:
                        continue
                    records[fold].append(
                        dict(
                            dataset=dataset,
                            cohort=dataset[:7],
                            stream=stream,
                            standard_compound_id=row.standard_compound_id,
                            **{"class": target["class"]},
                            residual_seconds=row.rt_seconds
                            - float(target.atlas_rt_seconds),
                        )
                    )
        print("STANDARD TRANSFER REFERENCES", dataset, flush=True)
    transfers = {}
    for fold in range(5):
        frame = pd.DataFrame(records[fold])
        frame.to_pickle(OUT / f"fold_{fold}_reference_residuals.pkl")
        train = frame[frame.dataset.map(folds).ne(fold)]
        test = {d[:7] for d, f in folds.items() if f == fold}
        assert not set(train.cohort) & test
        transfers[fold] = fit_standard_class_transfer(train)
        payload = dict(
            transfer=transfers[fold],
            training_groups=sorted(train.cohort.unique()),
            held_out_groups=sorted(test),
        )
        (OUT / f"fold_{fold}_standard_transfer.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    manual = pd.read_pickle(STD / "manual_standard_entries.pkl")
    diagnostics = []
    for dataset, fold in folds.items():
        parts = []
        for stream, targets in catalogs[dataset].groupby("stream"):
            auto = automatic[dataset][stream]
            primary = predict_standard_retention(
                targets, auto, libraries[fold], stream=stream
            )
            transferred = predict_standard_class_transfer(
                targets, auto, atlases[fold], transfers[fold], stream=stream
            )
            parts.append(
                targets.join(combine_standard_rt_sources(primary, transferred))
            )
        predicted = pd.concat(parts, ignore_index=True)
        lookup = {k: g for k, g in predicted.groupby(["stream", "name", "adduct_key"])}
        for row in manual[manual.dataset.eq(dataset)].to_dict("records"):
            g = lookup.get(
                (
                    row["stream"],
                    nominal_standard_name(row["name"]),
                    ion_key(row["adduct"]),
                )
            )
            if g is not None:
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
                    "RT_reference_source": p.standard_rt_reference_source
                    if p is not None
                    else "unavailable",
                }
            )
        print("STANDARD TRANSFER RT DIAGNOSTIC ONLY", dataset, flush=True)
    full = pd.DataFrame(diagnostics)
    full.to_csv(OUT / "held_out_standard_RT_diagnostics.csv", index=False)
    summary = []
    for (stream, correct), g in full.groupby(["stream", "before_same_peak"]):
        summary.append(
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
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "diagnostic_by_mode.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    (OUT / "provenance.json").write_text(
        json.dumps(
            dict(
                sources_sha256=hashes,
                annotation_outputs_generated=False,
                note="RT-only diagnosis, not annotation accuracy. Each evaluated group is excluded from new models. Standard masses cannot calibrate their own class transfer. Existing baseline has prior development exposure.",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
