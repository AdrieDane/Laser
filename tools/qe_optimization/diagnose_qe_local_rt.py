"""RT-only diagnostics: no annotation selections, no extras/accuracy claims."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys

import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ATLAS / "diagnostics_local_surfaces_v1"
sys.path.insert(0, str(REPO / "app"))
from matlab_annotation.local_rt_surface import predict_local_anchor_surfaces
from matlab_annotation.retention_atlas import predict_aligned_retention_atlas


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    snapshot = OUT / "sources"
    snapshot.mkdir()
    hashes = {}
    for name in ["local_rt_surface.py", "retention_atlas.py"]:
        source = REPO / "app/matlab_annotation" / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, snapshot / name)
    observations = pd.read_pickle(ATLAS / "observations.pkl")
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    results = []
    for dataset, fold in folds.items():
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        frame = observations[observations.dataset.eq(dataset)].reset_index(drop=True)
        atlas = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())["atlas"]
        for stream, group in frame.groupby("stream"):
            targets = group[["class", "carbon", "db"]]
            automatic = result["streams"][stream]["annotations"]
            local = predict_local_anchor_surfaces(targets, automatic)
            existing = predict_aligned_retention_atlas(
                targets, automatic, atlas, stream=stream
            )
            group = group.join(local)
            group["atlas_error_seconds"] = existing.atlas_rt_seconds - group.rt_seconds
            group["local_surface_error_seconds"] = (
                group.local_surface_rt_seconds - group.rt_seconds
            )
            results.append(group)
        print("RT DIAGNOSTIC ONLY", dataset, flush=True)
    full = pd.concat(results, ignore_index=True)
    full.to_csv(OUT / "RT_predictions.csv", index=False)
    full.to_pickle(OUT / "RT_predictions.pkl")
    for keys in [("stream",), ("dataset", "stream"), ("stream", "class")]:
        rows = []
        for key, g in full.groupby(list(keys)):
            if not isinstance(key, tuple):
                key = (key,)
            local = g.local_surface_error_seconds
            rows.append(
                dict(zip(keys, key))
                | dict(
                    entries=len(g),
                    local_available=int(local.notna().sum()),
                    atlas_within10=int(g.atlas_error_seconds.abs().le(10).sum()),
                    local_within10=int(local.abs().le(10).sum()),
                    either_within10=int(
                        (g.atlas_error_seconds.abs().le(10) | local.abs().le(10)).sum()
                    ),
                    local_median_absolute_seconds=float(local.abs().median()),
                )
            )
        table = pd.DataFrame(rows)
        table.to_csv(OUT / ("by_" + "_".join(keys) + ".csv"), index=False)
        if keys == ("stream",):
            print(table.to_string(index=False), flush=True)
    note = dict(
        sources_sha256=hashes,
        datasets=33,
        method="Fixed local RT plane from >=8 OTHER distinct strong automatic anchor compositions; target cannot calibrate itself. Analytic LOO diagnostics describe final trimmed anchor set, not external validation.",
        annotation_outputs_generated=False,
        note="RT-only development diagnostics, NOT annotation accuracy or candidate selection. Curated RTs are read only AFTER independent automatic-anchor prediction. No production behavior changed.",
    )
    (OUT / "provenance.json").write_text(json.dumps(note, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
