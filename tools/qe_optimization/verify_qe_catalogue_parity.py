"""Read-only native chemical-reference parity; NOT annotation selections."""

from pathlib import Path
import hashlib
import json
import shutil
import sys

import pandas as pd

from context import REPO, WORKSPACE as ROOT

QE = ROOT / "outputs/qe_optimization_20261022"
OUT = QE / "native_catalogue_parity_v2"
sys.path.insert(0, str(REPO / "app"))
from matlab_annotation.catalogue_targets import catalogue_missing_targets
from matlab_annotation.standard_rt_library import ion_key


def main():
    OUT.mkdir(exist_ok=False)
    inputs = []
    references = {s: [] for s in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")}
    for path in sorted((QE / "baseline/runs").glob("*/full_annotation.pkl")):
        result = pd.read_pickle(path)
        for stream, payload in result["streams"].items():
            reference = payload["reference"][["name", "adduct", "ref_mz"]].copy()
            reference = reference[reference.ref_mz.gt(0) & reference.ref_mz.notna()]
            references[stream].append(reference)
        with path.open("rb") as handle:
            inputs.append(
                dict(
                    dataset=path.parent.name,
                    sha256=hashlib.file_digest(handle, "sha256").hexdigest(),
                )
            )
    assert len(inputs) == 33
    envelope_path = QE / "experiment_003/fold_0_composition_envelope.json"
    envelope = json.loads(envelope_path.read_text())["envelope"]
    rows = []
    for stream, frames in references.items():
        old = pd.concat(frames, ignore_index=True).drop_duplicates()
        old["ion"] = old.adduct.map(ion_key)
        empty = pd.DataFrame(columns=["name", "adduct"])
        std = pd.DataFrame(columns=["adduct", "ref_mz", "internal_standard_quantity"])
        native, _ = catalogue_missing_targets(
            empty,
            REPO / "reference_data/matlab_pipeline",
            envelope,
            stream=stream,
            declared_standards=std,
        )
        native["ion"] = native.adduct.map(ion_key)
        merged = old.merge(
            native[["name", "ion", "ref_mz"]],
            on=["name", "ion"],
            suffixes=("_old", "_native"),
            validate="many_to_one",
        )
        difference = (merged.ref_mz_old - merged.ref_mz_native).abs()
        conflicts = merged[difference.gt(1e-6)]
        rows.append(
            dict(
                stream=stream,
                surviving_baseline_chemical_pairs=len(old),
                native_targets=len(native),
                overlapping_pairs=len(merged),
                mass_conflicts_above_1_microDa=len(conflicts),
                max_absolute_mass_difference_Da=float(difference.max())
                if len(difference)
                else None,
            )
        )
        conflicts.to_csv(OUT / (stream + "_private_mass_conflicts.csv"), index=False)
    summary = dict(
        datasets=33,
        run_kind="Read-only chemical target parity, no candidates/annotation selections changed",
        tolerance_Da=1e-6,
        by_mode=rows,
        total_overlaps=sum(r["overlapping_pairs"] for r in rows),
        total_mass_conflicts=sum(r["mass_conflicts_above_1_microDa"] for r in rows),
        note="Uses frozen fold0 envelope only as enumeration bounds. Not held-out annotation accuracy or proof of all chemical identities. Missing references/rare classes remain explicit.",
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                REPO / "app/matlab_annotation/catalogue_targets.py",
                envelope_path,
                REPO / "reference_data/matlab_pipeline/DataBase/lipids.tsv",
            ]
        },
        baseline_input_sha256=inputs,
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), OUT / "runner_source.py")
    print(pd.DataFrame(rows).to_string(index=False))
    assert summary["total_overlaps"] > 0 and summary["total_mass_conflicts"] == 0, (
        "Investigate native chemistry parity before using fallback"
    )


if __name__ == "__main__":
    main()
