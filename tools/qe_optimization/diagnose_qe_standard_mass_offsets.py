"""Read-only diagnostic: standards-only offsets vs existing discovery gaps.

No candidate admissions or annotations are changed. Raw feature IDs/private
diagnostics remain outside Git. Models saved BEFORE loading evaluation gaps.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from context import REPO, WORKSPACE as ROOT

QE = ROOT / "outputs/qe_optimization_20261022"
OUT = QE / "standard_mass_offset_diagnostic_v1"
sys.path[:0] = [str(REPO / "app"), str(Path(__file__).resolve().parent)]
from matlab_annotation.chemical_targets import compile_additional_targets
from run_qe_atlas_experiment import species, adduct


def main():
    OUT.mkdir(exist_ok=False)
    rows = []
    for path in sorted(
        (QE / "experiment_009/runs").glob("*/automatic_annotations.csv")
    ):
        a = pd.read_csv(path, low_memory=False)
        a = a[a.is_is.fillna(False).astype(bool)].copy()
        a["ppm"] = (a.feat_mz / a.ref_mz - 1) * 1e6
        for stream, g in a.groupby("stream"):
            # Each distinct chemical standard mass votes once, not each XCMS
            # split feature; this pilot does not fit any slope/RT dependence.
            p = g.groupby("ref_mz").ppm.median()
            center = float(p.median())
            deviation = (p - center).abs()
            mad = float(deviation.median())
            p90 = float(deviation.quantile(0.9))
            rows.append(
                dict(
                    dataset=path.parent.name,
                    stream=stream,
                    n=len(p),
                    offset_ppm=center,
                    mad_ppm=mad,
                    p90_ppm=p90,
                    enabled=len(p) >= 6
                    and mad <= 0.6
                    and p90 <= 1.5
                    and abs(center) <= 2,
                )
            )
    model = pd.DataFrame(rows)
    model.to_csv(OUT / "blind_standards_only_offsets.csv", index=False)
    # Diagnostic labels begin here; they cannot change the frozen offset model.
    gaps = pd.read_csv(
        QE / "experiment_007/full_comparison/discovery_diagnostics.csv",
        dtype={"feature_id": str},
    )
    gaps = gaps[gaps.reason.eq("mass_window")].copy()
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    output = []
    for dataset, part in gaps.groupby("dataset"):
        result = pd.read_pickle(QE / "baseline/runs" / dataset / "full_annotation.pkl")
        envelope = json.loads(
            (
                QE / f"experiment_003/fold_{folds[dataset]}_composition_envelope.json"
            ).read_text()
        )["envelope"]
        for stream, g in part.groupby("stream"):
            payload = result["streams"][stream]
            generated, _, _ = compile_additional_targets(
                payload["reference"], envelope, stream=stream
            )
            reference = pd.concat([payload["reference"], generated], ignore_index=True)
            reference["species_key"] = reference.name.map(species)
            reference["adduct_key"] = reference.adduct.map(adduct)
            groups = {k: v for k, v in reference.groupby(["species_key", "adduct_key"])}
            features = payload["features"].set_index(
                payload["features"].feature_id.astype(str)
            )
            m = model[model.dataset.eq(dataset) & model.stream.eq(stream)].iloc[0]
            offset = m.offset_ppm if m.enabled else 0.0
            for r in g.to_dict("records"):
                targets = groups[(r["species"], r["adduct"])]
                observed = float(features.loc[r["feature_id"], "mz"])
                theory = float(
                    targets.loc[
                        ((targets.ref_mz - observed) / targets.ref_mz).abs().idxmin(),
                        "ref_mz",
                    ]
                )
                before = (observed / theory - 1) * 1e6
                after = (observed / (1 + offset * 1e-6) / theory - 1) * 1e6
                output.append(
                    dict(
                        **r,
                        standard_offset_enabled=bool(m.enabled),
                        offset_ppm=float(offset),
                        signed_ppm_before=before,
                        signed_ppm_after=after,
                        now_inside3ppm=abs(after) < 3,
                        also_existing_atlas_RT_within10=abs(
                            r.get("atlas_RT_error_seconds", np.nan)
                        )
                        <= 10,
                    )
                )
    frame = pd.DataFrame(output)
    frame.to_csv(OUT / "private_mass_gap_diagnostics.csv", index=False)
    summary = frame.groupby("stream").agg(
        mass_gaps=("feature_id", "size"),
        inside3ppm_after_offset=("now_inside3ppm", "sum"),
    )
    frame["inside_mass_and_existing_atlas_RT"] = (
        frame.now_inside3ppm & frame.also_existing_atlas_RT_within10
    )
    summary["inside_mass_and_existing_atlas_RT"] = frame.groupby(
        "stream"
    ).inside_mass_and_existing_atlas_RT.sum()
    summary.to_csv(OUT / "potential_by_mode.csv")
    print(summary.to_string())
    print("WINDOW POTENTIAL ONLY, NOT NEW CORRECT ANNOTATIONS", flush=True)


if __name__ == "__main__":
    main()
