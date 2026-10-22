"""Read-only missing-feature discovery diagnosis, not alternative annotations."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
EXP = ROOT / "outputs/qe_optimization_20261022/experiment_007"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
sys.path[:0] = [str(EXP / "source_snapshot/app"), str(Path(__file__).resolve().parent)]
from matlab_annotation.retention_atlas import predict_aligned_retention_atlas
from matlab_annotation.chemical_targets import compile_additional_targets
from run_qe_atlas_experiment import species, adduct


def main():
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    manual = pd.read_pickle(EXP / "reports_v2/all_manual_comparison.pkl")
    rows = []
    for dataset, fold in folds.items():
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        pool = pd.read_pickle(EXP / "prepared" / f"fold_{fold}" / (dataset + ".pkl"))[
            "pool"
        ]
        exact = {
            (s, species(n), adduct(a), str(f))
            for s, n, a, f in pool[
                ["stream", "name", "adduct", "feat_feature_id"]
            ].itertuples(index=False, name=None)
        }
        atlas = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())["atlas"]
        envelopes = json.loads(
            (
                ROOT
                / f"outputs/qe_optimization_20261022/experiment_003/fold_{fold}_composition_envelope.json"
            ).read_text()
        )["envelope"]
        for stream, group in manual[
            manual.dataset.eq(dataset) & ~manual.same_manual_peak
        ].groupby("stream"):
            part = result["streams"][stream]
            generated, ranges, _ = compile_additional_targets(
                part["reference"], envelopes, stream=stream
            )
            reference = pd.concat([part["reference"], generated], ignore_index=True)
            reference["species"] = reference.name.map(species)
            reference["adduct_key"] = reference.adduct.map(adduct)
            needed = {(r.species, r.adduct) for r in group.itertuples()}
            reference = reference.loc[
                [
                    k in needed
                    for k in reference[["species", "adduct_key"]].itertuples(
                        index=False, name=None
                    )
                ]
            ].copy()
            refgroups = {
                k: g
                for k, g in reference.groupby(["species", "adduct_key"], sort=False)
            }
            observed = part["features"].set_index(
                part["features"].feature_id.astype(str)
            )
            pairs = part["all_pairs"][
                ["name", "adduct", "feat_feature_id", "ppm_error", "rt_error"]
            ].copy()
            pairs["species"] = pairs.name.map(species)
            pairs["adduct_key"] = pairs.adduct.map(adduct)
            pairs["fid"] = pairs.feat_feature_id.astype(str)
            pg = {
                k: g
                for k, g in pairs.groupby(["species", "adduct_key", "fid"], sort=False)
            }
            predictions = predict_aligned_retention_atlas(
                reference, part["annotations"], atlas, stream=stream
            )
            for r in group.itertuples():
                data = dict(
                    dataset=dataset,
                    stream=stream,
                    lipid_class=r.lipid_class,
                    species=r.species,
                    adduct=r.adduct,
                    feature_id=r.feature_id,
                )
                if not r.mapped:
                    reason = "manual_feature_unresolved"
                elif r.internal_standard:
                    reason = "internal_standard"
                elif (stream, r.species, r.adduct, r.feature_id) in exact:
                    reason = "exact_candidate_already_available"
                else:
                    composition = re.search(
                        r"\((?:[dtm]|[OP]-|O\+P-)?(\d+):(\d+)\)", r.species
                    )
                    decision = (
                        ranges.decide(
                            r.lipid_class,
                            *map(int, composition.groups()),
                            species=r.species,
                        )
                        if composition
                        else None
                    )
                    ref = refgroups.get((r.species, r.adduct))
                    if decision is not None and not decision.allowed:
                        reason = "composition_limit"
                    elif ref is None:
                        reason = "identity_absent_from_theory_reference"
                    else:
                        feature = observed.loc[r.feature_id]
                        errors = (float(feature.mz) - ref.ref_mz) / ref.ref_mz * 1e6
                        index = errors.abs().idxmin()
                        data["abs_mass_error_ppm"] = float(errors.abs().min())
                        data["atlas_RT_error_seconds"] = float(feature.rt) * 60 - float(
                            predictions.loc[index, "atlas_rt_seconds"]
                        )
                        data["fitted_reference_RT_error_seconds"] = (
                            float(feature.rt) - float(ref.loc[index, "ref_rt"])
                        ) * 60
                        existing = pg.get((r.species, r.adduct, r.feature_id))
                        if existing is not None:
                            data["fitted_pair_RT_error_seconds"] = float(
                                existing.rt_error.abs().min() * 60
                            )
                            data["fitted_pair_mass_error_ppm"] = float(
                                existing.ppm_error.abs().min()
                            )
                        if data["abs_mass_error_ppm"] >= 3:
                            reason = "mass_window"
                        elif not np.isfinite(data["atlas_RT_error_seconds"]):
                            reason = "RT_atlas_unavailable"
                        elif abs(data["atlas_RT_error_seconds"]) > 10:
                            reason = "RT_prediction_window"
                        else:
                            reason = "other_discovery_or_protected_feature"
                rows.append({**data, "reason": reason})
        print("DISCOVERY DIAGNOSED", dataset, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(EXP / "full_comparison/discovery_diagnostics.csv", index=False)
    print(frame.groupby(["stream", "reason"]).size().to_string())
    gaps = frame[~frame.reason.eq("exact_candidate_already_available")]
    print(
        gaps.groupby(["stream", "lipid_class", "reason"])
        .size()
        .sort_values(ascending=False)
        .head(50)
        .to_string()
    )


if __name__ == "__main__":
    main()
