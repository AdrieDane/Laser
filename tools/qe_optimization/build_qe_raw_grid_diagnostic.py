"""RT-only raw mass-series alignment diagnostic; NO annotation selections."""

from __future__ import annotations

import hashlib
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE as ROOT, REPO

BASE = ROOT / "outputs/qe_optimization_20261022/baseline"
PREVIOUS = ROOT / "outputs/qe_optimization_20261022/experiment_007"
ATLAS = ROOT / "outputs/qe_optimization_20261022/rt_atlas_development"
OUT = ROOT / "outputs/qe_optimization_20261022/raw_grid_development_v1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qe_atlas_experiment import species, adduct


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(parents=True, exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        hashes = {
            p.relative_to(snapshot).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        dump(
            OUT / "protocol.json",
            dict(
                name="RT-ONLY-raw-mass-series-grid-v1",
                annotation_selections=False,
                source_snapshot_sha256=hashes,
                max_search_seconds=90,
                anchor_ppm=1.5,
                support=">=8 OTHER compositions, Cspan>=4, >=2DB; training prevalence>=25% and >=3 groups; ambiguity withheld; declared-spike masses excluded; new source requires >=5s shift.",
                validation="Same five held-out project groups; frozen other-group RT atlases and composition prevalence. No manual RT/runtime curation inputs. Manual comparisons occur ONLY AFTER raw-data alignment predictions. Not annotation accuracy or external validation. Inherited baseline has prior development exposure.",
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, expected in protocol["source_snapshot_sha256"].items():
        assert (
            hashlib.sha256((OUT / "source_snapshot" / name).read_bytes()).hexdigest()
            == expected
        ), name
    sys.path.insert(0, str(OUT / "source_snapshot/app"))


def main(only_dataset=None, *, summarize=True):
    initialize()
    from matlab_annotation.raw_grid_alignment import infer_raw_grid_alignment
    from matlab_annotation.retention_atlas import predict_aligned_retention_atlas
    from matlab_annotation.chemical_targets import compile_additional_targets
    from matlab_annotation.declared_standards import declared_standard_targets

    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    rows = []
    folder = OUT / "predictions"
    folder.mkdir(exist_ok=True)
    for dataset, fold in sorted(folds.items()):
        if only_dataset is not None and dataset != only_dataset:
            continue
        final = OUT / (dataset + ".pkl")
        if final.exists():
            rows.append(pd.read_pickle(final))
            continue
        result = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
        payload = json.loads((ATLAS / f"fold_{fold}_atlas.json").read_text())
        assert dataset[:7] not in payload["training_groups"]
        atlas = payload["atlas"]
        prevalence = json.loads(
            (
                ROOT
                / f"outputs/qe_optimization_20261022/experiment_001/fold_{fold}_selection_model.json"
            ).read_text()
        )
        assert dataset[:7] not in prevalence["training_cohorts"]
        envelope = json.loads(
            (
                ROOT
                / f"outputs/qe_optimization_20261022/experiment_003/fold_{fold}_composition_envelope.json"
            ).read_text()
        )["envelope"]
        catalog = declared_standard_targets(
            OUT / "source_snapshot/reference_data/matlab_pipeline",
            result["internal_standards_expected"],
        )
        predictions = []
        for stream, part in result["streams"].items():
            spikes = catalog[catalog.stream.eq(stream)]
            generated, ranges, _ = compile_additional_targets(
                part["reference"], envelope, stream=stream, declared_standards=spikes
            )
            targets = (
                pd.concat([part["reference"], generated], ignore_index=True)
                .drop_duplicates(["name", "adduct"])
                .copy()
            )
            allowed = [
                ranges.decide(
                    r["class"], r["carbon"], r["db"], species=r["name"]
                ).allowed
                for r in targets.to_dict("records")
            ]
            targets = targets.loc[allowed].reset_index(drop=True)
            predicted = predict_aligned_retention_atlas(
                targets, part["annotations"], atlas, stream=stream
            )
            targets = targets.join(predicted)
            shift = infer_raw_grid_alignment(
                targets,
                part["features"],
                prevalence["composition_priors"],
                stream=stream,
                declared_standard_masses=spikes.ref_mz,
            )
            targets = targets.join(shift).assign(stream=stream)
            predictions.append(targets)
        predictions = pd.concat(predictions, ignore_index=True)
        # Freeze blind predictions BEFORE loading this dataset's evaluation labels.
        predictions.to_pickle(folder / (dataset + ".pkl"))
        manual = pd.read_pickle(
            PREVIOUS / "evaluation_v2" / dataset / "manual_comparison.pkl"
        )
        manual = manual[~manual.internal_standard]
        pool = pd.read_pickle(
            PREVIOUS / "prepared" / f"fold_{fold}" / (dataset + ".pkl")
        )["pool"]
        previous = {
            (s, species(n), adduct(a), str(f))
            for s, n, a, f in pool[
                ["stream", "name", "adduct", "feat_feature_id"]
            ].itertuples(index=False, name=None)
        }
        predictions["species"] = predictions.name.map(species)
        predictions["ion"] = predictions.adduct.map(adduct)
        lookup = {
            k: g
            for k, g in predictions.groupby(["stream", "species", "ion"], sort=False)
        }
        features = {
            s: p["features"].set_index(p["features"].feature_id.astype(str))
            for s, p in result["streams"].items()
        }
        detail = []
        for r in manual.itertuples():
            data = dict(
                dataset=dataset,
                stream=r.stream,
                lipid_class=r.lipid_class,
                species=r.species,
                adduct=r.adduct,
                feature_id=r.feature_id,
                mapped=r.mapped,
                previous_correct=r.same_manual_peak,
                previous_exact_candidate=(r.stream, r.species, r.adduct, r.feature_id)
                in previous,
                raw_grid_prediction_available=False,
                strict_new_RT_window=False,
                newly_inside_window=False,
            )
            target = lookup.get((r.stream, r.species, r.adduct))
            if r.mapped and target is not None and len(target) == 1:
                t = target.iloc[0]
                f = features[r.stream].loc[r.feature_id]
                ppm = abs((float(f.mz) - float(t.ref_mz)) / float(t.ref_mz) * 1e6)
                err = float(f.rt) * 60 - float(t.raw_grid_rt_seconds)
                data.update(
                    raw_grid_prediction_available=np.isfinite(err),
                    mass_error_ppm=ppm,
                    new_RT_error_seconds=err,
                    original_atlas_error_seconds=float(f.rt) * 60
                    - float(t.atlas_rt_seconds),
                    raw_grid_shift_seconds=t.raw_grid_shift_seconds,
                    raw_grid_status=t.raw_grid_status,
                    raw_grid_support=t.raw_grid_support,
                    strict_new_RT_window=bool(
                        np.isfinite(err) and abs(err) <= 10 and ppm < 3
                    ),
                )
                data["newly_inside_window"] = (
                    data["strict_new_RT_window"]
                    and not data["previous_exact_candidate"]
                )
            detail.append(data)
        detail = pd.DataFrame(detail)
        detail.to_pickle(final)
        rows.append(detail)
        print(
            "RT ONLY RAW-GRID",
            dataset,
            "new strict-window POTENTIAL",
            int(detail.newly_inside_window.sum()),
            "NOT selected annotations",
            flush=True,
        )
    if not summarize:
        return
    summarize_outputs()


def summarize_outputs():
    folds = json.loads((ATLAS / "fold_assignment.json").read_text())
    combined = pd.concat(
        [pd.read_pickle(OUT / (d + ".pkl")) for d in folds], ignore_index=True
    )
    combined.to_csv(OUT / "RT_window_diagnostics.csv", index=False)
    summary = combined.groupby(["stream", "lipid_class"]).agg(
        manual=("mapped", "size"),
        predicted=("raw_grid_prediction_available", "sum"),
        within_new_strict_window=("strict_new_RT_window", "sum"),
        new_window_potential=("newly_inside_window", "sum"),
    )
    summary.to_csv(OUT / "RT_window_by_class.csv")
    print(
        "RT DIAGNOSTIC ONLY\n" + summary.groupby("stream").sum().to_string(), flush=True
    )


def one_dataset(dataset):
    main(dataset, summarize=False)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    if args.workers == 1:
        main()
    else:
        initialize()
        shutil.copy2(Path(__file__), OUT / "runner_source_parallel.py")
        projects = sorted(json.loads((ATLAS / "fold_assignment.json").read_text()))
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            for completed in executor.map(one_dataset, projects):
                print("CHECKPOINT RT-ONLY", completed, flush=True)
        summarize_outputs()
