"""Fresh, frozen-source runs on the active collection; no curation inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

from context import REPO, SOURCE

STREAMS = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_one(job):
    dataset, out_text, snapshot_text, experimental = job
    out = Path(out_text)
    snapshot = Path(snapshot_text)
    target = out / "runs" / dataset
    target.mkdir(parents=True, exist_ok=True)
    if (target / "complete.json").exists():
        return json.loads((target / "complete.json").read_text())
    sys.path.insert(0, str(snapshot / "app"))
    from matlab_annotation.engine import run_matlab_annotation
    from matlab_annotation.automatic import AutomaticAnchorSettings

    start = time.monotonic()
    print("START", dataset, flush=True)
    try:
        bucket = -1
        with (
            (target / "run.log").open("a", encoding="utf-8") as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):

            def progress(fraction, message):
                nonlocal bucket
                new = int(fraction * 8)
                if new > bucket:
                    bucket = new
                    print(f"{fraction:.1%} {message}", flush=True)

            settings = AutomaticAnchorSettings(
                use_experimental_pe_selector=experimental
            )
            result = run_matlab_annotation(
                out / "inputs" / dataset,
                phases=("RP", "NP"),
                anchor_settings=settings,
                progress=progress,
            )
            annotations = []
            candidates = []
            for stream, payload in result["streams"].items():
                a = payload["annotations"].copy()
                assert not a["feat_feature_id"].astype(str).duplicated().any(), (
                    dataset,
                    stream,
                )
                annotations.append(a.assign(stream=stream))
                candidates.append(payload["all_pairs"].assign(stream=stream))
            import pandas as pd

            combined = pd.concat(annotations, ignore_index=True)
            combined.to_csv(target / "automatic_annotations.csv", index=False)
            pd.concat(candidates, ignore_index=True).to_pickle(
                target / "candidates.pkl"
            )
            for key in (
                "matlab_pe_candidate_decisions",
                "matlab_pe_selection_audit",
                "matlab_post_regression_audit",
                "matlab_anchor_audit",
                "matlab_compound_outcomes",
            ):
                if isinstance(result.get(key), pd.DataFrame):
                    result[key].to_csv(target / (key + ".csv"), index=False)
            temp = target / "full_annotation.partial.pkl"
            with temp.open("wb") as handle:
                pickle.dump(result, handle, pickle.HIGHEST_PROTOCOL)
            temp.rename(target / "full_annotation.pkl")
            record = dict(
                dataset=dataset,
                status="complete",
                seconds=time.monotonic() - start,
                counts={s: len(p["annotations"]) for s, p in result["streams"].items()},
                input_dir=str(out / "inputs" / dataset),
                source_snapshot=str(snapshot),
                automatic_settings=asdict(settings),
                fresh_raw_input_run=True,
                curation_available_to_annotation=False,
            )
            dump(target / "complete.json", record)
        print(
            "DONE",
            dataset,
            record["counts"],
            round(record["seconds"], 1),
            "seconds",
            flush=True,
        )
        return record
    except Exception:
        record = dict(
            dataset=dataset,
            status="failed",
            seconds=time.monotonic() - start,
            error=traceback.format_exc(),
        )
        dump(target / "error.json", record)
        print("FAILED", dataset, record["error"], flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--experimental-pe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    projects = sorted(p for p in SOURCE.iterdir() if p.is_dir())
    assert len(projects) == 33, (
        len(projects),
        "The active collection changed; reinventory before running",
    )
    if args.resume:
        manifest = json.loads((out / "run_manifest.json").read_text())
        assert manifest["datasets"] == [p.name for p in projects]
        assert manifest["experimental_pe"] == args.experimental_pe
        for f in manifest["source_files"]:
            assert digest(SOURCE / f["relative"]) == f["sha256"], f["relative"]
    else:
        out.mkdir(parents=True, exist_ok=False)
        snapshot = out / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "reference_data", "methods"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        frozen = {
            p.relative_to(snapshot).as_posix(): digest(p)
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        files = []
        print("Verifying source collection and preparing raw-only inputs", flush=True)
        for project in projects:
            collection = json.loads(
                (project / "collection_manifest.json").read_text(encoding="utf-8-sig")
            )
            for item in collection["files"]:
                path = project / item["name"]
                assert path.is_file() and path.stat().st_size == item["size"], path
                sha = digest(path)
                assert sha == item["sha256"], path
                files.append(
                    dict(
                        relative=path.relative_to(SOURCE).as_posix(),
                        sha256=sha,
                        size=item["size"],
                    )
                )
            staged = out / "inputs" / project.name
            staged.mkdir(parents=True)
            allowed = {f"peakgrouplist_{s}.txt" for s in STREAMS} | {
                "internal_standards.tsv",
                "group_definition.tsv",
            }
            allowed.update(p.name for p in project.glob("*_RP.xlsx"))
            allowed.update(p.name for p in project.glob("*_NP.xlsx"))
            allowed.update(p.name for p in project.glob("*_RP.csv"))
            allowed.update(p.name for p in project.glob("*_NP.csv"))
            for name in sorted(allowed):
                path = project / name
                if path.is_file():
                    (staged / name).symlink_to(path)
            assert not (staged / "MS_data.xlsx").exists()
            assert not list(staged.glob("*.mat"))
        manifest = dict(
            created_at=datetime.now(timezone.utc).isoformat(),
            source_root=str(SOURCE),
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            datasets=[p.name for p in projects],
            source_files=files,
            snapshot_files=frozen,
            experimental_pe=args.experimental_pe,
            fresh_raw_input_runs=True,
            curation_available_to_annotation=False,
            note="Only original peak lists, standards and sample metadata are staged. No MS_data.xlsx, MAT files, per-project lipid definitions or curator scripts are exposed to annotation.",
        )
        dump(out / "run_manifest.json", manifest)
    snapshot = out / "source_snapshot"
    for relative, sha in manifest["snapshot_files"].items():
        assert digest(snapshot / relative) == sha, relative
    records = []
    jobs = [(p.name, str(out), str(snapshot), args.experimental_pe) for p in projects]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            dump(out / "progress.json", records)
            print("COLLECTION PROGRESS", len(records), "/", len(jobs), flush=True)
    for item in manifest["source_files"]:
        assert digest(SOURCE / item["relative"]) == item["sha256"], item["relative"]
    dump(
        out / "run_summary.json",
        dict(
            results=records,
            all_sources_hashes_unchanged=True,
            complete=sum(r["status"] == "complete" for r in records),
            failed=sum(r["status"] == "failed" for r in records),
        ),
    )
    print("ALL RUNS FINISHED", len(records), flush=True)


if __name__ == "__main__":
    main()
