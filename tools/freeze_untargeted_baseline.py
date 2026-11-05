"""Freeze a private Stage 1 checkpoint and replay the actual browser selector.

No study paths or reference identities are embedded in this tool. Run outside the
repository's output tree; study snapshots and evaluation rows must not enter Git.
The worker loads frozen LASER code and receives feature keys, never gold labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time

MODES = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
BASELINE = "678ca0cab12e2e4a01b3197bb32e6b56cb202279"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def feature_id(value):
    value = str(value).strip()
    return value[:-2] if value.endswith(".0") else value


def ratio(numerator, denominator):
    return {"n": numerator, "N": denominator,
            "percent": 100 * numerator / denominator if denominator else None}


def project_files(project):
    """Current saved workflows and metadata; archives are not new acquisitions."""
    for item in sorted(Path(project).iterdir()):
        if item.name.startswith(".") or item.name == "Archive":
            continue
        if item.is_file() or item.name in ("XCMS", "Raw", "LASER", "Untargeted", "metadata", "Calibration"):
            files = [item] if item.is_file() else sorted(item.rglob("*"))
            for path in files:
                if path.is_symlink():
                    raise ValueError(f"Snapshot requires regular files, not symlinks: {path}")
                if path.is_file() and not path.name.startswith("."):
                    yield path
    # Preserve the producing job's request/log, without importing old job runs.
    manifest = Path(project) / "Untargeted/run_manifest.json"
    if manifest.is_file():
        job_input = json.loads(manifest.read_text()).get("inputs", {}).get("input_snapshot")
        if job_input:
            parts = Path(job_input).parts
            job = (Path(project).joinpath(*parts[parts.index(".laser"):-1])
                   if ".laser" in parts else Path(job_input).parent)
            if job.is_relative_to(Path(project)) and job.is_dir():
                for path in sorted(job.iterdir()):
                    if path.is_file() and path.suffix in (".json", ".log"):
                        yield path


def fingerprint(files, root):
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in files}


def reference_ledger(frame):
    """Keep every source row; mark duplicate/ambiguous metric units explicitly."""
    frame = frame.fillna("").copy()
    required = {"stream", "feature_id", "mapped", "name", "lipid_class", "adduct",
                "formula", "source_sheet", "source_excel_row"}
    if not required.issubset(frame):
        raise ValueError(f"Reference is missing columns: {sorted(required - set(frame))}")
    frame["feature_id"] = frame.feature_id.map(feature_id)
    frame["mapped"] = frame.mapped.map(lambda x: str(x).lower() in ("true", "1"))
    if not set(frame.stream).issubset(MODES):
        raise ValueError("Unexpected reference measurement mode")
    source_keys = list(zip(frame.stream, frame.source_sheet, frame.source_excel_row.astype(str)))
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Repeated source worksheet row; resolve reference duplication before freezing")
    frame["query_id"] = [hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                         for key in source_keys]
    frame["split"] = "development"
    frame["mapping_reason"] = frame.mapped.map({True: "mapped_original_feature", False: "unresolved_original_feature"})
    frame["formula_reason"] = ["available_tentative_reference" if str(x).strip() else "reference_formula_missing"
                               for x in frame.formula]
    frame["adduct_reason"] = ["available_tentative_reference" if str(x).strip() else "reference_adduct_missing"
                              for x in frame.adduct]
    frame["reference_tier"] = "curated_development_not_independent_standard"
    frame["feature_reference_count"] = frame.groupby(["stream", "feature_id"]).feature_id.transform("size")
    frame["unique_ion_query"] = frame.mapped & ~frame.duplicated(["stream", "feature_id"])
    # A formula/adduct conflict on one measured ion must not silently become gold.
    conflicts = frame.groupby(["stream", "feature_id"])[["formula", "adduct"]].transform("nunique").max(axis=1).gt(1)
    frame["ambiguous_reference"] = frame.mapped & conflicts
    frame["ion_metric_eligible"] = frame.unique_ion_query & ~frame.ambiguous_reference
    return frame


def score_representatives(queries, representatives, reference):
    """Four slots per source query; target positives alone define PC regression."""
    rows = []
    for query, replay in zip(queries, representatives, strict=True):
        if (query["stream"], feature_id(query["feature_id"])) != (replay["stream"], feature_id(replay["feature_id"])):
            raise ValueError("Replay query order/key mismatch")
        reps = {r["mode"]: r for r in replay["representatives"]}
        for mode in MODES:
            expected = reference[(reference.stream == mode) & (reference.name == query["manual_PC"])]
            mapped = expected[expected.mapped]
            found = reps.get(mode, {})
            fid, adduct = feature_id(found.get("feature_id", "")), found.get("adduct", "")
            same = mapped[mapped.feature_id == fid] if fid else mapped.iloc[:0]
            if not len(expected):
                status = "unconfirmed_extra" if fid else "no_reference_no_detection"
            elif not len(mapped):
                status = "unresolved_reference_mapping"
            elif not fid:
                status = "missing"
            elif len(same) and adduct in set(same.adduct):
                status = "correct_feature_and_adduct"
            elif len(same):
                status = "wrong_adduct"
            else:
                status = "different_peak_unconfirmed_identity"
            rows.append({"query": query["manual_PC"], "source_mode": query["stream"],
                         "source_feature_id": feature_id(query["feature_id"]), "target_mode": mode,
                         "expected_reference_count": len(expected), "mapped_reference_count": len(mapped),
                         "expected_feature_ids": ";".join(mapped.feature_id),
                         "expected_adducts": ";".join(sorted(set(mapped.adduct))),
                         "feature_id": fid, "adduct": adduct, "status": status,
                         "counterpart_query": mode.endswith("Pos") and mode != query["stream"] and bool(len(expected))})
    eligible = [r for r in rows if r["counterpart_query"]]
    return rows, ratio(sum(r["status"] == "correct_feature_and_adduct" for r in eligible), len(eligible))


def worker(source, project, keys, output, node):
    """Runs in a fresh process: no reference CSV or identity labels are loaded."""
    sys.path.insert(0, str(source / "app"))
    import pandas as pd
    from standard_lipidomics.project_store import load_untargeted_workflow
    from standard_lipidomics.untargeted_compounds import assign_compound_ids
    from presentation.untargeted_mode import build_mode_payload
    result, saved = load_untargeted_workflow(project)
    analysis = result["untargeted_analysis"]
    write_json(output / "saved_settings.json", {"criteria": analysis["criteria"], "settings": saved["settings"],
        "saved_versions": {k: analysis[k] for k in ("compound_version", "evidence_version", "lipid_grid_version")}})
    start = time.monotonic()
    analysis = assign_compound_ids(analysis, result=result)
    elapsed = time.monotonic() - start
    payload = build_mode_payload(result, analysis)
    for name in ("compound_features", "adduct_matches"):
        analysis[name].to_csv(output / f"{name}.csv", index=False)
    write_json(output / "payload.json", payload)
    assets = source / "app/presentation/untargeted_mode"
    (output / "viewer.mjs").write_text((assets / "model.js").read_text() + "\n" + (assets / "viewer.js").read_text())
    (output / "representatives.mjs").write_text('''import {readFileSync,writeFileSync} from 'node:fs';
import {ModeViewerModel,featureKey,sampleFigures} from './viewer.mjs';
const model = new ModeViewerModel(JSON.parse(readFileSync(new URL('./payload.json',import.meta.url))));
const queries = JSON.parse(readFileSync(new URL('./feature_keys.json',import.meta.url)));
const out = queries.map(q => ({...q, representatives:sampleFigures({model,
 selected:featureKey(q.stream,q.feature_id),canvas:'#fff',ink:'#111',grid:'#ddd'})
 .filter(x=>x.mode!=='empty').map(x=>({mode:x.mode,key:x.key,...model.point(x.key),...model.rows.get(x.key)}))}));
writeFileSync(new URL('./representatives.json',import.meta.url),JSON.stringify(out,null,2));
''')
    shutil.copy2(keys, output / "feature_keys.json")
    subprocess.run([str(node), str(output / "representatives.mjs")], check=True, cwd=output)
    streams = {}
    from standard_lipidomics.grid_browser import standard_sample_columns
    for mode, item in result["streams"].items():
        frame = item["features"]
        samples = standard_sample_columns(frame)
        groups = frame.attrs.get("sample_group_map", {})
        streams[mode] = {"original_features": len(frame), "sample_columns": samples,
                        "sample_group_counts": dict(Counter(str(groups.get(s, "Unassigned")) for s in samples)),
                        "feature_columns": list(frame.columns), "attrs": frame.attrs}
    write_json(output / "runtime_inventory.json", {"streams": streams,
        "tested_features": len(analysis["features"]), "raw_p_features": len(analysis["raw_p_features"]),
        "fold_change_filtered_features": len(analysis["significant_features"]),
        "compound_ions": len(analysis["compound_features"]),
        "compounds": int(analysis["compound_features"].compound_id.nunique()),
        "grouping_seconds": elapsed,
        "profile_status_counts": analysis["adduct_matches"].profile_status.value_counts().to_dict(),
        "runtime_versions": {k: analysis[k] for k in ("compound_version", "evidence_version", "lipid_grid_version")},
        "sample_metadata_audit": result.get("untargeted_sample_metadata_audit", pd.DataFrame()).fillna("").to_dict("records")})


def freeze(args):
    import pandas as pd
    project, out = args.project.resolve(), args.output.resolve()
    if out.is_relative_to(ROOT) or out.is_relative_to(project) or project.is_relative_to(out):
        raise ValueError("Output must be outside the repository and source project")
    if out.exists():
        raise ValueError("Checkpoint directory already exists; use a new version")
    node = Path(args.node or shutil.which("node") or "")
    if not node.is_file():
        raise ValueError("Node is required; provide --node with an executable path")
    revision = subprocess.check_output(["git", "rev-parse", args.revision + "^{commit}"], cwd=ROOT, text=True).strip()
    out.mkdir(parents=True)
    baseline = out / "source"
    baseline.mkdir()
    archive = out / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", revision], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(baseline, filter="data")
    print("Freezing project and source fingerprints", flush=True)
    paths = list(project_files(project))
    before = fingerprint(paths, project)
    snapshot = out / "project"
    for path in paths:
        target = snapshot / path.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if fingerprint(list(project_files(snapshot)), snapshot) != before:
        raise ValueError("Copied snapshot differs from original")
    refs = out / "reference"
    refs.mkdir()
    shutil.copy2(args.reference, refs / "manual_mapping.csv")
    shutil.copy2(args.queries, refs / "regression_queries.json")
    ledger = reference_ledger(pd.read_csv(refs / "manual_mapping.csv", dtype=str))
    ledger.to_csv(refs / "query_ledger.csv", index=False)
    counts = ledger.groupby(["lipid_class", "stream"]).size().unstack(fill_value=0).reindex(columns=MODES, fill_value=0)
    counts.to_csv(refs / "class_mode_scope.csv")
    queries = json.loads((refs / "regression_queries.json").read_text())
    keys = [{"stream": q["stream"], "feature_id": feature_id(q["feature_id"])} for q in queries]
    if len({(q["stream"], q["feature_id"]) for q in keys}) != len(keys):
        raise ValueError("Regression queries contain duplicate source ions")
    write_json(out / "feature_keys.json", keys)
    # Repeat both inference and the JS selector; require identical scientific outputs.
    replay_hashes = []
    for index in (1, 2):
        run = out / f"replay_{index}"
        run.mkdir()
        print(f"Frozen baseline replay {index}/2", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(baseline),
                        str(snapshot), str(out / "feature_keys.json"), str(run), str(node.resolve())],
                       check=True, cwd=baseline)
        replay_hashes.append({name: digest(run / name) for name in
                             ("compound_features.csv", "adduct_matches.csv", "payload.json", "representatives.json")})
    if replay_hashes[0] != replay_hashes[1]:
        raise ValueError("Non-deterministic replay; preserve checkpoint and investigate")
    representatives = json.loads((out / "replay_1/representatives.json").read_text())
    rows, metric = score_representatives(queries, representatives, ledger)
    pd.DataFrame(rows).to_csv(out / "pc_four_mode_replay.csv", index=False)
    scope = {"protocol": "laser-untargeted-evaluation/1", "acquisitions": 1,
             "development_query_count": len(ledger), "final_validation_queries": [],
             "split_reason": "All ALD reference classes were already inspected during development; independent acquisition required.",
             "additional_acquisition_policy": "Explicit recorded scope decision required; historical annotation optimization stays paused.",
             "regression_queries": len(queries), "class_labels": len(counts),
             "mode_reference_counts": ledger.stream.value_counts().to_dict(),
             "unresolved_mappings": int((~ledger.mapped).sum()),
             "missing_formulas": int(ledger.formula.eq("").sum()),
             "ambiguous_reference_rows": int(ledger.ambiguous_reference.sum())}
    write_json(out / "evaluation_scope.json", scope)
    after = fingerprint(list(project_files(project)), project)
    if after != before:
        raise ValueError("Source project changed during freeze/replay; checkpoint not valid")
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "packages": dict(sorted((d.metadata["Name"], d.version) for d in metadata.distributions() if d.metadata["Name"])),
                   "node": subprocess.check_output([str(node), "--version"], text=True).strip(),
                   "node_sha256": digest(node)}
    write_json(out / "environment.json", environment)
    write_json(out / "input_fingerprints.json", before)
    write_json(out / "source_fingerprints.json", fingerprint([p for p in baseline.rglob("*") if p.is_file() and "__pycache__" not in p.parts], baseline))
    provenance = {"protocol": scope["protocol"], "created_utc": datetime.now(timezone.utc).isoformat(),
                  "baseline_revision": revision, "source_archive_sha256": digest(archive),
                  "runner_sha256": digest(__file__), "manual_mapping_sha256": digest(refs / "manual_mapping.csv"),
                  "regression_queries_sha256": digest(refs / "regression_queries.json"),
                  "project_unchanged": True, "snapshot_verified": True,
                  "two_replays_identical": True, "replay_hashes": replay_hashes,
                  "positive_counterpart_recovery": metric, "external_validation": False,
                  "statistics_and_isotope_formula_searches_rerun": False}
    write_json(out / "provenance.json", provenance)
    print(json.dumps({"counterpart_recovery": metric, "scope": scope, "project_unchanged": True}), flush=True)
    if args.expected_correct is not None and (metric["n"], metric["N"]) != (args.expected_correct, args.expected_total):
        raise ValueError("Regression checkpoint differs from expected count; retain this failed run")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*(Path(x) for x in sys.argv[2:]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=BASELINE)
    parser.add_argument("--node")
    parser.add_argument("--expected-correct", type=int)
    parser.add_argument("--expected-total", type=int)
    args = parser.parse_args()
    if (args.expected_correct is None) != (args.expected_total is None):
        parser.error("Supply both expected regression counts or neither")
    freeze(args)


if __name__ == "__main__":
    main()
