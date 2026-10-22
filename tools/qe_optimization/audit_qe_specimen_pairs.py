"""Read-only explicit acquisition-metadata pairing audit; no annotation changes."""

from pathlib import Path
import hashlib
import json
import shutil
import sys

import pandas as pd
from scipy.io import loadmat

from context import WORKSPACE as ROOT, REPO, SOURCE

QE = ROOT / "outputs/qe_optimization_20261022"
OUT = QE / "explicit_specimen_pair_audit_v1"
sys.path.insert(0, str(REPO / "app"))
from matlab_annotation.specimen_pairs import explicit_specimen_pairs
from annotator.peak_evidence import sample_roles


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    datasets = json.loads((QE / "baseline/run_manifest.json").read_text())["datasets"]
    rows = []
    receipts = []
    (OUT / "private_maps").mkdir()
    for dataset in datasets:
        path = SOURCE / dataset / "MS_metadata.mat"
        baseline = QE / "baseline/runs" / dataset / "full_annotation.pkl"
        before = sha(path)
        data = loadmat(path, variable_names=["MS_metadata"], simplify_cells=True)[
            "MS_metadata"
        ]
        result = pd.read_pickle(baseline)
        features = {s: p["features"] for s, p in result["streams"].items()}
        try:
            maps, audit = explicit_specimen_pairs(data, features)
            status = "explicit_metadata_links_checked"
            bio = {}
            for stream, f in features.items():
                _, roles, _ = sample_roles(f)
                bio[stream] = {
                    sid
                    for col, sid in maps[stream].items()
                    if roles.get(col) == "biological"
                }
            common = set.intersection(*bio.values()) if bio else set()
            for stream, counts in audit.items():
                rows.append(
                    dict(
                        dataset=dataset,
                        stream=stream,
                        status=status,
                        **counts,
                        mapped_biological_specimens=len(bio[stream]),
                        biological_specimens_shared_all_four=len(common),
                        error="",
                    )
                )
            (OUT / "private_maps" / (dataset + ".json")).write_text(
                json.dumps(maps, indent=2), encoding="utf-8"
            )
        except ValueError as error:
            for stream in features:
                rows.append(
                    dict(
                        dataset=dataset,
                        stream=stream,
                        status="metadata_conflict_or_unsupported",
                        error=str(error),
                    )
                )
        assert sha(path) == before
        receipts.append(
            dict(dataset=dataset, metadata_sha256=before, baseline_sha256=sha(baseline))
        )
        print("PAIRING AUDITED", dataset, rows[-1]["status"], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "by_dataset_mode.csv", index=False)
    summary = dict(
        datasets=len(datasets),
        dataset_modes=len(frame),
        checked_dataset_modes=int(
            frame.status.eq("explicit_metadata_links_checked").sum()
        ),
        purpose="NON-ANNOTATION metadata audit. Explicit shared sample_id/filename/phase/mode/group links ONLY. Not guessed numerical filename offsets or curation labels.",
        curation_runtime_input=False,
        source_sha256=sha(REPO / "app/matlab_annotation/specimen_pairs.py"),
        inputs=receipts,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "runner_source.py")
    print(
        frame[
            [
                "dataset",
                "stream",
                "status",
                "mapped_columns",
                "raw_sample_columns",
                "biological_specimens_shared_all_four",
                "error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
