"""Read-only, feature-aware evaluation of exported annotations and manual maps.

Optional pickle input is restricted to explicitly trusted local result files.
The annotation CSV is authoritative; old diagnostic artifacts never replace it.
No annotation runtime, calibration or model fitting is invoked.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from output_io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    runtime_environment,
    sha256_file,
)
from standard_lipidomics.validation import (
    AnnotationComparison,
    compare_manual_annotations,
    summarize_comparison,
)


def _csv(path: Path) -> pd.DataFrame:
    # IDs such as 0017 and NA are identifiers, not numbers or missing values.
    return pd.read_csv(
        path,
        dtype={"xcms_feature_id": str, "feat_feature_id": str},
        keep_default_na=False,
        low_memory=False,
    )


def _fingerprint(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _revision() -> dict:
    root = Path(__file__).resolve().parents[1]
    sources = {
        "tools/compare_manual_annotations.py": sha256_file(Path(__file__)),
        "app/standard_lipidomics/validation.py": sha256_file(
            root / "app/standard_lipidomics/validation.py"
        ),
    }
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "revision": revision,
        "working_tree_dirty": dirty,
        "source_sha256": sources,
    }


def _diagnostics(run: Path, args, fingerprints: list) -> tuple:
    candidates, decisions = None, None
    if args.result_name:
        path = run / args.result_name
        # User explicitly opts in. Never use this option on downloaded or
        # untrusted pickle files: deserialization can execute arbitrary code.
        with path.open("rb") as source:
            result = pickle.load(source)
        if not isinstance(result, dict) or not isinstance(result.get("streams"), dict):
            raise ValueError(f"Not a LASER result dictionary: {path}")
        parts = []
        for stream, payload in result["streams"].items():
            frame = payload.get("all_pairs")
            if not isinstance(frame, pd.DataFrame):
                raise ValueError(f"Candidate table unavailable for {stream} in {path}")
            parts.append(frame.assign(stream=stream))
        candidates = pd.concat(parts, ignore_index=True) if parts else None
        fingerprints.append(_fingerprint(path))
        del result
    if args.candidates_name:
        path = run / args.candidates_name
        candidates = _csv(path)
        fingerprints.append(_fingerprint(path))
    if args.pe_decisions_name and (run / args.pe_decisions_name).is_file():
        path = run / args.pe_decisions_name
        decisions = _csv(path)
        fingerprints.append(_fingerprint(path))
    return candidates, decisions


def _overview(summary: pd.DataFrame, manual: pd.DataFrame) -> str:
    cells = []
    for row in summary.to_dict("records"):
        pct = row["percent_mapped_manual_same_peak"]
        rate = f"{pct:.1f}%" if pd.notna(pct) else "Not available"
        cells.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(v))}</td>"
                for v in [
                    row["stream"],
                    row["manual_same_peak"],
                    row["unique_mapped_manual_annotations"],
                    rate,
                    row["manual_same_lipid_different_peak"],
                    row["manual_peak_other_lipid"],
                    row["laser_not_in_manual_list"],
                    row["unresolved_manual_rows"],
                ]
            )
            + "</tr>"
        )
    counts = manual[
        manual.include_in_pooled.eq(True)
        & manual.mapped.eq(True)
        & manual.duplicate_manual_entry.eq(False)
    ].outcome.value_counts()
    pattern = "".join(
        f"<li>{escape(k.replace('_', ' '))}: {int(v):,}</li>" for k, v in counts.items()
    )
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Manual annotation comparison</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1250px;margin:35px auto;padding:0 22px;color:#20364a}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d4dee5;text-align:right}
th{background:#20364a;color:white}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}
</style></head><body><h1>Manual annotation comparison</h1>
<p>A match requires the same lipid name, adduct, measurement stream, internal-standard identity and original feature.
The table pools only datasets explicitly included in the summary. Detailed exports retain excluded datasets.</p>
<div class="scroll"><table><thead><tr><th>Stream</th><th>Same manual peak</th><th>Mapped manual entries</th>
<th>Same peak %</th><th>Manual lipid on another peak</th><th>Other lipid on manual peak</th>
<th>LASER names not in manual list</th><th>Unresolved manual entries</th></tr></thead><tbody>"""
        + "".join(cells)
        + """</tbody></table></div>
<p>Unresolved manual entries are not in the mapped denominator. The separate all-manual percentage in the CSV
counts unresolved entries as misses. Two disagreement columns can overlap; they must not be added together.
Duplicate manual and LASER rows are recorded separately and do not inflate unique-pair agreement.</p>
<h2>Observed selection differences</h2><ul>"""
        + pattern
        + """</ul>
<h2>How to interpret the candidate evidence</h2>
<p>Presence means that exact manual lipid and feature appear in the supplied final or PE discovery table.
Absence from those tables does not prove the pair was never considered earlier. Recorded failed checks can
coexist with a later accepted decision and are not reconstructed causes. Missing diagnostics remain unknown.
The PE local-support field reports observed counts, not an inferred score threshold.</p>
<p>Names absent from the manual list are unconfirmed additions, not proven false lipids. These examples have
informed development. This report is agreement with manual curation, not prospective analytical accuracy.</p>
<h2>Detailed results</h2><ul>
<li><a href="manual_disagreements.csv">Manual entries not on the same peak</a></li>
<li><a href="manual_comparison.csv">All manual entries and candidate evidence</a></li>
<li><a href="laser_comparison.csv">All LASER calls compared with the manual list</a></li>
<li><a href="by_dataset_stream_class.csv">Results by dataset, stream and class</a></li>
<li><a href="by_dataset_stream.csv">Results by dataset and stream</a></li>
<li><a href="pooled_by_stream_class.csv">Pooled results by stream and class</a></li>
<li><a href="pooled_by_stream.csv">Pooled results by stream</a></li>
<li><a href="validation_manifest.json">Input fingerprints and evaluation provenance</a></li>
</ul></body></html>"""
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="One subfolder per annotated dataset",
    )
    p.add_argument(
        "--curation-root",
        type=Path,
        required=True,
        help="Matching subfolders containing curated_feature_crosswalk.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or empty report folder; inputs are never overwritten",
    )
    p.add_argument("--annotations-name", default="final_automatic_annotations.csv")
    p.add_argument(
        "--candidates-name",
        help="Optional final candidates CSV in every selected run folder",
    )
    p.add_argument(
        "--pe-decisions-name",
        default="full_candidate_decisions.csv",
        help="Optional PE table; missing files remain unknown",
    )
    p.add_argument(
        "--result-name",
        help="Optional local result pickle filename for candidate diagnostics",
    )
    p.add_argument(
        "--trust-local-pickles",
        action="store_true",
        help="Explicitly trust local result files; pickle can execute code",
    )
    p.add_argument(
        "--dataset",
        action="append",
        help="Dataset folder to include; repeat for a subset",
    )
    p.add_argument(
        "--exclude-from-pooled",
        action="append",
        default=[],
        help="Retain this dataset's detail but exclude its pooled contribution",
    )
    p.add_argument(
        "--annotation-revision",
        help="Operator-declared revision of the annotation outputs, not the evaluation tool",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.result_name and not args.trust_local_pickles:
        parser.error(
            "--result-name requires --trust-local-pickles; use only known, trusted local results"
        )
    if args.result_name and args.candidates_name:
        parser.error("Choose either --result-name or --candidates-name, not both")
    for value in [
        args.annotations_name,
        args.result_name,
        args.candidates_name,
        args.pe_decisions_name,
    ]:
        if value and Path(value).name != value:
            parser.error("Artifact names must be filenames, not paths")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error("Output must be a new or empty directory")
    if not args.runs_root.is_dir() or not args.curation_root.is_dir():
        parser.error("Run and curation roots must be existing directories")
    available = {
        p.name: p
        for p in args.runs_root.iterdir()
        if p.is_dir() and (p / args.annotations_name).is_file()
    }
    chosen = sorted(set(args.dataset) if args.dataset else available)
    if not chosen or set(chosen) - set(available):
        parser.error(
            "No datasets found, or a selected dataset has no annotations export"
        )
    if set(args.exclude_from_pooled) - set(chosen):
        parser.error("Every pooled exclusion must name a selected dataset")
    tool_provenance = _revision()
    results, provenance = [], []
    for dataset in chosen:
        run = available[dataset]
        annotations_path = run / args.annotations_name
        truth_path = args.curation_root / dataset / "curated_feature_crosswalk.csv"
        fingerprints = [_fingerprint(annotations_path), _fingerprint(truth_path)]
        candidates, pe = _diagnostics(run, args, fingerprints)
        report = compare_manual_annotations(
            _csv(truth_path),
            _csv(annotations_path),
            candidates=candidates,
            pe_decisions=pe,
            dataset=dataset,
            include_in_pooled=dataset not in args.exclude_from_pooled,
        )
        results.append(report)
        provenance.append(
            dict(
                dataset=dataset,
                include_in_pooled=dataset not in args.exclude_from_pooled,
                inputs=fingerprints,
                final_candidate_table_supplied=candidates is not None,
                PE_discovery_table_supplied=pe is not None,
            )
        )
        print(
            f"Compared {dataset}: {len(report.manual)} manual entries, {len(report.laser)} LASER calls",
            flush=True,
        )
    combined = AnnotationComparison(
        pd.concat([r.manual for r in results], ignore_index=True),
        pd.concat([r.laser for r in results], ignore_index=True),
    )
    pooled = summarize_comparison(combined, group_by=("stream",), pooled_only=True)
    outputs = {
        "manual_comparison.csv": combined.manual,
        "manual_disagreements.csv": combined.manual[
            ~combined.manual.same_manual_peak.eq(True)
        ],
        "laser_comparison.csv": combined.laser,
        "by_dataset_stream_class.csv": summarize_comparison(
            combined, group_by=("dataset", "stream", "lipid_class")
        ),
        "by_dataset_stream.csv": summarize_comparison(combined),
        "pooled_by_stream_class.csv": summarize_comparison(
            combined, group_by=("stream", "lipid_class"), pooled_only=True
        ),
        "pooled_by_stream.csv": pooled,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        atomic_write_csv(args.output / name, frame)
    atomic_write_text(args.output / "overview.html", _overview(pooled, combined.manual))
    atomic_write_json(
        args.output / "validation_manifest.json",
        dict(
            schema_version=1,
            complete=True,
            evaluation_tool=tool_provenance,
            runtime_environment=runtime_environment(),
            annotation_revision=args.annotation_revision,
            annotation_revision_source="operator_declared"
            if args.annotation_revision
            else "not_supplied",
            selection_source="annotations_CSV",
            excluded_from_pooled=args.exclude_from_pooled,
            datasets=provenance,
            outputs={
                p.name: sha256_file(p) for p in args.output.iterdir() if p.is_file()
            },
            caveats=[
                "Candidate tables are observations, not complete pipeline event history",
                "Unlisted annotations are not proven false lipids",
                "Development examples are not untouched external validation",
            ],
        ),
        allow_nan=False,
    )
    print(pooled.to_string(index=False))
    print(f"Wrote read-only comparison to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
