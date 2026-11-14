#!/usr/bin/env python3
"""Command-line runner for LASER Standard Lipidomics."""

from __future__ import annotations

import argparse
from pathlib import Path

from application.standard_workflow import StandardAnnotationRequest
from annotator.standard_annotated_inputs import write_standard_annotated_inputs
from composition import standard_annotation_service
from output_io import atomic_write_csv
from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_lipidomics.workflow import discover_standard_streams


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laser-standard",
        description="Run the standalone standard XCMS lipidomics workflow.",
    )
    parser.add_argument("project", type=Path, help="XCMS project or Annotation folder")
    parser.add_argument(
        "--engine",
        choices=("matlab", "laser"),
        default="matlab",
        help="Annotation engine (default: matlab)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/standard_lipidomics"),
        help="Directory for result CSV files",
    )
    parser.add_argument(
        "--phase",
        choices=("both", "RP", "NP"),
        default="both",
        help="Chromatography phase for the MATLAB-equivalent engine",
    )
    parser.add_argument(
        "--moon",
        type=Path,
        metavar="FILE",
        help="Also write the final annotations as a Moon annotation document "
        "(.moon, moon-annotation/1) for the MATLAB pipeline, plus FILE_report.json",
    )
    parser.add_argument(
        "--experimental-pe",
        action="store_true",
        help="Enable optional MATLAB-engine PE peak correction (off by default)",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.experimental_pe and args.engine != "matlab":
        parser.error("--experimental-pe is available only with --engine matlab")
    references = Path(REFERENCE_DATA_DIRECTORY)
    phases = ("RP", "NP") if args.phase == "both" else (args.phase,)
    request = StandardAnnotationRequest(
        engine=args.engine,
        project=args.project,
        reference_path=references / "4d_lipid_reference.csv",
        matlab_reference_path=references / "matlab_pipeline",
        matlab_policy_path=references / "matlab_lipid_policy.tsv",
        np_rt_reference_path=references / "lipid_wizard_retention_times.csv",
        settings={
            "phases": phases,
            "only_accepted": True,
            "align_rp_rt": False,
            "adapt_class_rt": True,
            "use_grid_coherence": True,
            "use_experimental_pe_selector": args.experimental_pe,
        },
    )
    result = standard_annotation_service.run(request)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables.items():
        atomic_write_csv(output / f"{name}.csv", table, index=False)
    annotated_inputs = write_standard_annotated_inputs(
        discover_standard_streams(args.project),
        result.tables["automatic_annotations"],
        output / "annotated_input_files",
    )
    print(f"Wrote {len(result.tables)} result tables to {output}")
    print(
        f"Wrote {len(annotated_inputs)} annotated peak-group lists to "
        f"{output / 'annotated_input_files'}"
    )
    if args.moon is not None:
        _write_moon(args, result.tables["automatic_annotations"])
    return 0


def _write_moon(args: argparse.Namespace, annotations) -> None:
    import json

    from moon_format import annotations_to_moon, write_moon
    from output_io import atomic_write_text
    from version import __version__

    path = args.moon.expanduser().resolve()
    document, report = annotations_to_moon(
        annotations,
        discover_standard_streams(args.project),
        laser_info={
            "engine": args.engine,
            "phase": args.phase,
            "laser_version": __version__,
            "project": str(args.project.expanduser().resolve()),
        },
    )
    write_moon(document, path)
    atomic_write_text(
        path.with_name(path.stem + "_report.json"),
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    skipped = sum(report["skipped_by_reason"].values())
    print(
        f"Wrote {report['features']} Moon seed annotations to {path}"
        + (f" ({skipped} rows not written; see {path.stem}_report.json)" if skipped else "")
    )


if __name__ == "__main__":
    raise SystemExit(main())
