#!/usr/bin/env python3
"""Command-line runner for LASER Standard Lipidomics."""

from __future__ import annotations

import argparse
from pathlib import Path

from application.standard_workflow import StandardAnnotationRequest
from composition import standard_annotation_service
from output_io import atomic_write_csv
from resource_paths import REFERENCE_DATA_DIRECTORY


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
    return parser


def main() -> int:
    args = _parser().parse_args()
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
        },
    )
    result = standard_annotation_service.run(request)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables.items():
        atomic_write_csv(table, output / f"{name}.csv", index=False)
    print(f"Wrote {len(result.tables)} result tables to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
