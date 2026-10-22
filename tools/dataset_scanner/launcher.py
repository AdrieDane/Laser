"""Launch the portable browser UI, or run explicitly separated CLI stages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .core import Limits, copy_approved, scan, write_reports
from .server import serve


def main(resources=None):
    root = Path(resources or Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        type=Path,
        help="Stage 1 only: source containing 2024/2025/2026 year folders",
    )
    parser.add_argument("--report-dir", type=Path, help="Fresh stage-1 report folder")
    parser.add_argument(
        "--copy-plan", type=Path, help="Stage 2 only: saved scan_plan.json"
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Approved PASS dataset ID; repeat as needed",
    )
    parser.add_argument("--confirm-copy", action="store_true")
    parser.add_argument(
        "--settings",
        type=Path,
        help="JSON screening limits; does not affect annotation",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.scan and args.copy_plan:
        parser.error("Scan and copy are separate stages")
    if args.scan:
        if not args.report_dir:
            parser.error("Stage 1 requires a fresh --report-dir")
        settings = (
            json.loads(args.settings.read_text()) if args.settings else asdict(Limits())
        )
        plan = scan(
            args.scan, root / "reference_masses.csv", Limits(**settings), progress=print
        )
        write_reports(plan, args.report_dir)
        approved = [d for d in plan["datasets"] if d["status"] == "PASS"]
        print(
            f"Stage 1 complete: {len(approved)} pass; {sum(d['total_bytes'] for d in approved):,} bytes eligible. Nothing copied."
        )
    elif args.copy_plan:
        if not args.destination or not args.confirm_copy:
            parser.error("Stage 2 requires --destination and --confirm-copy")
        plan = json.loads(args.copy_plan.read_text(encoding="utf-8"))
        result = copy_approved(
            plan, args.destination, args.dataset, confirm=True, progress=print
        )
        print(f"Copied and verified {len(result)} datasets.")
    else:
        reports = (
            Path(os.environ.get("LOCALAPPDATA", str(Path.cwd())))
            / "LASER Dataset Scanner"
            / "reports"
        )
        serve(root, reports, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
