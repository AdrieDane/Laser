"""Turn a Moon annotation document (.moon) into a manual crosswalk.

    python tools/moon_crosswalk.py ANNOTATION.moon PROJECT --output curation/DATASET/curated_feature_crosswalk.csv

The .moon usually comes from the MATLAB pipeline through Moon
``getannotation --all --out x.moon``: the human-curated annotation of the same
project LASER annotated. Its feature IDs are exact rows of the project's
peak-group lists, so mapped rows get ``crosswalk_status = exact_feature_id``.
Feed the folder to ``tools/compare_manual_annotations.py --curation-root``.
Read-only for the inputs; only M+0 features are written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from moon_format import PeakIndex, moon_to_crosswalk, read_moon  # noqa: E402
from output_io import atomic_write_csv  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("moon", type=Path, help="Moon annotation document (.moon or .json)")
    p.add_argument("project", type=Path, help="Project or Annotation folder with the peak-group lists")
    p.add_argument("--output", type=Path, required=True, help="Crosswalk CSV to write (not overwritten)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; inputs and earlier results are never overwritten")
    document = read_moon(args.moon)
    crosswalk = moon_to_crosswalk(document, PeakIndex.from_project(args.project))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(args.output, crosswalk, index=False)
    counts = Counter(crosswalk.crosswalk_status)
    print(f"Wrote {len(crosswalk)} M+0 rows to {args.output}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
