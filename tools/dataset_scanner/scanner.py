"""Portable bundle entrypoint; uses only the included Python standard library."""

from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from dataset_scanner.launcher import main  # noqa: E402

if __name__ == "__main__":
    main(root)
