"""Build an offline, no-install Windows x64 scanner ZIP from reviewed sources.

The only download is the official, pinned Python embeddable runtime. Its hash
is recorded in build_manifest.json. No executable is run during packaging.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]
VERSION = "3.13.16"
URL = f"https://www.python.org/ftp/python/{VERSION}/python-{VERSION}-embed-amd64.zip"
RUNTIME_SHA256 = "97dae5274cc54867065e8d5a3226e48c35017ed332a0fdb0e27d5b5821961297"


def build(output, cached_runtime=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime_zip = output / "python-runtime-original.zip"
    if cached_runtime:
        shutil.copyfile(cached_runtime, runtime_zip)
    else:
        with (
            urllib.request.urlopen(URL, timeout=60) as response,
            runtime_zip.open("xb") as handle,
        ):
            shutil.copyfileobj(response, handle)
    digest = hashlib.sha256(runtime_zip.read_bytes()).hexdigest()
    if digest != RUNTIME_SHA256:
        raise ValueError("Official runtime does not match the reviewed SHA-256 pin")
    root = output / "LASER_Dataset_Scanner_Windows"
    root.mkdir()
    sources = REPO / "tools/dataset_scanner"
    package = root / "dataset_scanner"
    package.mkdir()
    for name in ("__init__.py", "readers.py", "core.py", "server.py", "launcher.py"):
        shutil.copy2(sources / name, package / name)
    shutil.copytree(sources / "web", root / "web")
    for name in ("Start Scanner.cmd", "scanner.py", "README.txt"):
        shutil.copy2(sources / name, root / name)
    # Windows command interpreters reliably handle CRLF, including ZIP extraction.
    launcher = root / "Start Scanner.cmd"
    launcher.write_bytes(
        launcher.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    )
    with (
        (REPO / "reference_data/4d_lipid_reference.csv").open(
            encoding="utf-8-sig", newline=""
        ) as src,
        (root / "reference_masses.csv").open("w", encoding="utf-8", newline="") as dst,
    ):
        writer = csv.DictWriter(dst, fieldnames=["name", "adduct", "mz"])
        writer.writeheader()
        for row in csv.DictReader(src):
            writer.writerow({key: row[key] for key in writer.fieldnames})
    runtime = root / "runtime"
    runtime.mkdir()
    with zipfile.ZipFile(runtime_zip) as archive:
        for info in archive.infolist():
            if Path(info.filename).name != info.filename or info.is_dir():
                raise ValueError("Unexpected path in official embeddable runtime")
            (runtime / info.filename).write_bytes(archive.read(info))
    # Record all distributed files; data files and runtime are redistributable
    # under their respective included licences. Keep Python's LICENSE.txt.
    records = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    (root / "build_manifest.json").write_text(
        json.dumps(
            dict(
                runtime_version=VERSION,
                runtime_source=URL,
                runtime_sha256=digest,
                source_files=records,
                network_required_when_running=False,
                allowed_years=[2024, 2025, 2026],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact = output / "LASER_Dataset_Scanner_Windows.zip"
    with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, root.name + "/" + path.relative_to(root).as_posix())
    print(f"Ready: {artifact} ({artifact.stat().st_size:,} bytes)")
    print(f"Python runtime source SHA-256: {digest}")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-zip",
        type=Path,
        help="Optional previously downloaded official runtime; same pinned hash required",
    )
    args = parser.parse_args()
    build(args.output_dir, args.runtime_zip)
