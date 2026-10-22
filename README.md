# LASER Standard Lipidomics

Standalone LASER tool for conventional XCMS lipidomics. This repository was
extracted from the standard-lipidomics module in
[`YRJJaspers/LASER-4D-lipidomics`](https://github.com/YRJJaspers/LASER-4D-lipidomics)
at commit `e447bac` on branch `codex/future-ready-review-fixes`.

This tool contains only the standard workflow and its shared scientific
dependencies. It does not expose the LASER 4D, MZmine-processing, Bruker,
targeted-extraction, or dataset-specific workflows.

## Included workflow

- Independent XCMS precursor-mass calibration for RP/NP and POS/NEG streams.
- Automatic MATLAB-equivalent XCMS annotation (no MATLAB installation needed).
- LASER reference-matcher annotation.
- Composition-range and grid-coherence policies.
- Interactive annotation review, reports, exports, saved projects, and
  post-annotation untargeted comparison.
- A small headless CLI for either annotation engine.

## Run the app

Python 3.10–3.14 is supported. Windows with Python 3.14 is tested in CI.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
streamlit run app/app.py
```

On macOS, you can instead double-click `Launch LASER Standard.command`. On
Windows, run `Setup (Windows).bat` once and then `Launch LASER Standard.bat`.
The Windows launcher listens on all network interfaces on port **8085** and
does not open its own browser tab, so it can be started by the CFM Analysis
Hub launcher. Set `LASER_STANDARD_HOST` or `LASER_STANDARD_PORT` before launch
to override those defaults.

Set `LASER_STANDARD_PROJECT` to prefill the XCMS project folder.

## Windows dataset scanner — ready-to-run download

[Download LASER_Dataset_Scanner_Windows.zip](https://github.com/Core-Facility-Metabolomics-AUMC/LASER-standard-lipidomics/raw/refs/heads/main/downloads/LASER_Dataset_Scanner_Windows.zip)

This is the separate dataset-collection scanner, not the annotation app. It
includes its own Windows x64 Python runtime: **no setup or Python installation
is needed**. Download the ZIP on the Windows PC, or transfer the complete ZIP
there using USB or a shared folder.

1. Right-click the downloaded ZIP and choose **Extract All…**.
2. Open the extracted `LASER_Dataset_Scanner_Windows` folder.
3. Double-click **Start Scanner.cmd**. Keep the `runtime` folder beside it;
   do not launch from inside the ZIP or copy only the `.cmd` file.

Stage 1 scans only **2024–2026**, checks all four measurement modes, and shows
the selected files and total copy size. Stage 2 copies only after you select
a destination and approve. The scanner does not change the original datasets.
See the [scanner instructions](tools/dataset_scanner/README.txt) for quality
criteria and limitations.

The updated scanner shows the current folder/file, live counters, elapsed time,
time since the last progress update and recent activity. Long pauses are flagged
as possible network waits or stalls; server responsiveness is shown separately.
Close an older running scanner and extract the new ZIP into a fresh folder to
use this update. The source datasets and screening thresholds are unchanged.

## Headless use

```bash
laser-standard /path/to/Processing/Annotation \
  --engine matlab \
  --phase both \
  --output results/standard_lipidomics
```

For the input contract and scientific behavior, see
[`docs/STANDARD_LIPIDOMICS_MODE.md`](docs/STANDARD_LIPIDOMICS_MODE.md) and
[`docs/STANDARD_LIPIDOMICS_ANNOTATION_REFERENCE.md`](docs/STANDARD_LIPIDOMICS_ANNOTATION_REFERENCE.md).

## Development

The agreed next steps and validation gates are in the
[annotation improvement plan](docs/ANNOTATION_IMPROVEMENT_PLAN.md).
The [manual annotation comparison tool](docs/ANNOTATION_VALIDATION.md) evaluates
saved results on the original features without changing annotation decisions.

```bash
python -m pip install -e '.[dev]'
pytest
ruff check app
```

The source repository is not modified by this extraction. Future changes can
be developed and released independently in this repository.
