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
  optional untargeted comparison (off by default).
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

### Project modules and results

The start page has five modules: **Calibration**, **Annotation**, **Untargeted
analysis**, **Open completed annotation**, and **Open completed untargeted
analysis**. Enter the **project folder** in any module; LASER discovers the XCMS
input files and saved results. The XCMS input subfolder can have another name
when it is the only input folder. Direct peak lists, `metadata/`, and the older
`Processing/Annotation` input layout remain supported.

```text
project folder/
  XCMS/                     # original peak-group lists and sample workbooks
  Calibration/              # current calibrated lists and model audit
  LASER/                    # annotation exports and saved GUI workspace
    LASER_PROJECT.json
    annotated_peaklists/
    tables/
  Untargeted/               # untargeted exports and saved GUI workspace
    LASER_PROJECT.json
    tables/
  Archive/
    Calibration/<timestamp>/
    LASER/<timestamp>/
    Untargeted/<timestamp>/
  metadata/                 # portable copies of original inputs and metadata
```

**Calibration** supports independent m/z and RT calibration, or both, for each
available RP/NP and positive/negative stream. Review the standards and click
**Save calibration**. **Annotation** offers **Raw XCMS data** or **Calibrated XCMS
data**. For calibrated data, choose the current export or an archived version
from **Calibration version**; dates and available streams identify each version.
The selected files are recorded with the saved annotation.

RT calibration brings over the four models and standard-review diagnostics from
[LASER-4D `a188f72`](https://github.com/YRJJaspers/LASER-4D-lipidomics/tree/a188f72a4366055238b474ab0826964430f33ee5):
constant shift, linear, robust linear, and monotonic piecewise linear. Its model
maps library RT to observed RT. For the standard XCMS workflow, the export
applies the inverse to express measured peak RTs on the library scale, including
RT boundaries. This lets both annotation engines consume a saved calibration.
Original RT values are retained in `LASER_original_*` audit columns; original
XCMS files are unchanged. RT models, reviewed anchors, fit metrics and
leave-one-standard-out diagnostics are saved in the calibration audit. Existing
automatic annotation-time RT models still operate on the selected inputs.

Annotation settings offer **Grid coherence V1** (the existing default) and
**Grid coherence V2**, copied from the same LASER-4D commit. V2 measures the worst
axis deviation from the local median using robust within-cluster spread and
native-unit noise floors. Duplicate candidates at one composition node cannot
increase support, and sparse nodes remain unassessed. Standard lipidomics uses
m/z and RT evidence without CCS. The run settings and saved tables record the
selected method. The LASER-4D internal method identifier `local_median_mad_v3`
is retained for provenance; this tool labels the option **Grid coherence V2**.

Calibration, annotation and untargeted results each have a **Save** button with
**Archive old project** (default) and **Overwrite old project** options. Failed
writes restore the previous result. Annotation and completed untargeted analyses
also save automatically, archiving the previous version. Their start-page
**Open completed…** modules restore the saved GUI workspace without rerunning
annotation or statistics. Older saves under `LASER/Annotation` and
`LASER/Untargeted` remain readable. To reopen an archived annotation or
untargeted result, enter its timestamped result folder in the corresponding
**Open completed…** module. Saved annotation peak lists retain the exact chosen
input, so resaving cannot silently substitute a newer calibration.

The **Minimal** and **Midnight** appearances, controls and page headers follow
the same newest LASER-4D branch (`codex/annotation-viewer-performance`, commit
`a188f72`, 4 November 2026). Change appearance using **Theme** on the start page.

These changes have software and synthetic-data validation. They do not establish
a new study-data annotation benchmark or an accuracy improvement for V2 or RT
calibration.

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
