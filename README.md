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
  metadata/                 # sample-group workbooks and internal_standards.csv
```

**Calibration** supports independent m/z and RT calibration, or both, for each
available RP/NP and positive/negative stream. Review the standards and click
**Save calibration**. **Annotation** offers **Raw XCMS data** or **Calibrated XCMS
data**. For calibrated data, choose the current export or an archived version
from **Calibration version**; dates and available streams identify each version.
The selected files are recorded with the saved annotation.

Place sample-group workbooks and **internal_standards.csv** in `metadata/`.
The CSV has `name,concentration,unit` columns. When present, it supplies the
project's internal-standard list and quantities for calibration and both
annotation engines. Names must match the bundled standard catalogue; malformed
or unknown entries produce an error. Saved calibrations retain their own copy,
so selecting an archived calibration also selects its original standard list.
Projects without a CSV retain the legacy internal-standard behaviour.

Annotation runs in a separate background Python process and saves its outputs
automatically. Closing the browser does not stop it; keep the computer and
worker process running. Reopen Annotation for the same project to view progress.
Logs and run status are stored in `.laser/jobs/annotation/`. Completion shows
**Annotation complete** and **Back to start**; open the saved project from the
start page when ready to review it.

The annotation workspace contains the reviewer with a top-right **Back to
start** button. Both axes offer RT, m/z and CH₂ Kendrick mass defect. Sample
intensity bars use metadata groups, with sample names available on hover.
**Previous class** and **Next class** follow the class dropdown order. Mounted
plots and local linked selection follow the approach in
[LASER-4D `be478d6`](https://github.com/YRJJaspers/LASER-4D-lipidomics/tree/be478d6):
click to select, Ctrl/Command-click to toggle, and Shift-click to select a range.
Selected points and diagnostic rows highlight immediately while the reviewer
updates without rerunning the surrounding page.

**Untargeted analysis** is a pipeline launcher. Choose the sample groups and
criteria, then run it in an independent background process. It uses the saved
annotation when available and excludes accepted lipid features from testing.
Without a completed annotation it explicitly offers analysis of all raw XCMS
features. Inputs, sample groups and accepted-feature exclusions are frozen at
launch. Results save in `Untargeted/`; logs and status are under
`.laser/jobs/untargeted/`. Completion shows a message and **Back to start**.
**Open completed untargeted analysis** is the separate viewer and never reruns
statistics or chemical searches. Its Save button preserves review changes with
the archive/overwrite choice.

The viewer lists all unannotated features below the selected **unadjusted raw
p-value** threshold. A separate column shows whether each also passes the
fold-change and detection filters used by the volcano/map. Select a list row
to see sample bars with metadata group colours, isotope partners, candidate
formulas, adduct links, and CH₂-series neighbours. Non-significant and annotated
features remain searchable as supporting counterparts.

Adduct searches infer neutral masses under H, NH₄, Na, K and water-loss positive
ions, plus deprotonated, formate, acetate, chloride, Na−2H and water-loss negative
ions. Doubly protonated/deprotonated ions are also considered; isotope spacing
constrains charge when available. Matches require mass tolerance and co-elution
within the same chromatography. RP↔NP links compare mass without equating their
retention times. Every compatible source/target adduct hypothesis is retained;
none is silently converted to an identification. The common ion families follow
[LIPID MAPS](https://www.lipidmaps.org/tools/ms/gp_ox_form.php).

Isotope evidence searches co-eluting M+1 and M+2 carbon-13 spacings at charge one
or two. Candidate neutral CHNOPS formulas use isotope-derived carbon counts
(rounded estimate ±3), exact mass, charge, and a closed-shell DBE screen,
including phosphorus. The bounded search uses N=0–3, O=1–14, P=0–2 and S=0–2;
the five highest-ranked candidates are displayed with total candidate count.
Approximate carbon-13 M+1/M+2 ratios rank candidates and account for adduct
carbon. Heteroatom isotope fine structure is not modelled. These are hypotheses;
mixed peaks, other isotope contributions,
labelled compounds and formulas outside the search bounds can remain unresolved.
Carbon isotope spacing follows [CIAAW](https://ciaaw.org/carbon.htm).

Local homologous-series evidence searches ±1–4 CH₂ increments (including C₂H₄),
scaled by isotope-supported charge or an assumed single charge. Links require
both the configured ppm/absolute mass limits and RT neighbourhood. Connected
components of three or more features are candidate series; isolated pairs are
labelled as pairs. Annotated neighbours supply class context without assigning
the unknown feature to that class. Full adduct links, homologous links, raw-p
hits and statistics are saved as project tables and restored in the viewer.
Older projects open with their stored evidence; rerun the pipeline to generate
the expanded searches.

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
the LASER-4D branch (`codex/annotation-viewer-performance`, theme snapshot
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
