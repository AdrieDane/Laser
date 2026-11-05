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

## Planned untargeted improvements

The [staged formula prediction and adduct annotation plan](docs/UNTARGETED_FORMULA_ADDUCT_PLAN.md)
defines nine implementation stages, their validation requirements and review
checkpoints. [Stage 1 is complete](docs/validation/untargeted_stage1/README.md):
the baseline and evaluation definitions are frozen, and the selected-PC 10/14
counterpart result was reproduced in two identical replays. Later
stages cover acquisition-specific adduct chemistry, full-feature ion grouping,
complete isotope patterns, external formula engines, joint inference and optional
raw chromatogram/MS/MS evidence. [Stage 2](docs/validation/untargeted_stage2/README.md)
adds project method settings and experimental method-aware adduct scoring;
its development comparison is complete. It retained 10/14 PC counterparts but
lost one all-class reference link, so current scoring remains the default.
[Stage 3](docs/validation/untargeted_stage3/README.md) adds experimental full-feature
strict/soft ion grouping. Its complete development comparison increases reference
coverage but regresses selected-PC counterparts to 9/14; current grouping remains
the default. The native Binner benchmark is still pending a trusted executable.
[Stage 4](docs/validation/untargeted_stage4/README.md) adds optional complete-ion
isotope evidence using IsoSpec, per-sample/QC ratios and explicit quality flags.
It preserves existing statistics, compound assignments and legacy formula labels;
the current isotope model remains the default.

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
All five modules also share a saved per-user `project_folder` preference. On
macOS it is stored in `~/Library/Application Support/LASER Standard Lipidomics/settings.json`;
Windows uses AppData/Roaming and Linux uses the XDG configuration directory.
`LASER_STANDARD_PROJECT` takes precedence over this saved preference.

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

Annotation and untargeted progress pages have an always-visible **Live terminal**
below the progress bar. It refreshes every two seconds, shows the latest worker
output, CPU time and time since the last log message, and detects stopped
processes. Logs remain visible on completion or failure. Future untargeted runs
also report feature counters during the longer cross-mode and adduct searches.

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
fold-change and detection filters. Select a list row
to see sample bars with metadata group colours, isotope partners, candidate
formulas, adduct links, and CH₂-series neighbours. Non-significant and annotated
features remain searchable as supporting counterparts.

Adduct searches infer neutral masses under H, NH₄, Na, K and water-loss positive
ions, plus deprotonated, formate, acetate, chloride and water-loss negative
ions. Doubly deprotonated ions remain supported; doubly protonated and Na−2H
negative ions are excluded. Isotope spacing constrains charge when available.
Matches require mass tolerance and co-elution
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

The completed untargeted viewer links a mirrored RT/m/z plot, the sample bars,
and a detection table in the browser. Selection, mode switches and highlights
do not rerun Streamlit. Initially only significant points are shown, with no
automatic selection. Choose **RP POS and NEG**, **NP POS and NEG**, or
**All 4 modes** (separate RP/NP panels). POS and NEG zoom together within each chromatography, with mirrored mass ranges; RP and NP zoom independently. Each pair has its own reset button. Grey non-significant background points
are off by default; enabling them shows a fixed sample of up to 20,000 points.
Related cluster and adduct detections are always shown, including non-hits.
Selected features and a representative possible counterpart per other mode
are bright green and slightly larger; significant increases are red and decreases blue (B versus A), cluster members purple, and
additional adduct candidates yellow. Significant cluster neighbours have an orange
fill and purple border. **Colour intensity by p-value** optionally makes smaller raw p-values stronger red/blue; selected, cluster and adduct colours stay unchanged. Hover shows original feature, compound and cluster IDs, plus accepted lipid identities when available. The table can select detections and filter by compound,
feature, mode, predicted formula or m/z. **Review selected feature** opens the saved review controls. Volcano plots are omitted from this viewer.
Purple highlights follow a connected carbon/double-bond grid when a coherent two-dimensional family is supported. Occupied neighbouring cells, an RT trend and a maximum per-edge RT gap are required. Sparse cases retain direct-neighbour evidence. Arbitrary transitive mass-link components are not highlighted; selecting another feature replaces the family.

Tentative ion groups receive shared `compound_0001` IDs across adducts and modes;
each table row retains its original feature ID, m/z, RT, likely adduct and mode.
Grouping uses one consistent neutral-mass hypothesis and a co-eluting ion family
within each chromatography; equal-mass RT isomers are not collapsed together.
Compound grouping v5 ranks candidate families using paired biological sample
profiles and mass errors. When at least six unique specimens are evaluable,
discordant abundance profiles are retained as candidate evidence but excluded
from selected groups. QC and blank injections never support this ranking.
Water-loss assignments require a coeluting intact parent in the same mode;
singleton adducts remain unresolved. Existing projects update these tentative
groups when opened, without rerunning statistics or changing accepted lipids.
The detection and evidence tables include predicted neutral chemical formulas
from the existing isotope-supported candidates, conditional on the assigned
adduct. A linked ion inherits a formula only when its compound has one consistent
candidate; unavailable or conflicting predictions remain marked as such.
Ambiguous adduct alternatives remain available as evidence. These display IDs
are not structural identifications. Homologous relationships between distinct
compounds receive separate `cluster_0001` IDs, propagated to their adducts and
mode counterparts. Both label tables are saved with the untargeted project.
Existing results receive labels on opening without rerunning group tests; use
Save untargeted analysis to persist them. Non-significant cluster counterparts
use a browser-local mass/RT lookup against the already loaded feature coordinates.

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

Selecting a compound automatically shows one sample abundance plot for each detected mode, using that mode's original sample values and group colours. The significant-feature table stays immediately below the mode plots. Saved evidence with distant same-mode RT values is excluded from coloured adduct highlights.

Mode-viewer points use circles, with selected features slightly larger. Significance direction is comparison group B relative to reference group A. Selection overrides significance colour; selected-cluster neighbours are purple, with significant neighbours orange and bordered purple.

Untargeted evidence version 3 uses carbon isotope abundance odds, M+2-supported
charge selection, and a physical carbon-count bound to reject impossible isotope
partners. Formula candidates use a wider vectorized CHNOPS search and soft
lipid-headgroup/H:C priors. They remain tentative ranked candidates; the code is
not a full implementation of SIRIUS or the Seven Golden Rules. Same-mode adduct
coelution defaults to 0.05 min, independently configurable from opposite-polarity
coelution. QC/blank/standard comparisons are rejected by the analysis backend,
even when conflicting metadata labels them as study samples. Their abundance
bars remain visible. The evaluation runner preserves original-feature mappings,
full manual denominators, source/input hashes and all four modes.

For further method development, [CAMERA](https://pmc.ncbi.nlm.nih.gov/articles/PMC3658281/)
and [Ion Identity Molecular Networking](https://www.nature.com/articles/s41467-021-23953-9)
support combining coelution, isotope and abundance evidence in consistent ion
networks. [Seven Golden Rules](https://doi.org/10.1186/1471-2105-8-105) and
[SIRIUS](https://pmc.ncbi.nlm.nih.gov/articles/PMC2639009/) motivate chemical and
isotope constraints. Exported feature tables lack scan-level peak shapes and
MS/MS spectra, so full chromatographic/MS/MS methods require richer inputs.

Double-click an axis or its tick labels to reset only that dimension for its linked POS/NEG pair. The other dimension and chromatography retain their zoom.

Click and drag pans the mode plots by default; the scroll wheel zooms and the toolbar can switch between Pan and box Zoom. POS/NEG panning remains linked within a chromatography.

Current untargeted ion forms (evidence version 4): positive `[M+H]+`,
`[M+NH4]+`, `[M+Na]+`, `[M+K]+`, `[M+H-H2O]+`; negative `[M-H]-`,
`[M+HCOO]-`, `[M+CH3COO]-`, `[M+Cl]-`, `[M-H-H2O]-`, `[M-2H]2-`.
Water-loss forms are marked as possible in-source fragments. Unsupported ion
forms are filtered out when regrouping older saved evidence for display.


Sample abundances use a fixed 2 × 2 panel: RP POS and NP POS above RP NEG and NP NEG, with a
single shared group legend below. Missing modes retain an empty slot. Each
panel header includes m/z, RT, tentative adduct, compound ID and feature ID.
Connected lipid-grid version 1 adds H2/double-bond differences and mixed CH2/H2
steps to local family discovery. It requires a two-axis seed, a fitted RT trend,
occupied adjacent grid cells and no RT-gap jumping. No manual identities enter
the search. Formula and adduct predictors are unchanged by this grid extension.

Annotated lipids within the selected cluster use a purple centre with a green outline; hover retains their accepted lipid identity. Selected detections remain larger and bright green, and significant cluster members remain orange with a purple outline.

Abundance panels use compact three-line headers: mode; m/z, RT and adduct; then compound and feature IDs. The scatter and abundance plots share aligned top and bottom row edges. Each POS/NEG scatter pair displays one shared retention-time tick scale in the middle, with the negative-mode label and retention-time title below the bottom panel. RP and NP remain independent.
