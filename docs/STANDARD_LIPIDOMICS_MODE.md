# Standard lipidomics mode

For the neutral comparison of both standard-lipidomics engines and shared
terminology, start with
[`STANDARD_LIPIDOMICS_ANNOTATION_REFERENCE.md`](STANDARD_LIPIDOMICS_ANNOTATION_REFERENCE.md).
This document is the detailed reference for the **LASER reference matcher**.

LASER Standard Lipidomics is a standalone workflow for conventional XCMS
feature tables without ion mobility/CCS or feature-linked MS/MS. It was
extracted from LASER 4D so it can be installed and released independently;
the 4D workflow is not exposed by this application.

## Two annotation engines

The **Annotation engine** control at the top of standard mode chooses between:

- **Automatic MATLAB engine (XCMS)** — the default workflow. It reconstructs
  independent NP and RP projects from their positive/negative peak-group lists,
  derives conservative phase-specific RT anchors, and runs the vendored MATLAB
  staged annotation logic separately for each phase. RP-only and NP-only runs
  remain available. It does not require MATLAB or
  `MS*.mat` files and accepts identified lipids only at strict `<3 ppm`. See
  [`MATLAB_ANNOTATION_ENGINE.md`](MATLAB_ANNOTATION_ENGINE.md).

- **LASER reference matcher** — the workflow this document describes. It matches
  XCMS peak-group lists against LASER's chemical reference under the MATLAB
  `lipids.tsv` policy, with guarded class RT adaptation on RP and the anchored
  Lipid-Wizard grid on NP. It works from peak-group lists alone, including
  manual uploads.

Both produce the same result shape, so the run summary, the exports and the
grid browser below work identically for either.

## Input contract

Use one or more tab-separated XCMS peak-group lists whose filenames contain:

- `NP_Neg`
- `NP_Pos`
- `RP_Neg`
- `RP_Pos`

The files produced for the MATLAB workflow normally use names such as
`peakgrouplist_RP_Pos.txt`. Required columns are detectable feature ID, m/z,
and RT columns; the standard XCMS names are `name`, `mzmed`, and `rtmed`.
Retention time is interpreted as minutes.

The automatic MATLAB engine processes RP and NP by default as isolated phase
runs; either phase can be selected alone. The LASER reference matcher also
supports all four streams. A project folder is sufficient; the GUI detects the
available lists automatically.

Standard mode uses the bundled `reference_data/matlab_lipid_policy.tsv`
policy by default. A different file can be entered under Annotation settings;
clearing that override makes LASER try the project's usual
`Data/Meta/MS_database/lipids.tsv` location. If the project contains
`analyse_MS_data.m`, LASER also
reads its `R_group.FA.range` setting; otherwise the MATLAB defaults of 8–42
carbons and 0–8 double bonds per fatty-acyl R group are used.

Columns beginning with `Group_` are XCMS group-detection summaries and are not
treated as sample intensities. Numeric columns after the structural/group
columns are retained as per-sample intensities.

## MATLAB-compatible policy

Before mass matching, LASER expands `lipids.tsv` into stream-specific rules and
requires all of the following:

- the exact class/composition variant is defined;
- the rule is enabled for the NP/RP and positive/negative stream;
- the candidate adduct matches a listed MATLAB measurement;
- the phase-specific priority is positive;
- total carbon and double-bond counts are compatible with the configured
  per-R-group composition range.

MATLAB's phase-dependent grouping is retained. Ether/plasmalogen candidates can
use an `O+P` rule in the phase where those forms are combined, and positional
lyso candidates can use the unpositioned lyso rule there; the preferred phase
uses the separate rules when their priority is higher.

Every candidate export records the selected MATLAB policy class, measurement,
priority, R-group count, composition range, rule index, and source files. A
separate policy-audit CSV includes every reference row considered and reports
whether it was allowed or rejected because of class variant, stream, adduct,
priority, or composition.

## Configurable total-composition limits

The collapsed **Carbon and double-bond annotation limits** panel contains an
editable table for the major lipid groups and another for every observed lipid
class. The defaults are the inclusive total-carbon and total-double-bond
extrema observed in the four example `MS_data.xlsx` datasets, combining their
raw Primary and Equivalents sheets and excluding internal standards. They are
stored in `reference_data/standard_lipid_composition_ranges.tsv`.
Ordinary PE is the updated exception: its default carbon range is 28--50
after checking the broader example collection. Other class defaults retain
their original bounds. These are adjustable method limits, not universal
biological cut-offs.

By default both the group range and class range must pass. Turn off **Apply
major lipid-group limits** to keep only class-specific limits. Clear **Use
group** for one class to exempt only that class from the group layer, or clear
**Filter enabled** to exempt it from every composition limit. The master
**Enable carbon and double-bond limits** checkbox disables the policy
completely. Internal standards are protected in every configuration.

Headless runs use the same defaults. A compact configuration can enable or
disable the layers, while optional `groups` and `classes` mappings override
only the named defaults:

```json
{
  "annotation": {
    "composition_ranges": {
      "enabled": true,
      "use_group_ranges": true,
      "groups": {
        "Two acyl/alkyl chains": {
          "carbon_min": 28,
          "carbon_max": 54,
          "db_min": 0,
          "db_max": 12
        }
      },
      "classes": {
        "PC": {
          "enabled": true,
          "use_group_range": false,
          "carbon_min": 26,
          "carbon_max": 66,
          "db_min": 0,
          "db_max": 13
        }
      }
    }
  }
}
```

## Annotation evidence

- RP positive and negative streams start from the bundled historical RP RT
  database. Global RT alignment is disabled in the automatic workflow.
- For each class/adduct, LASER searches for unambiguous mass-matched seed
  nodes. A class offset is applied only with at least five distinct nodes,
  sufficient carbon/double-bond span, a bounded shift, and a material
  leave-one-out improvement over the historical RT. Rejected models leave the
  original class reference untouched.
- NP positive and negative use the bundled Lipid-Wizard NP reference. Its
  `NP_RT` values are stored in seconds and converted to minutes on loading.
  LASER maps positional lyso and phase-combined `O+P` policy names to the
  reference identity before matching.
- LASER estimates one dataset offset from the dominant cluster of unambiguous
  mass anchors. It activates the NP grid only with at least 12 anchors from
  three classes, at least two minutes of reference-RT span, an offset no larger
  than one minute, anchor MAD no larger than 0.15 min, and 90th-percentile
  centered error no larger than 0.30 min. A failed calibration leaves the NP
  stream as explicitly withheld mass-only candidates.
- The identity RT values remain frozen: the anchor fit shifts the grid but
  cannot learn identities from the dataset being annotated.
- CCS and MS/MS evidence are disabled in standard mode.
- Polarity is fixed from the stream name before matching.
- Grid coherence is calculated from the complete local m/z/RT candidate
  neighbourhood for both engines. The standard-mode controls expose **Use grid
  coherence as an annotation filter**, enabled by default, and a minimum score
  of `0.45`; assessed local outliers are excluded before automatic selection,
  while sparse/edge nodes that cannot be assessed remain retained. Disabling
  the filter retains the scores as diagnostics without excluding candidates.
- PC follows the same staged mapper and grid-coherence workflow as the other
  lipid classes; there is no separate RP protonated-PC completion pass.

Each exported row carries `stream`, `chromatography`, `polarity`,
`rt_evidence_mode`, MATLAB policy provenance, class-model
coefficients/support, candidate ranks, and an automatic annotation status.

## Automatic decisions and abstention

The automatic resolver:

1. ranks identities for each feature;
2. selects at most one feature per exact lipid/adduct identity;
3. refuses to select a feature with multiple passing isobaric identities;
4. requires sufficient RP RT support or a validated NP anchor calibration;
5. chooses one primary quantitative feature for an accepted lipid across
   streams and retains other accepted measurements as equivalents.

Possible statuses include:

- `high_confidence_auto`
- `class_grid_supported`
- `mass_rt_supported`
- `high_confidence_np_grid`
- `np_grid_supported`
- `ambiguous_isobar`
- `duplicate_feature`
- `alternative_identity`
- `low_reference_support`
- `np_grid_low_confidence`
- `np_grid_unvalidated`
- `mass_only`

The first five are automatically accepted. NP mass-only or unvalidated-grid
matches remain visible in the all-candidates export but are not promoted to
identified lipids. Automation means making a reproducible decision when the
evidence is sufficient and reproducibly abstaining otherwise.

## Independent consensus mode used by the MATLAB engine

The Automatic MATLAB engine uses this matcher internally to build an
independent whitelist of ordinary endogenous RT-anchor identities. This is a
tighter invocation of the same LASER algorithm, not a benchmark and not a read
of `MS_data.xlsx`.

The consensus pass:

1. loads the selected phase's XCMS peak-group lists and reference;
2. applies the MATLAB-compatible class, stream, adduct, priority and
   composition policy;
3. generates reference pairs within 3 ppm;
4. for RP, applies the historical or validated class-specific RT model and a
   0.50-minute window; for NP, applies the independently anchored Lipid-Wizard
   grid and a 0.30-minute window;
5. applies grid coherence at the default minimum 0.45;
6. ranks identities per feature and features per identity/adduct; and
7. retains only `high_confidence_auto` rows for RP or
   `high_confidence_np_grid` rows for NP.

For RP, the two-dimensional score is:

```text
score = sqrt((mass error ppm / 3)^2 + (RT error minutes / 0.50)^2)
```

`high_confidence_auto` requires all of the following:

- the candidate is the best identity for its XCMS feature;
- it is the best XCMS feature for that exact identity and adduct;
- no second passing identity makes the feature an ambiguous isobar;
- `score <= 0.75`;
- the reference identity carries at least ten RT observations; and
- the candidate passed the active policy and coherence filter.

For NP, `high_confidence_np_grid` additionally requires the guarded NP
dataset-offset calibration to validate. NP identity-level RTs are supplied by
the Lipid-Wizard grid; they are not borrowed from the RP reference. The MATLAB
engine then uses the matching feature/identity/adduct only as a whitelist for
its separate NP RTDB anchor and still runs the NP MATLAB regression.

Peak support contributes only a small tie-breaker when one identity has
multiple candidate features. It never relaxes the m/z or RT window.

The consensus output is a set of exact `(XCMS feature, normalized species,
adduct)` tuples. It does not directly annotate the MATLAB-engine result. A
MATLAB RTDB candidate must agree with the tuple before it can become an ordinary
endogenous regression anchor, and the staged MATLAB loop must still process it.
Internal standards and RP positional LPC/LPE sibling groups use the exceptions
documented in
[`MATLAB_ANNOTATION_ENGINE.md`](MATLAB_ANNOTATION_ENGINE.md#automatic-rt-anchor-selection).

## Graphical workflow

Launch LASER normally and select **Standard lipidomics** on the first data
screen. The next screen exposes two independent module toggles:

- **XCMS m/z calibration** reviews one model per available stream and exports
  `peakgrouplist_<stream>.txt` files plus a
  `LASER_standard_mass_calibration.json` audit. It does not annotate.
- **Annotation** reads original or calibrated XCMS files and does not run or
  offer precursor-mass calibration.

Select both to calibrate first and pass the exported folder to annotation, or
select either module on its own. In Annotation, enter either the MATLAB/XCMS
project root, its `Processing/Annotation` folder, or a calibrated export folder,
then click **Detect project inputs** or run directly. LASER finds all available
peak-group lists automatically. Manual per-stream uploads and custom reference
paths remain available under **Annotation settings**.

The primary download, `LASER_standard_identified_lipids.csv`, contains one
automatically accepted primary feature per lipid. A separate equivalents
download preserves other accepted measurements of those lipids, and the
all-candidates download preserves every passing or withheld candidate with its
reason. The MATLAB policy and NP reference/calibration audits can be downloaded
from their own expanders.

Under **Annotation settings**, RP positional LPC/LPE processing can be switched
from the current **Post-regression resolver** to **Two-pass resolver and RT
refit (experimental)**. The experimental choice is available in both standard
annotation and is stored in headless configurations. It is
never selected automatically and does not change the NP workflow. Completed
runs export `matlab_positional_refit_audit.csv` so the provisional memberships
used by the second regression remain reproducible.

## Grid browser

After annotation, the **Lipid class grid browser** provides a focused
conventional-MS review surface. It uses a three-column layout, a predicted-RT
mesh over the carbon ×
double-bond surface, the same anchor confirmation, click and box selection,
per-sample statistics, diagnostic strips and same-lipid-ID projection. It drops
the CCS axis and 3D view, the MS/MS viewer and fragment criteria, mobilograms,
and the CCS and RT calibration stages. For MATLAB-engine/XCMS results, selecting
a point also shows a diagnostic chromatogram below the grid. The main trace
overlays all discovered samples (or one selected sample); XCMS RT bounds and
integration fields are shown as review metadata. If the same lipid ID is carried
by multiple XCMS features, their chromatograms appear in a second panel below
the main trace, with one colour per feature and dotted lines for review-only
candidates. The legend contains only the numeric XCMS feature IDs; sample names
remain available in hover text and the optional single-sample selector. These
traces are visual evidence only and never change annotation.
Automatic MATLAB mode calculates the same local grid-coherence score using m/z
and RT residuals only. With its default filter enabled, assessed rows below
0.45 are excluded from automatic annotations; sparse/edge rows that cannot be
assessed are retained. Scores and decisions are displayed and exported.

- **Left column** — the plotted dimensions, live m/z, RT, and grid-coherence
  review criteria, and the grid rendering options (predicted grid, C:DB labels,
  grey feature cloud). The column collapses to give the plot more width. Its
  initial controls reproduce the annotation run, including whether grid
  coherence was enabled and the threshold that was used.
- **Centre** — the grid itself, plus the selected-feature chromatogram and
  same-lipid-ID chromatogram panel when a MATLAB/XCMS point is selected. Green
  points are accepted annotations, blue points are candidates that passed the
  original run criteria, and light-blue points are extra candidates exposed by
  relaxing the viewer criteria. Yellow is reserved exclusively for RT anchors
  selected by the active annotation engine. A manually confirmed review anchor
  is a purple diamond. Yellow and red connectors show whether each observed
  feature elutes after or before its grid node.
- **Right column** — per-sample intensities for the selected lipid, the class
  and adduct selectors, one diagnostic row per species (intensity heatmap, grid
  coherence, ΔRT, Δmass), the selection actions, and the inlier/outlier and
  anchor tables.

RP uses the historical or validated class-shifted grid; NP uses the
dataset-anchored Lipid-Wizard grid. If NP calibration fails, the browser labels
the unshifted reference as diagnostic rather than active.

The review controls are **view-only**. The m/z, RT, and grid-coherence controls
decide which candidates the browser draws; their defaults reproduce the
original annotation criteria. Turning grid coherence off or lowering its
threshold can reveal originally excluded candidates in light blue. Confirming
anchors moves only the displayed grid, and deleting an annotation hides it for
the rest of the browsing session. None of these actions touch the automatic
annotations or any export — those remain exactly what the run produced. Because
every m/z candidate is retained, the RT window
may be widened past the run's tolerance to inspect near-misses; the m/z window
can only be tightened.

## Command line

Run either annotation engine directly from an XCMS project or Annotation
folder:

```bash
laser-standard /path/to/Processing/Annotation \
  --engine matlab \
  --phase both \
  --output results/standard_lipidomics
```

Use `--phase RP` or `--phase NP` for one chromatography phase. Select the
alternative matcher with `--engine laser`. Each result table is written as a
CSV in the output directory.

## Saved standard project, optional report, and annotated input copies

Every completed GUI annotation run writes `LASER/Annotation/LASER_PROJECT.json`
alongside machine-readable result tables. **Save LASER project** regenerates
these files. Opening the saved annotation restores the standard grid browser,
retained annotations, candidates, audits, and settings without rerunning
annotation. Untargeted results are saved separately.

`standard_lipidomics_report.html` remains available as an optional convenience
export. It is not required to reopen the project. The report follows the main
review structure while remaining faithful to conventional XCMS
evidence. It includes selected primary identities, equivalent features, review
candidates, per-class and per-stream summaries, per-sample intensity profiles,
mass/RT/grid residuals, RT-m/z maps, RT-anchor and grid audits, policy/range
decisions, and the exact annotation settings.

Standard peak-group lists do not contain raw chromatograms, mobility, CCS,
feature-linked MS/MS, or MZmine shape metrics. The report marks those evidence
types as unavailable rather than inferring them or displaying empty negative
results.

The workflow also creates `LASER/Annotation/annotated_peaklists/`, containing one copy of
each discovered `peakgrouplist_<phase>_<polarity>.txt`. Each output preserves
the input filename with `_annotated` inserted before `.txt` (in both the GUI
and CLI), so it is distinguishable from the original. It preserves
the original columns, row ordering and field text exactly, then inserts these
as columns 2 and 3:

- `laser_annotation`
- `adduct`

Unannotated feature rows receive two empty appended fields. If one XCMS feature
has more than one accepted identity/adduct pair, position-matched values are
semicolon-separated in both columns. The GUI exposes the same four files under
**Annotated copies of the input peak-group lists**.

The direct CLI writes both output types by default. Their locations can be
changed with `--report-output` and `--annotated-input-directory`.

## Optional untargeted comparison

The GUI offers untargeted analysis as a separate option, off by default. It
can run from XCMS peak-group lists alone or after annotation. When run after
annotation, LASER first
removes every XCMS feature present in the accepted automatic-annotation table,
including the selected primary feature and accepted equivalent features. It
then compares the remaining features between two biological groups. Review-only
lipid candidates remain eligible because they were not accepted annotations.

After annotation finishes, LASER opens a **Standard lipidomics results** page
instead of leaving the annotation-input form above the results. It provides
separate review workspaces and an annotation Save control. When untargeted
analysis is enabled:

- **Open annotation review** opens the lipid grid, anchor review, annotation
  audits and annotation exports;
- **Open untargeted analysis** opens the two-group criteria, volcano plot,
  significant unknown-feature table and a dedicated RT-m/z feature map below
  the volcano. The map contains only tested, unannotated features: grey points
  did not pass the current differential gates and coloured diamonds did. It
  does not open the annotation lipid-class grid browser.

Both pages use the same completed annotation result. **Back to results** moves
between them without repeating annotation, while **Start a new annotation**
explicitly clears the completed result and returns to the input form.

Both workflows save automatically under the XCMS project folder:
`LASER/Annotation` and `LASER/Untargeted`. Annotation includes the annotated
peak-group lists in `LASER/Annotation/annotated_peaklists`; untargeted includes
its statistical tables in `LASER/Untargeted/tables`. The start screen opens
either saved workflow in the GUI without recomputing it. Each workspace has a
Save control that regenerates the exports from the loaded result. By default,
the previous export moves to `LASER/Archive/<workflow>/<timestamp>`; the user
can choose to overwrite it instead. LASER copies discovered XCMS lists, sample
workbooks and `internal_standards.tsv` to `metadata/` on save while leaving
their original source files in place.

### Authoritative sample groups

Sample groups come from the project Excel metadata workbooks, for example
`26P0026_ALD rat brain_NP.xlsx`. LASER joins:

- workbook `File Name` to the XCMS intensity-column name, exactly;
- workbook `Mode` to the stream (`RP_Pos`, `RP_Neg`, `NP_Pos` or `NP_Neg`);
- workbook `Group Label` to the statistical group;
- workbook `Group Name` to the user-facing label; and
- workbook `Sample Type` to decide whether the injection is biological.

Only `study`, `sample` or `biological` sample types can be compared. QC,
pooled-QC, blank, internal-standard and calibration injections are excluded
from both statistical groups. The report and headless output contain an Excel
join audit with the workbook path and number of matched sample columns. When
no suitable workbook is available, LASER can use a validated XCMS `Group_*`
column-order mapping as a visible fallback; it never guesses groups from file
names.

### Statistics and display

For every stream independently, LASER applies a Welch two-sample t-test to
log2 intensities. The pseudocount for each feature is half its smallest
positive intensity. The untargeted screen uses the unadjusted, two-sided
p-value; no multiple-testing correction is applied. The default highlight
gates are:

- raw p-value strictly below 0.001;
- absolute log2 fold change at least 1.0; and
- non-zero detection in at least 50% of either biological group.

Four volcano panels (`RP_Pos`, `RP_Neg`, `NP_Pos`, and `NP_Neg`) are shown
above the dedicated untargeted feature map. They use synchronized axes but do
not mix distributions from different acquisition modes. To keep projects with
hundreds of thousands of features responsive, the four panels share a fixed
budget of at most 12,000 non-significant background features and the map draws
at most 20,000 per selected chromatography mode. Every raw-p-screened hit
remains visible, and no rows are removed from calculations, tables or exports.
Red means higher in group B and blue means higher in group A. Within either
direction, points just below the active raw-p cutoff are pale and points become
more saturated as p decreases. Full saturation is reached two orders of
magnitude below the cutoff, making colour strength comparable across runs.

Because raw p-values are not corrected for the number of tested features,
LASER labels these points as **raw-p hits** rather than multiplicity-controlled
discoveries. They are intended for exploratory prioritization and require
orthogonal follow-up.

The feature map shares one retention-time axis across ionization modes.
Positive-mode m/z values are above zero and negative-mode values are mirrored
below zero with a display-only minus sign, so RP positive and RP negative (or
NP positive and NP negative) can be inspected together.

Selecting a coloured raw-p hit in this map opens its sample-level abundance
pattern directly below the map. The view toggles between the original XCMS
injection order and a group-wise violin/box plot with every sample point; either
view can use raw or `log2(intensity + 1)` values. A companion table records each
biological group's sample count, detected count and percentage, median, first
and third quartile, and IQR. The selected feature also shows its raw p-value and
log2 fold change. This diagnostic does not recalculate the p-value or change
annotation.
The active feature remains marked by a large purple diamond and labelled
callout, independently of its red/blue p-value-scaled display colour.

LASER provides five interpretation aids without changing identity status:

1. local class context from nearby accepted lipids in RT-m/z space;
2. nearby unannotated features separated by one to four CH2 mass steps within
   a narrow RT neighbourhood;
3. connected CH2-spaced pairs or series formed specifically by other raw-p
   hits in the same acquisition stream;
4. co-eluting M+1 and M+2 candidates at the expected carbon-13 spacing, with
   their observed relative intensities, a charge hypothesis and a deliberately
   approximate carbon-count estimate from M+1/M. When this evidence is
   available, LASER also reports a short list of mass-matched, isotope-
   constrained CHNOPS elemental-formula hypotheses and their mass errors; and
5. supporting candidates in the other chromatography phase or in the opposite
   ionization polarity. Cross-phase evidence uses mass only because NP and RP
   retention times are not directly comparable. Opposite-polarity evidence
   requires a corresponding adduct hypothesis and similar RT in the same
   chromatography phase. When at least three biological specimens can be
   uniquely paired, LASER also calculates a Spearman abundance correlation on
   `log2(intensity + 1)`. Pairing uses an explicit Excel specimen identifier
   first, exact sample names second and a conservative unique normalized
   acquisition name last. Correlation is used before mass/RT residuals to rank
   otherwise compatible counterparts and is shown with its pair count and
   pairing method.

Each selected hit can be added to a persistent feature-review basket as
**keep**, **reject** or **follow up**, with a free-text note. The complete basket
is downloadable as CSV and is embedded in the HTML report. Keep and follow-up
rows also form a targeted reacquisition CSV containing m/z, RT, phase, polarity,
feature ID and a default plus/minus 0.10-minute RT window. A new standard run
starts with an empty basket so decisions cannot leak between projects.

All are hypotheses for follow-up. They do not create a lipid annotation or an
exact molecular formula. Neutral-mass alternatives and isotope-derived carbon
estimates remain explicitly labelled as unresolved composition hypotheses.
The HTML report contains a dedicated **Untargeted differences** tab, and
headless runs export `differential_unannotated_features.csv` plus
`untargeted_sample_metadata_audit.csv`. The standard workflow template records
all thresholds under `annotation.untargeted` for reproducibility.

## NP reference provenance and limitations

The bundled `reference_data/lipid_wizard_retention_times.csv` is an exact copy
of Metabo-Wizard's `data/retention_times.csv` at source commit
`b79150b9c3778980d8a3b699c0d1be6a8de7edbc`; its SHA-256 is
`9af33d996f7e6495394e8edfd9c908b74c9a8590200e0b8ed3884701c4562fb7`.
It contains 5,550 identities with NP RT values.

The bundled `reference_data/matlab_lipid_policy.tsv` is the standard policy
used for the four validation projects and is selected by default. The bundled
copy makes the workflow self-contained when only peak-group lists are
available; a project-specific policy can be selected under Annotation settings.

The NP result is an accurate-mass plus anchored-RT annotation, not a
structure-confirmed identification. Every NP class with O and P variants is
reported as a combined `O+P` identity; the workflow intentionally does not
distinguish ether from plasmalogen forms that coelute in NP. In the shared
notation, `O-n:d` and `P-n:(d-1)` map to `O+P-n:d`; source names/classes remain
in audit columns. Positional lyso isomers remain combined where the MATLAB
phase policy combines them. Isobars not separated by the reference remain
withheld without orthogonal evidence.
