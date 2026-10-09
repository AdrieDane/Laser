# Automatic MATLAB annotation engine

For the neutral comparison of both standard-lipidomics engines and shared
terminology, start with
[`STANDARD_LIPIDOMICS_ANNOTATION_REFERENCE.md`](STANDARD_LIPIDOMICS_ANNOTATION_REFERENCE.md).
This document is the detailed reference for the **Automatic MATLAB engine
(XCMS)** only. The alternative LASER reference matcher is documented in
[`STANDARD_LIPIDOMICS_MODE.md`](STANDARD_LIPIDOMICS_MODE.md).

LASER standard-lipidomics mode can run the Amsterdam UMC MATLAB annotation
logic directly from XCMS peak-group lists. MATLAB and `MS*.mat` files are not
runtime inputs.

The automatic workflow supports reverse-phase (RP) and normal-phase (NP)
positive and negative data. Both are processed by default, but as two isolated
phase jobs: each phase constructs its own RTDB calibration, priorities, adduct
set, anchors and regression grids before the result tables are combined. This
standalone application does not expose the LASER 4D workflow.

## Input

Point LASER at a project root or `Processing/Annotation` directory containing
one or both polarities for the selected phase:

- `peakgrouplist_RP_Pos.txt`
- `peakgrouplist_RP_Neg.txt`
- `peakgrouplist_NP_Pos.txt`
- `peakgrouplist_NP_Neg.txt`

One polarity may be omitted, although both are recommended. The GUI defaults
to **Both**; select **RP** or **NP** only when a single-phase run is wanted.
LASER reconstructs independent MATLAB project structures from the selected
XCMS tables and its bundled
`reference_data/matlab_pipeline` catalogue. No analyst-created MATLAB
memberships are read.

Precursor-mass calibration is a separate standard-lipidomics module, not an
annotation pre-step. Every available XCMS stream opens its own reviewed page in
that module, so RP positive, RP negative, NP positive and NP negative never
share a correction. Each page can search labelled standards,
unlabelled standards, Agilent tune-mix ions, or a combined panel; lets the
operator confirm the matched features; and previews constant-ppm, linear,
robust-linear or guarded-quadratic correction with before/after RMS error.
Calibration can be skipped independently for any stream. RP lipid searches use
the RP reference RT window; NP lipid searches use the `RT_NP` values from the
bundled internal-standard RT library. A lipid standard without an RT for its
chromatography is reported as unavailable instead of being matched by mass
alone. Tune-mix ions retain their early-RT guardrail. The module exports
annotation-ready XCMS files and a JSON audit manifest. Raw m/z and model
provenance remain in the exported tables; when those files are selected, the
calibrated value is used consistently by both the MATLAB mapper and its
independent LASER anchor-consensus pass.

If the XCMS annotation folder contains `internal_standards.tsv`, LASER loads
that study-specific spike panel automatically. Otherwise it uses the bundled
26-standard Amsterdam UMC lipidomics panel. Declared standards are protected
anchors: MATLAB's negative internal-standard codes are retained ahead of the
endogenous per-class anchor cap. Output names explicitly carry `(IS)`, while
the audit also records the supplied compound name and concentration. The GUI
shows every expected standard as accepted, review-only, anchor-only, or not
recovered, so missing controls are never silently omitted.

The bundled quantities describe the amount added per 100 µL. The 4D
quantitative-QC workflow uses the same 26-compound unlabelled panel as its
default normalization allowlist. Entries in the broader chemical catalogue
that are marked as possible `(IS)` compounds are not eligible for correction
unless a run explicitly supplies them through
`normalization.internal_standard_allowlist` or `normalization.class_mapping`.

## Automatic annotation

The adapter performs the following steps:

1. reconstructs the 228 MATLAB lipid definitions, measurements, composition
   axes, priorities and R-group rules from the bundled `lipids.tsv`;
2. removes theoretical endogenous targets outside the active inclusive
   class/major-group carbon and double-bond limits while retaining all declared
   internal standards and the original MATLAB composition radices;
3. combines the selected phase's positive and negative features in MATLAB's
   signed-mass domain without importing the other phase;
4. generates conservative RT-database anchor candidates;
5. requires an independent, high-confidence LASER reference match for an
   anchor to seed the MATLAB regression;
6. resolves competing identities globally per XCMS feature and requires enough
   anchors and carbon/double-bond span for each sibling group;
7. runs the unchanged vendored MATLAB staged regression engine;
8. retains candidates from every mapper stage and propagates the final
   calibrated RT model back onto early-stage candidates;
9. optionally, in RP only, completes the protonated PC grid for monoisotopic
   candidates within strict mass and short RT limits, keeping these rows in a
   distinct `matlab_rt_grid_completed` confidence tier; this compatibility
   fallback is disabled by default, so PC normally follows the same staged
   mapper as other lipid classes;
10. in RP only, resolves positional LPC/LPE isomers as coupled chromatographic
   pairs, with 2-acyl required to elute before 1-acyl; and
11. splits the result back into the selected phase's POS/NEG streams, retaining
    the original XCMS feature IDs and sample intensity columns.

Only the RP ion channels used in the curated MATLAB exports are promoted:
`[M+H]+`, `[M+NH4]+`, `[M-H]-`, and `[M-2H](2-)`.

NP instead follows the NP catalogue channels: `[M+H]+`, `[M+H-H2O]+`,
`[M+NH4]+`, `[M+Na]+`, `[M-H]-`, `[M-2H](2-)`, `[M-4H](4-)`, `[M+Cl]-`, and
`[M+CHO2]-`.

## Phase separation and NP identity rules

NP is not an RP grid with a different tolerance. The reconstructed project
contains only an `NP` phase, uses `priority_NP` and `accepted_NP` from the
MATLAB catalogue, calibrates only against NP observations in the MATLAB RTDB,
and fits independent NP sibling grids. Selecting **Both** orchestrates two
phase-local engine calls and merges only their finished streams and export
tables, so their RT models cannot be mixed.

NP LPC and LPE positional forms are represented as one combined phase identity
and one chromatographic peak. The RP early-2-acyl/late-1-acyl resolver is never
called for NP. The orphan-2-acyl rule is also RP-specific. The NP
positional-isomer audit records `not_applicable` rather than implying those RP
decisions were evaluated.

### Optional two-pass positional RT refit

The production default remains `post_regression`: MATLAB first fits its RT
surfaces and LASER resolves RP LPC/LPE positional peaks afterwards. Annotation
settings expose `iterative_refit` as an explicitly experimental alternative:

1. fit the normal provisional MATLAB RT surfaces;
2. resolve LPC/LPE early/late membership inside the normal automatic-anchor
   envelope (normally 30 seconds); this pass supplies memberships only and
   cannot directly accept an annotation;
3. preserve every non-positional membership, replace only RP 1-/2-acyl LPC/LPE
   memberships with those provisional selections, and refit the RT surfaces;
4. run the resolver again with the unchanged strict 10-second RT envelope;
5. recalculate position-specific grid coherence and apply the normal 0.45
   threshold.

This ordering tests whether correcting isomer identity before the final RT fit
improves the fitted positional grids. It does not widen final acceptance, alter
mass tolerance, combine 1-acyl and 2-acyl coherence, or affect NP. The extra
regression increases runtime, so the mode is not enabled by default.

## Mass tolerance

The automatic engine uses 3 ppm for anchor discovery and applies a strict final
acceptance rule:

```text
absolute ppm error < 3.0
```

Candidates at exactly 3 ppm or above can remain in the all-candidates audit,
but cannot appear in the identified-lipid output.

## PC handling

PC uses the same staged regression, mapping, competition, and grid-coherence
workflow as the other lipid classes. LASER does not run a separate
protonated-PC grid-completion pass.

## GUI and headless use

In the GUI choose **Standard lipidomics** and leave **Automatic MATLAB engine
(XCMS)** selected in the **Annotation** module. **Chromatography** defaults to
**Both**; choose **RP** or **NP** to limit the run. Enter the original XCMS
project folder or an output folder from the separate **XCMS m/z calibration**
module; matching peak-group lists are detected automatically. Annotation never
opens a calibration review. Controls are available under **Annotation
settings**, with 3 ppm as both the default and enforced final annotation
maximum.

The default **Unified validated evidence** setting runs one annotation engine
and one class RT model. Internal standards and the top 15 biologically expected
compositions define the stable anchor panel. Positional LPC additionally admits
composition-wide trusted RTDB/XCMS seeds because those seeds are subsequently
checked by its explicit abundant-late-1-acyl/earlier-2-acyl resolver. LPE does
not use an abundance ordering and retains the biological panel. The older
**Biological abundance prior** and **Automatic regression inliers** settings
remain available only for controlled diagnostic comparisons. They are not two
annotation engines.

## Automatic RT-anchor selection

The name **Automatic regression inliers** describes how retained matches are
stored: they become positive MATLAB regression memberships. It does not mean
that every mass match is fitted first and classified afterward from its
regression residual. Anchor selection is a guarded pre-regression process.

### 1. Independent LASER consensus

Before generating MATLAB anchors, LASER runs its separate reference matcher on
the same selected-phase XCMS lists. RP uses the chemical reference and a
0.50-minute RP window and accepts only `high_confidence_auto` rows. NP uses the
anchored Lipid-Wizard NP identity grid and a 0.30-minute window and accepts only
`high_confidence_np_grid` rows. Both use the MATLAB-compatible lipid policy,
3 ppm and grid-coherence minimum 0.45.

For an ordinary endogenous lipid, the MATLAB RTDB candidate must agree with
that pass on all three fields:

- XCMS feature ID;
- normalized lipid species; and
- adduct.

This is an anchor whitelist, not a direct annotation transfer. The staged
MATLAB engine must still accept the target. Internal standards bypass this
consensus because their declared spike identity is protected. RP positional
LPC/LPE sibling groups also bypass it because the chemical reference does not
independently represent both chromatographic positions. NP has no positional
sibling exception because it uses one combined LPC/LPE identity.

The exact LASER confidence criteria are documented under
**Independent consensus mode** in
[`STANDARD_LIPIDOMICS_MODE.md`](STANDARD_LIPIDOMICS_MODE.md).

### 2. RTDB candidate window and score

For every accepted MATLAB sibling definition, the selector enumerates nonzero
stage-1/2 targets with a finite theoretical mass and RTDB prediction. Only the
selected phase's catalogue channels are eligible. An observed XCMS feature
must be within 3 ppm and 30 seconds of the target.

Candidates are ranked using:

```text
score = sqrt((mass error ppm / 3)^2 + (RT error seconds / 30)^2)
```

The best candidate must have `score <= 1.0`. This combined cutoff is stricter
than the surrounding rectangular window near both limits.

### 3. Anchor-mode eligibility

After consensus, **Biological abundance prior** retains internal standards and
expected-abundant endogenous compositions. **Automatic regression inliers**
retains every trusted candidate at this step. It removes only the abundance
restriction; it does not remove consensus, exclusions, ambiguity resolution,
per-grid limits or span requirements.

**Unified validated evidence**, the production default, applies the biological
rule to ordinary classes and LPE, and the trusted composition-wide rule to
positional LPC. Expected-abundance rows are ordered before supplemental LPC
rows. This class-aware combination was selected from the four bundled example
cohorts: it improved LPC recovery while preserving the more accurate LPE RT
model.

The biological prior is generated from weighted common mammalian fatty-acyl
chains. Compositions are listed below in prior order:

| MATLAB/grid family | Eligible compositions |
|---|---|
| 1-acylLPC | 16:0, 18:1, 18:2, 18:0, 20:4, 22:6, 16:1, 18:3, 20:3, 22:4, 14:0, 20:1, 20:0 |
| Other single-chain LPC/LPE/LPS/LPG/LPI/LPA/MG/CE/AC/FA/FA-OH/AC-OH | 16:0, 18:1, 18:2, 18:0, 20:4, 22:6, 16:1, 18:3, 20:3, 22:5, 22:4, 14:0, 20:1, 20:0 |
| Two-chain PC/PE/PS/PG/PI/PA/BMP/DG/SLBPA/MGDG/DGDG/CDP-DG/PGP/DLCL | 36:4, 36:2, 36:3, 38:6, 34:1, 34:2, 38:5, 32:0, 40:8, 40:7, 38:4, 40:6, 36:1, 34:0, 34:3 |
| TG and MLCL | 54:6, 54:5, 54:4, 56:6, 52:4, 56:7, 56:8, 58:8, 54:3, 52:3, 58:9, 52:2, 56:5, 58:10, 50:2 |
| CL | 72:6, 72:7, 74:8, 72:8, 74:9, 76:10, 72:5, 70:6, 70:5, 74:7, 70:4, 76:9, 72:4, 74:10, 76:11 |
| d/m sphingolipids | 34:1, 42:2, 42:1, 40:1, 36:1, 41:1, 38:1, 44:2, 44:1, 40:2 |
| t sphingolipids | 34:0, 42:1, 42:0, 40:0, 36:0, 41:0, 38:0, 44:1, 44:0, 40:1 |
| SPH-d/m and S1P-d/m | 18:1 |
| SPH-t and S1P-t | 18:0 |

Ether/plasmalogen grid variants currently use the corresponding glycerolipid
composition prior. `1-acylLPC` and `1-acylLPE` map to the ordinary LPC/LPE
prior. In RP, a 2-acyl positional class is not biological-prior eligible. RP
positional LPC 20:5, 22:1 and 22:5 are explicitly excluded in both positional
spellings and in both anchor modes. These exclusions do not apply to NP's
combined LPC identity. The 22:5 entry remains available to other single-chain
classes.

These lists define eligibility only. A listed composition is not an anchor
unless the dataset contains a candidate that passes every other rule.

### 4. Global ambiguity and per-grid selection

Duplicate catalogue rows for the same observed feature and theoretical target
are collapsed first. The remaining candidates are resolved globally per XCMS
feature so an isobaric feature cannot independently seed multiple lipid grids.

- The lowest-score identity wins.
- If another identity exists, the winning score must be at least 0.20 lower
  than the next score; otherwise the feature is rejected as ambiguous.
- Only one observed feature is retained per theoretical identity.
- Candidates are ordered by score, with representative intensity as the
  secondary preference.
- Internal standards are retained before endogenous candidates.
- At most 20 anchors are retained per sibling grid.

The complete sibling grid is activated only with at least five retained
anchors and either a carbon span of at least four or a double-bond span of at
least two. If the grid fails this requirement, none of its tentative matches is
inserted as a regression-anchor set.

### 5. Regression membership and audit

Retained anchors are written into the reconstructed project as positive
memberships and enter MATLAB's staged regression with inlier weight 1. The
result records the selected mode in `anchor_settings`, every retained row in
`matlab_automatic_anchors.csv`, and per-grid counts and rejection reasons in
the RTDB seed audit. Consequently, the two anchor modes are reproducible and
can be compared without changing downstream rules.

For code-driven runs, generate a standard configuration:

```bash
python app/laser_cli.py template standard --output standard_phase.json
python app/laser_cli.py run standard_phase.json
```

The relevant configuration is:

```json
{
  "workflow": "standard",
  "engine": "matlab",
  "inputs": {"project": "/path/to/Processing/Annotation"},
  "annotation": {
    "phases": ["RP", "NP"],
    "anchor_selection_mode": "unified_evidence_anchors",
    "anchor_ppm": 3.0,
    "anchor_rt_seconds": 30.0,
    "anchor_score_margin": 0.2,
    "minimum_anchors": 5,
    "maximum_anchors": 20,
    "positional_isomer_rt_seconds": 10.0,
    "positional_isomer_gap_seconds": 8.0,
    "positional_isomer_minimum_separation_seconds": 3.0,
    "positional_isomer_score_margin": 0.2,
    "positional_isomer_local_rt_adjustment_seconds": 10.0,
    "positional_isomer_local_support_groups": 2,
    "positional_isomer_local_composition_distance": 4,
    "positional_isomer_workflow_mode": "post_regression",
    "use_grid_coherence": true,
    "grid_coherence_threshold": 0.45
  }
}
```

Headless output includes identified and equivalent lipids, all candidates, the
MATLAB compound audit, RTDB seed audit, independent-reference seed audit, and
the retained automatic anchor table. It also writes
`matlab_internal_standard_audit.csv` with one row per declared spike.
`matlab_grid_coherence_audit.csv` records the calculated threshold, assessed
rows, unassessed sparse/edge rows and diagnostic outliers. With the default
filter enabled, assessed rows below 0.45 are excluded from automatic
annotations; unassessed rows are retained. Coherence columns remain available
on candidate and annotation exports.

For RP, `matlab_positional_isomer_audit.csv` contains one row per LPC/LPE
composition and adduct considered by the resolver. Seed evidence is keyed by
feature, adduct, family, carbon and double bonds; the resolver validates this
identity again before using it. LPC first selects the strongest 1-acyl feature
inside the strict mass/RT envelope and then accepts a distinct earlier 2-acyl
feature only when order and expected separation agree. LPE uses no abundance
preference. It accepts a complete ordered pair, or a unique 1-acyl singleton
with an exact seed or a bounded neighbouring-grid RT correction; 2-acyl never
survives without 1-acyl. The normal RT error is 10 seconds. Two or more nearby
compositions in the same family/adduct series may supply at most a 10-second
local correction, rather than globally widening the RT window. The audit lists
the selected feature IDs, seed identity validity, local support/correction,
pre-grid selection and post-grid retention. For NP the audit says
`not_applicable`, because LPC/LPE positional forms elute as one combined peak
and are not resolved.

When `iterative_refit` is selected,
`matlab_positional_refit_audit.csv` lists every resolved identity, feature and
RT membership supplied to the second regression. The normal positional audit
also retains the provisional status and feature IDs beside the strict final
decision. A `disabled`, `not_applicable`, or `no_resolved_memberships` status
is explicit; the absence of a refit must never be inferred from an empty plot.

## Validation and limitations

The legacy `.mat` replay entry point is retained only as a development oracle
for regression testing. `app/tools/compare_matlab_annotation.py` tests whether
the port reproduces a previously curated MATLAB project; it is not the
production XCMS-only workflow.

The XCMS-only engine is conservative and should report uncertainty rather than
force an identity when anchors or isomer discrimination are insufficient. It
currently has no CCS or feature-linked MS/MS evidence. RP positional lyso
groups and mass-identical lipid classes remain important review cases; NP
intentionally reports its combined positional identity. The automatic seed
and candidate audit files must therefore be retained alongside the
identified-lipid CSV.

## Provenance

`app/matlab_annotation/vendor/` is a pinned copy of the reverse-engineered
MATLAB annotation modules and must remain unchanged. See
[`app/matlab_annotation/UPSTREAM.md`](../app/matlab_annotation/UPSTREAM.md).
LASER-specific XCMS reconstruction, automatic anchoring, orchestration and
result shaping live outside that directory.
