# Stage 7: original-feature chromatographic evidence

Authorized 6 November 2026. **Software/internal comparison complete; CURRENT
remains the default after a routine-workflow regression.** Native CAMERA is still
finishing its bounded mode attempts; UFA comparison is open after an input-adapter
contract failure, and native MZmine activation remains pending. Stage 8 has not
started. This is one development acquisition, not independent validation.

## Completed LASER comparison

Matched 3,413-query original-ion/adduct agreement (all 3,668 manual rows retained):

| Mode | Table-based before | Peak shapes after | Unconfirmed extras before → after |
| --- | --- | --- | --- |
| RP positive | 46.6% (790/1694) | 47.0% (797/1694) | 1201 → 1200 |
| RP negative | 48.3% (328/679) | 49.2% (334/679) | 1102 → 1107 |
| NP positive | 66.8% (506/757) | 67.4% (510/757) | 882 → 874 |
| NP negative | 67.7% (364/538) | 68.2% (367/538) | 908 → 900 |

These are correct adduct calls on the same original feature, not lipid-identity
accuracy. There are 36 gains and 16 losses: net +20; extras fall by 12 overall.
Expected original-feature/adduct pairs improve from 446/960 to 451/960. Matched
PC counterpart recovery stays 9/14. Four-mode, all-class counts, full manual
and assigned-ion denominators, abstentions and all 252 class/mode cells are in
[the aggregate tables](all_class_mode_adduct_recovery.csv).

The separate **normal 762-query workflow regresses**, so the option is experimental:

| Mode | Before | After | Unconfirmed extras before → after |
| --- | --- | --- | --- |
| RP positive | 2.7% (46/1694) | 2.7% (45/1694) | 461 → 454 |
| RP negative | 0.4% (3/679) | 0.4% (3/679) | 464 → 456 |
| NP positive | 4.9% (37/757) | 4.5% (34/757) | 171 → 170 |
| NP negative | 0.9% (5/538) | 0.9% (5/538) | 194 → 193 |

Routine all-manual percentages are coverage: most manual entries are accepted
lipids or not normal raw-p query roots. They are **not conditional adduct accuracy**.
The aggregate table separately reports agreement among assigned reference ions
and assignment coverage. Four correct calls are lost with no gains; extras fall
by 17. Pair recovery falls 21/960 → 19/960. Normal PC counterparts stay 10/14.
Extras here require nonempty adduct calls; the original saved display included
219 additional unassigned rows, whose historical counts are preserved separately.

All 251,431 original statistical rows/evidence columns are exact. The real routine
trace analysis and viewer payload are exact after save/reopen. All 254,324 original
coordinates and all 94 original prepared inputs are retained. Five derived trace
metadata files were added to the inputs-only test project; no saved results added.
Independent QA found 31 inconsistent original apex/window rows (11/4/2/14 in
RPpos/RPneg/NPpos/NPneg). None were queried by the trace measurements or conditional
QC formula inputs. The release rejects those windows explicitly; original metadata
is retained without repair. Tests: 1,100 passed, nine skipped, 91 subtests. An initial
wrong-interpreter test invocation lacked the isolated isotope dependency; the
failed invocation is retained privately and the declared runtime passes.

Trace inference source 3f5a84c; replay 28a45a4 includes numeric-key and original-column
checker compatibility only. Current control was copied byte-identically from the
completed prior replay and exactly reproduces Stage6 sequential assignments.
Source24c5edc adds the unused-invalid-window safeguard. No weights, thresholds,
query scopes or class/mode outputs were selected after scoring. All failed setup,
checker and adapter attempts remain preserved outside Git. Model-only selector
timing does not establish full browser-render latency.

**Native UFA trial is not an accuracy benchmark.** Three low-carbon charge controls
passed, but an independent C120 ammonium-ion control exposed a format mismatch:
original monoisotope anchor yields zero native hits; the required most-abundant
isotope anchor yields the correct formula. The prototype supplied original
monoisotopic XCMS coordinates to native `MassMAIso` inputs. That trial was stopped;
partial outputs remain private, with all query statuses retained. No native formula
accuracy is reported. Source dcb753f withholds scores for pools requiring a different
anchor. An appropriate raw-envelope anchor bridge and reusable database preparation
are future work; per-query database generation also confounded this prototype's
runtime. This is an adapter limitation, not evidence that native UFA predicts poorly.

The protocol below was fixed before study scoring. Native account/format/timeout
limitations must remain open; do not describe Stage7 as fully benchmarked.

## Installation and frozen protocol

The optional **Raw peak shapes (experimental)** route uses original XCMS
`groupidx` memberships, per-injection peak boundaries and the original global
raw-to-corrected RT maps. Export this read-only bridge with
`Rscript tools/export_xcms_trace_metadata.R PROJECT ORIGINAL_XCMS_RESULTS NEW_OUTPUT`,
then place the five exported files in `PROJECT/metadata/trace_sources/`.
All original group coordinates and sample filenames must match; XCMS group RT
dispersion is never substituted for a chromatographic peak width. Canonical
`Raw/STREAM` files must be present. R is needed for this one-time legacy export,
not for normal LASER workers or reopening saved results.

At launch LASER freezes hashes of the bridge, original coordinate tables and raw
files. The detached job verifies these before measurement. It reads each eligible
injection once, extracts 4-ppm EICs inside the original corrected boundaries with
0.05-minute padding, and persists per-sample quality, pair summaries and first-QC
representative traces. Biological samples and QC provide technical evidence;
blank/IS injections do not. Selection in the viewer uses saved summaries only.
Missing metadata preserves the table-based route with an explicit warning.

## Prospective rules, fixed before study scoring

- Within the SAME injection only: at least five points above 10% of the
  baseline-corrected apex, two on each edge, and a shared observed apex window.
  Baseline is the 20th intensity percentile; the SNR proxy uses median absolute
  first-difference deviations, with minimum ratio five. This is not a measured
  detection limit or calibrated signal-to-noise estimate.
- Reject gap-filled/multiple native peaks, competing mass centroids, dominant
  neighbouring peaks, multimodal traces, nonmonotone scans and scan gaps over
  three times the median interval. Insufficient coverage remains unavailable.
- Pearson shape correlation at least 0.85 AND normalized overlap at least 0.5
  in at least 80% of usable injections, with at least two usable injections,
  supply support. Overlap is sum(min(x/sum(x), y/sum(y))) on common actual scans.
  Median correlation below 0.5 OR overlap below 0.25 with at least two usable
  injections supplies discordance; intermediate evidence stays ambiguous.
- No pair-specific shifts, stretching or interpolation to create shape agreement.
  The original global RT map locates windows; no artificial scans are generated.
  POS/NEG and RP/NP are different injections: no shape correlation across them.
- Supported same-mode links sort ahead of unknown links, then by the unchanged
  mass error. Existing group quality gains 0.25 × median shape correlation.
  Discordant same-mode links are vetoed. Unavailable/ambiguous links retain their
  existing quality. Other rules are unchanged. This first experiment supports
  current grouping only, isolating trace evidence from joint-inference changes.
- Isotope-pair shapes are a diagnostic sidecar. Existing isotope/formula outputs
  are unchanged. No verified isotope-member gold exists, so recognition precision
  and recall cannot be claimed.

The five-point/two-edge/0.85 choices are informed by the
[MZmine metaCorrelate documentation](https://mzmine.github.io/mzmine_documentation/latest/module_docs/id_ion_networking/metacorr/metacorr.html).
LASER's overlap and replicated-injection decision rules are its own prospective
implementation, not a claim of native MZmine equivalence.

## Frozen comparison

Use the same ALD development acquisition and original coordinates: all 3,413
natural formula-query keys, all 3,668 manual rows (including 46 unmapped), all
63 classes × four modes. Blind workers receive physical coordinates and sample
measurements, never reference names/classes/formulas/adducts/charges. Original
physical isotope endpoints may be measured without supplying their charge to
ion grouping. Replay current and trace arms on identical keys; separately repeat
the normal 762-query workflow and fixed 15-PC viewer regression. Compare actual
JavaScript representative choices. Preserve old statistical/evidence columns
exactly; declared display/group labels may change. Verify typed save/reopen and
source/input hashes. No post-hoc threshold tuning, class/mode winner splicing or
automatic default promotion.

Report all-manual original-ion/adduct agreement with counts and percentages,
assignment coverage and abstentions, conditional agreement, wrong original peaks,
reference pairs, group consistency, unconfirmed extras, gains AND losses,
trace availability/quality reasons, isotope diagnostic coverage and processing
cost. These are curated-ion agreement measurements, not confirmed lipid identity,
independent validation, analytical FDR or calibrated confidence.

## Native engines and limits

CAMERA 1.58.0, xcms 4.0.2 and mzR 2.36.0 are isolated in a private R 4.3.2
environment. This is the compatible Bioconductor 3.18 release, not latest CAMERA.
Raw-reader startup is checked before study work. Retain loader version warnings.
Use original xcmsSet groups and the first QC injection per mode, `groupFWHM`
sigma=6/perfwhm=0.6, `findIsotopes` maxcharge=2/maxiso=4/ppm=4/mzabs=0/minfrac=0.5,
and `groupCorr` correlation=0.85/pval=0.05/hcs, CiS enabled, CaS disabled.
`findAdducts` ppm=4/mzabs=0/multiplier=3/max_peaks=100 uses configured LASER common
monomer ions, preserving native rule priorities where available; acetate=14 and
water-loss=99. Excluded user ions and multimers remain excluded. Preserve all
native alternatives; multiple adduct labels are unresolved, never selected using
gold. Report full native coverage plus query-touching pseudogroups separately.
Native CAMERA is within-mode; do not fabricate cross-mode links. Its one-QC
evidence footprint differs from LASER's repeated-injection support, so neither
is an intrinsic algorithm-only comparison. Budget: 1,800 seconds per mode,
one native thread, at most two mode processes; failed/time-limited modes remain
explicitly unperformed in full denominators. No success-only pooling.

IDSL.UFA 2.0 and IDSL.IPA 2.9 are installed in the same isolated environment.
Conditional raw-profile formula comparison will use supplied-adduct inputs and
the complete unranked LASER formula candidate pool, not reference formulas or a
LASER isotope-ranked shortlist. It is native reranking of a shared candidate
universe, not independent candidate generation or blind adduct accuracy.
Use the first QC per mode; actual centroid spectra, scan indices and observed
apex intensity inside the native boundaries are required. Fixed native controls:
peak spacing 0.005 Da, intensity cutoff min(C,10)%, mass accuracy 0.01 Da,
maxNEME=5, minPCS=950, minNDCS=2, minRCS=5, score coefficients c(1,1,1,1,1),
RT tolerance disabled, one thread. Native source implements its NDCS window as
intensity >=20% of the apex (the top 80% of height); preserve this native behavior.
Budget 1,800 seconds per mode. Missing mappings,
ambiguous peaks, candidate-empty cases and native failures stay in all 3,413
denominators. Native charge/electron conventions must pass synthetic controls
before actual study calls; otherwise record the adapter as unavailable.

Native MZmine 4.11.0 still requires a compatible login; no account is created or
study spectra uploaded. Its study comparison remains pending, as do previous
Binner/SIRIUS comparisons. Stage 7 must not be called fully benchmarked while
required native comparisons are unperformed. No independent isotope gold or MS/MS
was introduced. Detailed study outputs and raw/native mappings stay outside Git;
only code, protocol and safe aggregates are published.
