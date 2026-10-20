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

After the staged mapper fits its final RT surfaces, a calibrated completion
step evaluates the retained monoisotopic proposals inside the strict `<3 ppm`
and 10-second envelope. It requires an allowed phase/adduct, passing grid
coherence, one unambiguous best identity per feature, and a confidence of at
least `0.50` for new calls. Confidence combines mass and RT residuals, local
grid coherence, relative signal strength, feature/identity competition, adduct,
and a separate class effect for each RP/NP/polarity stream. It also scores total
carbon and double-bond composition jointly within each lipid class. The fixed coefficients in
`reference_data/standard_post_regression_confidence.json` were fitted against
15,804 candidates from four independently curated example cohorts and do not
read `MS_data.xlsx` during annotation. RP positional LPC/LPE proposals must
still have been selected by the dedicated positional-isomer resolver, and
declared internal standards retain their protected identity. Decisions are
recorded in `matlab_post_regression_audit.csv` and each evaluated candidate
carries `post_regression_confidence`.

The v2 model adds smooth composition evidence to the existing configurable hard
carbon/double-bond limits. Its class-specific ellipses are fitted on curated
training compositions, with variance floors of four carbon units squared and
one double bond squared. Outlying combinations receive a graded penalty;
odd-chain lipids and compositions absent from training remain eligible. Classes
with fewer than 12 curated observations receive no learned distance penalty.
These are empirical cohort priors, not universal chemical validity rules.

To improve assignments without reducing or inflating annotation totals, v2
reselects at the frozen v1 model's count **per stream**. It reserves declared
internal standards first, takes the best qualifying v2 calls up to that count,
and fills any remaining places with supported v1 calls. Such retained rows are
explicitly marked `baseline_count_retained`, even when their v2 score is below
`0.50`; a new subthreshold candidate is never forced into the output. If feature
or identity conflicts prevent filling the count, the baseline selection is
retained. The audit records the baseline count and retained-call count, and all
demoted proposals are marked as review candidates.

Calibration can be reproduced from the curated benchmark exports:

```bash
python -m pip install -e '.[calibration]'
PYTHONPATH=app python tools/calibrate_standard_confidence.py \
  --benchmark-root /path/to/LASER_standard_benchmark \
  --reference-model reference_data/standard_post_regression_confidence.json \
  --output-model /path/to/recalibrated_confidence.json \
  --report /path/to/cross_validation.csv
```

The bundled model's leave-one-dataset-out evaluation re-fits both the
coefficients and soft composition priors for every fold. On the four stored
strict candidate pools it increases exact curated agreement from 7,979 to
8,418 of 9,557 calls (83.49% to 88.08%), with the same call count in every
stream. These are candidate-stage comparisons, not four complete engine
reruns. The upstream hard ranges and candidate pools were already fixed using
the original example projects. A missing curated match is **unconfirmed**, not
proof of an incorrect biological annotation, and this agreement rate is not an
analytical false-discovery estimate. Evaluation on additional datasets excluded
from training is reported separately.

The aggregate per-stream fold results are saved in
[`validation/standard_confidence_v2_cross_validation.csv`](validation/standard_confidence_v2_cross_validation.csv).
Additional full updated-pipeline runs were evaluated against workbook-to-XCMS
crosswalks requiring corrected profile correlation of at least `0.995`:

| Dataset excluded from model fitting | Calls, before and after | Exact curated matches, v1 → v2 | Agreement, v1 → v2 |
| --- | ---: | ---: | ---: |
| 26P0015_Cells | 2,774 | 2,249 → 2,382 | 81.07% → 85.87% |
| 25P0031_FishLarvae | 2,164 | 1,491 → 1,643 | 68.90% → 75.92% |

For these two runs, the frozen v1 scorer was replayed on the identical fitted
candidates; its cells assignments were also checked against a separate full
v1 pipeline run and were identical. Counts remain equal separately in all four
streams. Cells was an exploratory additional evaluation; fish was evaluated
after the v2 model and selection policy had been fixed. Neither dataset was
added to model fitting or composition-prior estimation.

### Independently supported fitted-grid completion

After v2 selection, `fitted_grid_completion.py` checks the fitted reference
surfaces for feature–target pairs omitted by the staged candidate mapper. It
uses the existing carbon/double-bond ranges, allowed phase/adducts, strict
`<3 ppm` and 10-second limits; it does not extrapolate missing RT models or
generate new positional LPC/LPE assignments. Each new pair needs at least two
distinct, nearby carbon/double-bond nodes from the **pre-existing** accepted
selection, with grid coherence at least `0.45`. Proposed pairs cannot support
one another. Original XCMS feature IDs and protected internal standards are
preserved.

The separate model `reference_data/standard_fitted_grid_confidence.json` adds
class-specific carbon and double-bond effects to the continuous v2 evidence.
It was fitted on the same four original cohorts, not on the additional cells,
fish, ALD brain or heart datasets. A newly introduced identity also needs a
confidence margin of `0.10` over its competing XCMS feature; competing lipid
identities retain the existing `0.10` margin. Selection preserves the preceding
v2 count separately in each stream. Baseline fallback calls remain explicitly
labelled `baseline_count_retained`, rather than being represented as newly
confident calls. This count constraint preserves assignment volume, not the
number of distinct lipid species.

The `matlab_fitted_grid_audit` table records proposed, independently supported,
selected and retained counts. An unavailable model or baseline pair leaves
the preceding selection unchanged and records the reason. Disable this stage
with `use_fitted_grid_completion=False`; disabling post-regression completion
also disables it.

The model fit and leave-one-dataset-out scoring can be reproduced with the
calibration command above plus `--fitted-grid`, using a separate output model
and report. Every fold re-fits its composition priors and coefficients without
the validation cohort. This candidate-stage cross-validation does not include
regenerating omitted fitted pairs; the integrated comparisons below do.

Eight datasets were compared against curated identities **and the same original
XCMS feature**. Workbook rows were mapped to features using mass/RT and
protein-corrected sample profiles with correlation at least `0.995`; unmapped
rows were excluded. Counts are unchanged in all 32 streams:

| Dataset | Excluded from model fitting | Calls, unchanged | Exact matches, v2 → refined | Same identity, different feature, v2 → refined |
| --- | --- | ---: | ---: | ---: |
| 26P0025 ALD brain | Yes | 2,833 | 2,482 → 2,578 | 48 → 32 |
| Rat brain | No | 3,185 | 2,810 → 2,926 | 58 → 33 |
| Cells | Yes | 2,774 | 2,382 → 2,440 | 37 → 32 |
| Fish larvae | Yes | 2,164 | 1,643 → 1,753 | 205 → 152 |
| Rat plasma | No | 1,776 | 1,551 → 1,576 | 20 → 18 |
| Rat spinal cord | No | 2,919 | 2,480 → 2,573 | 35 → 22 |
| Pex5 mouse | No | 2,056 | 1,826 → 1,862 | 18 → 18 |
| Heart | Yes | 2,741 | 2,310 → 2,349 | 56 → 53 |

The final stage was replayed on cached full v2 runs for all eight datasets.
Complete updated-pipeline reruns on ALD brain and heart produced identical
assignment sets to those replays. Heart was evaluated after the rules were
fixed; the other additional datasets informed development, so exclusion from
fitting alone should not be interpreted as a fully untouched test set. These
results are curated agreement, not analytical accuracy or false-discovery
estimates. The aggregate results are stored in
[`validation/fitted_grid_completion.csv`](validation/fitted_grid_completion.csv).

For ALD brain, missing exact curated assignments fell from 1,252 to 1,156;
unconfirmed extra identities fell from 303 to 223. Remaining missing
assignments include 432 eligible but unselected pairs, 255 outside the strict
RT window, 214 absent from the fitted candidate set, 139 failing grid coherence,
108 outside the mass window, four lacking a fitted RT model, three with another
feature selected, and one blocked by positional resolution. Broader windows or
carbon ranges were not enabled merely to reproduce the curated workbook.
Missing curated support does not by itself establish that an extra is wrong.

### RP isomer-aware anchor refinement

The final RP-only pass in `rp_isomer_anchors.py` resolves abundant isomer
pairs before predicting weaker members of their series. It supplements, rather
than globally refits, the staged mapper. The input reference grid holds the
calibrated **RTDB prior**; it is not the final staged-regression prediction.
This distinction matters when the staged O/P or positional curve has drifted.

The method uses the validated primary channels: `[M+H]+` for PC-O/PC-P and
1-/2-acyl LPC, and `[M-H]-` for PE-O/PE-P and 1-/2-acyl LPE. Other channels
retain ordinary engine eligibility but do not receive this additional
isomer-based promotion. NP is an exact no-op, retaining its combined identities
and separate RT methodology.

1. Generate monoisotopic feature/target matches inside the existing strict
   `<3 ppm` mass limit and the 30-second anchor-search window.
2. Jointly assign distinct peaks to the two forms, requiring a minimum
   three-second separation, agreement with the reference gap within eight
   seconds, and an assignment-score margin of `0.15`. LPC/LPE require 2-acyl
   before 1-acyl. O/P ordering comes from the method-specific RTDB, not a
   universal literature ordering. O-C:D is compared with P-C:(D-1), and the
   theoretical masses must actually agree.
3. Rank pairs by the weaker member's intensity, retain at most 20 pairs per
   family/adduct, and reject inconsistent residuals using a median/MAD rule.
   Neither member is assumed to be the more abundant isomer.
4. Predict each target using at least four **other** nearby pair compositions
   in the same family/adduct. The target's own composition cannot support its
   prediction. Support must span at least four carbons or two double bonds,
   and satisfy `|delta carbon| + 2 |delta DB| <= 8`.
5. Apply separate local median RT residual corrections to the two subclasses,
   each bounded by the 30-second anchor envelope. Reject crossing corrected
   curves. Rejoin and resolve features inside the unchanged strict 10-second
   final RT window, check grid coherence and feature/identity competition.
   Missing partners are not manufactured; a new singleton 2-acyl cannot survive
   without its 1-acyl counterpart.
6. Reserve declared standards and their feature IDs, then reselect at the
   preceding annotation count per stream. Standards cannot seed endogenous
   isomer pairs. Baseline calls can fill remaining positions; failures leave
   the preceding selection unchanged with an audited reason.

`rp_isomer_evidence_score` is a deterministic **ranking utility**, not an
analytical probability: `0.95 - 0.10 |ppm|/3 - 0.10 |RT seconds|/10
- 0.10 (1 - coherence)`. Unassessed coherence uses zero in this utility. The
underlying `post_regression_confidence` remains the cohort-model score and is
not replaced by this utility; a physically supported new call can therefore
have a cohort-model score below `0.50`. A selected physical call is labelled
`rp_isomer_anchor_supported`. Raw anchor pairs, per-target prior/corrected RTs,
independent support counts and final selection totals are downloadable as
`matlab_rp_isomer_anchors`, `matlab_rp_isomer_rt_models`, and
`matlab_rp_isomer_anchor_audit` respectively. Disable the pass with
`use_rp_isomer_anchor_refinement=False`; it is also disabled when
post-regression completion is disabled.

Comparison with commit `a8eca37`, using the same curated profile-matched
identity/adduct/original-feature definition as above:

| Dataset | Calls, unchanged | Exact matches before/after | Different-feature calls before/after | Unconfirmed extras before/after |
| --- | ---: | ---: | ---: | ---: |
| ALD brain | 2,833 | 2,578 / 2,616 | 32 / 26 | 223 / 191 |
| Rat brain | 3,185 | 2,926 / 2,975 | 33 / 21 | 226 / 189 |
| Cells | 2,774 | 2,440 / 2,462 | 32 / 31 | 302 / 281 |
| Rat plasma | 1,776 | 1,576 / 1,602 | 18 / 17 | 182 / 157 |
| Rat spinal cord | 2,919 | 2,573 / 2,605 | 22 / 17 | 324 / 297 |
| Pex5 mouse | 2,056 | 1,862 / 1,884 | 18 / 15 | 176 / 157 |
| Fish larvae | 2,164 | 1,753 / 1,761 | 152 / 139 | 259 / 264 |
| Heart | 2,741 | 2,349 / 2,390 | 53 / 51 | 339 / 300 |

This comparison replays the final pass on eight cached complete baseline runs.
Complete updated-pipeline reruns on ALD brain and heart produced exactly the
same assignment sets as the replays. NP tables are unchanged, and assignment
counts are checked unchanged separately in all 32 streams.
Heart was checked after the assignment rules were fixed. The other seven
informed development; exclusion from the earlier confidence-model fitting
does not make them untouched tests. Aggregate results are in
[`validation/rp_isomer_anchors.csv`](validation/rp_isomer_anchors.csv).

The gains are largest for positional lysolipids and PE ethers, not uniformly
for every class or stream. For ALD brain, exact LPC positional matches increase
from 11 to 52 of 85 curated assignments, LPE from 20 to 44 of 59, PE-P from 6
to 44 of 71, and PE-O from 19 to 29 of 53. At a fixed total count, those gains
can displace other previously correct calls: the overall improvement is a
**net** 38 exact matches, not the sum of class-specific gains. Fish has five
more unconfirmed extras despite improving exact matches and reducing
different-feature assignments. Ordinary RP-positive PE is not addressed by
this pass. These are curated-agreement results, not structural confirmation
or an analytical false-discovery estimate.

Only the RP ion channels used in the curated MATLAB exports are promoted:
`[M+H]+`, `[M+NH4]+`, `[M-H]-`, and `[M-2H](2-)`.

NP instead follows the NP catalogue channels: `[M+H]+`, `[M+H-H2O]+`,
`[M+NH4]+`, `[M+Na]+`, `[M-H]-`, `[M-2H](2-)`, `[M-4H](4-)`, `[M+Cl]-`, and
`[M+CHO2]-`.

## Earlier conservative RP-positive PE completion

This subsection records the earlier conservative checkpoint, retained as a
diagnostic fallback. Its metrics and limited NP control describe that earlier
checkpoint; current full four-stream results are reported separately below.

The final `pe_completion` pass addresses ordinary diacyl PE separately from the
preceding isomer pass. It adds supported `[M+H]+` PE on unused RP-positive
features; it never removes an existing annotation, changes another class, or
modifies NP/RP-negative assignments. The preceding passes still preserve their
counts, but this PE-only pass can make a small evidence-supported increase.
Keeping an exact global call count would otherwise exclude good PE simply
because other classes already filled that count.

Candidates must already satisfy the original `<3 ppm`, `<=10 s` fitted RT,
monoisotopic, adduct and grid-coherence gates. Declared internal standards and
their features are excluded. Up to 20 abundant, clearly preferred PE peaks
inside `1.5 ppm / 5 s` establish local mass/RT residuals. Each target requires
at least four **other** compositions within `|delta carbon| + 2|delta DB| <= 8`,
covering at least four carbons or two double bonds. It must fit that local
pattern within `1 ppm / 5 s`; the original mass/RT gates remain mandatory.
Ambiguous split peaks and strict competing non-PE identities are withheld.
An already annotated PE identity is not duplicated at another peak.

When accepted RP-negative PE is available, positive anchors estimate a local
run-to-run RT offset. Both local positive predictions and cross-polarity
alignment exclude the target composition itself. Negative-mode support is an
additional model feature, not a mandatory identity whitelist or an assumption
that two runs have identical retention times. Positive anchors can establish
support without a negative-mode observation. Newly proposed calls cannot
propagate to support one another within the same pass.

The PE-specific evidence model in `standard_pe_confidence.json` uses local
mass/RT residuals, coherence, abundance, independently aligned negative-mode
evidence and broad carbon/DB plausibility. It does not reuse the old global
class-confidence score and imposes no hard odd-carbon whitelist. Scores are
classifier outputs, not structural confirmation or measured false-discovery
probabilities. The default score threshold is `0.90`.
The dedicated `pe_completion_score` decides PE additions; the old
`post_regression_confidence` is retained separately, not overwritten to make
the old model appear more confident.

Only the four original fitting cohorts (rat brain, plasma, spinal cord and
Pex5 mouse) fit this model. There are 108 locally supported training candidates,
85 on the same curated feature. Score-level leave-one-cohort-out checks selected 44
candidates, of which 40 matched the manual peak. Unlisted identities were
treated as unconfirmed negatives for model fitting, not as proven false lipids.
The data are small and related; this is not an analytical accuracy estimate.

The frozen rule was then evaluated on all 15 example folders (14 unique
datasets; duplicate fish is excluded from pooled results). Same-feature PE
matches increased from **6 to 205 out of 1,057 mapped curated PE peaks**.
The pass added 224 annotations: 199 on the same manual peak, nine on a
different peak for a curated identity, and 16 identities absent from the full
curated list. Existing calls were retained. Total annotations rose by 0.60%.
Results are in [`validation/pe_completion.csv`](validation/pe_completion.csv).
Most curated RP-positive PE remains missing; this conservative pass is a first
improvement, not a complete PE resolution. EDT brain and lengthening spinal
cord have weaker agreement among additions and warrant particular review.

Full updated RP runs from the original inputs were completed for all 15
folders. Every run reproduced the final-stage PE choices exactly and preserved
all pre-existing RP-positive and RP-negative assignments. NP was not rerun
for every folder: NP uses an unchanged baseline, exact no-op unit tests and a
separate full ALD-brain NP control that matched its baseline. The new PE calls
were also checked for accepted status in the actual candidate/grid-browser
tables, not merely their presence in an offline benchmark. The test suite
passes 256 tests and 82 subtests; saved-project round trips and the packaged
module/model were checked. Original inputs and previous TXT exports remain
unchanged.

The app exposes `matlab_pe_completion_audit`, `matlab_pe_anchors` and
`matlab_pe_candidate_decisions`, persists them in saved projects and offers
CSV downloads. The candidate `pe_decision` describes local-pattern checks;
the final selection additionally requires the PE score and feature-competition
checks. The incoming candidate flags in this diagnostic table are not final
acceptance flags; the updated stream `all_pairs` and `annotations` are final.
`use_pe_completion=false` disables PE completion at the configured composition
limits, and disabling
post-regression completion also disables this pass. The independent PE score
can be controlled with `pe_completion_confidence_threshold`. Cache version 22
ensures older cached runs are not mistaken for results of this change.

`tools/calibrate_pe_completion.py` reproduces the model and leave-one-cohort-out
checks from exported PE candidate diagnostics and curated crosswalks. Supply
only development cohorts to `--datasets`; do not fit on evaluation cohorts.
All curated comparisons occur after annotation. No curated workbook is read
by the PE runtime pass.

## Expanded PE peak selection (default)

`pe_rebuild` replaces the earlier PE completion method by default. Separate
models support RP-positive, RP-negative and NP-positive PE. NP-negative
retains its preceding method. Each model searches its original feature list
again, so a candidate need not survive
the old fitted-grid or 10-second RT filter. It does not reuse curated peaks
as runtime anchors or read curated workbooks during annotation.

The new pass uses:

- A bounded 40-second RT discovery window. Sparse reference-grid RTs are
  interpolated/extrapolated locally from the existing method reference.
- A strict **less than 3 ppm corrected** mass window. Its residual offset is
  the median error of at least 20 independently accepted non-PE features
  within the old 3 ppm / 5-second envelope. With insufficient support the
  correction is zero. Raw mass and original RT errors remain in the result;
  they are never overwritten to make a candidate appear closer.
- A local RT residual plane varying with carbon count and double bonds.
  Up to 30 abundant candidate compositions establish support. Every target's
  prediction excludes its own composition and requires at least five others.
- Raw opposite-mode peaks at the corresponding PE `[M-H]-` / `[M+H]+` mass,
  within 2/5/10/20-second cross-mode neighbourhoods, plus accepted
  opposite-mode PE when available. Presence in its annotation list is not
  mandatory. These windows
  are evidence features, not an assertion that both runs have identical RT.
- Abundance, corrected mass error, local/reference RT errors and broad
  composition evidence in a small, portable boosted-tree agreement model.
  No dataset name, feature identifier, curated label, or species one-hot
  input enters the model. Odd carbon is evidence, not an automatic exclusion.

The curator guide `annotatie_gateways.pptx` (slides 12, 15, 17, 20, 35)
motivated comparing abundant peaks, local mass/RT patterns, edge flexibility
and stronger negative-mode PE evidence. Its workflow instructions are not
executed by LASER. Its conditional biology notes are not hard-coded as
universal cut-offs. Missing grid nodes are not automatically filled.

One peak is chosen per new PE composition, with a minimum score advantage
over the second choice. Existing accepted annotations, their features and
declared internal standards are protected. Old non-PE *review candidates*
are not an unconditional veto: PE-specific evidence can resolve a candidate
that failed the earlier global decision. No existing annotation is replaced.
The models cover ordinary `[M+H]+` PE in RP-positive and NP-positive mode and
`[M-H]-` PE in RP-negative mode. Ether/plasmalogen PE and other classes are
not processed by this pass. Counterpart annotations are snapshotted before
completion: additions in one stream cannot reinforce themselves through
the other. RP/NP references, RT predictions, evidence and models remain
separate. The `pe_negative_*` feature names are retained for compatibility;
in the RP-negative model they describe the **positive** counterpart.

Carbon/DB limits apply to regenerated targets as well as the earlier engine.
The default ordinary PE carbon range is now 28--50, with 0--13 double bonds.
This shared range update can also change PE candidates in NP and RP-negative
mode; fresh validation must therefore check all four streams. Other class ranges are
unchanged.

Public settings and matching setup controls:

| Setting | Default | Meaning |
| --- | --- | --- |
| `use_pe_rebuild` | `true` | Use expanded PE selection; false restores the earlier PE method |
| `pe_rebuild_score_threshold` | `0.10` | Coverage-oriented evidence threshold, **not** a 10% probability/FDR |
| `pe_rebuild_feature_margin` | `0.03` | Required score advantage over another peak for the same composition |
| `pe_rebuild_search_seconds` | `40` | Discovery half-window, limited to 60 seconds |
| `use_pe_completion` | `true` | Master enable switch for either PE method |

The generic post-regression master switch also disables both PE methods.
The earlier `pe_completion_confidence_threshold=0.90` applies only when
`use_pe_rebuild=false`; it is not the threshold for the new model. Cache
version 25 invalidates older app results. Missing or wrong-stream model files fail closed.
The runtime uses NumPy/Pandas; sklearn remains calibration-only.

A separate final safeguard handles explicitly declared PE internal standards,
not endogenous PE. It restores a standard only when exactly one monoisotopic
feature meets the original raw `<3 ppm` and `<=10 s` limits with the declared
primary adduct. Existing identities and occupied features are not replaced.
Ambiguous standards remain for review. The endogenous class-coherence filter
does not reject these independently supported spikes; its original score and
flags remain available in the audit. This final step cannot feed back into
earlier fitting and is disabled by either PE/post-regression master switch.

The existing `matlab_pe_completion_audit`, `matlab_pe_anchors` and
`matlab_pe_candidate_decisions` tables carry the new method's evidence.
Unlike the earlier diagnostics, the new `pe_final_selected` and
`pe_decision=accepted` fields explicitly indicate final selection, including
score, peak-choice margin and occupied-feature checks. Final accepted rows
also have `automatic_accepted=true`, `automatic_candidate=true` and
`rank_for_feature=1`, so they are full annotations in the review grid and
exports. Scores are agreement with manual choices, not structural proof.
The review grid prioritizes the run's accepted feature over a closer-to-old-
reference alternative (manual feature overrides still take precedence).
Accepted PE is displayed using its local RT prediction and independently
corrected mass error, while preserving the raw errors in the stored results.

`tools/calibrate_pe_rebuild.py` fits the portable JSON model from independently
generated `raw_candidates.pkl` tables and curated feature crosswalks supplied
on the command line. Four original development cohorts are used for the
three shipped models. `--stream` selects the stream when recalibrating.
Other example cohorts and leave-one-dataset-out checks assess
agreement, but the collection has informed development and is not a new,
prospectively untouched validation set. The fish duplicate is excluded from
pooled metrics and grouped calibration.

The packaged models are `standard_pe_rebuild.json` (RP-positive),
`standard_pe_rebuild_rp_negative.json` and
`standard_pe_rebuild_np_positive.json`. Each stream's model is fitted
separately on the same four development cohorts. A stream cannot use another
stream's model. Leave-one-dataset-out checks hold out the PE scorer and final
PE selector; preceding cached engine models are not refitted inside those
folds. Treat these checks as development estimates, not fully independent
end-to-end validation.

### Four-stream PE checkpoint, 20 October 2026

All 15 example folders received full RP/NP core runs. Pooled figures exclude
the duplicate fish folder (14 unique datasets). The denominator includes
every manual ordinary PE entry, including unresolved original-peak mappings
counted as misses. A match requires the same name, adduct, stream and
original feature, not just the same lipid name.

| Stream | Same manually chosen peak | All manual PE entries | Agreement |
| --- | ---: | ---: | ---: |
| RP-positive | 980 | 1,071 | 91.5% |
| RP-negative | 1,213 | 1,267 | 95.7% |
| NP-positive | 919 | 945 | 97.2% |
| NP-negative | 910 | 1,000 | 91.0% |

The target is met for each pooled stream, **not every individual dataset**.
There remain 36/31/6/8 wrong-peak choices and 184/144/126/28 PE names absent
from the corresponding stream's manual list, respectively in table order.
Unlisted identities remain unconfirmed, not proven false annotations.
The next priority is reviewing these additions and the weaker individual
datasets without losing coverage. All examples informed development; this
is agreement with this collection, not an external accuracy estimate.

The late, independent declared-standard safeguard was replayed on all 15
completed core outputs. An additional fresh final-version plasma RP/NP run
verified exactly the same annotations in all four streams. It cannot affect
earlier models. The final checkpoint passes 299 tests and 83 subtests.
The wheel's runtime modules and all PE model bytes match source.

See [`validation/pe_multistream_summary.csv`](validation/pe_multistream_summary.csv),
[`validation/pe_multistream_by_dataset.csv`](validation/pe_multistream_by_dataset.csv)
and [`PE_HANDOFF.txt`](PE_HANDOFF.txt) for the continuation state and local
diagnostic report locations. Existing TXT exports were not regenerated.

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
