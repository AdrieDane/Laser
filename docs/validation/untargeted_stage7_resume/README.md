# Native formula preparation and comparison follow-up — 7 November 2026

## Completed comparison — 7 November 2026

**Keep CURRENT unchanged; do not promote native UFA.** The preparation repair
works at full scope. All1007 batches covering257407 formula/adduct requests were
attempted and persisted, and all four matching processes completed within their
unchanged job-timer budgets. Native UFA did not improve this configured conditional
formula workflow. All3413 formula keys remain in these primary denominators:

| Mode | LASER lipid-aware | Native msbuddy | Native UFA, reported top1 |
|---|---:|---:|---:|
| RP_Pos | 1527/1592 (95.92%) | 1500/1592 (94.22%) | 317/1592 (19.91%) |
| RP_Neg | 443/585 (75.73%) | 462/585 (78.97%) | 10/585 (1.71%) |
| NP_Pos | 670/724 (92.54%) | 705/724 (97.38%) | 98/724 (13.54%) |
| NP_Neg | 444/512 (86.72%) | 437/512 (85.35%) | 25/512 (4.88%) |

Combined reference-formula agreement: LASER3084/3413 (90.36%), msbuddy3104/3413
(90.95%), UFA450/3413 (13.18%). These are **supplied-adduct/charge conditional**
results, not adduct annotation accuracy, confirmed lipid identities or the accuracy
of the unchanged CURRENT default. LASER's lipid-aware and msbuddy options were
already evaluated in the frozen Stage5 comparison; their previous outputs are
reused unchanged. No native result was selected per mode or class.

UFA returned a top-ranked formula for2112/3413 queries:450 agree and1662 disagree
with curated formulas. It could not evaluate1144 queries:337 missing/multiple/
gap-filled QC mappings,694 insufficient/ambiguous original monoisotopic traces,
87 strong interfering/unverified channels,19 ambiguous major anchors and7 requiring
a missing native profile. Another157 completed without a retained native candidate.
All these cases remain in the primary denominator; they are not dropped.
Native ranked truth recall1760/3413 and reported-rank≤5 recall1059/3413 are separate
from shared generation recall3409/3413. Ten requested profiles were not returned
by the native generator (7 deprotonated,3 protonated); queries requiring an omitted
profile abstain rather than silently rank an incomplete pool.

Against LASER lipid-aware, UFA gains5 and loses2639 original same-query top1 matches.
The library-absent127-query subset is retained: LASER106/127, msbuddy106/127,
UFA2/127. This same acquisition/library exposure is development evidence, not
independent validation. Full [all-mode results](all_mode_formula_comparison.csv),
[63-class/four-mode results](all_class_mode_formula_metrics.csv),
[gains and losses](all_class_mode_transitions.csv),
[availability reasons](all_mode_status_counts.csv),
[observed anchor counts](observed_anchor_counts.csv),
[requested/returned profiles](profile_universe.csv).

### Ranking interpretation and evidence limits

Inspection of the pinned native source confirms that IDSL.UFA2.0 sorts its
IdentificationScore decreasing, then emits sequential ordinal ranks1..N and
omits the underlying scores. Top1 here preserves that native order; uniqueness
of an underlying score cannot be established. The adapter retains fractional
ranks if supplied, but zero reported rank ties in this version does not mean
zero underlying score ties. This is a reporting clarification discovered during
source inspection, not a reranking or changed prediction. The initial private
scorer's “unique best” terminology and class manual-denominator labelling were
corrected; all per-query predictions and top1 counts are unchanged.

UFA receives first-QC raw chromatographic windows with its native filters and
complete shared generated pool. LASER uses sample-integrated isotope evidence,
lipid-aware priors and budget32; msbuddy has its own candidate/eligibility rules.
Those different footprints, priors, budgets and QC availability prevent an
intrinsic algorithm ranking. The result rejects promotion of **this configured
workflow**, not every possible UFA configuration. No weights, filters, tolerances,
anchor rules or budgets were tuned after study results. No independently verified
isotope-member gold, analytical FDR, calibrated confidence or MS/MS is claimed.

### Runtime and integrity

Preparation used4 workers,501.19 seconds on the Python monotonic job timer
(limit600). The Mac slept from approximately09:04 to09:37 America/Denver:
UTC freeze-to-cache-pin elapsed2446.35s and native R-reported elapsed2445.54s.
Both clocks are retained;501s is **not the wall-clock duration**. Matching job timers:
RP positive44.71s,RP negative12.99s,NP positive13.31s,NP negative11.17s, each below120s;
2 modes at a time. No manual budget extension or retry after scoring.
[Timing details](timing.json), [frozen provenance](provenance.json).

All94 prepared files,5 derived trace metadata inputs, original raw sources,IS,
method,2212 original/staged native package files and5 saved statistics/annotation
Parquet snapshots retain their hashes. Profile cache hashes match before/after
matching. Actual pruning equals full native matching on16 fixed eligible queries,
four in every mode. Production app source is unchanged fromdd0331d; current
annotation, statistical and grouping workflows were not rerun or overwritten.
Project result folders remain empty. [Validation](validation_checks.json).

Software regression suite:1105 passed,9 skipped,91 subtests passed; one detached-worker15-second timing check failed amid host cache reads and passed on a focused unchanged-limit retry in4.27s with the external bytecode cache inherited by child processes. No application or test source was changed. [Test results](test_results.json).

Full native formula comparison is now complete for this configured experiment.
MZmine/SIRIUS/Binner access, prior CAMERA positive-mode timeouts and independent
isotope/identity validation remain open. Stop for review before Stage8.


User authorized this follow-up with “oke do it”. CURRENT remains default; no
Stage 8, historical annotation optimization, trace threshold tuning or new
biological/statistical pipeline run. This is the same development acquisition.

## Frozen implementation and controls

Cache schema3 pins R/native versions, native constants SHA256, JIT-stable native
function hashes and the exact cloned optimizer. Two native temporary allocations
that are overwritten before any read are omitted; the native peak-merging work
array remains intact. Pure element-sorter results are memoized by both arguments
in cloned environments, leaving installed packages and arithmetic unchanged.

Native256-formula batches have atomic RDS plus SHA256 commits. Uncommitted orphan
files are rebuilt, corrupt committed records fail explicitly, live writers are
protected, and a killed writer's stale lock is recovered. Every batch is retained
for a later resume. Four workers share the entire queue, including the large
ammonium pool; one worker is not reserved per adduct. No formula filtering or
candidate budget reduction is used for speed. Complete universes are required
before study matching, with eight fixed unmodified-native profile probes/adduct.

High-carbon/both-polarity/charge2/interference controls and522-profile native parity
pass the prior30/5/10s gates (overall120s cap). The actual killed-writer test retains
one batch unchanged and computes only two missing batches; cold/warm/parallel /
serial equality passes for3072 profiles and256 fixed unmodified-native probes.
Corruption, orphan, missing-universe and live-lock checks pass. The initial3072-
profile unmodified-engine test exceeded120s and remains privately recorded; the
final test checks all3072 serial/parallel results plus256 fixed native probes.
No study outcomes were inspected to make this test-efficiency change.

A separate broad synthetic32768-formula/adduct preparation gate finished in93.31s
(limit120s), warm4.69s (limit30s), with32 original-native probes exact and all warm
cache hashes identical. The largest test pool is twice the smaller positive pool.
The local isolated native runtime is byte-identical to2212 source package files.

The adapter preserves fractional native ranks instead of rounding tied ranks to
integers. The evaluator will distinguish a unique best formula from a tied-best
set; ties cannot be silently counted as uniquely correct. A pruning parity failure
is fatal, rather than an ordinary unavailable query. These are adapter/reporting
safeguards, not changes to native ranking, filters or raw-anchor settings.

## Prospective study protocol

Freeze and push source, all runtime/constants/raw/config hashes and cache recipe
before preparation or matching. Use the exact existing full unranked generated
candidate pools for all3413 formula-query keys, all63 classes and four modes;
retain all3668 manual rows,46 unmapped and all252 class/mode cells. No curated
formula/class/name labels enter native workers. Native known adduct and charge
are explicitly supplied: this is conditional formula prediction, not blind
adduct assignment. Full generated pool recall is separate from ranked survival.

One complete preparation process:600s maximum,4 one-thread workers. Matching:
120s per mode,2 one-thread workers concurrently. No expanded timeouts after study
outcomes. Each completed cache is pinned before any study query scoring. First
four successful queries per mode must exactly match unpruned native execution.
Missing, ambiguous, failed, unattempted and tied-best cases remain explicit in
full denominators. No tuning/per-mode winner splicing after scoring.

Compare unique-best/top5, tied-best possible matches, ranking coverage, wrong
formulas, abstentions, full generated pool recall and runtime against existing
lipid-aware LASER and native msbuddy on the same3413 keys. Retain every class/mode
cell and library-known/absent splits. Do not infer intrinsic algorithm superiority:
UFA receives first-QC raw profiles, LASER uses sample-integrated evidence/budget32,
and engines have different candidate generation and filtering. No independently
verified isotope-member gold or MS/MS is available. Outputs remain tentative.

All original data, current workflow predictions/statistics and accepted annotations
stay unchanged. Only preparation/cache/adapter tools are changed. All raw study
files and detailed feature/formula crosswalks remain private. Publish safe
aggregates, update installed main, and stop before Stage8.
