# Stage 4: complete isotope patterns and measurement quality

5 November 2026. Software validation and internal comparison complete; verified
isotope-member validation remains open. Current isotope evidence and
current grouping remain the defaults. Stage 5 has not started. Binner's separate
Stage 3 native benchmark remains pending.

## Implementation and scope

**Feature-evidence search settings → Isotope model** offers an experimental
complete-ion model. It adds saved candidate scores, component observations and
quality summaries. Existing statistical values, accepted annotations, compound
assignments and legacy formula labels are preserved. The viewer reads the saved
summary; point selection does not run an isotope search.

[IsoSpecPy](https://github.com/MatteoLacki/IsoSpec) **2.6.1** generates fine natural
isotopologues to 99.999% cumulative probability. Every allowed ion includes its
complete elemental delta, signed electron correction and charge. Formate/acetate
carbon, ammonium nitrogen/hydrogen, chloride and potassium isotopes are included.
The two user-prohibited ions remain unsupported. Labelled formulas and known
internal-standard annotations are excluded from natural-abundance scoring.
Unrecognized labelled features cannot be identified reliably from areas alone.

The full calculation is retained in memory; components above 0.1% of the parent
are searched. A bounded two-FWHM span approximates centroid merging within each
nominal isotope. This is an explicit approximation, not an instrument line-shape
fit or proof of experimentally resolved fine structure. Unknown instrument laws
or resolution references use nominal aggregation and receive an uncertainty flag.

The [Q Exactive Plus specification](https://documents.thermofisher.com/TFS-Assets/CMD/Specification-Sheets/PS-63912-LC-MS-Q-Exactive-Plus-Orbitrap-PS63912-EN.pdf)
defines its resolving power at m/z 200. For that named model, a missing reference
is inferred from the manufacturer specification and recorded; project metadata
is not overwritten. The [Orbitrap resolution law](https://assets.thermofisher.com/TFS-Assets/CMD/Technical-Notes/tn-64287-hram-orbitrap-ms-tn64287-en.pdf)
scales local resolving power inversely with the square root of m/z. Generic
instruments do not inherit this assumption. OpenMS's
[fine/coarse distribution distinction](https://openms.de/documentation/html/classOpenMS_1_1IsotopePatternGenerator.html)
was reviewed; no native OpenMS performance benchmark is claimed.

Matching uses mass error, capped peak-width/coelution windows and per-sample
ratios. Biological specimens supply primary technical ratio evidence; QC provides
a separate diagnostic and fallback when fewer than three biological pairs exist.
Blanks and standard injections supply neither biological nor isotope-ratio fits.
At least three positive finite pairs and log-ratio MAD at most 0.5 are needed for
an intensity fit. Ratios within a factor of two are called compatible; this is a
heuristic, not a calibrated probability. Competing feature alternatives, shared
components, variable ratios, possible lower monoisotopic parents and missing peaks
remain recorded. Mass and intensity errors contribute jointly to the score.

The fifth percentile of positive sample feature areas is used only as a missingness
proxy. It is not a measured noise floor or LOD. Saturation is acted on only when
an explicit feature flag exists; otherwise it is recorded as unassessable from
integrated areas. Missing/noisy M+1 does not eliminate a hypothesis. Existing
candidate formulas are supplemented by a mass-only search of the existing public
formula library, with its SHA256 recorded. This is not a new de-novo formula
generator or the Stage 5 external-engine comparison. All candidates are retained;
the best additive score does not replace the legacy formula/adduct assignment.

## Evaluation contract

`tools/evaluate_complete_isotopes.py` freezes committed source and protocol before
comparing current and complete evidence. Workers receive no manual formulas or
identities. Every original statistical/compound value must remain identical, and
current compound assignments must reproduce the Stage 1 replay. Detailed study
rows and source/input fingerprints stay outside Git.

Formula candidate recall and joint top-1/top-5 are measured on the SAME mapped,
unambiguous, formula-bearing raw-p queries in both arms. All four modes and all
252 reference class/mode cells remain visible, including zero-query cells.
These are development measurements with prior library exposure, not external
validation. Expanded candidate pools and isotope reranking are reported together;
any gains cannot be attributed solely to the isotope calculator.

A separate all-reference readout supplies the curated formula/adduct to inspect
conditional envelope quality. It retains all 3,668 reference rows, with explicit
unmapped, formula-missing, labelled, unsupported and mass-incompatible outcomes.
Those reference hypotheses never enter the prediction workers. The raw check
examines apex centroid snapshots from the three QC injections per mode within
±0.10 min of each original feature RT. It records absent/ambiguous centroids and
raw ratios, without fitting a trace model or treating an absent centroid as a
demonstrated false prediction. This limited measurement check is not the Stage 7
trace-evidence workflow.

**No independently verified isotope-member gold set exists for this acquisition.**
Consequently real isotope-recognition precision/recall must remain unavailable.
Controlled cases, reference-conditioned fits and raw QC observations must not be
substituted for those metrics. The implementation can be published while the
verified-envelope validation requirement remains open. Formula recall on the
declared real queries is still reported. Do not promote the experimental model
or claim Stage 4 fully validated from synthetic tests alone.


## Completed internal comparison

Committed producer `81e8aa2` completed both fixed arms without retuning. **Current
isotope evidence and current grouping remain the defaults.** All 251,431 original
feature/statistical rows and original compound fields are identical; current
assignments reproduce the frozen Stage 1 replay. Existing detections, accepted
annotations, formula labels and unconfirmed extras are unchanged. Results below
belong to the additive experimental candidate table, not replaced assignments.

Only **27 queries** meet the common raw-p, mapped, unambiguous, formula-bearing
scope. There are 17 RP-positive, 1 RP-negative, 9 NP-positive and **zero NP-negative**
queries; this is not a general four-mode accuracy estimate. All 252 class/mode
cells, all reference eligibility rows and every gained/lost query are private.

| Mode | Formula candidate recall: current → complete | Joint formula/adduct top 1: current → complete | Complete joint top 5 |
| --- | --- | --- | --- |
| RP positive | 88.2% (15/17) → 100.0% (17/17) | 17.6% (3/17) → 11.8% (2/17) | 100.0% (17/17) |
| RP negative | 0.0% (0/1) → 100.0% (1/1) | 0.0% (0/1) → 100.0% (1/1) | 100.0% (1/1) |
| NP positive | 77.8% (7/9) → 100.0% (9/9) | 0.0% (0/9) → 33.3% (3/9) | 88.9% (8/9) |
| NP negative | Unavailable (0 queries) | Unavailable (0 queries) | Unavailable (0 queries) |

Overall candidate recall is 22/27 →27/27; joint top-1 is 3/27 →6/27; complete
joint top-5 is 26/27. Five queries gain a correct joint top-1 and **two RP-positive
queries lose one**. Candidate-pool gains occur in five queries; none loses the
reference joint candidate from the pool. The small, library-exposed scope and
combined pool expansion/reranking prohibit attributing the gains solely to the
isotope model. Correct first-choice performance remains weak.

The all-reference, formula-conditioned diagnostic retains all 3,668 rows. Of
these, 255 are unmapped, formula-missing or internal standards; a further four
reference hypotheses fail the parent-mass tolerance. Compatible/partial fits
are not correct-identification counts. No independently verified isotope-member
reference set was manufactured from these fits.

All **12 raw QC files** were read, three per mode, yielding 53,218 diagnostic
rows including three absent/ambiguous parent cases. Unique centroids at predicted
component positions occur in 11,584/25,071 RP-positive snapshots, 4,248/9,243
RP-negative, 4,234/11,013 NP-positive and 3,006/7,888 NP-negative. These denominators
include weak higher isotopes; they are availability observations, **not recall**.
Absent components remain uncertain because of weak signal, extraction/alignment,
interference and centroid merging. Raw ratios and checksums form a review worklist.

Full suite: **1,059 passed, 8 skipped, 91 subtests passed**, including 20 focused
composition/quality/save/worker checks. The complete additive model takes **1.92 s**
for 762 significant-feature queries and 1,426 candidate hypotheses. This excludes
existing pipeline work and exports. On 60 uncached selections, browser-model
plot-description computation has median 3.2 ms and maximum 20.0 ms; actual browser
rendering, transfer and DOM latency are not included.

All frozen input fingerprints and all 94 prepared project files pass preservation
checks, including 68 raw acquisitions, the IS CSV and method JSON. The inputs-only
project result folders remain empty. The pinned dependency is installed in the
local app environment. Source and archive hashes, reports and detailed study rows
remain private outside Git. The experimental model is not promoted. **Stage 4's
verified real-isotope precision/recall requirement remains open; Stage 5 has not
started.** Binner's separate Stage 3 benchmark also remains pending.


Runtime coverage across all significant features (not a correctness measurement):

| Mode | Raw-p queries | Current candidate queries | Complete candidate queries | Compatible/partial best fit | Mass-only best fit |
| --- | --- | --- | --- | --- | --- |
| RP_Pos | 297 | 81 | 148 | 82 | 48 |
| RP_Neg | 286 | 85 | 153 | 89 | 41 |
| NP_Pos | 88 | 16 | 33 | 18 | 11 |
| NP_Neg | 91 | 20 | 48 | 21 | 21 |

Across all 762 queries, candidate coverage rises from 202 to 382; 380 remain
unresolved. Only 210 have compatible/partial isotope evidence for the best
additive candidate. These compatibility counts are not identification accuracy.
