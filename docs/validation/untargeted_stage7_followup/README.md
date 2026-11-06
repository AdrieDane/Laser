# Stage 7 repair and validation follow-up

## Follow-up outcome — 6 November 2026

**Keep CURRENT.** The conservative option restores all four correct routine calls
lost by legacy trace, but every assignment equals CURRENT exactly in both scopes.
It provides no demonstrated incremental original-ion/adduct gain here. No rule was
strengthened after scoring to manufacture a gain. The four inspected losses were
development cases, not an independent validation set.

| Scope | Mode | CURRENT = conservative, all-manual coverage | Unconfirmed extras, both |
|---|---|---:|---:|
| natural | RP_Pos | 790/1694 (46.64%) | 1201 |
| natural | RP_Neg | 328/679 (48.31%) | 1102 |
| natural | NP_Pos | 506/757 (66.84%) | 882 |
| natural | NP_Neg | 364/538 (67.66%) | 908 |
| routine | RP_Pos | 46/1694 (2.72%) | 461 |
| routine | RP_Neg | 3/679 (0.44%) | 464 |
| routine | NP_Pos | 37/757 (4.89%) | 171 |
| routine | NP_Neg | 5/538 (0.93%) | 194 |

All-manual percentages include every curated row, including accepted/non-significant
ions not offered a normal raw-p query and unresolved mappings. They are **not overall
adduct annotation accuracy**. Extras are unconfirmed, not measured false positives.
Full denominators: 3668 manual rows, 46 unmapped, 63 classes × four modes; natural
3413 artificial query keys and separately routine762 queries. The full current /
legacy / conservative comparison, assignment coverage, conditional agreement,
abstentions, wrong adducts and gains/losses are in
[all-mode results](all_mode_before_after.csv) and
[all class/mode results](all_class_mode_adduct_recovery.csv). Natural legacy gains36 /
losses16; routine legacy gains0/losses4. Conservative gains0/losses0 versus CURRENT.

Natural reference-pair recovery: current446/960, legacy451/960, conservative446/960.
Routine:21/960,19/960,21/960. Fixed PC counterpart regression remains natural9/14
and routine10/14 in every arm. [Pair details](reference_pairs.json),
[PC summary](PC_counterpart_summary.json),
[curated family consistency](curated_group_consistency.csv),
[actual viewer slot status counts](all_mode_slot_status_summary.csv).
These are curated original-ion/assignment agreement and coverage, not confirmed
molecular identities, independent validation or analytical FDR.

### Native formula comparison remains unavailable

Two complete caches passed eight fixed unmodified-native probes each: protonated
40396 profiles (495.5s), doubly charged negative32495 (573.8s). Deprotonated52307 and
ammonium132209 pools exceeded their fixed600s caps; no timeout was extended. They
reached45056 and40960 progress respectively, without complete validated persisted
caches. File-read stalls in native isotope constants were observed; total times
include host I/O. The entire required shared universe was unavailable, so **no
native study ranking or formula accuracy was computed**, in any mode. All3413
queries and all252 class/mode cells remain explicit, with unavailable values rather
than zero accuracy. [Preparation status](native_preparation_status.csv),
[complete conditional formula scope](native_all_class_mode_formula_comparison.csv).

The repaired observed-major-isotope bridge and lossless optimizations pass all
synthetic controls and exact parity with522 unmodified native profiles, including
C120 dominant M+1, both polarities, charge2, sulfur/chloride, interference and ties.
The release additionally pins the isotope-constants file in cache schema2 and uses
JIT-stable deparsed function hashes. A new check asserts cache descriptors remain
identical before and after native execution. These are descriptor safeguards only;
no profile/scoring arithmetic changed. A command-line reference-path mistake in
pilot006 was preserved; corrected007 and strengthened008 passed. Pilot008 cold512
profiles=2.821s, warm=0.008s,100 indexed queries=0.098s.
[Synthetic receipts](synthetic_controls.json), [source/protocol](provenance.json).
The scientific experiment remains frozen to0ca4bd0/cache schema1; the subsequent
schema2 descriptor guard was tested synthetically, not used to rerank study data.
The old initial package-file manifest omitted native isotope constants; an extra
fingerprint was recorded during preparation before any matching. This limit is
retained, not presented as an initial frozen-data hash.

### Preservation and limits

All254324 original coordinates and251431 statistical rows are exact, with762 raw-p
and387 fold-filtered rows. Each worker verifies original non-label statistical and
evidence columns; actual routine save/reopen and viewer payload are exact. A fresh
independent check verifies original-key/mass/RT constraints and all old statistics
against exact hash-verified frozen Parquet duplicates. Redundant178MB CSV reads in
the host filesystem stalled; failed read/clone checks and first scorer schema error
were preserved privately. No numerical tolerance was loosened, no inference rerun
or candidate/denominator filtering used for this recovery. All94 prepared files,
five derived metadata inputs, raw sources, native package code, IS and method
hashes match; project result folders remain empty. The full original saved project
was fingerprinted before workers. [Verification](validation_checks.json).

Software suite:1106 passed,9 skipped,91 subtests passed. Browser timing covers the
JavaScript model, not a new full-browser latency benchmark. Cached trace observations
were replayed with exact source/settings/pair-set proof; this is not new raw scanning
or biological testing. Native known-adduct formula comparisons remain conditional,
not blind adduct accuracy; first-QC raw profiles and LASER sample areas/budget32
have different evidence footprints. No verified isotope-member gold. CAMERA is not
rerun; prior timeouts, account-dependent engines and external validation stay open.
Stop for review before Stage8; historical annotation optimization remains paused.


Authorized 6 November 2026 with “oke do it”. CURRENT stays the default. This is
another development experiment on the same acquisition; the four lost routine
calls were inspected during development and are not an independent holdout.
No Stage8 work or historical annotation optimization is authorized.

## Diagnosis and fixed trace policy

Four correct routine original-ion/adduct calls disappeared after three shared
families lost a same-mode fragment/adduct link. Low median Pearson correlation
triggered the legacy OR veto despite substantial normalized overlap. A fragment
anchor could then lose its required intact parent and drop its entire family,
including otherwise supported cross-mode detections. Detailed case identities
and original-feature crosswalks remain outside Git.

Retain the legacy option for reproducibility; add **Conservative peak-shape veto
(experimental)**. Use the exact same measured EICs/quality gates and thresholds.
At least two usable injections and at least80% jointly having r<0.5 AND
normalized overlap<0.25 are required for a veto. Dissimilar but substantially
overlapping shapes stay ambiguous. Supported shapes receive zero ranking bonus
and no sorting promotion. Missing/ambiguous evidence retains table-based order
and quality. There are no lipid-class, mode or original-feature exceptions.
Settings are frozen before full predictions; no post-scoring tuning or winner
splicing. Cached original scan measurements may be reused with exact source,
settings and pair-set hashes; this is a cached-observation replay, not new raw
measurement or new biological testing.

## Native UFA adapter and lossless optimizations

Choose the anchor from observed coeluting isotope channels, independently of any
candidate formula or rank. Known adduct/charge are supplied only for this explicitly
conditional formula comparison. Search nominal13C isotope orders0–4 at4ppm,
retain five observed signal points/two edges, common-apex positions within one
scan, r≥0.85 and normalized overlap≥0.5. Missing/competing monoisotopic peaks,
strong unverified isotope signals, sparse scans, acquisition gaps or a dominant-
peak tie within5% produce an unavailable state. No lag shifts or interpolation
of artificial scans. The next observed channel supplies the native R13C input.
This does not independently validate isotope membership in real study data.

Native IDSL.UFA2.0/IDSL.IPA2.9 profiles use the unchanged0.005Da spacing,
min(C,10)% intensity cutoff, native abundance constants and1e-12 internal tail
cutoff. Native matching retains0.01Da, NEME5, PCS950, NDCS2, RCS5 and unit score
coefficients. Native IPA peak detection is not run: original native XCMS bounds
and actual raw centroid heights remain the bridge. Original feature IDs always
refer to the original monoisotopic feature, with anchor order/mass saved separately.

Build shared content-addressed native profile universes once per adduct, retaining
all generated candidates. Cache hits require identical native versions, generator
and profile-function hashes and numerical settings. Installed namespaces/packages
are not modified. A cloned native generator memoizes pure elemental computations
and replaces recursive Cartesian enumeration with the same left-associative
sum/product order. Partial branches already at/below the native cutoff cannot
survive later probability factors in[0,1]; dropping them retains exactly the
native final states. Native peak merging, abundance normalization and rounding
are unchanged. These optimizations are explicitly part of this configured native
comparison, not a claim of unmodified native runtime performance.

The native matcher receives the union of EVERY candidate/isotope tolerance window,
including machine-rounding padding. It still applies its original exact tolerance
and nearest-centroid rule. Original centroid order/ties are retained. Cache and
spectrum pruning must exactly reproduce unmodified native profiles/results.

## Pilot gates and scope

Synthetic controls cover positive/negative, charge2, C120 dominant M+1,
sulfur/chloride, sparse/competing/interfering signals, centroid ties, and cache
cold/warm equivalence. First pilot passed correctness but failed cold runtime:
512 profiles53.756seconds. Preserved, not overwritten. Lossless optimization then
passed exact parity against522 unmodified native profiles: eight controlled ions,
two high-heteroatom formulas and512 cached synthetic profiles. Final pilot cold
512=2.811s, warm=0.005s,100 indexed native queries=0.103s. Gate limits remain30s,
5s and10s respectively. These validate adapter/optimization mechanics, not real
formula accuracy or isotope precision/recall.

After these gates pass, freeze/push source, settings, all input/cache descriptors,
native code and query hashes before full study inference. Shared cache preparation
has a600-second process cap per adduct (four one-thread jobs concurrently); require
all caches before scoring. Native matching has120seconds per mode, two one-thread
jobs concurrently. The first eight fixed evenly spaced profiles per adduct must also exactly match
unmodified native generation; the first four eligible queries per mode must
match native unpruned scoring. No expanded timeout after results. Failed/preparation-excluded,
ambiguous and unattempted queries remain in the full primary denominators; if
shared preparation fails, report native comparison unavailable rather than score
an incomplete candidate universe.

Compare CURRENT, legacy trace and conservative trace on the same3413 natural
query keys and the normal762-query workflow. Retain all3668 manual rows,46
unmapped and all63classes × four modes, including zero-query cells; actual saved
JavaScript viewer choices and the fixed15-PC/14-counterpart regression. Report
gains AND losses, assigned/abstained counts, wrong adducts and original peaks,
reference-pair recovery, curated group consistency and unconfirmed extras.
Compare native UFA conditional formula top1/top5, candidate generation versus
ranking, support coverage and failures on all3413 keys against the earlier
lipid-aware known-adduct reference. Expose database/library and evidence-footprint
confounds. No independent validation, calibrated confidence or analytical FDR.

Preserve all original statistics/accepted annotations/inputs/defaults. Verify the
original387-row significant subset, typed save/reopen, source hashes and local
viewer selection. Publish safe aggregates only. More CAMERA runs are deferred;
prior timeouts, native account limits and lack of verified isotope gold remain
open. Stop at this follow-up checkpoint beforeStage8.
