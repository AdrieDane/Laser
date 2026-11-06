# Stage 7 repair and validation follow-up

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
