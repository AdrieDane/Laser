# Stage 5: lipid formulas and external engine comparison

Local four-arm comparison completed on source `5c193e9`; native msbuddy completed
with adapter/evaluator source `6e9801c`. SIRIUS and MZmine remain unavailable without
compatible login/activation. No default promotion. This stage does not start
joint cross-mode inference (Stage 6), resume historical annotation optimization,
or close the outstanding Binner/verified-isotope requirements.

## Candidate model

The optional **Lipid backbones and de novo formulas** setting requires the complete
isotope model. It uses the bundled public `lipids.tsv` chemistry, including ether
and plasmalogen templates, and +O0..4 variants. Chain totals range from two to 48
carbons per chain, capped at 180, with at most 36 double bonds and the physical
chain-carbon bound. Isoprenoid definitions use 1–36 repeat units. Blank/unknown
chemistry definitions are recorded, not invented. Fixed backbones also allow
+O0..4 and up to four additional H2 losses. The existing formula library remains
a candidate source. No study reference class, name, formula or RT fit is
used by the predictor.

A broader de-novo meet-in-the-middle search covers neutral C1..180, H0..400,
N0..6, O0..60, P0..3 and S0..3. Neutral valence checks allow phosphorus and sulfur
valence exceptions; the complete ion composition is checked after adding/removing
the adduct atoms. The two user-prohibited ions remain excluded. No isotope-carbon
estimate is required to generate candidates.

All structured formula/adduct/charge candidates are retained with origins and
possible lipid classes. A fixed prior favours backbone/library candidates over
generic-only formulas (cost +1), with +0.15 per extra oxygen. These are documented
heuristics, not trained probabilities or class-identification claims. Before
isotope scoring, candidates are ordered by that prior plus normalized mass error.
The default scores 32 candidates per feature with complete isotope evidence;
the configurable limit is 1–256. Non-scored candidates remain saved and explicitly
labelled, and the GUI reports when the scoring budget is reached. The evaluation
must measure both generated truth and truth surviving the scoring budget.

The experimental output is additive. Existing p-values, accepted annotations,
compound assignments and legacy formula labels remain unchanged. The GUI reads
saved evidence; selecting a point does not enumerate formulas or call an external
service. Formula settings, candidate tables and budget audits persist with jobs
and saved projects. External engines use isolated benchmark adapters, not automatic
GUI downloads or account creation.

## Fixed comparison contract

Preparation retains all 3,668 reference rows. There are **3,413 natural formula
queries**: RP positive 1,592, RP negative 585, NP positive 724, NP negative 512.
The 255 other rows have no formula or are internal standards. All primary formula
queries are mapped in this acquisition. Unsupported masses/charges, timeouts,
failed jobs and absent predictions stay in primary denominators. All 252 original
reference class/mode cells, including formula-empty cells, must remain visible.
Library-known and library-absent formulas are reported separately.

The primary comparison supplies the adduct explicitly to every engine. This is
conditional formula prediction, **not adduct-identification accuracy**. Separate
blind LASER inputs omit both adduct and charge; their MS1 window does not depend
on the reference charge. Stage 6's joint multi-mode inference remains separate.

All queries use the same original m/z/RT and feature-derived biological mean-area
MS1 window (6 Da above the parent, RT ±0.05 min), with no formula-based peak
selection. These are area pseudospectra, not isolated acquired raw MS1 spectra.
External programs receive aggregate intensities, whereas LASER can retain paired
sample areas. This conversion loses evidence, so it cannot establish an intrinsic
algorithm winner or the full potential of an external raw-data workflow. No MS/MS
exists in this acquisition; no fake MS/MS or synthetic fragmentation is supplied.

The requested shared bounds are C1..180/H0..400/N0..6/O0..60/P0..3/S0..3, with 4 ppm
mass error. Native current LASER's tighter hard-coded NOPS/carbon rules are retained
and recorded as a baseline limitation; they are not silently rewritten to call a
different method "current". Different engine tolerance conventions and native
filters must be stated. Query timeouts are 30 seconds, and candidates lost through
generation, element limits, output limits or isotope-scoring budgets remain visible.
No real-data-dependent weight tuning or class/mode winner splice is permitted.

## Native tools and reproducibility

Stage 1 pins are retained: msbuddy 1.0.0, SIRIUS 6.5.4 and MZmine 4.11.0. Native
artifacts must be checked against publisher hashes; dependency environments,
database/model hashes, command lines and input conversions are frozen before
execution. A separate supported Python environment isolates msbuddy from LASER.
The first Python 3.10/SciPy 1.15.3 setup failed during macOS library loading,
before any prediction. The completed native run uses Python 3.12.12/SciPy 1.17.1.

[msbuddy](https://msbuddy.readthedocs.io/en/latest/quickstart.html) receives explicit
adducts. Its instrument preset is unset to retain the requested 4 ppm tolerance;
the native precursor 1<m/z<1500 restriction must remain reported, including
unsupported queries. Supplied-adduct formula results cannot be presented as adduct
assignment results. Native estimated FDR is not an independently validated FDR
for this acquisition.

[SIRIUS](https://v6.docs.sirius-ms.io/adducts/) documents singly charged import.
Charge-two inputs remain unsupported in the primary view. An account/activation
requirement is an unavailable comparison, not zero analytical accuracy. The user
has not previously used SIRIUS; no account will be created automatically.

[MZmine](https://mzmine.github.io/mzmine_documentation/latest/module_docs/id_spectra_chem_formula/chem-formula-pred.html)
requires raising its 800 m/z formula cutoff for these queries. Explicit ions,
neutral formula bounds and MS1-only settings must be recorded. Native CLI license
activation and imported isotope representation must be validated before claiming
an MZmine result. A CDK-only surrogate is not a native MZmine benchmark.

Raw data, query crosswalks, study names and detailed predictions remain outside
Git. Public reporting contains only source/protocol descriptions, synthetic tests
and safe aggregates. Current defaults stay unchanged until validation justifies
promotion; implementing a selectable candidate model is not promotion.

## Frozen local results

[Complete five-arm class/mode table](formula_class_mode_results.csv), including
all 63 classes × four modes, library-known/library-absent strata and empty cells.
[All class/mode gains and losses](all_class_mode_transitions.csv) retain regressions.

All four arms completed all 3,413 predefined queries, without query timeouts.
The table reports **conditional neutral-formula top-1 agreement with the supplied
adduct**, not adduct accuracy or untouched validation. Reference counts include
all formula-bearing natural lipids; 255 other ledger rows remain documented.

| Mode | Current | Stage 4 | Lipid-aware | Native msbuddy | Lipid-aware gains / losses vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| RP positive | 74.1% (1179/1592) | 53.0% (843/1592) | 95.9% (1527/1592) | 94.2% (1500/1592) | +379 / -31 |
| RP negative | 49.7% (291/585) | 45.1% (264/585) | 75.7% (443/585) | 79.0% (462/585) | +215 / -63 |
| NP positive | 65.3% (473/724) | 58.1% (421/724) | 92.5% (670/724) | 97.4% (705/724) | +217 / -20 |
| NP negative | 61.3% (314/512) | 51.0% (261/512) | 86.7% (444/512) | 85.4% (437/512) | +170 / -40 |

Pooled top 1: 2257/3413 current, 1789/3413 Stage 4, 3084/3413 lipid-aware.
Native msbuddy top 1 is 3104/3413; generated truth is 3299/3413 and top 5 is
3294/3413. New-model generated truth is 3409/3413 and top 5 is 3357/3413. Four references
lie outside the frozen 4 ppm tolerance; no tolerance was relaxed after inspection.
None of the generated correct formulas was lost to the 32-candidate isotope budget,
although the budget was reached for 1522 known-adduct and 2953 blind queries.

The existing public library covers 3286 queries; 127 are library-absent. On those
127 queries, top 1 is 66 current, 34 Stage 4, 106 lipid-aware and 106 msbuddy.
These strata refer to LASER's pre-existing library, not the external database's
exposure. This is same-acquisition development evidence, not a held-out family benchmark.

With both adduct and charge hidden, lipid-aware **joint formula/adduct top 1** is
855/1592 RP positive, 72/585 RP negative, 49/724 NP positive and 56/512 NP negative:
1032/3413 overall. This ranking weakness prevents default promotion. No post-hoc
weights, thresholds, class/mode output splice or reference adduct was introduced.

The production additive overlay completed 762 raw-p queries in 68.60 seconds;
all original 251431 feature/statistical rows, 1624 compound detections and other
existing tables matched exactly. Accepted annotations, unconfirmed extras and
original-feature selections were unchanged. All 94 prepared copies, including
68 raw files, internal standards and method metadata, retain their hashes; the
inputs-only project's output folders remain empty. Full suite: 1072 passed,
8 skipped, 91 subtests. This is offline computation timing, not browser latency.

Attempt 001 is preserved: the current arm completed, but the Stage 4 process
could not import the isotope dependency because the launch omitted its path.
Attempt 002 reran all four arms with the same source and an explicit runtime path.
All 1827 extracted source files and the source archive are byte verified.

The verified SIRIUS 6.5.4 native synthetic test refused execution without login.
Verified MZmine 4.11.0 reports no compatible logged-in user; its legacy `.mzuser`
format is unsupported. Neither produced study predictions. These are unavailable
comparisons, not measured 0% accuracy. No account was created. Native msbuddy setup
is complete, as described below. Stage 5 is not fully closed while required native
comparisons remain unavailable; Stage 6 remains unstarted.

## Completed native msbuddy comparison

[Runtime and public model hashes](native_msbuddy_runtime.json) freeze msbuddy 1.0.0,
Python 3.12.12, NumPy 2.2.6, SciPy 1.17.1, LightGBM 4.6.0, scikit-learn 1.6.1,
all 27 resolved dependencies and the three official database/model files. The
native worker is byte-identical to the pre-evaluation `5c193e9` worker. The later
evaluator change represents missing native per-query timing as unavailable,
instead of a misleading zero. There were no scientific rule/weight changes.

Synthetic protonated/ammoniated/deprotonated/doubly deprotonated controls all
recover their expected top formula (4/4). Native attempt 001 preserves a synthetic
startup failure caused by a missing OpenMP loader path; no study data were scored.
Attempt 002 uses scikit-learn's hashed bundled OpenMP library via an isolated
loader path and then runs all study queries. No system library was installed.

All 3413 query IDs are retained: 3338 native completed queries and 75 outside the
native `1<m/z<1500` range (54 RP positive, 2 RP negative, 0 NP positive, 19 NP negative).
There were no study-batch failures/timeouts. Three supported queries returned no
candidates; 39 supported queries lack the reference formula in the candidate pool.
All 42 negative charge-two reference queries remain included; charge-two behavior
also passes the synthetic control. Native FDR values are not validated analytical
FDR for this acquisition and are not used as the primary evaluation.

Native gains/losses versus current top 1 are +364/-43 RP positive, +213/-42 RP
negative, +242/-10 NP positive, +161/-38 NP negative. Compared with the lipid-aware
arm, native gains/losses are +52/-79, +99/-80, +47/-12, +48/-55 in the same order.
No per-mode winner outputs were combined. The annotation loop took 6.41 seconds,
excluding initial database loading; individual query times were not measured.

Decision: retain the lipid-aware candidate generator as experimental and keep
msbuddy as a reproducible optional comparator. Neither result establishes a best
joint formula/adduct algorithm or supports changing defaults. LASER's larger pool
and top-5 coverage and msbuddy's slightly higher conditional pooled top 1 are both
useful evidence for the next separately authorized joint-inference stage. Different
library exposure, native mass scope and per-sample versus aggregate evidence
prevent an intrinsic algorithm-winner claim.
