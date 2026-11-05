# Stage 5: lipid formulas and external engine comparison

Implementation under validation; no default promotion. This stage does not start
joint cross-mode inference (Stage 6), resume historical annotation optimization,
or close the outstanding Binner/verified-isotope requirements.

## Candidate model

The optional **Lipid backbones and de novo formulas** setting requires the complete
isotope model. It uses the bundled public `lipids.tsv` chemistry, including ether
and plasmalogen templates, and +O0..4 variants. Chain totals range from two to 48
carbons per chain, capped at 180, with at most 36 double bonds and the physical
chain-carbon bound. Isoprenoid definitions use 1–36 repeat units. Blank/unknown
chemistry definitions are recorded, not invented. The existing formula library
Fixed backbones also allow +O0..4 and up to four additional H2 losses.
remains a candidate source. No study reference class, name, formula or RT fit is
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
execution. Python 3.10 isolates msbuddy from LASER's Python environment.

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
