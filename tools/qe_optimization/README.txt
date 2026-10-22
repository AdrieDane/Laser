QE collection research runners — 22 October 2026

NOT production annotation entry points. No app hook/default has been enabled.
These runners use the user's sole active 33-folder collection, and write all
raw-data staging, local trusted pickles, manual crosswalks and model fits OUTSIDE
the repository. Do not commit or upload those input/profile/feature tables.
Pickles must be locally generated trusted artifacts, never downloaded inputs.

Set explicit environment variables before invoking a runner from this repo:
  LASER_OPTIMIZATION_WORKSPACE: existing research workspace directory
  LASER_QE_DATASETS: existing QE_example_datasets directory
  OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1

Current local workspace:
  /Users/yorrick/Documents/Codex/2026-10-19/https-github-com-yrjjaspers-laser-standard
Current data root:
  /Users/yorrick/Library/Mobile Documents/com~apple~CloudDocs/Work/AI/example_data_CFM/QE_example_datasets
The original local runners remain under workspace/work/. These checked-in
copies replace only path configuration with explicit environment variables.
Frozen per-experiment sources/protocols are the authoritative executed versions.

Use Python with the project's runtime AND calibration extras installed. The
recorded experiments used Python 3.13, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1,
scikit-learn 1.8.0. Thread limits avoid BLAS oversubscription.

Outputs use fixed, versioned directories beneath the workspace. Existing
complete markers support resuming annotation runners; audit/reference-repair
directories deliberately refuse overwrites. Inspect existing outputs and live
processes first. NEVER start a concurrent duplicate against the same output.

Reproduction sequence in a NEW empty workspace, with the same 33 inputs:
  python tools/qe_optimization/audit_qe_examples.py
  python tools/qe_optimization/run_qe_baseline.py --output <workspace>/outputs/qe_optimization_20261022/baseline --workers 3
  python tools/qe_optimization/repair_qe_standard_crosswalks.py
  python tools/qe_optimization/build_qe_rt_atlases.py
  python tools/qe_optimization/run_qe_atlas_experiment.py
  python tools/qe_optimization/evaluate_qe_baseline.py <baseline output> --manual-reference <workspace>/outputs/qe_optimization_20261022/manual_reference_v2
  python tools/qe_optimization/evaluate_qe_baseline.py <experiment_001 output> --manual-reference <same v2 reference>
  python tools/qe_optimization/report_qe_experiment.py --experiment <experiment_001 output>

DELIVER all 132 dataset/mode before/after rows to the user, including unchanged
rows, before running the next annotation-changing experiment. Read AGENTS.md.
Then experiment 002 (fixed dual-source proposals and nonlinear evidence):
  python tools/qe_optimization/run_qe_dual_experiment.py
  python tools/qe_optimization/evaluate_qe_baseline.py <experiment_002 output> --manual-reference <same v2 reference>
  python tools/qe_optimization/report_qe_experiment.py --experiment <experiment_002 output>

Do not describe a replay as a fresh production/raw-input run. Baseline snapshots
were 4da94ef; the later launcher/Python-version changes do not alter annotation.
All model fits exclude each evaluated project group. Thirty project groups
span 33 folders; related acquisitions share folds. The first seven filename
characters are grouping metadata only, NEVER model predictors. Fold assignments
must be reviewed if projects/cohorts/acquisitions change. Known prior exposure
and inspection of development diagnostics preclude an untouched external test
claim. Outer training features use the outer training RT atlas, not fully nested
inner OOF atlases; conditional grouped scores are not probability calibration.

Reference v2 fixes ONLY previously unresolved INTERNAL STANDARDS with unique,
exact exported m/z/RT AND raw sample-profile correlation >=0.999999 across >=4
matched samples. The older mapper applied protein normalization to standards
whose exported profiles were uncorrected. It preserves every previous feature
mapping, all non-standard mappings and ALL manual denominators. The correction
is applied equally to baseline and experiments, never reported as an annotation
gain. Reports retain unresolved entries in the primary denominator.

The user target remains >=90% correct original-feature agreement by curated
class and mode, with controlled extras. Not achieved by experiment 001. A model
score estimates curator agreement, not analytical identification probability
or FDR. Manual-absent annotations are unconfirmed extras, not proven false hits.

verify_qe_preparation_parity.py is a historical engineering check against the
preserved old preparation cache. It needs that local cache; it is not an
additional scientific selection experiment or part of the clean pipeline.

Further fixed experiments, each AFTER the preceding full table was delivered:
  python tools/qe_optimization/run_qe_chemical_experiment.py
  python tools/qe_optimization/run_qe_context_experiment.py
Evaluate/report each output exactly as above using reference v2. They are
experiment_003 and experiment_004 respectively; neither changes app defaults.
003/004 known_limitations.txt explains the missing separate standard catalog.

Read-only diagnostics (NOT annotation performance experiments):
  python tools/qe_optimization/diagnose_qe_local_rt.py
  python tools/qe_optimization/build_qe_standard_rt.py --output <new empty diagnostic folder>
The latter has preserved local v1/v2/v3 methodology-development versions; do not
overwrite them or call their RT-window percentages annotation accuracy. Current
standard RT code learns other-group positional REPORT metadata for declared
spikes only; it does not alter benchmark isomer equivalence or relabel generic
endogenous LPC/LPE. Read docs/PE_HANDOFF.txt for the current checkpoint.

Experiment005 (full132-row table delivered before006) replays004 selections
with conservative declared-standard completion, phase-specific reporting and
channel handling. Endogenous models and benchmark equivalence are unchanged.
First build standard RT v3 and independent endogenous-class residual transfer:
  python tools/qe_optimization/build_qe_standard_rt.py --output <workspace>/outputs/qe_optimization_20261022/standard_rt_development_v3
  python tools/qe_optimization/build_qe_standard_transfer.py
  python tools/qe_optimization/run_qe_standard_experiment.py
Evaluate/report experiment_005 as above with reference v2. The old v1/v2 standard
diagnoses are historical, not necessary inputs to005. RT-only support is NOT
annotation accuracy. New calls require a unique ORIGINAL feature before any
occupancy filtering; conflicting RT sources withhold completion. Labelled
standards retain their isotope masses and do not suppress unlabeled lipids.

Experiment006, after delivering005's full table:
  python tools/qe_optimization/run_qe_specialist_experiment.py
Same004 evidence/candidates and005 standard models, new class/mode specialists
where TRAINING support allows. Unknown or under-supported classes retain004
global predictions; zero-positive training classes require >=6 groups before
withholding full calls. Threshold .5/margin .05 remain fixed. Evaluate/report
every dataset and mode, including regressions; do not adopt based on pooled90%
if individual modes/classes remain poor. Raw truth, local trained pickles and
source workbooks are NEVER repository inputs or files to commit.

006 was rejected: lower overall correct original-feature coverage than005.
After its full report,007 returns to the global004 model architecture and adds
OTHER-mode evidence, keeping the exact physical candidate universe:
  python tools/qe_optimization/run_qe_cross_mode_experiment.py
Evaluate/report experiment_007 as above. Matching validated biological GROUP
profiles are not individual-sample pairing; missing evidence stays unknown.
Phase RT offsets use OTHER compositions only; ether/positional classes distinct.

After007's full report,008 tests TRAINING-group-calibrated operating points:
  python tools/qe_optimization/run_qe_operating_experiment.py
Three inner project-group score folds within each outer training set choose a
per-stream cutoff, maximizing correct original-feature training matches within
a fixed9% endogenous extras/endogenous-manual budget. Standards separate, same005
logic. Frozen007 final score models, same physical candidates and ambiguity
margin. No evaluated outer labels choose cutoffs. This is NOT fully nested RT
cross-validation: outer-training atlas/evidence are inherited inside inner score
folds. No test-set extra/FDR guarantee. Full132-row table and class reports still
required before the next experiment, regardless of acceptance/rejection.

008's individual mode budgets worsened RPnegative;009 therefore uses JOINT
cutoffs from the SAME008 training tables, maximizing weakest-mode coverage within
a POOLED9.5% endogenous extras/endogenous-manual training budget:
  python tools/qe_optimization/run_qe_balanced_operating_experiment.py
No evaluated labels choose cutoffs. Same007 final scores/pools, same005 standards.
Evaluate/report all132 rows. Training budget is not a test-set/FDR guarantee.

After009's full table,010 isolates global model capacity:
  python tools/qe_optimization/run_qe_capacity_experiment.py
Only600 rounds instead of180; same007 evidence/candidates,009 frozen thresholds,
005 standard handling. Cutoffs are NOT recalibrated to600-round scores. No raw
grid RT source enabled; inspect held-out extras and wrong features before adoption.

Read-only raw RT mass-series diagnostic (NOT annotation accuracy):
  python tools/qe_optimization/build_qe_raw_grid_diagnostic.py --workers 3
Frozen other-group RT/prevalence, raw mass/RT/intensity only for inference.
>=8 OTHER compositions required; competing grids and mass-degenerate votes
withheld; declared-spike masses excluded. Predictions are frozen BEFORE reading
the evaluation labels. Runtime sources remain unchanged. Diagnostic per-feature
CSV/pickles stay OUTSIDE Git; only aggregate validation reports may be committed.
Current helper has an exact, unit-tested support prefilter absent from the first
frozen diagnostic source. Verify replay parity before future annotation admission.

010 lost coverage at its fixed cutoffs and was rejected. After its full table,
011 tests conditional PEAK choice with009 accepted identities held fixed:
  python tools/qe_optimization/run_qe_peak_choice_experiment.py
Group-held-out model,identity/cohort-balanced training,no composition occurrence
features,>=50 identities/>=6 groups per class/mode,score.75/margin.15. Protected
standard/manual rows and occupied/spike features cannot be borrowed.011 was
rejected(-3 correct);extras/counts identical in every132 comparisons. Historical
repair_qe_peak_storage.py / repair_qe_peak_standard_contract.py preserve failed
engineering snapshots/models/outputs. Do NOT run in a clean reproduction with
the current fixed helpers; they are provenance tools,not scientific parameters.

After011's full table,012 tests conservative NP series completion,preserving
EVERY009 call and keeping RPunchanged:
  python tools/qe_optimization/run_qe_np_series_experiment.py
ONLY blind raw-grid predictions,not root diagnostic truth. Unique raw hit within
<1.5ppm/<=5s BEFORE occupancy/standard filters,inadequate old local anchors,
>=25%/>=3 OTHER-group composition support. Existing identities/features,standards
and conflicting claims protected. No new classifier/score cutoff. Evaluate ALL33
and report132 rows,including extras/wrong peaks. Initial extras guard remains
the original9,697 total. No production entry point/default changed.

012 found422 more correct calls than009 but594 more extras and60 more wrong
peaks. Its9,910 extras fail the original9,697 safeguard; it is NOT adopted.
After its full132 table,013 tested a fixed neutral-ion ambiguity exclusion:
  python tools/qe_optimization/run_qe_ion_footprint_experiment.py
Only NEW012 NP-positive calls; invert native adduct/charge masses, withhold
different canonical classes within5ppm neutral/2s RT of old009 or one another.
Exclusion-only, not a wider admission window or proof of chemical error.
All33 finished; ZERO calls changed. All132 rows delivered. Not adopted.

After013's full table,014 tests pairwise ORIGINAL-feature choice starting009:
  python tools/qe_optimization/run_qe_pairwise_peak_experiment.py
Both-order true-vs-wrong feature pairs, identity/project-group balanced.
Numeric differences/means/observation flags, same identity categories. Old
selection/confidence shortcuts and occurrence descriptors removed. Symmetric
pairwise preference must beat EVERY competing peak by>=.75, no forced cycle
winner. Same>=50 identity/>=6 group support, occupied/spike protection and
fixed identity multiset as011. Model180 rounds/31leaves/minleaf25/L2=2/lr.08.
Every132 comparison MUST retain009 counts/extras; correct peaks may regress.
Evaluate/report with identical referencev2 before any next experiment.
