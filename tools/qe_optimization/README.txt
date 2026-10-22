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
