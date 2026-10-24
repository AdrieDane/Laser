QE collection research runners — 22 October 2026

Latest verified reference: 029, 82729/92102 correct (89.823239%), 9268 extras,
1604 different-feature choices. Actual vs028: +41/-78/-24. Retained research
only, NOT app default; 77/123 curated class/modes remain below90. RPnegative
loses48correct and NPnegative loses65, including84 lost NPnegativeBMP matches.
Full132table delivered BEFORE030. See docs/validation/qe_20261024_029.
  python tools/qe_optimization/run_qe_standard_mode_experiment.py
  python tools/qe_optimization/verify_qe_standard_mode_experiment.py
029 refits ALL four028 mode experts with15 genuinely nested independent STD
features; same labels/candidate pools/005STD cleanup/fixed027massstage.

Prospective030 additive NP BMP specialist (NO old024prediction splice):
  python tools/qe_optimization/run_qe_bmp_rescue_experiment.py
ALL029calls immutable; fresh25nested specialist fits with training-only partial
BMP labels and same029predictors. No mass/RT/range changes. Inner-only additional
extras budget10%ofendoNPBMPmanual; wrongfeatures<=2%inEACHNPmode. Final extras
not guaranteed. Whole132table plus BMP class-specific table required after.

Label-free related-standard descriptors (not used029/030):
  python tools/qe_optimization/prepare_qe_related_standard_context.py
Separate16context columns, no identity merging or assumed sharedRT/order,
source classes/coordinates immutable.363pool/old-column parity and reproduced
15oldexactSTDfeatures/rawmeasuredcoordinates. NOannotation decisions/gains.

Historical027COMPLETE,retained SMALL research addon:82408correct/9243extras/1589different-
features,versus022+23/+5/+1. All29newcalls=23correct+5extras+1differentfeature.
Full132table delivered BEFORE028. Superseded researchreference027,NOTproduction.
  python tools/qe_optimization/verify_qe_calibrated_mass_discovery.py
All132fixedrule/scope replays,33immutableoldtables,29rawcoordinates,276source/
148inputhashes passed;sourcefiles165unchanged checkedbyevaluator. No newscorefit.

028fixed FOUR mode experts (not selecting old phase winners):
  python tools/qe_optimization/run_qe_mode_expert_experiment.py
  python tools/qe_optimization/evaluate_qe_baseline.py <experiment_028> --manual-reference <samev2>
  python tools/qe_optimization/report_qe_experiment.py --experiment <experiment_028> --before <experiment_027>
  python tools/qe_optimization/verify_qe_mode_expert_experiment.py
100classifiers(180roundsEACH)in25nested parentmodels;ALL4experts refit. Same022
corepredictors/labels,005standards,EXACT027fixed label-free physicalmasspool and
sameadditionselectorafternewcore/stdcalls. Changed coreoccupancy may change
massadditions;wholealgorithmcomparedagainst027,also022forcorecontext. No025new
standardcontextfields/026weights/024labels. EqualprojectweightswithinMODE;
moretotalmodelcapacitythanoldshared180roundclassifier. OOFcutoffs/budget cover
core endogenouscalls ONLY,asin027; outer-only mass proposal caches NEVERenter
innercalibrationortraining. Finalextrasbudgetnotguaranteed. Full132before029.

026COMPLETE/REJECTED:82358correct,9301extras,1589differentfeatures vs022:-27/+63/+1.
Full132table actually delivered BEFORE027. No mode/class cherry-picking.
027fixed conservative new-mass discovery (not another scoring refit):
  python tools/qe_optimization/run_qe_calibrated_mass_discovery.py
No new score or cutoff fit. Independent RAWdeclared spike detector with EMPTY
endogenous proposals;query-excluded outer standard/RT/envelope references.
Actualneutralmassmedian>=6anchors,MAD<=.6ppm,p90<=1.5,abscenter<=2. Whole declared
spike target masses within5.01ppm withheld; target excludes own neutral groups.
Coarse corrected<3ppm/RT<=10s,>=3RTcohorts/in-domain/MAD<=10. ALLalternatives
compete BEFORE fixedfineRT<=3s/MAD<=5/OTHERmode strong exactclassCDB evidence,
novelrawabsppm>=3pair and unoccupiedfeature/absentidentity gates. ALL022calls
immutable,rawobservedmz/RT/I untouched. Samechemicaltarget/rangebounds;unknown
formulas cannot be guessed. Fixed thresholds NOT tuned against outer labels.
Evaluate/reportv2all33/baseline+022andfull132table before028. Helpersunusedinapp.

Autonomous cycles authorized24October; full132table still REQUIREDbeforeeachnext.
025(after024tableDELIVERED):
  python tools/qe_optimization/prepare_qe_standard_context.py
  python tools/qe_optimization/run_qe_standard_context_experiment.py
  python tools/qe_optimization/evaluate_qe_baseline.py <experiment_025> --manual-reference <samev2>
  python tools/qe_optimization/report_qe_experiment.py --experiment <experiment_025> --before <experiment_022>
  python tools/qe_optimization/verify_qe_standard_context_experiment.py
15numeric independent spike-context fields;25nested standard models,363exact
old-pool/predictor parity checks. Same labels and final005standard handling.
Completed82384correct/9335extras/1575different-features:versus022-1/+97/-13;
REJECTED globally. Full132table delivered before026. No class/modecherry-picking.

026(after025tableDELIVERED):
  python tools/qe_optimization/run_qe_class_balanced_experiment.py
  python tools/qe_optimization/evaluate_qe_baseline.py <experiment_026> --manual-reference <samev2>
  python tools/qe_optimization/report_qe_experiment.py --experiment <experiment_026> --before <experiment_022>
  python tools/qe_optimization/verify_qe_class_balanced_experiment.py
Fixedsqrt inverse candidate-class/mode cell weights,clipped[.25,4]withincohort,
then equal total cohort weights. Uses022oldfeatures and labels,NOT024/025combo.
Optionalweight_policy=None preserves original fit behavior; productionunused.
Same25nested fixed180HGB fits,newOOFcutoffs,9.5%endoextra trainingbudget,005STD.
All33/allclasses/full132table and heldout trade-offs required before adoption.

24 October resumption: experiment024 isolates partial BMP training supervision:
  python tools/qe_optimization/run_qe_partial_curation_experiment.py
  python tools/qe_optimization/evaluate_qe_baseline.py <experiment_024 output> --manual-reference <same v2 reference>
  python tools/qe_optimization/report_qe_experiment.py --experiment <experiment_024 output>
  python tools/qe_optimization/report_qe_experiment.py --experiment <experiment_024 output> --before <experiment_022 output>
  python tools/qe_optimization/verify_qe_partial_curation_experiment.py
Same363label-free022tables,25nested score fits,005standards and training-cutoff
objective. ONLY known BMP negatives in an uncurated opposite NP polarity become
UNKNOWN during permitted-group supervised fitting. All positives immutable;
no new positives, runtime polarity preference or manual-presence predictors.
Calibration/evaluation labels,92102manual denominator and ALL extras unchanged.
No new standard context in024. Existing standard-context preparation is unused.
Verifier checks25unchanged-positive/model-scope audits, heldout score reproduction,
132selector output comparisons and5calibration hashes/budgets. Extra compositions
also listed in the opposite NP manual list are diagnostic only, NOT excluded
from extras and NOT proof of this polarity/adduct/feature. Deliver the complete
132-row022 ->024table before another annotation-changing experiment.

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

Nested-training preparation (after the complete021 table was delivered):
  python tools/qe_optimization/verify_qe_unified_preparation.py
  python tools/qe_optimization/build_qe_nested_references.py
  python tools/qe_optimization/prepare_qe_nested_candidates.py
  python tools/qe_optimization/run_qe_nested_training_experiment.py
Evaluate/report experiment_022 with the same v2 manual reference and all132 rows
before any023 annotation experiment. The first three commands are engineering
preparation, NOT annotation accuracy experiments. The last command waits for
the complete363-table preparation summary before fitting/accepting anything.
The generic reporter also supports --before <other completed checkpoint>,
writing comparison_with_<checkpoint-name>/ separately from full_comparison/.
It verifies the two runs contain exactly the same dataset collection. For022,
both originalbaseline ->022 and009 ->022 reports are published; the latter is
the experiment delta, not a cumulative gain misattributed to the new change.

After022's FULL132 table was delivered,023tested class/mode cutoffs again,
this time on022's genuine nested four-inner-block OOFs (not008/007features):
  python tools/qe_optimization/diagnose_qe_acceptance_gates.py
  python tools/qe_optimization/run_qe_nested_class_operating_experiment.py
Evaluate/report023 as usual with referencev2,plus --before <experiment_022>.
First command is read-only recorded-score/gate auditing,NOT an annotation
experiment. All264selectorparitychecks passed. Gate audit outputv2 preserves
an emptyv1 directory from an overbroad whole-module byte guard; audited selector
ASTs are identical. Private manual row scores/FIDs must NEVER enter Git/models.
023reuses all022finalscores/pools/005standards,changes ONLY training-derived class
cutoffs. Exact022trainingreference and132uniformcompetitionchecks required;
15MILPstagesoptimal. No evaluated curation input, refit or new candidate window.
023was REJECTED:-156correct/+255extras/+2wrongfeatures versus022. It is retained
for provenance only,not as another default. Full132table required BEFORE024.
Use the SAME5 project blocks, with FOUR inner score folds. Reference masks
exclude the outer test, inner calibration and own-query project as applicable;
all learned endogenous features are generated before supervised flags enter.
Do not recycle the old self-informed training queries under a new fold name.
Inherited baseline/vendor exposure and unchanged005 standard handling remain
limitations; this is not a pristine external whole-pipeline test or an FDR.

CRITICAL resource isolation: these new runners freeze app/, reference_data/
AND methods/, set explicit process-local reference paths BEFORE app imports,
verify those resolve INSIDE the frozen snapshot, and record source/input hashes.
An app-only research snapshot silently fell back to older installed PE ranges
(30-48 instead of repository28-50). Existing-candidate parity caught this
BEFORE022 scoring/annotation. The invalid nested_reference_development_v1 and
partial nested_candidate_preparation_v1/debug_v1 outputs are preserved locally
for provenance ONLY; NEVER resume/reuse their tables. No source dataset or
production scientific default was changed. Current runners use new v2 paths.
The candidate preparer keeps at most3 numerical jobs queued, checks parity
before completing a dataset, and never supplies legacy cached label flags to
predictors. These flags exist in old parity caches but are explicitly ignored;
do not claim that absolutely no file containing labels was ever opened.

Read-only009 feature-occupancy diagnosis, not another accuracy experiment:
  python tools/qe_optimization/diagnose_qe_feature_occupancy.py --output <new empty directory>
Counts curator-matched endogenous features already in007 pools but not selected
by009, distinguishing features left unoccupied from features assigned another
label. Includes standards as possible occupants, but not in this diagnostic's
manual denominator. This never produces/revises annotations or promises gains.
Latest executedv2 explicitly reports zero manual feature-sharing conflicts and
agrees with every originalv1 count plus five candidate-ceiling invariants.
Only aggregate counts and hashed provenance may enter Git, not feature IDs.

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

014 finished:104 replacements,62 correct matches gained/30 lost,net+32 vs009.
Every132 comparison retained009 counts/extras.86 class/modes below90 vs85 in009;
small pooled improvement does NOT establish class-level success or app adoption.
After014's full table,015 tested one-pass strong automatic RT-anchor completion:
  python tools/qe_optimization/run_qe_strong_anchor_experiment.py
All modes,009 old calls unchanged. Score>=.95/<1.5ppm automatic endogenous
anchors,>=6 OTHER compositions,robust residual MAD<=2s/p90<=5s,shift<=60s,
nearby CDB; unique raw1.5ppm/5s hit BEFORE occupancy/spike filters. Other-group
composition>=25% and class/adduct>=50%,both>=3groups. No self-votes or iterative
cascades. Evaluate/report all33 with v2.015 was REJECTED:157 additional correct
matches but410 more extras/28 more wrong peaks vs009.9,726 extras exceeds initial
baseline safeguard by29; low correct-call yield is the more important concern.
FULL132 tables through015 delivered; app entrypoints/defaults still unchanged.

Read-only diagnostics, NEVER new annotation-performance claims:
  python tools/qe_optimization/diagnose_qe_standard_mass_offsets.py
  python tools/qe_optimization/diagnose_qe_class_mapping_failures.py
Both require NEW versioned output folders and refuse overwrite. The mass pilot
freezes standards-only offsets before reading discovery gaps.475007 missing
manual peaks would fit corrected3ppm AND old atlas10s; this is WINDOW POTENTIAL,
not475 additional correct calls. Future calibration needs actual declared-spike
validation,distinct-neutral-standard weighting and preserved original m/z.
The mapping audit counts failed DEFINITION/PHASE/FOLDER steps,not failed datasets.
304 such baseline failures; some rare classes also lack an outer-fold RTmodel.
Zero-match labels stay in the denominator; X4 has blank native chemistry and
cannot be responsibly guessed. See docs/PE_HANDOFF.txt for next-step hypotheses.

After015's full table,016 tested mass CONTEXT on EXACT007 physical pools:
  python tools/qe_optimization/run_qe_mass_context_experiment.py
Blind BASELINE automatic standards validated against actual positive-quantity
declared name/adduct/labelled mass. Neutral chemical masses vote once. Query
excludes own feature/chemical mass channels. >=6 standards,MAD<=.6ppm,p90<=1.5ppm,
abs median<=2ppm; unavailable staysNaN. Eight descriptors added,old raw ppm
retained. SAME180-round globalHGB,EXACT009 cutoffs(not recalibrated),005 standards.
All33 complete,full132 table delivered. REJECTED:136 fewer correct matches vs009
for only73 fewer extras. No default/candidate-coordinate/window changes. This
does not test a recalibrated operating point or expanded calibrated discovery.

Unused native-catalogue helper/chemical parity (NOT annotation performance):
  python tools/qe_optimization/verify_qe_catalogue_parity.py
Output native_catalogue_parity_v2,refuses overwrite. All33 baseline reference
unions,fold0 envelope as enumeration bounds.15,070 overlapping chemical targets,
zero mass differences>1microDa. Native codes/formulas/charges,no guessed masses.
V1 remains historical; V2 after equivalent-adduct spike-protection guard repair.
No full calls,curated feature mapping or original source writes in this check.
Next experiment must prove correct original-feature annotations and controlled
extras,not present reference enumeration/parity as annotation success.

After016's full table,017 tested unique restored native catalogue targets:
  python tools/qe_optimization/run_qe_catalogue_experiment.py
Missing from baseline+003 reference only; matching outer-fold native CDB envelope,
RT atlas and composition/channel support. Unique raw1.5ppm/5s hit BEFORE occupied
and declared-spike filtering; competing new claims all withheld. All009 calls
preserved. All33 complete:+19 correct,+34 extras,+1 different peak vs009. Total
82,022/92,102=89.055612%,9,350 extras. Not a production/default adoption.

After017's full table,018 tested a bounded RP residual-plane peak selector:
  python tools/qe_optimization/run_qe_rp_plane_experiment.py
Fixed009 identities/counts;>=12 distinct OTHER automatic CDB anchors,full-rank
interpolation hull,independent LOO checks,physical RP derivatives. Old peak>=7s
away,unique raw1.5ppm/3s alternative,occupied/spike protection,no cascades. All33
complete:ZERO replacements,exact009 results. Rejected as no improvement; don't
silently relax gates. Full132 tables through018 delivered before019.

Read-only explicit specimen metadata audit (no annotation selections):
  python tools/qe_optimization/audit_qe_specimen_pairs.py
Loads ONLY MS_metadata,not curated MS/MS_data. Exact filename/phase/mode/shared
sample_id plus validated group agreement.30/33 datasets pass; SCENIC_Retina group
conflict and NAD_Liver/NAD_Muscle unknown group mapping remain unavailable. No
guessing acquisition-number offsets. Private specimen maps remain outsideGit.

019, after018's full table:
  python tools/qe_optimization/run_qe_paired_sample_experiment.py
Same007 pools and global180-round HGB,plus8 paired-biological-specimen abundance
features;>=8 positive pairs,technical repeats averaged per specimen. OTHER-mode
frozen BASELINE automatic exact class/CDB anchors only,not feedback from009 or
test curation. Unknown metadata/profile evidence staysNaN. Refit final AND three
inner group-excluded score models per outer fold; recalibrate NEW score cutoffs
with EXACT009 joint weakest-mode objective/9.5% pooled endogenous training budget.
Same005 standards. Inherited OUTER-training RT features are not fully inner-
cross-fitted; known baseline exposure remains. Must evaluate all33/referencev2,
deliver full132 table,check held-out extras and class regressions BEFORE adoption.
Local frozen runner_source.py is authoritative for current execution; portable
runner path substitutions/formatting need a NEW clean workspace for reproduction.
019 complete/full132 table delivered:81,995 correct,9,382 extras,1,619 different-
feature choices;85/123 curated class/modes below90. vs009:-8 correct,+66 extras,
no net wrong-feature change. REJECTED; no default adoption or cherry-picking
improved modes/datasets. New paired metadata/evidence helpers remain unused by
production entrypoints. Inner/final model and source snapshots stay outsideGit.

Read-only discovery-versus-selection audit (not new annotation accuracy):
  python tools/qe_optimization/diagnose_qe_candidate_ceiling.py
All endogenous manual entries,including unresolved; standards excluded only
from this explicitly labelled diagnostic. Exact007 candidate presence is an
optimistic ROW-WISE ceiling,not necessarily attainable under feature/identity
competition.32 class/modes remain below90 even at this ceiling: scoring alone
cannot solve those classes. Do not change annotation denominators or drop them.

After019's full132 table,020 tests additive paired-profile-supported discovery:
  python tools/qe_optimization/run_qe_paired_discovery_experiment.py
Native monoisotopic catalogue/SAME003 other-group composition bounds,>=3RTcohorts,
MAD<=10s,raw<3ppm/<=30s. OTHER BASELINE strongautomatic exactclass/CDB support.
All physical alternatives compete before occupancy/novelty/profile gates; old007
queries remain competitors. Actual declared-spike masses exclude WHOLEtarget.
Frozen label-free evidence before attaching traininglabels; final/inner grouped
discovery HGB fits. New best identity AND best feature,margin.1,novelto007,
>=8pairedbiologicalsamples/logcorr>=.8,trainingcutoff. Every009 oldcall immutable.
New acceptance calibrated with matching008 INNER OOF007 old-policy occupancy,
NOT published training-dataset009 outputs from different outer models that could
include evaluated-project labels. Old009 cutoffs/RTfeatures not fullyinnerfitted;
conditional grouped replay with priorbaselineexposure,not external validation.
All33 finished:82,007correct,9,320extras,1,619differentfeatures. vs009:+4correct,
+4extras,0wrongfeatures. NOT adopted;50%newcallagreement/tinygain. Full132 table
delivered. Appdefaults unchanged. Frozen executedlocal runner authoritative.

Read-only proposal-limit audit (NOT annotation performance):
  python tools/qe_optimization/diagnose_qe_paired_discovery_limits.py
12,473queries/6,795novelto007;273novelmanualfeatures considered;98passpairgate,
74alsofree/identityabsent009. These are NOT correct acceptedannotations. Only
aggregatecounts/sourcehashes committed; privatefeature/sample/modeltables outsideGit.

UNUSED class_operating_point.py prototype+synthetic tests prepares a future021
class/mode cutoff experiment. No real021 annotations or productionhooks yet.
MILP cappedclasscoverage objective,pooledbudget,class+mode+totaltraining floors,
under-supportedreferencefallback,exactglobalwinners(no promotion of losinglabel).
Read docs/PE_HANDOFF.txt before building groupedOOFtrainingtables/realexperiment.

After020's full132 table,021 tests class/mode TRAINING-cutoff allocation:
  python tools/qe_optimization/run_qe_class_operating_experiment.py
EXACT007candidates/evidence/finalscores,005standards. SAME008 INNER OOFscoremodels;
all5exact009training-referenceparitychecks. MILP cappedclasscoverageobjective,
9.5%pooledendogenoustraining extrasbudget,class/mode/totalfloors andunsupported
fallbacks. No promotionoflosingfeaturelabels. All33finished/full132tableDELIVERED.
REJECTED:81,610correct,9,669extras,1,579differentfeatures;92/123class/modesbelow90.
vs009:-393correct,+353extras,-40wrongfeatures. All33finalscoresEXACT009unchanged.
No appdefaults changed; heldoutcountsnotguaranteedbytrainingconstraints.

Read-only paired feature-distribution diagnosis (NOT annotationperformance):
  python tools/qe_optimization/diagnose_qe_training_feature_shift.py
SAMEcorrectmanualquery inheldout007cache vsfourcacheswhoseatlastrainingincludes
thatproject. Conditionsonpresence inBOTH; missingcandidatecountsseparate. Small
pooledmedianRTdifference,butlarger forsomeminorityclasses. Not proof ofsolecause
of021failure. Aggregateonly,nomodel/annotationchanges or per-featurepairtables.

Unused nested_feature_folds.py +synthetictests prepares a future022properly
nested ENDOGENOUS featuregeneration experiment:25exclusionmasks overexisting5
projectblocks,4innerblocks. Everyquery,outertest andinner-validationproject
excludedfromrelevantRT/prior/envelopemodels. Actual33/30groupplanvalidated.
No022models/predictors/annotationsyet. Preservegroupchecks andprovegeneric007
preparationparity BEFORE runningnewmodels. See current docs/PE_HANDOFF.txt.
