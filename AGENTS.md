# Experiment reporting policy

## Current work status — RESUMED by user,25October2026

034COMPLETE,REJECTED;029retained. Full132TARGETscope table +wrongmode totals
DELIVERED before any035annotation.82729 ->82814/92100 (+85),89.8251900 ->89.9174810%;
extras9268 ->9405(+137),wrong1604 ->1610(+6). Priority81896 ->81981/90871;
43 ->44below90/86; alltarget76 ->77below90/122. Sparseunchanged833/1229.
RPnegative-12correct; NPpositiveBMP+63correct/+65extras,not90%. ALLintegrityPASS:
296source/481inputhashes,25UNCHANGEDmodels/100experts,ZEROclassifierfits,
528queryscopechecks,132INNER+33outerBITEXACTscores,5curves/choices reproduced,
660uniformselectorparities,132fullpipeline/rawcoordinatechecks,33manualhashes.
Reports docs/validation/qe_20261025_034; actualdelta versus_029. No appdefault.
Next needs better candidate discrimination,not just accepting more old scores.

034method: guarded class cutoffs on EXACT retained029 scores/models and
standard_context_preparation_v1. New helper guarded_class_operating_point.py
alongside UNCHANGED preexisting class_operating_point.py.725tests passed,
8skipped,83subtests. Protocol frozen,ALL5INNER calibrations completed BEFORE
outer outputs. Training gain109-155 per fold is NOT held-out accuracy. Every
eligible class keeps its baseline training correct count; each mode wrong
cannot rise; same9.5%training extras budget. EligibilityTRAININGonly>=100manual,
3projects,20correct winners from3projects. Sparse/unsupported keep029cutoff.
Rawscore positive-coverage quantiles can be<.05;ambiguitymargin still.05;
fixed feature/identity winners never promoted.02925models/100experts reused
deliberately,ALL132INNER+33outer scores reproduced BITEXACT,not alias fitcache.
No classifier refits/newcandidates/ranges/rejected033scores. Same005cleanup,
fixed027mass stage. Main runner run_qe_guarded_class_experiment.py; new integrity
tool verify_qe_guarded_class_experiment.py. No production change. Mustfinish
ALL33evaluation,full132table,integrity,retain/reject BEFORE any035annotation.
029remains retained until decision. Old033checkpoint below remains valid.

LATEST checkpoint:033COMPLETE,REJECTED;029retained. FULL132TARGETscope table
plus separate wrong-feature totals DELIVERED in this chat;doNOTrepeat merely
because another cycle starts.033target82729/92100=89.8251900% (NO net gain),
extras9268 ->9243 (-25),wrong1604 ->1611 (+7). Priority81896 ->81881/90871
(-15),43 ->45below90/86;capped90mean83.8606410 ->83.8573513. Sparse833 ->848.
Alltargetclass/modesbelow90:76 ->79/122. Both main+supplemental integrityALLPASS:
294source/531inputs,25models/100fresh180roundexperts,528queryexclusions,
33outer scores/132fulloutputs,33rawSTD/363ALLpool+predictorreproductions,
1105prepinputs,132rawcoordinate stream checks. Supplemental132INNERscores,
5complete countgrids/5cutoffchoices and33identicalmanualreferencehashchecks.
Full v2preparation363complete; six uncached datasets/66tables EXACTparity,
8086normalizations/100range dictionaries. No model cache/reuse.715tests passed,
8skipped,83subtests. All033annotation/evaluation/verification jobs finished.
Reports:docs/validation/qe_20261025_033;actualdelta versus_029 (not top-level).
766newouter candidate pairs,7630all masks are NOTaccepted matches/gains.
Correct evaluator reference argument is QE/manual_reference_v2,NOT
baseline/evaluation_v2 (the latter has comparison tables,not reference inputs).
No034fits/annotation decisions yet. Next: class-aware INNER-only acceptance
calibration from WHOLEretained029; unchanged candidates/scores/competition,
training-only support/extra+wrong guards; no rejected033winner splicing.
Do not change app annotation defaults yet. All older status paragraphs below
are historical; this checkpoint supersedes their in-progress033descriptions.

The user requested a pause at the next safe stopping point. Experiment032and
its complete integrity replay are finished;032was rejected and029remains the
retained research reference. Full132-row032table was delivered. No033annotation
fits started,no annotation jobs remain running,no production defaults changed.
User explicitly resumed on25October. Next cycle tests independent-standard
RT alignment from retained029; no production defaults change. Handoff and
next implementation notes:docs/validation/next_common_RT_experiment.txt;
verified results:docs/validation/qe_20261024_032/versus_029/.

25October033implementation:standard_aligned_candidates.py adds STRICT original-
feature discovery and8numeric independent-standard RTdescriptors. ALLold029
physical competitors retained;competition evidence recomputed. New scope guard
recognizes q_standard_aligned_ even without oldSTDcolumns.713app/tests passed,
8skipped,83subtests. Label-free preparationv1 is running across363query/masks;
first Sperm_cells dataset independently reconstructed all11masks including raw
spike detection/all predictors. This is integrity,NOT annotation accuracy.
Latest engineering checkpoint supersedes that preparation status: original
v1preparation intentionally interrupted after6complete dataset receipts to
cache deterministic text normalization/immutable range lookup ONLY. All files
preserved, no annotation fits.715app/tests passed/8skipped/83subtests. Cached
v2single Sperm_cells11mask EXACT full-output parity PASSED (pool,predictors,
metadata). Full v2preparer is now running, followed by a new queued launcher.
Original v1queued fit launcher was stopped BEFOREanyfits. Current workflow:
prepare_qe_standard_aligned_candidates_cached.py, then
verify_qe_composition_cache.py (ALL6complete v1datasets/66tables must match
exactly), then033run/evaluate/report/fullverification.033runner now requires
v2preparation ANDthe cache-equivalence receipt before fitting. No thresholds,
references, raw measurements, scores or labels enter the NEWtext/range cache;
pre-existing raw-quality caches unchanged. NO033performance claim yet.
Do not start a duplicate033runner or a next experiment before its fulltable.
Sources:prepare/run/verify_qe_standard_aligned*_experiment.py in tools (prepare
file is prepare_qe_standard_aligned_candidates.py).029remains retained.

Read-only selection diagnosis v2 completed ALL33/132; no annotation changes or
new cutoffs. Of8975priority missing:3935correct peaks win fixed competition but
fall below mode cutoff;1101lose identity competition;402lose labelcompetition;
193ambiguous;3257absent COREpool;59STDspecialist;28unresolved.3935includes449
scores<.05,1679in[.05,.2),1807>=.2. These are NOT guaranteed safe gains. Initial
v1count4552included617also blocked by identity competition;source85c206a kept.
See docs/validation/qe_20261025_selection_029/interpretation.txt. AFTER033full
table, class-aware INNER-only cutoff calibration is a promising next separate
experiment; do NOT start it yet or infer thresholds from outer query labels.

## Active example collection — user decision, 22 October 2026

The ONLY active example collection for future optimization and performance
evaluation is:
`/Users/yorrick/Library/Mobile Documents/com~apple~CloudDocs/Work/AI/example_data_CFM/QE_example_datasets`

The user confirmed that the datasets in this folder are now all we have.
Do not include the sibling `archive/`, older flat-layout example folders or
historical saved runs as additional current datasets. Do not report archived
datasets as missing inputs that need to be recovered for the active benchmark.
Historical experiments remain historical evidence, not the current baseline.

After the user's explicit removals on 22 October, the active inventory is
33 folders, 132 folder/mode combinations and 33 distinct original four-peak-list
sets. Removed from the active collection and moved to recoverable Trash:
all three `24P0038_Fibros...` versions, both `25P0058_Heart_V2` versions,
`26P0002_ZebraFish_PEX_Yorrick`, `25P0081_IPSC`, and
`24P0072_IONIS_plasma_serum`. Do not restore/include them without a new user
request. These are user-requested exclusions, not proof that all their data
failed quality checks. The earlier 41-folder/164-check/38-input-set audit is
historical and predates these removals.

Show every remaining folder in detailed results. Keep related or shared-
acquisition datasets together in development/test splits. Different peak-list
hashes alone do not establish independent biological cohorts. If future
additions share original inputs, resolve the authoritative reference version
and count each input set only once, never choosing by the best LASER score.

A new annotation-performance baseline must be established on this active
33-folder collection. Before/after experiments must use the same collection
and denominators. The quality audit is not an automatic-annotation performance run.
The old 15-folder/14-unique-dataset scores must not be presented as scores for
this collection. Prior use of any overlapping datasets still counts as prior
development exposure; changing the collection folder does not make them
untouched external validation data. Reinventory if the folder contents change.

## Reporting requirements

24 October user update: ignore X4 for the optimization/90% target. Two NPnegative
manual rows are excluded from a COMPANION target-scope view:92100manual rows,
122curatedclass/modes,currently76below90 in retained029. Historical all-manual
tables remain92102/123/77 for continuity; denominator change is NOT a gain.
No formula guess or curation relabeling. See report_qe_target_scope.py.

Deployment deliverable requested24October: once the ACTUAL integrated app is
ready, run fresh end-to-end annotation/peakgroup exports for all33datasets,
benchmark same original features by class/mode versuscuration, and produceONE
colleague-ready report. Not current cachedresearchreplays. AddONLYcolumns2/3
laser_annotation/adduct to originals, names *_annotated.txt, private output
root with dataset/mode subfolders. Detailed requirements recorded in
docs/validation/deployment_benchmark_plan.txt. Not yet deployment-ready.

24 October: user explicitly authorizes autonomous optimization cycles and
retain/reject decisions without asking between experiments. This does NOT
waive the complete user-facing before/after table requirement before starting
the next annotation-changing experiment, or authorize benchmark relabeling.

24 October latest priority: postpone dedicated optimization of classes with
very few manual entries. Working priority is class/mode combinations with at
least 100 manual entries across at least three independent project groups;
this cutoff is an agent-selected practical priority, not a user-defined
exclusion or a revised benchmark denominator. Keep sparse classes visible in
all full performance reports, and do not count their postponement as a gain.
The incomplete catalogue_candidate_preparation_v2 prototype is parked (six
dataset receipts, no annotation fits); do not present it as a completed
experiment or resume sparse-class work ahead of well-represented classes.

Latest retained research reference after029:82729/92102correct,9268unconfirmed
extras,1604different-feature choices (all33datasets/132modes). Versus028:
41morecorrect,-78extras,-24wrong. RPnegative loses48correct,NPnegative loses65;
NPnegativeBMP loses84 (146/520=28.1%). These regressions MUSTremain explicit.
77/123curatedclass/modesbelow90;goalNOTachieved. NOTproductiondefault.
Original9697extrasbaseline still429higher. Compare next algorithm against029.
025/026remain rejected;027fixedmassstage+028modeexperts retained with029's
15independent nested standardcontext columns. Full029table delivered before030.
030:NEWnested/OOF-calibrated NP BMP specialist additions,all029calls immutable.
No posthoc old024model/class output splice. Related-standard descriptors have
been prepared WITHOUTannotation decisions;notused029/030.

030completed_scoped_v2 REJECTED:82785correct/9430extras/1605differentfeatures:
versus029+56/+162/+1,219totaladditions. Initialstrictscopeattempt failed
incomplete13datasets/16fits;preservedunchanged. ALL25fits restarted explicitly
distinguishingallowedreferenceprojects from real supervised contributors;
no inventedlabels or weakenedexclusions. Full030table delivered before031.
031currentlytests16related-standard descriptors on ALLfour029coreexperts,
keeping029labels/exactSTDfeatures/candidates/005standards/fixed027massstage.
NOrejected030scores/labels/additions. Comparewhole031algorithm against029.

031alsoREJECTED:82728correct/9180extras/1592differentfeatures,vs029-1/-88/-12;
RPpositive-129correct;targetclass/modesbelow90rise76to78of122,cappedclassmean
declines. Full031TARGETscope132table delivered.029remainsresearchreference.
Currentread-only029candidateaudit:6215missingmanualcorrect-featurecandidates
alreadyinpool,3128mappedmissingfrompool,28unresolved(target92100);31/122class/
modescannotreach90evenoptimisticrowwisecandidatecoverage. See
docs/validation/qe_20261024_candidate_gaps_029. Prioritize discovery/selection
gaps in well-represented classes, not outer-test winner ensembles.
032annotation/evaluation COMPLETE,REJECTED:82652correct/9186extras/1623different-
features,versus029-77/-82/+19. Target82652/92100=89.7415852%;77of122below90.
All77lostcorrect are in86prioritycells:81896/90871 ->81819/90871;43 ->44below90;
capped90prioritymean83.8606410 ->83.8244654. Sparsecorrect833/1229unchanged.
Full032TARGETscope132user-facing rows and separate wrong-feature mode totals
HAVEbeen delivered;doNOTrepeat merely because another cycle starts.
363label-free old-pool/old-column parity preparation;12native-adduct-aware
same-acquisition measurements. Allfourmodeexperts refit with SAME029labels,
15exactSTDfields,OOFobjective/budget,005cleanup/fixed027massstage.029retained.
Complete032integrity ALLPASSED:289source/529inputhashes,25models,528query-
exclusions,33scores/132outputs/5calibrations,33raw-measurement reproductions,
363old/new prepared table checks,432preparation inputs,twoSTDaliasdemotions.
No033annotation fits yet. No appdefaults. Next implementation notes:
docs/validation/next_common_RT_experiment.txt.
See docs/validation/qe_20261024_032/versus_029 for actualdelta (top-level tables
compare original baseline for continuity). Currentworktree app/tests709passed,
8skipped,83subtests;these are engineering checks,not annotation accuracy.

Read-only032fit-symmetry audit:36of40entire-expert serialized pairs identical;
four0/1pair experts differ in estimator_preprocessor serialization. Other
individual estimator state fields and expert metadata matched in a key audit,
but full transform/score equivalence NOTproven. No fit-cache or model reuse
implemented;actual032still25parentfits/100modeclassifiers. Preserve provenance.

Read-only common-class029diagnosis COMPLETE:86priorityclass/modes,90871manual,
81896correct,43below90. Of8975missing:6085correct-feature candidates already
available,2275outside current nested RTprediction10s,409outside3ppm,138missing
exacttheoreticalidentity/adduct,40STDspecialist gaps,28unresolved. RTcategory
is relative to the atlas/fallback,not proof it is the only exclusion cause.
1043RTgaps are10-15s.104inputhashes verified. See
docs/validation/qe_20261024_common_discovery_029. No annotation changes.
Newunused standard_anchored_retention prototype+9synthetic tests: exact-class
independent PRIMARY-library spike RT residual versus OTHER-group isotope
transfer;no feedback from transfer-only detections,no ether/lyso merging,
strict reference exclusion,weak/disagreeing/out-of-domain anchors withheld.
Returns numerical predictions ONLY,no candidates/annotations;NOTused032.
Promising next discovery strategy after032table,not proven gains. Mainmanual
scope unchanged;92072mapped target entries have zero shared-feature conflicts.

BMP clarification (22 October 2026): LASER should annotate BOTH NP polarities
when each has supporting evidence. Do not impose a preferred-polarity filter
based on the manual list. Opposite-polarity manual support may be reported as
a diagnostic, but manual-absent calls remain unconfirmed extras in the primary
evaluation. Experiment024 tested training-only unknown labels for opposite-NP
BMP absence: BMP improved substantially, but all-class correct matches fell38
and extras rose32 versus022. It is NOT adopted as a global/default model; do not
silently combine022/024class outputs from outer-tested winners.

Current optimization target (22 October): the user requests at least 90%
correct same-original-feature coverage of manually curated lipid classes,
across all four modes, with controlled extras. This is not a PE-only target.
The user permits new strategies, composition limits and matrix-aware options.
They delegated the extras trade-off to the agent's judgment: use no net increase
in total extras as the initial guardrail, and report any small increase versus
a substantial gain explicitly before adopting it as a new default. Do not
claim analytical FDR from curator agreement. Do not feed a dataset's curated
workbook/saved fits or hard-coded dataset/feature identities into annotation.
Any learned priors, RT atlas or ranking model must be evaluated with grouped
held-out data, keeping related/shared acquisitions together and explaining
prior development exposure and rare-class limitations.

User policy established 21 October 2026: after every experiment that changes
annotation behaviour, report a complete before-and-after table for every
example dataset and every measurement mode. This applies to changes in code,
models, settings, thresholds, composition limits, acceptance or original-peak
selection, including experiments that are rejected or reverted. Do not replace
the complete table with pooled totals or only the rows that improved.

## Required report

- Include RP positive, RP negative, NP positive and NP negative for each dataset,
  even when a mode is unchanged. Mark unavailable or untested cells explicitly;
  never fill them with assumed zeroes or imply a test was performed.
- In the user-facing response, show the complete dataset-by-mode table with
  correct same-manual-peak PERCENTAGES before and after, relative to ALL manually
  curated annotations in that dataset/mode/class scope. Include the exact match
  counts and manual denominator alongside the percentages, for example
  90.0% (90/100). Keep extra annotation counts before and after as separate
  columns. Separate mode tables are acceptable. State the lipid-class
  scope explicitly. Include all-class results; for a class-specific experiment,
  also provide that class's complete table in the response or linked full report.
- A correct match requires the same lipid, adduct, measurement mode and ORIGINAL
  feature as manual curation. The right lipid on a different feature is not a
  correct match. Report different-peak choices separately.
- Percentage policy clarified by the user on 21 October 2026: correct/manual
  percentage = 100 × correct original-peak matches / all manual annotations.
  Include unresolved manual-feature mappings in the primary denominator and
  report their count separately; they remain unconfirmed matches, not proven
  wrong peaks. A mapped-only percentage may be a clearly labelled secondary
  diagnostic, never substituted for the full-manual-list percentage. Resolve
  duplicate manual entries explicitly. Pool counts and denominators before
  division; do not average dataset percentages. Display one decimal place and
  retain unrounded values and counts, so rounding cannot conceal a small gain.
- Extras are calls absent from that dataset and mode's manual list, not merely
  the increase in total annotation count. Call these unconfirmed extras, not
  proven false positives. Report totals already present, not just newly added
  extras. Do not collapse the same lipid across different datasets or modes.
- The saved report or CSV must also show manual denominators, before/after
  match percentages, different-peak choices, total annotation counts and manual
  entries not on the same peak. Clearly define missing/unresolved categories;
  do not add overlapping categories as though they were disjoint errors.
- Include each folder in the active collection in the detailed results. Flag
  shared original-input sets and exclude duplicate versions from pooled
  totals, following the active-collection rules above. Label the pooled input
  set count and the chosen reference versions explicitly. The historical
  duplicate-Fish/14-unique-dataset convention is no longer the current scope.
- Identify the before and after versions, experimental/default settings, tested
  scope and whether results come from full raw-input runs, saved-output replays
  or conditional cross-validation. State training/development-data limitations
  and missing source files. Never describe these examples as untouched external
  validation or describe cached replays as new full-input annotation runs.
- Preserve the exact prior checkpoint and record both gains and regressions.
  Save the tables and provenance with the experiment, link the report, and give
  the user its complete table before moving on to another experiment.

Documentation-only or formatting-only changes that cannot affect annotation
results do not require rerunning annotation. Reuse an explicitly identified,
verified comparison when reporting the same completed experiment again.

Latest historical annotation-performance table (not the active collection's baseline):
[Local calibration pilot, rejected for deployment](docs/validation/local_pe_calibration_20261021.md).
The app retains the [user-requested rollback](docs/validation/experiment_rollback_20261021.md).
