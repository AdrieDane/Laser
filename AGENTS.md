# Experiment reporting policy

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
gaps in well-represented classes, not outer-test winner ensembles. Candidate-
adduct-aware companion evidence is a prototype under test;032notyetannotation-
fitted and no annotation improvement has been established for that prototype.

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
