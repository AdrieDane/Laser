# Stage 1 checkpoint: frozen baseline and evaluation contract

Completed 5 November 2026. Protocol `laser-untargeted-evaluation/1`.
Baseline application revision: `678ca0cab12e2e4a01b3197bb32e6b56cb202279`.
Stage 2 has not started. This checkpoint changes evaluation tooling and
documentation; it does not change application predictions or defaults.

## Frozen scope and provenance

One ALD development acquisition, all four modes, and all 63 reference
class/subclass labels. The copied PC holdout project and its original project
have byte-identical original feature lists in all four modes; they count as one
acquisition. The historical 33-dataset annotation optimization remains paused.

| Mode | Original features | Manual reference rows | Mapped rows | Formula recorded |
| --- | ---: | ---: | ---: | ---: |
| RP positive | 85,091 | 1,694 | 1,680 | 1,593 |
| RP negative | 57,065 | 679 | 671 | 585 |
| NP positive | 53,314 | 757 | 745 | 724 |
| NP negative | 58,854 | 538 | 526 | 512 |
| Total | 254,324 | 3,668 | 3,622 | 3,414 |

The private checkpoint contains the source archive, complete installed-package
versions, Python/Node versions, copied current workflows, metadata, producing-job
settings/log, input hashes, accepted annotations, saved statistics, isotope/formula
evidence, original-feature crosswalk and explicit query lists. It verifies copied
bytes and checks that the source project remains unchanged. The 46 unresolved
reference mappings remain in the all-manual denominator. All 3,622 mapped rows
were found in the frozen original feature coordinates. Mapping mass/RT differences
are recorded for review; matching a feature key does not independently prove its
lipid identity.

Source worksheet rows uniquely identify reference entries. Repeated worksheet
rows fail the freeze instead of being silently removed. Repeated original ions
are retained for all-manual reporting but deduplicated for ion-level metrics;
conflicting reference formulas/adducts are marked ambiguous and excluded from
conditional agreement. No repeated source rows or ambiguous ion references were
found here. Detailed rows and study identifiers stay outside Git.

## Evidence inventory and limitations

- Twelve biological specimens pair uniquely across every mode pair: six in each
  biological group. Three pooled QC and two blank/IS injections per mode stay
  available in abundance plots but do not enter biological comparisons or profile
  matching.
- Saved statistics cover 251,431 features after accepted-lipid exclusions. There
  are 762 raw-p hits at the saved threshold 0.001; 387 also pass the saved absolute
  log2 fold-change threshold 1.0. The existing viewer's raw-p selection is the
  baseline; Stage 1 does not change this definition.
- The four saved mass-calibration audit rows say `not_applied`, with zero confirmed
  standards. No standalone calibration export or RT calibration model is present
  in the current project. Upstream corrections are unknown.
- Table-derived M+1/M+2 candidates are available. No independently verified
  isotope envelopes, raw chromatograms, or MS/MS files were found in the frozen
  current project. This says nothing about files elsewhere. Trace-dependent and
  MS/MS comparisons remain deferred.
- The IS catalog contains 26 names/concentrations. It does not independently
  establish elemental formulas, envelopes, purity or ion assignments.
- Curated adduct labels cover protonated (1,540), deprotonated (1,175), ammoniated
  (911), and doubly deprotonated (42) ions. There are no verified sodium, potassium,
  formate, acetate or chloride labels in this reference. There are no curated
  negative-mode PC references. Missing references cannot establish false positives.

All these ALD classes were previously inspected during development. **The entire
acquisition is development/regression data.** The final-validation query list is
explicitly empty pending an independent acquisition or independently verified
standards. Randomly splitting these already inspected ions would not create an
untouched test set. Future split decisions must occur at acquisition/compound
family level before inspection, with any training/library overlap recorded.

## Reproduced baseline

Two fresh processes loaded the frozen saved analysis, applied grouping v5 and
called the actual JavaScript `sampleFigures` selector. The inference worker
received only source mode/feature keys; manual PC labels and reference adducts
were loaded separately for scoring. Accepted runtime lipid annotations remain
available as the existing algorithm specifies; this is not a wholly blind
identification experiment. Statistics, isotope searches and formula searches were
preserved, not rerun.

Both replays produced identical compound tables, candidate-link tables, browser
payloads and selected representatives: 668 compounds, 1,624 ions. Grouping took
5.14 and 3.21 seconds on the recorded machine; these are worker times, not browser
selection latency measurements.

The selected 15-PC regression reproduces **10/14 (71.4%) exact positive-mode
counterparts**. Source-mode self-matches do not enter that denominator; one source
has no expected other-positive reference. All 60 source-by-mode slots are retained,
including negative modes with no curated PC reference. This reproduces the prior
checkpoint; it is not a new performance gain or general identification accuracy.

## Metric contract

Freeze query IDs, feature mapping, scope and eligibility before running a candidate.
Reference names/formulas/adducts are evaluation labels, never inference input in a
blind arm. A separate known-adduct formula arm may supply the reference adduct,
but must be labelled and never scored as joint adduct inference.

| Metric | Required denominator and failure treatment |
| --- | --- |
| Conditional adduct agreement | Correct / all assigned, reference-verifiable original ions. Wrong adducts stay in the denominator. Unknown or conflicting references are separate. This curated development agreement is not independently validated accuracy. |
| Assignment coverage | Assigned / predefined eligible mapped unique reference ions. Abstentions stay in the denominator; report production exclusion/selection reasons separately. |
| All-manual recovery | Exact original-ion/adduct matches / ALL manual rows, including unresolved mappings. This is coverage, not conditional accuracy. |
| Formula candidate recall, top 1, top 5 | Correct neutral formula generated or at stated rank / ALL predefined formula benchmark queries. Missing predictions, unsupported charges/masses and timeouts remain failures in the primary view; show a compatible-input secondary view separately. Missing reference formulas are declared before the run. |
| Joint correctness | Correct neutral formula, adduct and charge on the same original feature / ALL joint benchmark queries. Matching ion composition alone is insufficient. |
| Counterpart recovery | Correct original feature AND adduct in target mode / ALL expected target links. Separate missing, wrong-adduct, different-peak and unresolved-reference outcomes. |
| Isotope/group quality | Verified member/charge precision and recall; group purity/completeness, isomer merges and family splits. Currently unmeasured: no verified envelope/network gold set. |
| Unconfirmed extras | All predicted ions absent from the declared reference, including pre-existing extras. Do not call these false positives or analytical FDR. |
| Confidence/speed | Correctness by confidence range with abstention; fixed-input worker time/memory and browser selection latency, reported separately. |

Runtime assignment reasons include accepted-lipid exclusion, absence from saved
tests, raw-p selection, and unsupported assignment. These explain the small
coverage of a significance-seeded workflow; they must not be hidden by reporting
agreement alone. The baseline illustrates the difference:

| Mode, all classes | Adduct agreement among assigned | Assignment coverage | All-manual recovery | Unconfirmed extra ions |
| --- | ---: | ---: | ---: | ---: |
| RP positive | 90.2% (46/51) | 3.0% (51/1,680) | 2.7% (46/1,694) | 553 |
| RP negative | 100.0% (3/3) | 0.4% (3/671) | 0.4% (3/679) | 546 |
| NP positive | 75.5% (37/49) | 6.6% (49/745) | 4.9% (37/757) | 193 |
| NP negative | 83.3% (5/6) | 1.1% (6/526) | 0.9% (5/538) | 217 |

Small, selected denominators—especially the three negative-mode assignments—do
not support broad accuracy claims. These exact-feature adduct comparisons do not
establish lipid species identity. Formula ranking and verified isotope/group
quality have not been newly evaluated in Stage 1. Their existing outputs are
frozen for the later stages.

Every behavior-changing experiment must report before/after results for the
complete frozen scope in all four modes, with class-specific results where
applicable. Include lost correct calls, wrong peaks, abstentions, unconfirmed
extras and unavailable/untested cells. Show percentages to one decimal with n/N;
retain unrounded values. Pool counts before dividing. Do not change queries or
select class/mode winners after seeing results. Preserve unsuccessful attempts.

## External comparison readiness

[Version/artifact pins and compatibility checks](external_tools.json) cover
MZmine 4.11.0, SIRIUS 6.5.4, msbuddy 1.0.0, Binner 1.0.0 and optional IDSL.UFA 2.0.
No external engine was installed or benchmarked during this stage.

Pinned MZmine source confirms the 800 m/z formula cutoffs and shape-correlation
defaults (r 0.85, five points, two on each edge). The lipid comparison must raise
the mass cutoff and specify adducts/polarity explicitly. SIRIUS documents
single-charge import; charge-2 queries must remain visible as unsupported.
msbuddy does not identify adducts and requires explicit hypotheses; its wheel
metadata and documented Python limits differ, so use an isolated supported
environment rather than modifying LASER's Python environment. Binner's artifact
could not be retrieved with verified TLS; its JAR hash and JRE compatibility must
be established before running it. IDSL.UFA is deferred until isotope profiles exist.

The checks use official [MZmine parameters](https://github.com/mzmine/mzmine/blob/23a45f6c3aa9bee28312bed36fe83b224a049e80/mzmine-community/src/main/java/io/github/mzmine/modules/dataprocessing/id_formulapredictionfeaturelist/FormulaPredictionFeatureListParameters.java),
[SIRIUS adduct documentation](https://v6.docs.sirius-ms.io/adducts/),
[msbuddy input requirements](https://msbuddy.readthedocs.io/en/latest/quickstart.html),
[Binner release/input specification](https://binner.med.umich.edu/), and
[IDSL.UFA package metadata](https://CRAN.R-project.org/package=IDSL.UFA).

These are preflight findings, not executable benchmark adapters or performance
claims. Before each external comparison, freeze the full dependency environment,
model/database hashes, exported batch/configuration, input conversion, charge and
missing-value behavior, thread/memory limits and timeouts. Engine defaults alone
are not a fair comparison configuration.

## Reproduction

Use private paths outside this checkout. Python must provide the baseline LASER
dependencies; Node is required for the real viewer selector.

```sh
python3 tools/freeze_untargeted_baseline.py \
  --project /private/project \
  --reference /private/curated_original_feature_mapping.csv \
  --queries /private/regression_queries.json \
  --output /private/new-checkpoint \
  --node /path/to/node --expected-correct 10 --expected-total 14
python3 tools/audit_untargeted_checkpoint.py \
  --checkpoint /private/new-checkpoint \
  --duplicate-project /private/original-copy
```

The reference CSV carries stream, original feature, mapped flag, name, class,
adduct, formula, manual m/z and RT seconds, source sheet and source row. Regression
JSON carries stream, original feature and `manual_PC` identity. Worker input is
stripped to mode/feature keys. Existing checkpoint/audit directories are refused;
use a new version for every attempt. The first development attempt exposed a
bad import in the new runner; the second exposed producing-job path relocation.
Both failed attempts were retained privately, fixes tested, and the third
checkpoint completed both replays and integrity checks.

Validation: 14 focused evaluation/profile tests passed; lint and whitespace
checks passed. No application code changed, so this does not claim a new full
application-suite run or a new browser-latency measurement.

Stage 1 is complete. Review this checkpoint before starting Stage 2's acquisition
settings and adduct-chemistry changes.

## User-authorized scan-data supplement

After the checkpoint was pushed, the user supplied the full archive of the same
ALD acquisition and requested a new, inputs-only test project. Its four original
peak lists are byte-identical to the frozen acquisition. This expands available
evidence without adding an independent dataset or changing the original checkpoint.

The new project follows `XCMS/`, `metadata/`, `Raw/`, `Calibration/`, `LASER/`,
`Untargeted/` and `Archive/`. All output folders remain empty; no saved results or
annotation holdout were copied. Reference workbooks/libraries for evaluation are
kept separately under `evaluation/reference/`, outside runtime input discovery.
The previous project preference was not changed.

All 68 converted mzXML injections were copied, SHA256-verified and streamed through
an XML parser: 17 files per mode, 65,990 centroided MS1 scans, correct polarities,
and **no MS2 scans**. The source vendor-raw folder contains only a placeholder.
Thus scan-level MS1 traces/isotope evidence are now available for the optional
trace stage; MS/MS evaluation remains unavailable. No peak-shape extraction or
isotope/formula benchmark was run as part of copying these inputs.

LASER's own readers validate all four peak lists, all 17 sample metadata mappings
per mode, all matching acquisition files and the 26-standard CSV. Method settings,
source documentation and file-level provenance accompany the private project.
The freeze utility now includes canonical `XCMS/` and `Raw/` directories; a
synthetic regression protects these inputs while excluding old archives and gold
reference folders. Fifteen focused evaluation/profile tests pass after this
extension. Stage 2 remains unstarted.
