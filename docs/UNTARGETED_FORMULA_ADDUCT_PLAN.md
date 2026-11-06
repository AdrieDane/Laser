# Untargeted formula prediction and adduct annotation plan

Created 5 November 2026. Stages 1 and 2 completed; Stage 3 comparison published;
Stage 4 software/internal comparison published; verified isotope validation open.
See the [frozen baseline and evaluation contract](validation/untargeted_stage1/README.md).

Improve the interpretation of untargeted lipidomics features by combining ion
relationships, isotope patterns, lipid chemistry and quantitative evidence.
Compare established methods on identical inputs before choosing a new default.
The current custom grouping and formula ranking remain the reference implementation.

This plan covers the standard tool's untargeted evidence workflow. The historical
33-dataset annotation optimization remains paused. Raw XCMS data, accepted lipid
annotations and manual review decisions must be preserved. A formula candidate,
adduct hypothesis or matching mass is not a confirmed molecular identification.

## Stage tracking and checkpoints

Start a stage when the user requests it or explicitly asks to continue. Complete
that stage, publish its implementation and validation summary, and stop at its
review checkpoint before starting the next stage. Push completed work to `main`,
as requested by the user. Experimental algorithms remain selectable until their
validation supports a default change; merging code does not itself promote it.

| Stage | Deliverable | Depends on | Status |
| --- | --- | --- | --- |
| 1 | Frozen baseline and evaluation contract | None | Complete; checkpoint ready for review |
| 2 | Acquisition settings and adduct chemistry | 1 | Complete; experimental option, current default retained |
| 3 | Ion grouping across the full feature list | 1, 2 | LASER comparison complete; current default retained, Binner pending |
| 4 | Complete isotope patterns and measurement quality | 2, 3 | Software/internal comparison complete; verified isotope validation open |
| 5 | Lipid formula candidates and external engine comparison | 1, 4 | LASER/msbuddy comparisons complete; experimental only, SIRIUS/MZmine activation pending |
| 6 | Joint formula and adduct inference across modes | 3, 4, 5 | Authorized; experimental implementation and blind validation in progress |
| 7 | Optional chromatographic trace evidence | 1, 3 and available traces | Not started |
| 8 | Optional MS/MS evidence and formula comparison | 1, 5 and available spectra | Not started |
| 9 | Final validation, saved projects and default selection | 1–6; available parts of 7–8 | Not started |

Stage 7 may be brought forward after stage 3 if chromatographic traces are
available. Stages 7 and 8 are additional evidence routes. If their inputs are
unavailable, record them as deferred and complete the table-based route without
claiming that those comparisons were performed. No completion dates are assigned
until the stage 1 data inventory establishes the available scope.

## Shared design requirements

- Keep original feature IDs and acquisition modes throughout every transformation.
- Separate ion families of one proposed compound from homologous lipid series.
  A CH2 relationship joins different compounds; it must not merge their identities.
- Evaluate all features for ion relationships. P-values control biological
  reporting and highlighting, not eligibility to provide annotation evidence.
- Use accepted lipids as supporting observations when appropriate, while keeping
  their exclusion from the untargeted biological comparison. Record this evidence
  source. Keep held-out manual reference labels outside runtime inference.
- Preserve several plausible formula/adduct/charge combinations and an unresolved
  outcome. Do not silently convert a score or posterior into an accuracy percentage.
- Keep biological tests separate from annotation quality. QC and blanks are not
  biological comparison groups. Suitable QC chromatograms may supply technical
  peak-shape or isotope evidence; blanks may identify background signals.
- Run expensive calculations in the background worker and persist results.
  Point selection and mode changes must use cached data and update within one second
  in the documented reference project and machine configuration.
- Keep private study files, original-feature crosswalks, subject metadata and
  detailed evaluation rows outside Git. Publish code, schemas, synthetic fixtures,
  protocol descriptions and safe aggregate summaries only.

## Stage 1 Frozen baseline and evaluation contract

**Deliverables**

- Freeze source revision `678ca0c`, dependencies, settings, input fingerprints,
  accepted annotations, statistical results and evidence outputs.
- Inventory the current ALD project: four feature streams, specimen mapping,
  calibration provenance, available isotope information, raw chromatograms,
  MS/MS spectra and reference standards. Do not assume scan-level data exist
  because feature tables contain retention times.
- Create a versioned evaluation protocol and reproducible runner with explicit
  query lists, original-feature mappings, duplicate-reference handling, eligibility
  reasons, and separate development and final evaluation sets.
- Freeze the available dataset and class scope. Additional acquisitions may be
  added to this evidence evaluation only with a recorded scope decision; this
  does not resume the historical optimization collection.
- Reproduce the current 10/14 positive counterpart result on the selected PC
  development subset. Keep that subset for regression checks, not independent
  validation or a general estimate of adduct accuracy.
- Pin external tool versions and inspect their actual defaults, supported charges,
  input formats, platform compatibility and runtime requirements before comparison.

**Evaluation contract**

| Measurement | Denominator and interpretation |
| --- | --- |
| Adduct accuracy among assigned ions | Correct / (correct + incorrect) among assigned, reference-verifiable original ions. Report unknown reference identities and unresolved assignments separately. |
| Assignment coverage | Assigned / all predefined eligible reference queries, including abstentions in the denominator. Show input-ineligible and unresolved reference mappings separately. |
| All-manual reference recovery | Correct original-feature matches / ALL manual entries in the declared dataset/mode/class scope. Include unresolved manual mappings. Label this as recovery or coverage, never conditional prediction accuracy. |
| Isotope recognition | Precision and recall of charge and isotope-member assignments against verified envelopes, with missing and interfered envelopes reported separately. |
| Formula candidate recall | Queries whose correct neutral formula is generated / all benchmark queries. Distinguish an absent candidate from a ranking error. |
| Formula top 1 and top 5 | Correct neutral formula within the stated rank / all benchmark queries, including failed predictions. Report ties and adduct ambiguities explicitly. |
| Joint correctness | Correct neutral formula AND adduct AND charge on the same original feature / all benchmark queries. Matching total ion composition alone is insufficient. |
| Counterpart linking | Correct original feature and adduct in the target mode / all expected reference links. Wrong-peak choices, wrong adducts and missing links remain separate categories. |
| Group quality | Pairwise precision/recall and group purity/completeness, including erroneous merges of RT isomers and splits of true ion families. |
| Unconfirmed extras | Total calls outside the declared reference set, before and after. These are not automatically false positives or an analytical FDR estimate. |
| Confidence and speed | Observed correctness by score range, abstention, worker time/memory and browser update latency. |

For every behavior-changing experiment, follow [AGENTS.md](../AGENTS.md): preserve
the preceding checkpoint and provide complete before/after tables for every
dataset and all four modes, including unchanged, unavailable and untested cells.
Include all-class results and class-specific results for a class-specific change.
Show percentages to one decimal place with exact counts and denominators; retain
unrounded values in the detailed files. Do not average dataset percentages.

**Completion check:** the baseline is reproducible, the selected-PC regression is
reproduced, denominators are unambiguous, and the frozen evaluation scope and
development/test separation are ready for review.

## Stage 2 Acquisition settings and adduct chemistry

**Deliverables**

- Define a project metadata schema for RP and NP method settings: mobile-phase
  additives and concentrations, polarity, instrument/resolution settings and
  calibration selection. Store project-specific values in metadata and saved
  projects, not in source code or machine-specific defaults.
- Provide common, possible and disabled ion settings for each method/polarity.
  Distinguish a chemistry-based preference from an absolute prohibition.
- Preserve the user's exclusions of `[M+2H]2+` and `[M+Na-2H]-`.
  `[M-2H]2-` remains supported where justified.
- Introduce lipid-class adduct preferences with an unknown-class fallback.
  Source these preferences from documented chemistry or independent development
  data, not the manual labels of the project being evaluated.
- Explicitly resolve the formate/acetate ambiguity: their mass difference is
  `14.015650064 Da`, the same as CH2. Do not let an adduct pair manufacture a
  homologous-series edge. If additives are unknown, retain that uncertainty.
- Record the method configuration with every candidate and export.

**Completion check:** configuration round-trips through project save/load;
synthetic adduct-versus-CH2 examples are correctly distinguished; the complete
four-mode evaluation quantifies both recovered and lost assignments.

## Stage 3 Ion grouping across the full feature list

**Deliverables**

- Build candidate isotope/adduct relationships from the complete feature tables,
  including non-significant features. Preserve current accepted-lipid exclusions
  from biological testing and the original feature-level statistical definition.
- Group within each mode before linking opposite polarities in the same
  chromatography. Treat RP-to-NP associations separately; never compare their
  absolute RT values as though they share a chromatographic method.
- Keep strict coelution checks and separate competing RT isomers. Where peak-width
  information is available, test tolerances relative to peak width rather than
  assuming that one fixed window is appropriate for every method.
- Use correctly paired specimens, detection support, mass error and abundance
  profiles. Compare soft evidence scoring with the current hard correlation gates;
  control shared biological group effects without requiring all real adducts to
  have nearly identical sample abundances.
- Keep unmatched features and candidate links as inspectable evidence. Do not
  merge an entire transitive connected component into one compound automatically.
- Benchmark against Binner on compatible aligned feature tables. Document any
  format conversion, unavailable evidence or non-equivalent settings.

**Completion check:** all original features are retained; grouping is independent
of p-value selection; confirmed ion families remain distinct from homologous
series; false merges and missing members are measured against the frozen baseline.

## Stage 4 Complete isotope patterns and measurement quality

**Deliverables**

- Evaluate a maintained isotope calculator such as IsoSpec or a suitable OpenMS
  component instead of implementing another carbon-only approximation.
- Simulate the complete proposed ion composition, including charge, isotope
  contributions of heteroatoms and atoms introduced by the adduct.
- Compare simulated and measured patterns at the acquisition's resolving power.
  Do not assume that isotope fine structure is resolved merely because exact
  masses can be calculated.
- Score isotope mass and intensity errors jointly, with detection limits,
  missing peaks, overlap, saturation and uncertain monoisotopic selection recorded.
- Where possible, use reliable per-sample isotope ratios and suitable QC evidence
  instead of relying only on a ratio of aggregate intensities. Keep labelled
  internal standards separate from natural-abundance formula evaluation.
- Treat carbon count inferred from M+1 as uncertain evidence. A missing or noisy
  M+1 must not automatically prohibit all formula candidates.

**Completion check:** controlled envelopes cover CHNO, phosphorus/sulfur lipids,
carbon-containing and chloride adducts, supported charges, missing isotopes,
interference and saturation. Report recognition precision/recall and formula
candidate recall on real inputs, not only synthetic success rates.

## Stage 5 Lipid formula candidates and engine comparison

**Deliverables**

- Build lipid-aware candidate generation from supported backbones/headgroups,
  chain compositions, ether/plasmalogen variants, oxidation and other supported
  modifications. Include the existing formula library while retaining a broader
  de novo route for formulas absent from it.
- Apply elemental, valence and unsaturation constraints to the appropriate
  neutral or ionic composition. Do not apply unrelated generic filters blindly.
- Preserve structured formula/adduct/charge candidates until the complete scoring
  step. Measure truth lost through element limits, isotope assumptions, library
  restrictions, timeouts or candidate truncation.
- Compare the current engine, SIRIUS, msbuddy and MZmine formula prediction on
  identical queries with matched element sets, mass errors and available evidence.
  Report any native defaults in a separate, explicitly labelled comparison.
- Run MS1-only comparisons separately from MS/MS-supported comparisons.
  msbuddy supports queries without MS/MS but does not assign adducts: supply the
  tested adduct hypothesis explicitly, rather than accepting a hidden default.
- Check MZmine's documented `m/z 800` formula-search cutoff and raise/disable it
  for lipid coverage. Preserve multiply charged queries through a fallback when
  an external engine does not support them; do not silently remove them from
  the reported benchmark denominator.
- Record runtime, failure/timeout reasons and candidate counts. Isolate optional
  external engines from the main GUI environment when needed for compatibility.

**Completion check:** complete formula recall, top-1/top-5 and joint-adduct tables
are available for all four modes and declared classes, with separate library-known
and library-absent cases. Select an engine or combination from this evidence.

## Stage 6 Joint inference and cross mode consistency

Implementation and fixed comparison contract:
[Stage 6 validation](validation/untargeted_stage6/README.md).

**Deliverables**

- Score complete hypotheses containing a neutral formula, ion assignments,
  charge, supporting features and evidence provenance.
- Combine measurement-calibrated mass errors, isotope fit, coelution, specimen
  profiles and chemistry preferences. Account for evidence derived from the same
  measurement rather than treating all component scores as independent.
- Compare the current sequential selection with a component-wise joint selection
  or probabilistic approach that permits competing hypotheses and abstention.
- Require compatible neutral formulas/masses across linked adducts. Retain the
  intact-parent support requirement for water-loss interpretations.
- Model opposite-polarity links within RP and NP independently. Across RP and NP,
  retain one-to-many or unresolved associations when separation/isomer ambiguity
  prevents a unique counterpart. Formula agreement alone is not identity proof.
- Use homologous-grid and method-specific RT evidence as supporting context,
  without turning a homologous family into a single compound or using evaluation
  curation as a runtime prior.
- Expose evidence components, alternatives, score margins and unresolved reasons.
  Calibrate confidence on separate data before displaying accuracy probabilities.

**Completion check:** the complete comparison reports exact counterpart recovery,
wrong original peaks, wrong adducts, unresolved cases, group purity and extras.
Retain changes only when their measured trade-off is justified; do not improve
one positive mode while concealing losses in other modes or classes.

## Stage 7 Optional chromatographic trace evidence

**Deliverables**

- Import or extract chromatographic traces mapped to original XCMS feature IDs.
  Preserve peak boundaries, scan coverage, signal quality and provenance.
- Add within-sample peak-shape and overlap evidence. Distinguish this evidence
  from correlation of integrated intensities across biological samples.
- Benchmark MZmine metaCorrelate plus IIN and CAMERA, and optionally CliqueMS,
  using comparable raw inputs and documented settings.
- Evaluate difficult near-isobars, split peaks, coeluting compounds and alternative
  RP peaks explicitly. Missing or poor traces must produce an uncertainty state,
  not fabricated shape evidence.
- Add IDSL.UFA to the MS1 formula comparison where its raw isotope-profile input
  requirements can be satisfied.

**Completion check:** quantify the incremental gain over the table-only route,
including losses, acquisition requirements and processing cost. If traces are
unavailable, mark this stage deferred and retain the table-based results.

## Stage 8 Optional MS/MS evidence

**Deliverables**

- Map available spectra to original features and record precursor isolation,
  polarity, collision conditions, purity and potential chimeric spectra.
- Compare SIRIUS, BUDDY/msbuddy and MZmine using the same eligible spectra.
  Include lipid-class fragmentation evidence where justified.
- Test MIST-CF on its supported data domain rather than extrapolating its published
  positive-mode results to all four modes. Keep recent research such as FACET in
  an experimental comparison until reproducible performance supports adoption.
- Evaluate network formula refinement such as ZODIAC only where the required
  fragmentation evidence is available. It is not a substitute for missing MS/MS.
- Keep neutral-formula, adduct, lipid-class and structural-identification claims
  separate. MS1-only candidates must not inherit structural certainty from a name.

**Completion check:** the comparison reports gains over MS1-only inference,
unsupported cases and confidence calibration. If spectra are unavailable, mark
this stage deferred; do not fabricate a performance comparison.

## Stage 9 Validation and staged release

**Deliverables**

- Run the frozen evaluation on every dataset and all four modes in the agreed
  scope, including independent acquisitions and standards when available. Keep
  related acquisitions together in development/test splits.
- Publish all-class and class-specific before/after counts and denominators,
  group errors, total unconfirmed extras, failed predictions and runtime. A raw
  mass match or a different original peak must not count as an exact recovery.
- Select the default only after reviewing the full trade-off. Use no net increase
  in unconfirmed extras as the initial guardrail; report any proposed exception
  alongside its repeatable gain before adopting it.
- Version candidate tables, inference settings and compound/cluster assignments.
  Preserve the prior saved project and provide an ID crosswalk when regrouping
  changes display compound numbers. No silent irreversible migration.
- Save structured candidates, selected hypotheses, evidence scores, failure
  reasons, external-engine versions and input fingerprints with the project.
  Keep a table-only path when optional raw or MS/MS evidence is unavailable.
- Keep computation in the worker with progress, terminal output, cancellation
  and resumable/reopenable results. Verify Windows and macOS paths actually used
  by the application, including dependency failures and old-project loading.
- Verify marker meanings, formula uncertainty, linked-mode bars and exact feature
  selection. Measure representative selections and mode changes below one second;
  document the machine, project size and observed latency distribution.
- Run focused scientific regressions and the required application checks, then
  update documentation, source revision and installed package. Preserve rollback.

**Completion check:** a reviewed release record identifies the chosen method,
tested scope, limitations, before/after tables, saved-project compatibility and
measured UI performance. Optional evidence routes are clearly enabled or deferred.

## Research references

Documentation was consulted on 5 November 2026. Pin executable versions and
recheck applicable defaults at stage 1; current documentation is not a frozen
benchmark specification. Published benchmark results are not LASER performance.

- [MZmine correlation grouping](https://mzmine.github.io/mzmine_documentation/latest/module_docs/id_ion_networking/metacorr/metacorr.html)
  and [Ion Identity Networking](https://mzmine.github.io/mzmine_documentation/latest/module_docs/id_ion_networking/iin/iin.html).
- [MZmine ion-type ranking](https://mzmine.github.io/mzmine_documentation/latest/module_docs/filter_featurelistpreferences/feature_list_preferences.html),
  [formula prediction](https://mzmine.github.io/mzmine_documentation/latest/module_docs/id_spectra_chem_formula/chem-formula-pred.html)
  and [lipid annotation](https://mzmine.github.io/mzmine_documentation/latest/module_docs/id_lipid_annotation/lipid-annotation.html).
- [SIRIUS methods](https://v6.docs.sirius-ms.io/methods-background/)
  and [adduct handling and charge support](https://v6.docs.sirius-ms.io/adducts/).
- [msbuddy capabilities](https://msbuddy.readthedocs.io/en/latest/)
  and [input adduct requirement](https://msbuddy.readthedocs.io/en/latest/quickstart.html).
- [BUDDY formula annotation](https://doi.org/10.1038/s41592-023-01850-x).
- [Binner feature-table annotation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7828469/).
- [CAMERA](https://pmc.ncbi.nlm.nih.gov/articles/PMC3658281/)
  and [CliqueMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC6792096/).
- [IsoSpec2 isotope calculations](https://doi.org/10.1021/acs.analchem.0c00959).
- [Seven Golden Rules](https://pmc.ncbi.nlm.nih.gov/articles/PMC1851972/)
  and [IDSL.UFA MS1 formula annotation](https://pubmed.ncbi.nlm.nih.gov/36137231/).
- [MetAssign joint inference](https://doi.org/10.1093/bioinformatics/btu370).
- [MIST-CF](https://doi.org/10.1021/acs.jcim.3c01082),
  [ZODIAC](https://doi.org/10.1038/s42256-020-00234-6)
  and [FACET preprint](https://arxiv.org/abs/2611.01352).
- [Lipidomics annotation quality requirements](https://pmc.ncbi.nlm.nih.gov/articles/PMC8346590/).

## Stage completion record

Add a row after each completed stage. Record rejected experiments as well as
retained changes; detailed study reports stay private. Use anonymous report
identifiers rather than private filesystem paths or subject identifiers.

| Stage | Date | Source revision | Decision | Public summary and report identifier | Next stage |
| --- | --- | --- | --- | --- | --- |
| Plan | 5 November 2026 | See the commit adding this document | Plan published; no implementation stage started | This document | 1 |
