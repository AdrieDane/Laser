# Annotation improvement plan

Agreed 21 October 2026. Improve agreement with manual curation by choosing
both the correct lipid annotation and the same original XCMS feature. Build a
shared annotation and peak-selection engine, starting with PE and then the
other lipid classes. Additional strategies and tools are allowed when they
improve measured performance without an uncontrolled increase in annotations.

## Starting point

The current four-stream validation contains 15 folders and 14 unique datasets.
The duplicate fish folder must not inflate pooled results. Across all classes,
34,284 of 45,712 reliably mapped manual entries have the same lipid, adduct,
stream and original feature in LASER: 75.0%. Ordinary PE has better pooled
agreement, but 10 of 56 unique dataset and stream combinations are below 90%.

| Stream | Ordinary PE on the same manual peak | All manual PE entries | Percentage |
| --- | ---: | ---: | ---: |
| RP positive | 980 | 1,071 | 91.5% |
| RP negative | 1,213 | 1,267 | 95.7% |
| NP positive | 919 | 945 | 97.2% |
| NP negative | 910 | 1,000 | 91.0% |

The PE denominator includes unresolved manual-feature mappings as misses.
The all-class 75.0% denominator includes only reliably mapped manual entries.
These measures must stay separately labelled. The examples have informed
development, so their results are not prospective external accuracy estimates.

Reference checkpoints: [PE summary](validation/pe_multistream_summary.csv),
[PE by dataset](validation/pe_multistream_by_dataset.csv) and
[previous handoff](PE_HANDOFF.txt). The 21 October export fix in `582d13b`
is part of the new implementation baseline.

## Phase 1 Explain each disagreement

Create a reusable, evaluation-only comparison tool and automated tests. For
every manual entry, record whether LASER selected the same peak, selected a
different peak for that identity, selected another identity on the manual peak,
selected neither, or could not reliably map the manual entry to a source peak.
Report both kinds of conflict when they occur together.

Separately record the available candidate evidence. Distinguish the final
candidate table, the PE discovery table, and missing diagnostic information.
Absence from a final table is not proof that the peak was never considered.
Record rejected gates only when explicit recorded evidence supports them;
several gates may fail together. Missing values are unknown, not passes.
Distinguish missing local support from a low PE score.

Deliver per-entry disagreements, additional LASER calls, summaries by dataset,
stream and class, and pooled unique-dataset summaries. Reconcile totals,
preserve feature IDs and standards, and fingerprint input artifacts. Keep
manual labels and benchmark code outside the annotation runtime.

First evaluate the saved full outputs from all 15 folders. No new annotation
selection rules are needed for this checkpoint. Use the results to prioritize
candidate discovery versus peak selection, rather than optimizing a headline
percentage without understanding the cause.

## Phase 2 Add independent peak evidence

Build reusable evidence extraction for detection across samples, robust
abundance, pooled-QC consistency and blank contrast where metadata permits.
Do not let one unusually intense sample dominate every peak choice. Resolve
sample roles from explicit metadata and retain an unknown state when absent.

Group isotope peaks, charge states and chemically plausible adducts using
mass differences, coelution and sample profiles. Correlation alone must not
merge different lipids that change together biologically. Validate isotope
spacing, expected envelopes and charge rather than marking every regenerated
candidate as independently established monoisotopic evidence.

Where usable EICs are supplied, add actual chromatographic shape and integration
evidence. XCMS group RT bounds are not chromatographic peak shapes. EICs and
additional evidence are optional: missing companions must not make otherwise
supported peak-list-only runs fail.

Feature grouping precedent: [CAMERA](https://pmc.ncbi.nlm.nih.gov/articles/PMC3658281/).

## Phase 3 Improve independent RT and mass calibration

Require several independent clues for high-confidence anchors. Fit robust,
class-family and stream-specific RT patterns with uncertainty bounds. A target
composition must not support its own calibration. Allow local corrections
only where observations support them; avoid unrestricted extrapolation.

Keep RP and NP RT models separate. Model the RP ether/plasmalogen families
and LPC/LPE positional ordering using reliable anchors and observed method
behaviour. NP combined identities must not acquire unsupported RP positional
labels. Learn mass-offset variation with m/z or RT only if enough independent
anchors and held-out checks justify the extra flexibility.

Apply configurable carbon and unsaturation bounds and chemical plausibility.
Use unusual composition as evidence, not an unconditional ban on real odd
carbon or rare lipids. Never create a peak merely to fill a grid position.

RT guidance: [Lipidomics Standards Initiative](https://lipidomicstandards.org/separation-ms/).

## Phase 4 Resolve competing annotations together

Separate two questions: which peak best represents a lipid, and whether the
evidence is sufficient to annotate that lipid at all. Compare competing peaks
within an identity instead of relying only on an absolute candidate score.

Introduce a joint resolver for compatible peak and identity choices. It may
correct an earlier automatic call when stronger evidence supports another
choice. Manual confirmations and declared standards remain protected. Keep
alternatives visible and record every replacement with its evidence.

Constrain duplicated primary choices within the appropriate stream, identity
and adduct, while allowing explicitly represented positional isomers and
related adducts. Keep a no-annotation option. Cross-stream support requires
validated sample correspondence and independent phase-specific RT alignment;
absence in a less sensitive stream is not automatic evidence against a lipid.

Trial the resolver on PE and PC before expanding to TG/DG, other
phospholipids, ether/plasmalogen families and sphingolipids. Prioritize classes
by missed manual peaks, error pattern and candidate availability.

## Phase 5 Improve training labels and validation

Train with reliable same-feature matches and competing peak examples. Do not
label every identity absent from the manual list as a proven false lipid.
Select a small, diverse set of extra and disputed annotations for a one-time
curator review. Distinguish a wrong identity from a real but less suitable
quantification peak or an additional genuine isomer.

For development checks, hold out entire datasets or related acquisition
batches. Refit all learned pipeline stages within each fold, not just the
last PE scorer. Choose thresholds within development folds, never from the
held-out answers. Curated annotations are for training and evaluation, not
runtime answer lookups for the same validation dataset.

Obtain genuinely untouched datasets for a final external assessment. Report
uncertainty and sparse-class counts; an excellent pooled result must not hide
a weak dataset, stream or lipid family.

## Acceptance checks

For every candidate version, report the following by dataset, stream and class:

- Manual entries with the same lipid, adduct and original peak.
- The same manual lipid on another peak.
- Another lipid identity on the manually chosen peak.
- Manual entries still missing and entries whose original peak is unresolved.
- Additional identities absent from the manual list, separately from proven errors.
- Total calls, removed and added calls, and changes outside the targeted class.
- Candidate availability and explicit decision evidence, without invented causes.

Initial goal: make ordinary PE consistently reach at least 90% on the same
manual peak in each sufficiently populated dataset and stream, while reducing
wrong-peak choices and unconfirmed additions. Next aim for at least 90%
all-class same-peak agreement separately in each measurement stream. These
are acceptance targets, not promised outcomes or reasons to force a count.

Accept improvements based on more correct peaks with controlled errors and
additions. Reject a coverage gain obtained only by indiscriminately loosening
thresholds. Test one new evidence component at a time before combining them.
Always compare against the frozen checkpoint and the immediately preceding
accepted version. A failed gate must leave the preceding defaults available.

Preserve raw mass and RT values, original feature provenance and saved-project
compatibility. Do not claim more structural detail than the evidence supports.
Keep the existing TXT contract: insert only `laser_annotation` as column 2
and `adduct` as column 3, preserve all other content and order, and use the
`_annotated.txt` suffix. Never overwrite original input files.

## Implementation checkpoints

1. Push this plan with the export fix already in the baseline.
2. Ship and test the evaluation-only disagreement analysis on all 15 folders.
3. Add independent peak evidence behind an experimental option.
4. Evaluate PE/PC calibration and joint selection against the frozen baseline.
5. Expand only improvements that pass the class, stream and dataset checks.
6. Validate on untouched data before describing general annotation accuracy.
