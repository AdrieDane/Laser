# Experimental PE correction results

Later checkpoint: [strict RP-positive consensus](rp_pe_consensus_20261021.md)
adds three more correct manual-peak choices without increasing extras. Counts
below describe the initial experiment, not that follow-up.

21 October 2026. With the optional correction enabled, LASER selects the same
lipid, adduct and original peak as manual curation for 971 of 1,000 NP-negative
PE entries, up from 910. Every one of the 14 unique datasets now exceeds 90%
in NP negative. The remaining below-90% cases are all RP positive.

This is ordinary diacyl PE only, not PE-O or PE-P. The comparison uses the
preceding `c3dd7b3` code baseline and its `8857a1e` annotation checkpoint.
Results are from development examples, not an independent accuracy estimate.
The new option remains off by default.

## Same annotation on the same manually chosen peak

The denominator includes every manual PE entry, including entries whose
original feature could not be mapped reliably. The correct name on another
peak is not counted as a match. The duplicated fish folder is present in the
detailed results but excluded from the pooled totals.

| Measurement stream | Before | With correction | All manual PE entries | Before percent | With correction percent |
| --- | ---: | ---: | ---: | ---: | ---: |
| RP positive | 980 | 981 | 1,071 | 91.5% | 91.6% |
| RP negative | 1,213 | 1,215 | 1,267 | 95.7% | 95.9% |
| NP positive | 919 | 919 | 945 | 97.2% | 97.2% |
| NP negative | 910 | 971 | 1,000 | 91.0% | 97.1% |

No individual dataset and stream lost a correct PE match. Nevertheless, the
90% target is not achieved everywhere:

| Dataset | Remaining weak stream | Same manual peak | Percent |
| --- | --- | ---: | ---: |
| EDT brain | RP positive | 55 / 71 | 77.5% |
| Serhii brain | RP positive | 58 / 72 | 80.6% |
| Fish larvae | RP positive | 89 / 101 | 88.1% |
| Lipid lengthening spinal cord | RP positive | 55 / 62 | 88.7% |

The detailed [dataset results](experimental_pe_by_dataset_20261021.csv) and
[stream summary](experimental_pe_summary_20261021.csv) retain counts and clear
before/after labels. These compare the option on versus off, not two separate
new raw-data acquisitions.

## Added annotations and changed peaks

Across 14 unique datasets, NP negative gains 65 annotations: 61 match the
manual identity and original peak; four identities are absent from that
stream's manual list. Three existing automatic PE choices move to the
manually selected peak: one RP positive and two RP negative. No existing
identity is removed. NP positive, all non-PE choices and standards are
unchanged relative to the immediately preceding baseline.

Across all classes, annotation count rises from 38,822 to 38,887, an increase
of 0.17%. Same-peak matches rise from 34,284 to 34,348 of 45,712 reliably
mapped manual entries: 75.0% to 75.1%. The all-class denominator differs from
the all-manual PE denominator above; this first experiment does not solve the
large overall annotation gap.

The four additional NP-negative identities need curator review. All have
opposite-mode support but lack the upper-isotope support used by the admission
rule. Absence from the manual list does not establish that they are false.

| Dataset | Additional identity | Original feature ID |
| --- | --- | --- |
| Kidney summer | PE(43:4) | 32603 |
| Heart | PE(31:5) | 37131 |
| Lipid lengthening spinal cord | PE(45:5) | 44705 |
| ALD brain | PE(44:4) | 52934 |

Total PE identities absent from the corresponding stream's manual list change
from 482 to 486. [Every changed choice](experimental_pe_changes_20261021.csv)
includes its original feature IDs and supporting evidence.

## Validation scope

The final production selector was replayed on saved completed core outputs
for all 15 folders, representing 14 unique datasets and all four streams.
Candidate discovery and sample/isotope evidence were recomputed from saved
original feature tables. Manual comparison was performed after selection.
All newly accepted choices passed the actual grid-display check on their
selected original feature; NP-positive tables were exactly unchanged, and
non-PE/standard selections were unchanged in every folder.

Additional fresh full RP/NP workflow runs passed for heart, EDT brain and rat
brain with the experiment enabled. Their final choices in all four streams
matched the replay. A fresh default-mode plasma run matched the preceding
baseline. The fish and Serhii source folders are no longer present at the
supplied location; their replay uses retained original-feature data and prior
full outputs. These are not 15 new full raw-input runs.

Four development cohorts fit the new models: rat brain, rat plasma, rat spinal
cord and Pex5 mouse. Selector-only leave-one-cohort-out checks give RP positive
226 to 226 same-peak matches, RP negative 280 to 281, and NP negative 187 to
195, with no increase in unlisted identities in those folds. Earlier pipeline
models were not refitted, thresholds were not chosen through nested folds,
and all examples informed rule development. In particular, an initial
RP-positive rule improved heart but harmed worms; requiring both mass and
RT improvement removed that regression. Untouched external data are still
needed before claiming general performance.

Software verification passes 392 tests and 83 subtests, changed-file lint and
the whitespace check. The built wheel matches all 156 corresponding packaged
source/data files byte for byte, including the three new models. The option-off
and CLI export regression tests pass. A push does not install or restart the
user's app. No previous TXT export is regenerated by this checkpoint.
All 60 previous TXT exports retain their hashes; all 48 original peak lists
still available at their recorded paths also retain their hashes. The other
12 source lists are in the absent fish, duplicate-fish and Serhii folders.

## Next improvement

Prioritize the four weak RP-positive cohorts. Separate wrong-peak choices
from genuinely absent annotations, then review their independent evidence
and a small set of disputed or additional calls with a curator. Test better
RT uncertainty and peak-quality evidence before allowing more replacements
or new RP-positive identities. Keep the current no-regression and controlled-
addition checks; do not raise coverage simply by accepting every candidate.

Local detailed artifacts are under
`outputs/experimental_pe_selector_20261021/production_replay_final/` in the
task workspace, with fresh controls in `fresh_controls/`. The compact CSVs
and [method and usage guide](../EXPERIMENTAL_PE_SELECTION.md) are versioned;
large feature tables and candidate caches remain local.
