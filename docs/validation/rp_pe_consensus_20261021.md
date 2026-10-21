# RP positive PE consensus results

21 October 2026. The new strict consensus rule adds three RP-positive PE
annotations on the same original peak as manual curation, with no additional
unlisted identities, wrong-peak choices or removals. This is a small improvement,
not the large RP-positive gain still needed. The experiment remains off by
default and does not install or restart the user's application.

## Correct matches and extra annotations

These are sums across 14 unique datasets and four measurement streams. The
duplicate fish folder is excluded from pooled counts but retained in the
detailed CSV. A match requires the same lipid, adduct and original feature.
Manual entries with unresolved original-feature mappings count as misses.
The before column is the preceding `a06782f` experiment enabled, not the
option-off default method.

| Stream | Same manual peak before | Same manual peak now | All manual PE entries | Same peak percent now | Extra identities now | Different peak now |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RP positive | 981 | 984 | 1,071 | 91.9% | 184 | 32 |
| RP negative | 1,215 | 1,215 | 1,267 | 95.9% | 144 | 29 |
| NP positive | 919 | 919 | 945 | 97.2% | 126 | 6 |
| NP negative | 971 | 971 | 1,000 | 97.1% | 32 | 7 |
| Total | 4,086 | 4,089 | 4,283 | 95.5% | 486 | 74 |

An extra identity is a LASER call absent from that dataset and stream's manual
list; it is not a proven false lipid. Different peak means that the manual
identity is present but LASER selects another original feature. These counts
are separate from correct matches and remain unchanged in this step. The
same lipid in different datasets or streams contributes separate entries.
Another 43 PE calls match a manual identity whose original feature could not
be mapped reliably. Those cannot be judged as the same or a different peak;
they are not included in the 486 extras or 74 different-peak calls.

Across all lipid classes, there remain 3,652 unlisted calls and 534 calls on a
different peak from the curator's choice. Correct same-peak matches rise from
34,348 to 34,351 of 45,712 reliably mapped manual entries, still 75.1%. Total
calls increase from 38,887 to 38,890. No previous call is removed; standards,
other lipid classes and the other three measurement streams are unchanged.

| Dataset | Newly matched PE | Original RP-positive feature |
| --- | --- | --- |
| Serhii brain | PE(38:1) | 102087 |
| V2 | PE(38:5) | 52759 |
| Cells | PE(40:5) | 67159 |

The [stream summary](rp_pe_consensus_summary_20261021.csv),
[all dataset results](rp_pe_consensus_by_dataset_20261021.csv) and
[new choices with their supporting feature IDs](rp_pe_consensus_additions_20261021.csv)
provide the underlying counts and provenance.

## Accepted rule

A new RP-positive identity requires strong upper-isotope evidence, tight
corrected mass and local RT errors, supported sample intensities and measured
blank contrast. The same PE must already be selected in both NP streams and
have an accepted RP-negative peak within five seconds. Confirmations are
snapshotted before the experiment; they cannot be created by its own new calls.
The preferred peak must itself pass admission, without falling back to a
lower-ranked alternative. See [the method guide](../EXPERIMENTAL_PE_SELECTION.md)
for every threshold and limitation.

No new model weights are shipped. The existing four-cohort pair models remain
unchanged. The new rule was developed using these examples, so its all-dataset
comparison is not untouched external validation.

## Rejected experiments

Simply admitting RP-positive peaks with good mass, local RT, rank and isotope
evidence would add 43 calls: seven on the manual peak, 35 absent from the manual
list and one on another peak. That rule was rejected. This shows why a high
candidate score or plausible isotope envelope is not sufficient admission.

A second experiment added XCMS grouping extent, original detection counts and
signed mass/RT errors to a ranker trained with more example cohorts. In
conditional leave-one-cohort-out checks across 14 cohorts, the existing strict
replacement guard permitted no changes. A less restrictive guard at preference
0.90 produced three correct-peak gains but lost one previously correct choice.
It was also rejected. Earlier pipeline models were not refitted in those folds;
this is not fully independent validation. Neither experiment changes runtime.

## Remaining RP positive gap

| Dataset | Same manual peak now | Percent |
| --- | ---: | ---: |
| EDT brain | 55 / 71 | 77.5% |
| Serhii brain | 59 / 72 | 81.9% |
| Fish larvae | 89 / 101 | 88.1% |
| Lipid lengthening spinal cord | 55 / 62 | 88.7% |

Some missed manual peaks fall outside the current corrected mass or local RT
limits. Other disagreements involve very close competing original features.
Neither issue is resolved by confirming that the lipid exists in another mode.
The next substantial PE experiment should test target-independent, local
mass/RT calibration with uncertainty, and inspect these competing features.
Actual EIC/integration evidence would help when supplied. Do not widen gates
or relabel every candidate solely to reach the numerical target.

## Verification scope

The final production selector was replayed on all 15 retained full core outputs,
representing 14 unique datasets, with sample/isotope evidence recomputed from
saved original feature tables. Comparison to manual curation followed runtime
selection. Every preceding selected annotation is retained. Newly accepted
choices are full annotations, pass the actual original-feature grid-display
check and retain historical mass, RT and grid scores alongside the new gates.

This is not 15 new raw-input runs. Fish and Serhii source folders remain absent
at the supplied location; their replay uses the original-feature tables retained
from the earlier full runs. New fresh full-workflow controls cover available
datasets. Software tests cover every new admission gate, missing and non-finite
evidence, stricter user limits, protected choices and no alternate-peak fallback.
Fresh full-workflow controls for V2 and cells matched the final replay in all
four streams. A further cells control fingerprinted the final runtime sources
before and after its full run and also passed. The suite passes 426 tests and
83 subtests; changed-file lint and whitespace checks pass. The built wheel
matches all 157 corresponding source/data files byte for byte, including the
new module and unchanged model files. Saved-project round trips retain the
new gates, accepted flags and NP original-feature provenance.

The [provenance manifest](rp_pe_consensus_provenance_20261021.json) fingerprints
the runtime, models, cached core outputs, preceding annotations, manual
crosswalks and new annotation exports. All 48 currently available originals
and all 60 previous TXT exports retain their hashes. Twelve originals remain
unavailable in the absent fish, duplicate-fish and Serhii folders.

Detailed artifacts remain local under
`outputs/rp_pe_consensus_20261021/production_replay_final/` and `fresh_controls/`.
Rejected ranking artifacts are under `outputs/rp_pe_next_20261021/`, and their
labels are evaluation-only, never runtime inputs. Previous TXT exports are not
regenerated by this checkpoint.
