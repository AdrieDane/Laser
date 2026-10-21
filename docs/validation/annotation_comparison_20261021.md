# Annotation comparison findings

## Scope and baseline

The evaluation-only comparison covers the saved four-stream outputs for all
15 example folders. The duplicate fish folder remains in the detailed report
but is explicitly excluded from pooled results, leaving 14 unique datasets.
No annotation rules, models, feature choices or TXT exports changed.

The annotations are from checkpoint `8857a1e`, including its independent final
standard safeguard. The candidate diagnostics come from the saved full core
runs and PE decision exports. The annotations CSV is authoritative; earlier
diagnostic tables cannot replace its choices. The comparison tool is from
`05bb11d`, which also includes the user's CLI export fix `582d13b`.

The same original feature, lipid name, adduct, stream and standard identity
must agree. There are 45,712 reliably mapped manual entries, of which 34,284
match LASER on the same peak: 75.0%. A further 543 manual entries have unresolved
original-peak mappings. Counting those as misses gives 74.1% of all 46,255
manual rows. Neither measure is a prospective analytical accuracy estimate.

| Stream | Same manual peak | Mapped manual entries | Same peak percentage |
| --- | ---: | ---: | ---: |
| RP positive | 15,831 | 21,885 | 72.3% |
| RP negative | 5,481 | 7,734 | 70.9% |
| NP positive | 7,537 | 9,244 | 81.5% |
| NP negative | 5,435 | 6,849 | 79.4% |

The machine-readable stream summary is
[annotation_comparison_20261021.csv](annotation_comparison_20261021.csv).

## What the remaining differences show

Of the 11,428 mapped manual entries not on the same peak:

- 10,810 have neither the requested lipid selected elsewhere nor another label
  on the manually chosen peak.
- 524 have the requested lipid selected on another peak only.
- 81 have another lipid selected on the manual peak only.
- 13 have both conflicts at once.

The exact manual lipid and peak already appear in at least one available
candidate table for 9,332 of these 11,428 differences, or 81.7%. The remaining
2,096 are absent from the supplied tables. Absence does not establish that a
candidate was never considered earlier, and presence does not prove that its
evidence is sufficient for acceptance.

## PE priorities

All 216 remaining mapped ordinary PE disagreements appear in the saved
candidate tables. Their selection outcomes are:

| Stream | PE not selected | Same PE on another peak | Total |
| --- | ---: | ---: | ---: |
| RP positive | 44 | 33 | 77 |
| RP negative | 13 | 31 | 44 |
| NP positive | 10 | 6 | 16 |
| NP negative | 72 | 7 | 79 |

This supports prioritizing independent evidence, peak ranking and existing
rejection checks rather than assuming PE requires a broader discovery window.
The current additive PE method protects earlier automatic choices as well as
their occupied features. A later joint resolver must distinguish such
automatic choices from manual confirmations and protected standards.

Recorded earlier checks include post-regression ineligibility and grid
coherence failures, but these observations do not establish the causal reason
for the final result. Actual stage events need to be recorded in future runs.

## Other class priorities

| Class | Manual entries not on the same peak | Already in available candidate tables |
| --- | ---: | ---: |
| PC | 1,098 | 933 |
| TG | 884 | 804 |
| DG | 510 | 393 |
| PS | 424 | 404 |
| Ether TG | 365 | 293 |
| 1 acyl LPC | 312 | 123 |

PC is a suitable next pilot alongside PE. TG/DG and PS have substantial
numbers of represented but unselected manual pairs. LPC also needs attention
to candidate availability and positional handling, not just its final score.

## Verification and next checkpoint

The comparison reproduces the preceding all-class selected-feature counts
without changing annotations. The software suite passes 334 tests and 83
subtests, including the user's export regression test and 34 new comparison
tests. The wheel contains the validation module and current CLI export fix.

The detailed report and per-entry evidence stay in the local workspace under
`outputs/annotation_disagreements_20261021_verified/`. Its manifest fingerprints
the annotation CSVs, manual crosswalks, candidate artifacts, evaluation source
and generated outputs. Large data artifacts are not committed to Git.

Next: instrument actual rejection events and add independent isotope/adduct
and sample-consistency evidence. Evaluate their effect on PE/PC choices
against this frozen baseline before changing default selection behaviour.
