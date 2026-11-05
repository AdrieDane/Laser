# Stage 3: full-feature ion grouping

5 November 2026. Implementation and frozen LASER comparison complete; current default retained.
Native Binner comparison is unavailable pending a trusted executable. Stage 3 is
not fully closed and Stage 4 has not started. Current grouping remains the default.

## Implementation

The untargeted launcher's **Feature-evidence search settings → Ion grouping**
offers current grouping, experimental full-feature strict profiles, and experimental
full-feature soft evidence. Acquisition/adduct settings remain a separate control.

The full-feature engine builds measured ion relationships before consulting any
statistical result. It uses every original stream row, including accepted lipids
and non-significant features. It retains invalid-coordinate rows and unmatched
features with an explicit status in `ion_feature_ledger`. Duplicate original
keys fail instead of being silently dropped. Accepted ions remain excluded from
biological testing; QC/blanks are excluded from biological profile evidence.

The search proceeds through within-mode adduct relations, opposite polarities
within one chromatography, and finally RP-to-NP relations. RP and NP retention
times are never directly compared. Mass search is vectorized/chunked. Within a
chromatography, RT windows are capped by the configured tolerance and, where
valid `rtmin`/`rtmax` intervals exist, by half the sum of the pair's peak widths
(floor 0.005 min). Unknown/invalid widths use the configured cap. These grouped
feature widths are not equivalent to scan-level chromatographic shape evidence.

Pairing uses unique specimen metadata, then conservative acquisition-name
fallbacks. All eligible biological groups are used, independently of the selected
contrast. Log-area correlations are calculated on co-detected positive values;
missing/zero values are not median-imputed. Within-group centring controls shared
group-wide effects. At least six co-detected paired biological specimens are needed.

Two fixed evidence rules are compared:

- Strict: Spearman rho ≥0.6 and centred Pearson correlation ≥0.2.
- Soft: rho ≥0.2, centred correlation ≥0.05, and combined score ≥0.45.
  Score weights are 0.30 positive rho, 0.40 positive centred correlation, 0.15 ppm
  compatibility and 0.15 RT compatibility. RP-to-NP omits RT and renormalizes the
  remaining weights; it receives no RT bonus. These are heuristic evidence scores,
  not calibrated probabilities.

Family assembly processes within-mode edges first. Every merge must preserve
the full neutral-mass span, complete within-phase RT constraints, and one original
monoisotopic ion per mode/adduct. Thus a chain of pairwise-compatible mass/RT edges
cannot automatically merge an entire transitive component. Equal-mass RT isomers
cannot both occupy the same mode/adduct slot. Alternative hypotheses and rejection
reasons remain in `ion_relationships`. Water-loss assignments require an intact ion
in the same mode. The user's two ion exclusions and method-specific disables apply.
Stage 2 chemistry penalties, when selected, remain soft preferences; the Stage 3
comparison holds acquisition scoring at legacy to isolate the grouping change.

Edge ranking subtracts 0.01 per catalog priority position and 0.05 per chemistry
penalty. At each opposite-polarity/cross-chromatography stage, existing coeluting
family support adds 0.025 per additional established member (capped at three).
This favours supported families over isolated equal-mass RT isomers without
using p-values or manual evaluation identities.

Full-list M+1/M+2 mass/profile candidates are exported in
`ion_isotope_candidates`. Conditional attachment to an already-supported parent
does not merge two existing ion families and is explicitly labelled as unverified
envelope evidence. Complete heteroatom/isotope-envelope validation is Stage 4.
Reference adduct evaluation excludes conditional isotope members from monoisotopic
matches. No formula engine or biological test has been replaced.

Compound IDs are assigned before significance filtering, including unmatched
singletons. Threshold changes cannot change this complete ledger. Only afterwards
are significant families selected for browser payloads. The full graph is retained
on disk, while the browser uses saved family memberships and compact family links;
it does not repeat a full search when a point is clicked. Full-feature mode disables
the old display-only mass search that could invent unsupported neighbours.

All three evidence tables, engine/settings version and a fingerprint of source
coordinates, levels and pairing metadata persist in saved projects/background jobs.
The hard-to-soft evaluation cache must match these fingerprints and all physical
settings. It reuses only mass/profile candidates, never reference labels or selected
groups. Saved empty-hit projects still retain their complete feature ledger.

## Frozen evaluation

`tools/evaluate_full_feature_groups.py` freezes committed source and protocol before
running three fixed arms against the Stage 1 checkpoint: current grouping, full
strict profiles and full soft evidence. Current assignments must match the complete
Stage 1 table. Statistics and existing formula/isotope outputs are checked exactly;
full-list isotope candidates are additional, separately labelled evidence.

Each full arm must retain all 254,324 original features. Evaluation covers every
manual row, all 63 reference class/subclass labels and all four modes, including
the 46 unresolved original mappings. It reports assigned adduct agreement, reference
recovery/coverage, gains/losses, all 60 selected-PC mode slots, and reference-identity
pair agreement/recovery. Unknown extras do not become false positives or FDR.

For continuity, the historical extra-ion column counts displayed detections in
significant families. The complete unmatched feature ledger is not a set of new
annotation calls. Full-ledger assignment counts are reported separately. Formula
ranking, verified isotope accuracy and molecular-identification accuracy are not
claimed by this grouping comparison. This is development data, not independent
validation. No per-class winner splice, threshold tuning or default promotion is
planned after inspecting results.

```sh
python3 tools/evaluate_full_feature_groups.py \
  --checkpoint /private/stage-1-checkpoint \
  --method /private/project/metadata/acquisition_method.json \
  --output /private/new-stage-3-attempt --node /path/to/node
```

## Completed native comparison and decision

Producer `5625abe`; all three fixed arms completed against the same frozen
acquisition. **Keep current grouping as the default.** Exact selected-PC positive
counterparts regress from **10/14 to 9/14** in both full-feature engines. Wider
reference eligibility increases recovery, while conditional adduct agreement is
weaker in several modes. Each full engine loses 15 previous original-ion/adduct
reference matches (RP positive 2, RP negative 2, NP positive 9, NP negative 2).
No class/mode splice, weight tuning or promotion followed evaluation.

All-manual original-ion/adduct recovery (3,668 rows, including 46 unresolved
mappings; not molecular-identification accuracy):

| Mode | Current | Full strict | Full soft |
| --- | --- | --- | --- |
| RP_Pos | 2.7% (46/1694) | 48.5% (822/1694) | 55.4% (939/1694) |
| RP_Neg | 0.4% (3/679) | 39.0% (265/679) | 44.3% (301/679) |
| NP_Pos | 4.9% (37/757) | 62.0% (469/757) | 65.3% (494/757) |
| NP_Neg | 0.9% (5/538) | 62.1% (334/538) | 66.4% (357/538) |

The complete ledger is eligible for assignment in the full engines, whereas the
current engine's evaluated output contains significance-selected families. This
scope expansion contributes to recovery gains; the table is not an accuracy gain
on identical assigned subsets.

Conditional assigned-adduct agreement:

| Mode | Current | Full strict | Full soft |
| --- | --- | --- | --- |
| RP_Pos | 90.2% (46/51) | 78.5% (822/1047) | 74.2% (939/1265) |
| RP_Neg | 100.0% (3/3) | 56.9% (265/466) | 54.7% (301/550) |
| NP_Pos | 75.5% (37/49) | 74.1% (469/633) | 71.4% (494/692) |
| NP_Neg | 83.3% (5/6) | 71.4% (334/468) | 70.4% (357/507) |

Assignment coverage among eligible reference ions:

| Mode | Current | Full strict | Full soft |
| --- | --- | --- | --- |
| RP_Pos | 3.0% (51/1680) | 62.3% (1047/1680) | 75.3% (1265/1680) |
| RP_Neg | 0.4% (3/671) | 69.4% (466/671) | 82.0% (550/671) |
| NP_Pos | 6.6% (49/745) | 85.0% (633/745) | 92.9% (692/745) |
| NP_Neg | 1.1% (6/526) | 89.0% (468/526) | 96.4% (507/526) |

Unconfirmed extras in displayed significant families, with the historical scope
held constant (unknown identities, not established false positives):

| Mode | Current | Full strict | Full soft |
| --- | --- | --- | --- |
| RP_Pos | 553 | 583 | 607 |
| RP_Neg | 546 | 574 | 580 |
| NP_Pos | 193 | 179 | 205 |
| NP_Neg | 217 | 201 | 218 |

Full PC reference recovery, separate from the selected counterpart regression:

| Mode | Current | Full strict | Full soft |
| --- | --- | --- | --- |
| RP_Pos | 9.1% (20/220) | 67.7% (149/220) | 72.3% (159/220) |
| RP_Neg | No reference | No reference | No reference |
| NP_Pos | 11.7% (24/206) | 66.0% (136/206) | 66.0% (136/206) |
| NP_Neg | No reference | No reference | No reference |

Compatible reference-pair recovery is 21/960 current, 395/960 strict and 407/960
soft. Conditional pair agreement is 21/35 (60.0%), 395/564 (70.0%) and 407/593
(68.6%); discordant pairs are 14, 169 and 186 respectively. Exact-name reference
pairs are a surrogate, not independently verified ion-family purity or an FDR.
Both full ledgers retain all 254,324 original keys. They assign 96,129 and 142,088
monoisotopic ions respectively, including accepted and non-significant observations;
these counts are not confirmed identifications. All 756 class/mode/arm cells and
all three 60-slot PC tables are retained privately, including losses.

The shared candidate pool contains 4,852,684 adduct relationships and 155,681
unverified isotope candidates. Grouping times are 2.77 s current, 156.32 s full
strict and 176.90 s full soft; export/report time is additional. An initial attempt
was preserved and stopped during strict assembly after over 11 minutes: scalar
writes to Arrow-backed status strings repeatedly copied a large column. Batched
status recording fixes that cost without changing evidence rules. The new source
was committed before a fresh complete comparison. A cross-mode pair-denominator
checker correction and one-representative-per-mode viewer fix are included.

The full suite passes **1,039 tests**, with 8 skipped and 91 subtests passed.
All 251,431 original statistical/evidence rows are unchanged. Source and input
fingerprints pass; all prepared input/raw/IS copies are unchanged and the new
project remains inputs-only. Independent full-ledger checks verify original keys,
unique monoisotopic mode/adduct slots, neutral-mass spans and complete RT bounds
against saved settings. The report checker reads the frozen 4 ppm tolerance;
its initial hard-coded 3 ppm check was corrected without changing predictions.

Browser-model tests on 60 uncached selections per full engine verify one focus
per mode and build both plot descriptions below one second. These timings exclude
browser Plotly rendering/DOM, transfer and initialization; they do not establish
end-to-end browser latency. Native Binner execution remains unperformed. Stage 3
is not fully closed and Stage 4 has not started.

## Binner compatibility and remaining blocker

The [official Binner release page](https://binner.med.umich.edu/) and
[manual](https://docs.google.com/document/d/1_7f9rv-gXfUY5XoDR4B7Awrzkjr7nIIW/mobilebasic)
describe a Java GUI operating on aligned feature tables. The pinned version is
1.0.0. `tools/export_binner_inputs.py` prepares four private feature tables with
positive m/z, RT in minutes, original biological sample areas, and lossless
feature/sample crosswalks. All original rows remain; no manual reference identities
enter these inputs. The canonical ion library retains allowed charges and excludes
the two user-prohibited ions. Its native import mapping still needs verification.

The official download hosts failed certificate verification. Inspection found a
server certificate expiring **17 February 2026** and an obsolete/mismatched supplied
intermediate. Verified requests and the system TLS client both failed. No trusted
JAR/hash is available, so no native Binner computation or performance result is
claimed. Source access is by author request; no email was sent. The export/preflight
record is complete, but the benchmark remains pending a trusted distribution.

The planned native comparison must record Binner's own filtering, imputation,
transformation and annotation settings. It will not silently substitute a local
reimplementation. Its default RT gap is not assumed equivalent to LASER's capped
pairwise window. The available 12 biological specimens meet the documented minimum
of three but fall below the recommendation of 20; this limitation must accompany
any eventual Binner result. The source project and all study outputs remain outside Git.
