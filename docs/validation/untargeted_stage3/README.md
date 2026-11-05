# Stage 3: full-feature ion grouping

5 November 2026. Implementation ready; frozen LASER comparison pending.
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
