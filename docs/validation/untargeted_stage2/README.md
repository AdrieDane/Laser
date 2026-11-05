# Stage 2: acquisition settings and adduct chemistry

5 November 2026. Stage 2 complete; checkpoint ready for review.
Stage 3 has not started. The current scoring remains the default. Method-aware
scoring is experimental and selectable in the untargeted launcher.

## Project method settings

`metadata/acquisition_method.json` stores the method, using the
[version 1 schema](../../schemas/acquisition_method.schema.json).
An [example](../../examples/acquisition_method.example.json) captures the
[user-supplied published acquisition protocol](https://doi.org/10.1038/s43856-024-00605-9).
It keeps RP/NP solvent recipes and gradients separate, alongside column, flow,
temperature, LC/MS instruments, resolving power, resolution reference m/z, scan
range, spray voltage, polarities and per-polarity m/z/RT input calibration records.

These are acquisition provenance fields, not instructions to recalibrate data.
Unknown values remain null/unknown. The reference m/z for the reported resolving
power is unspecified. Additive percentage basis is preserved as unspecified `%`.
Recipe concentrations are not interpreted as fixed concentrations throughout a
gradient. The project's existing IS catalog remains authoritative; the protocol's
specimen extraction and IS amounts do not overwrite it.

The launcher has an **Acquisition and adduct settings** editor, with a save button.
It supports **Current scoring** and **Method-aware (experimental)**, separate A/B
recipes, instrument settings and common/possible/disabled ions for each phase and
polarity. **Automatic** ion settings derive preferences from the recipe. Explicit
overrides stay local to their phase and polarity. Invalid settings fail explicitly
instead of silently falling back to generic chemistry.

Workers freeze the selected method in their input snapshot. Each completed
untargeted project stores its own `acquisition_method.json` and method object/hash
in `run_manifest.json`. Reopening a saved result uses its frozen method even if
the live project metadata subsequently changes. Candidate/compound/cluster exports
carry `acquisition_method_id` and `adduct_policy_version`; the complete JSON export
resolves that ID. Adduct candidate tables also retain ion states, class hints and
chemistry penalties. Legacy projects without method metadata reopen with unknown
chemistry and current scoring.

## Chemistry and safeguards

- Formic acid/formate supports a formate preference; ammonia/ammonium supports
  an ammonium preference. Acetic acid/acetate supports acetate. These are soft
  preferences. Absence from a recipe does not prohibit a contaminant/background ion.
- Sodium, potassium, chloride, water-loss and charge-2 hypotheses remain possible
  unless explicitly disabled. Water-loss intact-parent support, isotope charge,
  ppm, coelution and biological-profile checks remain in place.
- `[M+2H]2+` and `[M+Na-2H]-` remain excluded and cannot be re-enabled through
  configuration. `[M-2H]2-` remains available.
- Literature-based class preferences favour protonated PC/LPC/SM in positive mode,
  counterion adducts for these classes in negative mode, ammoniated neutral lipids
  such as TG/DG/CE, and deprotonated acidic phospholipids/FA. CL/MLCL retain their
  negative charge-2 alternative. Unsupported classes use generic recipe preferences.
- Only existing accepted runtime annotations may supply a class hint. No manual
  evaluation labels, ALD class frequencies or fitted study-specific priors enter
  inference. Unknown untargeted classes stay unknown; a neighbour is not assigned
  a class merely by proximity.

The class preferences follow [glycerophospholipid ion chemistry](https://doi.org/10.1194/jlr.D001149)
and [lipid annotation quality guidance](https://doi.org/10.1038/s41467-021-24984-y).
They are not independently calibrated probabilities. Recipe state contributes
penalty 0 for common and 1 for possible; a known-class preference mismatch adds 1.
Group ranking subtracts 0.10 times the anchor penalty and 0.025 times each member
penalty from the preceding support score. Disabled ions are removed before scoring.
These constants are fixed before the Stage 2 comparison, not optimized on the PC
regression subset. Current scoring has zero chemistry penalty.

Formate and acetate offsets differ by **14.015650064 Da**, exactly CH2.
[MZmine documents the same ambiguity](https://mzmine.github.io/mzmine_documentation/latest/module_docs/id_lipid_annotation/lipid-annotation.html).
A coeluting singly charged negative one-CH2 pair is therefore marked
`adduct_alias_ambiguous` while both ion explanations are allowed. Method-aware
scoring excludes that edge from homologous clustering; grid growth also avoids
such ambiguous carbon edges. The candidate remains in the exported evidence
table. Disabling one explanation or exceeding the same-mode coelution window
removes this particular ambiguity. This conservative rule can lose genuine
coeluting homologues: it does not establish which explanation is correct.
Mixed charge and different-adduct hypotheses are not declared homologues by this
mass difference alone. Later joint inference/trace evidence can resolve ambiguity.

The browser's cached lookup uses the same allowed ion sets for each mode, including
non-significant cluster neighbours. Changing a point does not launch a new pipeline.

## Validation protocol

The new schema, unknown/supplied settings UI, save/reopen and detached workers are
tested with synthetic inputs. Regression tests cover disabled ions in source and
target directions, the user's hard exclusions, class preferences, unknown fallback,
non-finite/invalid settings and the exact formate/acetate-versus-CH2 case.

The development comparison used the Stage 1 frozen acquisition, query ledger,
accepted exclusions, saved biological tests, isotope/formula outputs and all four
modes. It replays grouping and the actual browser selector in two fixed arms:
current scoring and method-aware scoring with the supplied method. The complete
current-scoring assignment table must match the Stage 1 baseline. The method-aware
arm reports all-class and PC original-ion/adduct agreement/recovery, coverage,
unconfirmed extras, gained/lost matches, all reference class/mode cells and all
selected-PC mode slots. No additional acquisitions or manual labels are introduced.

```sh
python3 tools/evaluate_untargeted_method.py \
  --checkpoint /private/stage-1-checkpoint \
  --method /private/project/metadata/acquisition_method.json \
  --output /private/new-stage-2-attempt \
  --node /path/to/node
```

The runner archives the committed source and freezes its protocol before inference.
Gold references are read separately from the worker, which receives feature keys
and method settings. Original non-label feature columns must remain identical.
The source checkpoint is fingerprinted before/after. Existing output directories
are refused; failed attempts must remain available. Detailed study results remain
outside Git. External engines, new formula ranking, scan-trace extraction and
full-feature ion grouping are not part of this stage.

The new inputs-only project has the supplied method in its metadata, with current
scoring retained. It still has no completed annotation or untargeted result.

## Source-bound results and checkpoint decision

Inference implementation `3db7c78`; completed evaluation revision `8cad478`.
The latter changes only the evaluation key-format checker/test, not inference.
Attempt 001 stopped when an integer feature key was compared to its text form.
It was retained, not overwritten, and produced no completed scientific comparison.
The corrected checker normalizes only feature-key representation; all remaining
original evidence/statistical columns and dtypes are compared exactly. A regression
test verifies that even a 1e-15 p-value change fails the check.

Attempt 002 completed both fixed arms. The complete 1,624-ion current-scoring
assignment table (original feature, adduct, compound, representative, neutral mass,
cluster) matches Stage 1 exactly. Both arms preserve all 251,431 original non-label
test/evidence rows and the checkpoint's file fingerprints. Formula/isotope
searches and biological comparisons were not rerun or replaced.

All-class original-ion/adduct **reference recovery**, not identification accuracy:

| Mode | Before | Method-aware | Unconfirmed extra ions before → after | Gained / lost reference matches |
| --- | ---: | ---: | ---: | ---: |
| RP positive | 2.7% (46/1,694) | 2.7% (45/1,694) | 553 → 551 | 0 / 1 |
| RP negative | 0.4% (3/679) | 0.4% (3/679) | 546 → 546 | 0 / 0 |
| NP positive | 4.9% (37/757) | 4.9% (37/757) | 193 → 195 | 0 / 0 |
| NP negative | 0.9% (5/538) | 0.9% (5/538) | 217 → 217 | 0 / 0 |

The denominator includes all 3,668 frozen manual rows, including 46 unresolved
original mappings. Extra ions are outside the reference set, not proven false
positives. Their pooled total stays 1,509; report the opposite RP/NP changes rather
than presenting that pooled total alone.

Conditional agreement and coverage remain separate:

| Mode | Assigned-ion adduct agreement before → after | Assignment coverage before → after |
| --- | ---: | ---: |
| RP positive | 90.2% (46/51) → 90.0% (45/50) | 3.0% (51/1,680) → 3.0% (50/1,680) |
| RP negative | 100.0% (3/3) → 100.0% (3/3) | 0.4% (3/671) → 0.4% (3/671) |
| NP positive | 75.5% (37/49) → 75.5% (37/49) | 6.6% (49/745) → 6.6% (49/745) |
| NP negative | 83.3% (5/6) → 83.3% (5/6) | 1.1% (6/526) → 1.1% (6/526) |

These small, significance-selected development denominators do not establish
general adduct accuracy. Most curated accepted lipids are intentionally excluded
from untargeted biological testing; Stage 3 addresses full-feature ion evidence.

PC-specific original-ion/adduct reference recovery:

| Mode | Before | Method-aware | Class-specific unconfirmed extras |
| --- | ---: | ---: | --- |
| RP positive | 9.1% (20/220) | 9.1% (20/220) | Unavailable: unconfirmed ions have no inferred lipid class |
| RP negative | Unavailable: no curated PC reference | Unavailable | Unavailable |
| NP positive | 11.7% (24/206) | 11.7% (24/206) | Unavailable: unconfirmed ions have no inferred lipid class |
| NP negative | Unavailable: no curated PC reference | Unavailable | Unavailable |

The separate 15-PC browser regression stays **10/14 (71.4%)** exact positive-mode
counterparts in both arms. All 60 mode slots are retained privately, including
the 12 RP-negative and 5 NP-negative displayed ions without curated PC reference.
No loss of a saved annotation is implied: the one lost RP-positive match is a
link to an existing accepted reference ion, not removal of that annotation.

The ambiguity guard marks 67 of 3,484 pair candidates. Grid candidates change
10,411 → 9,632; current grids contain 40 directly ambiguous root edges, versus one
in the method-aware export (that edge remains flagged/excluded from clustering).
Cluster-member records change 7,319 → 6,970, with 238 → 249 clusters. These are
structural changes, not measured cluster precision/recall: no verified network gold
set exists. Grouping took 6.71 s for current scoring and 7.07 s for method-aware;
these are background worker times, not point-selection latency measurements.

**Decision:** retain current scoring as the default. Method-aware scoring remains
an inspectable experimental option. It adds reproducible method chemistry and
uncertainty handling, but this development comparison shows no gain and one lost
reference link. No post-hoc class/mode winner splice or weight tuning was applied.

Private reports include both arms, all 252 class/mode cells (63 labels × four
modes), exact transitions, all-PC slots, source/method/input hashes and failed
attempt provenance. The full application suite passed 1,028 tests with 8 skips
and 91 subtests; the checker correction additionally passed 13 focused tests.
Schema/example validation, detached-worker round trips, browser policy checks,
lint and whitespace checks passed. Stage 2 is complete. Stage 3 has not started.
