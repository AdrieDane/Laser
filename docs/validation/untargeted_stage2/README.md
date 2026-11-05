# Stage 2: acquisition settings and adduct chemistry

5 November 2026. Implementation ready; frozen development comparison pending.
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

The development comparison will use the Stage 1 frozen acquisition, query ledger,
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

Final comparison results and the checkpoint decision will be added here after the
source-bound evaluation completes. Do not interpret this implementation push as
completion of Stage 2 or permission to start Stage 3.
