# Standard lipidomics annotation reference

This is the entry point for LASER's conventional, XCMS-based lipidomics
workflow. Standard lipidomics has two annotation engines. They share the same
peak-group inputs, result browser and export shape, but they do not use the same
annotation algorithm.

Precursor-mass calibration is not part of either annotation engine. The
standard workflow exposes **XCMS m/z calibration** as a separate, optional
module beside **Annotation**. Its outputs are ordinary discoverable XCMS
peak-group lists with calibrated m/z, retained raw m/z and fixed-model
provenance; both engines can consume those files without recalibration.

The detailed, authoritative references are deliberately kept separate:

- [`STANDARD_LIPIDOMICS_MODE.md`](STANDARD_LIPIDOMICS_MODE.md) describes the
  **LASER reference matcher**, including RP/NP matching, policy enforcement,
  class RT adaptation, confidence states and automatic abstention.
- [`MATLAB_ANNOTATION_ENGINE.md`](MATLAB_ANNOTATION_ENGINE.md) describes the
  **Automatic MATLAB engine (XCMS)**, including unified RTDB anchor selection,
  legacy diagnostic modes, regression, LPC/LPE positional resolution and audits.
- Benchmarking remains in the source LASER 4D repository and is intentionally
  outside this standalone annotation tool.

## Choosing an engine

| Property | LASER reference matcher | Automatic MATLAB engine (XCMS) |
|---|---|---|
| GUI label | `LASER reference matcher` | `Automatic MATLAB engine (XCMS)` |
| Chromatography | RP and NP | RP and NP by default; each phase runs independently |
| Feature input | XCMS peak-group lists | XCMS peak-group lists for the selected phase |
| Main identity source | LASER chemical reference | MATLAB catalogue, target model and RTDB |
| RT treatment | Historical RP reference with guarded class offsets; anchored Lipid-Wizard NP reference | MATLAB RTDB spline plus automatically selected regression anchors |
| Policy | MATLAB-compatible class, stream, priority, adduct and composition policy | MATLAB catalogue definitions, priorities, siblings and stages |
| Automatic ambiguity behavior | Abstains when one feature has multiple passing identities | Resolves global feature conflicts before anchoring, then uses MATLAB's staged sibling regression |
| Positional LPC/LPE | Reference/policy behavior documented in the matcher reference | RP: coupled early-2-acyl/late-1-acyl resolver; NP: one combined peak identity |
| MATLAB or `.mat` required | No | No |

The MATLAB engine is the default standard-lipidomics engine in the GUI. The
LASER matcher remains available as the alternative engine and is also used
internally as an independent high-confidence consensus check when the MATLAB
engine selects ordinary endogenous RT anchors.

## Shared terminology

- **XCMS feature**: one observed peak-group row. Its feature ID identifies the
  observed signal, not a lipid identity.
- **Identity**: the normalized `(lipid class, species, adduct)` combination.
- **Candidate**: an identity/feature pairing that passed the initial scientific
  search window. A candidate is not necessarily an annotation.
- **Automatic annotation**: a candidate that passed the active engine's
  ranking, ambiguity and evidence requirements.
- **Review-only candidate**: visible evidence that was not promoted to an
  automatic annotation. Review controls never silently change the run export.
- **Anchor/inlier**: an observed feature assigned to a known target and used to
  fit or shift an RT model. In the MATLAB engine, automatically retained
  anchors are inserted as positive regression memberships.
- **Equivalent**: another accepted measurement/adduct for an already accepted
  lipid identity. It is retained separately from the primary quantitative row.

Identity and feature must not be confused. One XCMS feature can be chemically
compatible with more than one identity, and one identity can have more than one
candidate feature. Benchmark colours therefore require the same normalized
identity and adduct; a shared feature ID alone never transfers a colour.

## Shared inputs

The standard workflow accepts one or more tab-separated XCMS files whose names
contain `NP_Neg`, `NP_Pos`, `RP_Neg` or `RP_Pos`. The usual project filenames
are `peakgrouplist_<stream>.txt`. Required information is a feature ID, m/z and
retention time in minutes. Numeric sample columns are retained as intensities;
XCMS `Group_` detection summaries are metadata rather than sample profiles.

The automatic MATLAB engine processes both phases by default, using separate
phase-local runs before combining their outputs. It can also be limited to RP
or NP. The LASER matcher can use all four streams in one run. Neither engine
reads `MS_data.xlsx` while annotating.

## Default decision settings

These are production defaults, not universal scientific constants. Every run
records the effective settings and should retain its audits.

| Setting | LASER matcher | MATLAB engine | Benchmark |
|---|---:|---:|---:|
| Precursor tolerance | 4 ppm default; configurable up to 20 ppm | 3 ppm anchor search; final annotation strictly `<3 ppm` | Strict `<3 ppm` crosswalk |
| RP RT window | 0.50 min | 30 s anchor search; 10 s grid-coherence scale | Strict `<10 s` crosswalk |
| Grid coherence | Enabled, minimum 0.45 | Enabled, minimum 0.45 | Enabled, minimum 0.45 |
| Carbon/double-bond ranges | Example-data class extrema plus major-group layer | Same limits applied before target competition | Uses MATLAB-engine defaults |
| Automatic anchor ambiguity margin | n/a | 0.20 score units | n/a |
| Anchors per MATLAB sibling grid | n/a | Minimum 5, maximum 20 | Uses selected MATLAB-engine setting |
| Benchmark profile correlation | n/a | n/a | `log1p` Pearson `r >= 0.995` |

The MATLAB engine's independent LASER consensus pass deliberately tightens the
LASER matcher to 3 ppm and accepts only `high_confidence_auto` rows. Its exact
criteria and exceptions are documented in both engine-specific references.

## Shared carbon and double-bond limits

Both annotation engines use the expandable **Carbon and double-bond annotation
limits** settings. The bundled per-class limits are the inclusive minima and
maxima observed across `Primary list (raw)` and `Equivalents (raw)` in all four
example `MS_data.xlsx` workbooks; rows marked `(IS)` were excluded. The exact
version-controlled table is
`reference_data/standard_lipid_composition_ranges.tsv`.

The policy has two layers. A candidate must pass its major-group range and its
class range. The major groups are single acyl chain, two acyl/alkyl chains,
sphingolipids, three acyl/alkyl chains, and four acyl chains. Their defaults are
the union of the member-class extrema, so the more specific class range is the
limiting layer until a user deliberately narrows a group.

Controls are independent:

- **Enable carbon and double-bond limits** disables or enables the complete
  policy.
- **Apply major lipid-group limits** disables only the global group layer;
  per-class ranges continue to apply.
- **Use group** opts one class out of its major-group limit while retaining
  that class's own range.
- **Filter enabled** opts one class out of both layers.

All bounds are inclusive. Internal standards are always retained. For the
MATLAB engine, disallowed theoretical endogenous targets are removed before
RT-anchor generation, regression and feature competition without changing the
MATLAB composition radices. For the LASER matcher, reference candidates are
filtered before mass/RT matching. This makes the same setting scientifically
equivalent across both engines.

## Provenance and audit contract

A reproducible run should retain its identified, equivalent and all-candidate
tables together with the available policy, anchor, grid-coherence, internal-
standard and positional-isomer audits. The effective settings in
the run manifest are part of the result.

Important source-of-truth modules are:

| Behavior | Source |
|---|---|
| LASER reference matching and confidence states | `app/standard_lipidomics/workflow.py` |
| MATLAB-compatible allow-list | `app/standard_lipidomics/policy.py` |
| Biological abundance prior | `app/annotator/fa_model.py` |
| MATLAB automatic anchors | `app/matlab_annotation/automatic.py` |
| MATLAB orchestration and final filtering | `app/matlab_annotation/engine.py` |
| Positional LPC/LPE resolver | `app/matlab_annotation/isomer_resolver.py` |
| Standard result browser | `app/standard_lipidomics/grid_browser.py` |

When any default or decision rule changes, update the relevant engine-specific
reference, this overview table, the headless configuration template and the
tests in the same commit.
