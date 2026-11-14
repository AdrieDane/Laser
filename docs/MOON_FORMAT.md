# LASER and Moon annotation documents (`.moon`)

[Moon](https://github.com/Core-Facility-Metabolomics-AUMC/Moon) is a Python
layer around the MATLAB MS pipeline. It moves annotations in and out of the
MATLAB project as a small JSON document, `moon-annotation/1`; a `.moon` file is
that JSON with a user-facing extension. This page explains what LASER reads and
writes, how that maps onto `.moon`, and the `.moon` export and import added to
LASER (`app/moon_format.py`).

## 1. LASER input and output in short

**Input** (`laser-standard PROJECT` or the *Annotation* module):

| What | Where | Notes |
|---|---|---|
| XCMS peak-group lists | `peakgrouplist_{RP,NP}_{Pos,Neg}.txt` in the project, `XCMS/` or `Processing/Annotation/` (`discover_standard_streams`) | Tab-separated: an ID column (`name`/`feature`/`feature_id`/`id`), `mzmed`, `rtmed` (minutes), optional `mzmin/mzmax/rtmin/rtmax/npeaks`, `Group_*`, then sample intensities |
| Reference data | `reference_data/` (shipped) | Lipid catalogue, MATLAB pipeline reference and policy, NP retention times |
| Optional metadata | `metadata/` | Sample groups, `internal_standards.csv` |
| Settings | CLI flags / GUI | Engine (`matlab` = MATLAB-equivalent, default; `laser` = reference matcher), phase (`both`/`RP`/`NP`) |

LASER does **not** take earlier annotations as input for annotating: manual
truth is used only for evaluation (`standard_lipidomics/validation.py`, never
imported by annotation selection).

**Output** (CLI `--output DIR`):

| File | Content |
|---|---|
| `automatic_annotations.csv` | The final annotations: one row per (feature, lipid, adduct). Key columns: `stream`, `name`, `class`, `adduct`, `feat_feature_id`, `feat_mz`, `feat_rt` (min), `isotopologue`, `is_is`, `quantitative_role`, `auto_annotation_status`, `ppm_error`, `matlab_compound`, `matlab_code`, many evidence columns |
| `identified_lipids.csv`, `equivalent_lipids.csv`, `all_candidates.csv` | Identities per lipid, equivalent (secondary) adducts, every candidate considered |
| `*_audit.csv`, `matlab_*` | Engine diagnostics (anchors, grid coherence, PE, isomers ...) |
| `annotated_input_files/*_annotated.txt` | The input lists with `laser_annotation` and `adduct` inserted after the first column |
| **new** `--moon FILE` | `FILE` (moon-annotation/1) + `FILE_report.json` (rows not written and why) |

The GUI keeps its own project folders (`LASER/`, `Calibration/`,
`Untargeted/`, `Archive/`, saved-project manifest); the `.moon` export is
currently CLI/Python only.

## 2. The shared `.moon` contract

A UTF-8 JSON object: optional `"format": "moon-annotation/1"`, a `features`
list, and any extra fields. Per feature MATLAB (through Moon) needs:

| Field | Required | LASER column / rule |
|---|---|---|
| `mode` | yes | `stream` (`RP_Pos`, `RP_Neg`, `NP_Pos`, `NP_Neg`) |
| `feature` | yes | Moon `FT<n>`: the 1-based row of the peak-group list (or its `feature` column). LASER's `feat_feature_id` is mapped to it through the same list |
| `name` | yes | `name` without ` (IS)`, in the spelling MATLAB's `get_compound_index` finds (`PC(34:1)`, `LPC(O+P-18:2)`). LASER's names are the same except positional lyso isomers: the pipeline's `lipids.tsv` composes `"1-acyl " + class`, so the export writes `1-acyl LPC(16:0)` for LASER's `1-acylLPC(16:0)` (LASER's own name stays in `laser.name`; the import converts back). Without the space MATLAB reports `unknown_compound` (seen on production, 14 Nov 2026) |
| `adduct` | yes, unless `set_as` is `inlier` | `adduct`, as `[M+H]+`, `[M-2H](2-)` |
| `internal_standard` | no | `is_is` or ` (IS)` in the name |
| `isotope` | no (default `M+0`) | only M+0 is exported |
| `mz`, `rt` | no (checks) | from the peak-group list; `rt` in **seconds** |
| `set_as` | no | LASER writes `inlier`: seed peaks, mapped and accepted in the MATLAB GUI |
| `source` | no | document `source: "Laser"`; Moon keeps it per feature until changed in MATLAB |
| `role` | no | MATLAB's primary/equivalent/other (export from MATLAB) |
| `laser` | no | LASER-specific: per feature `role` (quantitative_role), `status`, `ppm`, `matlab_compound`, `matlab_code`, original `name`; per document `engine`, `phase`, `laser_version`, `project`, `rows`, `skipped_by_reason` |

MATLAB exports (Moon `getannotation`) add `matlab` (row, code, i_compound),
`role`, `is_reference`, `rtp` and a `classes` section; LASER keeps reading them
and ignores what it does not need. Extra fields never break the other side;
documents never carry `$schema`. Moon's validator is authoritative; LASER's
`validate_moon` checks only this shared minimum.

IDs only make sense for the project the document was made for: both sides map
`FT<n>` and `feat_feature_id` through that project's peak-group lists.

## 3. Use

**Export** (LASER -> MATLAB):

```bash
laser-standard /path/to/Processing/Annotation --engine matlab --output results --moon results/laser.moon
```

then in Moon: `python -m moon setannotation <project> results/laser.moon`
(or the Moon web page). Rows that cannot be placed exactly (unknown stream or
feature, isotope, m/z or RT not matching the list, a second label on a feature)
are listed in `laser_report.json`, never guessed.

**Import** (MATLAB -> LASER, evaluation):

```bash
python -m moon getannotation <project> --all --out curated.moon          # in Moon
python tools/moon_crosswalk.py curated.moon /path/to/project --output curation/DATASET/curated_feature_crosswalk.csv
python tools/compare_manual_annotations.py --runs-root runs --curation-root curation --output report
```

Moon feature IDs are exact, so these rows get `crosswalk_status =
exact_feature_id`, which `compare_manual_annotations` now treats as mapped,
like `profile_match`.

**Python:**

```python
from moon_format import PeakIndex, annotations_to_moon, write_moon, read_moon, moon_to_frame, moon_to_crosswalk
peaks = PeakIndex.from_project(project)
doc, report = annotations_to_moon(result.tables["automatic_annotations"], peaks)
write_moon(doc, "laser.moon")
frame = moon_to_frame(read_moon("curated.moon"), peaks)     # LASER columns + moon_status
```

## 4. Changes to LASER (branch `moon-format`)

| File | Change |
|---|---|
| `app/moon_format.py` | New: peak-list ID mapping, export, validation, import, crosswalk. No new dependencies |
| `app/standard_cli.py` | Optional `--moon FILE`; without it nothing changes |
| `app/standard_lipidomics/validation.py` | `exact_feature_id` counts as mapped next to `profile_match` (one line) |
| `tools/moon_crosswalk.py` | New: `.moon` -> `curated_feature_crosswalk.csv` |
| `app/tests/test_moon_format.py` | Synthetic tests (export, skips, round trip, BOM, invalid documents, import, comparison, CLI, tool) |

The branch starts from `main` (upstream) and touches no annotation logic, so it
can go to the upstream repository on its own branch or straight to `main`.

## 5. Verified

- `app/tests/test_moon_format.py` 20 passed, `test_validation_cli.py` passes;
  ruff clean on the changed files. The full suite on Windows shows the same 10 failures as `main` without this branch (Windows portability, unrelated).
- A real run (anonymised DEV set, engine `matlab`, 1935 annotations): the
  export gives exactly the features of Moon's own LASER converter (only when
  the CSV is re-parsed by pandas the last digit of `ppm` can differ: float
  parsing); Moon reads the file and remaps all 1935; importing it back gives
  the same LASER feature IDs; Moon's file imports as 1935 `exact_feature_id`
  crosswalk rows.

## 6. Possible next steps

- Moon calls `laser-standard --moon` instead of converting
  `automatic_annotations.csv` itself (keep the converter for older LASER).
- An *Export .moon* button in the Annotation module and a *Compare with .moon*
  view.
- Not planned: using `.moon` truth inside annotation selection; LASER keeps
  evaluation separate from selection.
