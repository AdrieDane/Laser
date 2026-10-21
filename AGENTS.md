# Experiment reporting policy

User policy established 21 October 2026: after every experiment that changes
annotation behaviour, report a complete before-and-after table for every
example dataset and every measurement mode. This applies to changes in code,
models, settings, thresholds, composition limits, acceptance or original-peak
selection, including experiments that are rejected or reverted. Do not replace
the complete table with pooled totals or only the rows that improved.

## Required report

- Include RP positive, RP negative, NP positive and NP negative for each dataset,
  even when a mode is unchanged. Mark unavailable or untested cells explicitly;
  never fill them with assumed zeroes or imply a test was performed.
- In the user-facing response, show the complete dataset-by-mode table with
  correct same-manual-peak counts before and after and extra annotation counts
  before and after. Separate mode tables are acceptable. State the lipid-class
  scope explicitly. Include all-class results; for a class-specific experiment,
  also provide that class's complete table in the response or linked full report.
- A correct match requires the same lipid, adduct, measurement mode and ORIGINAL
  feature as manual curation. The right lipid on a different feature is not a
  correct match. Report different-peak choices separately.
- Extras are calls absent from that dataset and mode's manual list, not merely
  the increase in total annotation count. Call these unconfirmed extras, not
  proven false positives. Report totals already present, not just newly added
  extras. Do not collapse the same lipid across different datasets or modes.
- The saved report or CSV must also show manual denominators, before/after
  match percentages, different-peak choices, total annotation counts and manual
  entries not on the same peak. Clearly define missing/unresolved categories;
  do not add overlapping categories as though they were disjoint errors.
- Include each supplied folder in the detailed results. Flag the duplicate fish
  dataset and exclude it from pooled totals, leaving 14 unique datasets unless
  the example collection changes. Label the pooled sample count explicitly.
- Identify the before and after versions, experimental/default settings, tested
  scope and whether results come from full raw-input runs, saved-output replays
  or conditional cross-validation. State training/development-data limitations
  and missing source files. Never describe these examples as untouched external
  validation or describe cached replays as new full-input annotation runs.
- Preserve the exact prior checkpoint and record both gains and regressions.
  Save the tables and provenance with the experiment, link the report, and give
  the user its complete table before moving on to another experiment.

Documentation-only or formatting-only changes that cannot affect annotation
results do not require rerunning annotation. Reuse an explicitly identified,
verified comparison when reporting the same completed experiment again.

Latest full example table:
[RP-positive consensus comparison](docs/validation/experiment_02270eb_all_datasets.md).
