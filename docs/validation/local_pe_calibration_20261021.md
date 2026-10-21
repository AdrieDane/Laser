# Local PE calibration pilot rejected for deployment

The calibration trial adds only **1 correct manual-peak match**, alongside
**5 unconfirmed extras and 2 different-peak choices**, across 14 unique datasets.
It fails the acceptance criteria and is not enabled in the app. Existing
annotations were preserved throughout the trial. The earlier method remains.

The user-requested rollback is already applied in `3323003`: the three preceding
RP-positive approaches stay rejected. This is a separate, frozen experiment.
The baseline retains the earlier optional PE correction, including NP-negative
gains. The optional correction is ON for both sides of this comparison; the
default and optional application workflows do not enable this new prototype.

## What was tested

Only RP-positive endogenous PE could be added. No existing annotation, protected
standard or manually confirmed choice could be replaced. Carbon/unsaturation
limits, the old portable PE scorer, its 0.10 threshold and 0.03 ambiguity margin
were unchanged. Corrected mass and RT acceptance windows were not widened.

Mass calibration uses a robust m/z-and-RT surface from strong automatic non-PE
anchors, excluding the PE family. It must improve predictions with whole lipid
classes held out. Local RT calibration uses high-confidence automatic PE anchors,
excluding every anchor of the target composition. It requires local support,
interpolation rather than unrestricted extrapolation, and better held-out
predictions. Prediction uncertainty must fit inside the existing 3 ppm/10 second
acceptance windows. Unsupported calibrations fall back to the preceding method.
Curated labels, dataset names and manual feature IDs never fit these corrections.

## Interpretation and limits

Correct means the same lipid, adduct, mode AND original feature as manual curation.
Extras are annotations absent from that dataset/mode's manual list: unconfirmed,
not proven false. Different-peak choices are separately reported in the CSV.
These categories must not be added to manual misses as independent errors.

All 15 supplied folders were evaluated using retained full-core outputs and
original-feature tables. These are not 15 fresh raw-input runs. The duplicate fish
folder is displayed but excluded from pooled totals (14 unique datasets). The
Fish, duplicate Fish and Serhii raw-input folders are absent from the input root;
their retained tables are used. These examples are development data, not untouched
external validation. Other classes and the other three modes are exact controls.

The all-class percentage denominator is reliably mapped, unique manual entries.
The PE-class denominator is all manual PE rows, including unresolved rows and
explicitly marked PE standards, consistent with the preceding numeric benchmark.
Only endogenous PE can change in this experiment. Unresolved manual mappings
remain misses, but do not establish a wrong original peak. The CSV also reports
total calls, unresolved mappings, denominators and manual entries not on the same
peak. No original input or previously exported peakgroup TXT is rewritten.

Two preliminary report attempts stopped on the first dataset because stream-level
tables lack consistent root-level stream metadata. Their partial artifacts were
preserved. The evaluator now assigns the explicit parent stream and verifies
all baseline selected keys using the root `stream` field. This changes reporting,
not candidate selection. The scientific module, candidate generator and portable
model remained fingerprint-frozen across the completed pilot. No thresholds
were retuned against its results.

**All classes, 14 unique datasets:** correct 34,348→34,349; extras 3,652→3,657; different-peak calls 534→536; total calls 38,887→38,895.

**PE class, 14 unique datasets:** correct 4,086→4,087; extras 486→491; different-peak calls 74→76; total calls 4,689→4,697.

The predeclared acceptance criteria require no correct-peak losses, no additional
wrong-peak choices, and at least 4 correct gains per extra. The pilot fails these
criteria and is NOT enabled. The runtime remains at the restored earlier method.
The research implementation and audit are retained separately for reproducibility.

The sole correct addition is Worms PE(30:0), original feature 55722. The two
different-peak choices are Heart PE(42:7), feature 74019, and Fish PE(35:5), feature
96016; the duplicate Fish result is not counted twice. The extras are Kidney
PE(35:0), Lengthening PE(44:2), and ALD brain PE(46:3), PE(45:5) and PE(47:5).
These are trial results, not additions retained by the app.

## What this means for the next experiment

In the retained baseline there are 76 mapped, endogenous RP-positive PE entries
not annotated on the manual peak: 32 use another peak and 44 are not selected.
The curator's exact feature is present in this trial's candidate table for 62 of
those 76 entries (25 of the 32 different-peak cases and 37 of the 44 unselected
cases). Another 4 endogenous PE manual features cannot be reliably mapped and
are excluded from that diagnostic count. These are 14-unique-dataset totals.

The next priority is therefore comparing competing peaks and the option not to
annotate, rather than treating a better calibration fit as sufficient evidence
to add a label. This is a prioritization based on observed candidate availability,
not proof that all 62 can be safely corrected. Preserve standards and manual
locks, use independent isotope/adduct/profile evidence, and test any replacement
rule on held-out datasets with correct-peak losses and extras reported separately.
The local-calibration pilot is not silently retuned against this result.

## All lipid classes

### RP positive

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 1,165 → 1,165 | 76 → 77 |
| Heart | 1,329 → 1,329 | 75 → 75 |
| Worms | 1,144 → 1,145 | 127 → 127 |
| Fish larvae | 680 → 680 | 31 → 31 |
| Fish larvae (duplicate) | 680 → 680 | 31 → 31 |
| Serhii brain | 586 → 586 | 31 → 31 |
| EDT brain | 1,338 → 1,338 | 105 → 105 |
| Lengthening spinal cord | 1,267 → 1,267 | 151 → 152 |
| V2 | 1,294 → 1,294 | 60 → 60 |
| Cells | 1,204 → 1,204 | 65 → 65 |
| ALD brain | 1,271 → 1,271 | 88 → 91 |
| Rat plasma | 959 → 959 | 23 → 23 |
| Rat spinal cord | 1,194 → 1,194 | 107 → 107 |
| Rat brain | 1,452 → 1,452 | 66 → 66 |
| Pex5 mouse | 949 → 949 | 28 → 28 |
| Total, 14 unique datasets | 15,832 → 15,833 | 1,033 → 1,038 |

### RP negative

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 353 → 353 | 35 → 35 |
| Heart | 328 → 328 | 30 → 30 |
| Worms | 400 → 400 | 48 → 48 |
| Fish larvae | 323 → 323 | 31 → 31 |
| Fish larvae (duplicate) | 323 → 323 | 31 → 31 |
| Serhii brain | 514 → 514 | 24 → 24 |
| EDT brain | 451 → 451 | 50 → 50 |
| Lengthening spinal cord | 527 → 527 | 38 → 38 |
| V2 | 262 → 262 | 90 → 90 |
| Cells | 500 → 500 | 91 → 91 |
| ALD brain | 441 → 441 | 55 → 55 |
| Rat plasma | 137 → 137 | 8 → 8 |
| Rat spinal cord | 422 → 422 | 24 → 24 |
| Rat brain | 496 → 496 | 16 → 16 |
| Pex5 mouse | 329 → 329 | 59 → 59 |
| Total, 14 unique datasets | 5,483 → 5,483 | 599 → 599 |

### NP positive

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 487 → 487 | 142 → 142 |
| Heart | 540 → 540 | 135 → 135 |
| Worms | 474 → 474 | 92 → 92 |
| Fish larvae | 383 → 383 | 112 → 112 |
| Fish larvae (duplicate) | 383 → 383 | 112 → 112 |
| Serhii brain | 509 → 509 | 130 → 130 |
| EDT brain | 683 → 683 | 121 → 121 |
| Lengthening spinal cord | 725 → 725 | 142 → 142 |
| V2 | 677 → 677 | 59 → 59 |
| Cells | 419 → 419 | 79 → 79 |
| ALD brain | 613 → 613 | 44 → 44 |
| Rat plasma | 378 → 378 | 59 → 59 |
| Rat spinal cord | 627 → 627 | 82 → 82 |
| Rat brain | 682 → 682 | 72 → 72 |
| Pex5 mouse | 340 → 340 | 30 → 30 |
| Total, 14 unique datasets | 7,537 → 7,537 | 1,299 → 1,299 |

### NP negative

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 357 → 357 | 61 → 61 |
| Heart | 303 → 303 | 53 → 53 |
| Worms | 435 → 435 | 75 → 75 |
| Fish larvae | 534 → 534 | 102 → 102 |
| Fish larvae (duplicate) | 534 → 534 | 102 → 102 |
| Serhii brain | 514 → 514 | 58 → 58 |
| EDT brain | 317 → 317 | 19 → 19 |
| Lengthening spinal cord | 381 → 381 | 19 → 19 |
| V2 | 408 → 408 | 48 → 48 |
| Cells | 440 → 440 | 53 → 53 |
| ALD brain | 396 → 396 | 24 → 24 |
| Rat plasma | 165 → 165 | 71 → 71 |
| Rat spinal cord | 461 → 461 | 71 → 71 |
| Rat brain | 443 → 443 | 21 → 21 |
| Pex5 mouse | 342 → 342 | 46 → 46 |
| Total, 14 unique datasets | 5,496 → 5,496 | 721 → 721 |

## PE class only

### RP positive

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 67 → 67 | 4 → 5 |
| Heart | 86 → 86 | 9 → 9 |
| Worms | 131 → 132 | 3 → 3 |
| Fish larvae | 89 → 89 | 10 → 10 |
| Fish larvae (duplicate) | 89 → 89 | 10 → 10 |
| Serhii brain | 58 → 58 | 9 → 9 |
| EDT brain | 55 → 55 | 22 → 22 |
| Lengthening spinal cord | 55 → 55 | 33 → 34 |
| V2 | 71 → 71 | 14 → 14 |
| Cells | 77 → 77 | 19 → 19 |
| ALD brain | 66 → 66 | 29 → 32 |
| Rat plasma | 20 → 20 | 3 → 3 |
| Rat spinal cord | 63 → 63 | 13 → 13 |
| Rat brain | 75 → 75 | 15 → 15 |
| Pex5 mouse | 68 → 68 | 1 → 1 |
| Total, 14 unique datasets | 981 → 982 | 184 → 189 |

### RP negative

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 80 → 80 | 2 → 2 |
| Heart | 99 → 99 | 7 → 7 |
| Worms | 127 → 127 | 5 → 5 |
| Fish larvae | 109 → 109 | 8 → 8 |
| Fish larvae (duplicate) | 109 → 109 | 8 → 8 |
| Serhii brain | 81 → 81 | 7 → 7 |
| EDT brain | 82 → 82 | 11 → 11 |
| Lengthening spinal cord | 84 → 84 | 15 → 15 |
| V2 | 96 → 96 | 18 → 18 |
| Cells | 90 → 90 | 31 → 31 |
| ALD brain | 86 → 86 | 20 → 20 |
| Rat plasma | 23 → 23 | 2 → 2 |
| Rat spinal cord | 76 → 76 | 12 → 12 |
| Rat brain | 95 → 95 | 5 → 5 |
| Pex5 mouse | 87 → 87 | 1 → 1 |
| Total, 14 unique datasets | 1,215 → 1,215 | 144 → 144 |

### NP positive

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 75 → 75 | 5 → 5 |
| Heart | 77 → 77 | 6 → 6 |
| Worms | 118 → 118 | 16 → 16 |
| Fish larvae | 75 → 75 | 27 → 27 |
| Fish larvae (duplicate) | 75 → 75 | 27 → 27 |
| Serhii brain | 66 → 66 | 2 → 2 |
| EDT brain | 51 → 51 | 21 → 21 |
| Lengthening spinal cord | 71 → 71 | 14 → 14 |
| V2 | 87 → 87 | 4 → 4 |
| Cells | 64 → 64 | 0 → 0 |
| ALD brain | 61 → 61 | 4 → 4 |
| Rat plasma | 19 → 19 | 0 → 0 |
| Rat spinal cord | 55 → 55 | 12 → 12 |
| Rat brain | 52 → 52 | 12 → 12 |
| Pex5 mouse | 48 → 48 | 3 → 3 |
| Total, 14 unique datasets | 919 → 919 | 126 → 126 |

### NP negative

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 76 → 76 | 2 → 2 |
| Heart | 79 → 79 | 2 → 2 |
| Worms | 118 → 118 | 6 → 6 |
| Fish larvae | 83 → 83 | 5 → 5 |
| Fish larvae (duplicate) | 83 → 83 | 5 → 5 |
| Serhii brain | 70 → 70 | 1 → 1 |
| EDT brain | 57 → 57 | 1 → 1 |
| Lengthening spinal cord | 75 → 75 | 1 → 1 |
| V2 | 87 → 87 | 0 → 0 |
| Cells | 69 → 69 | 1 → 1 |
| ALD brain | 62 → 62 | 3 → 3 |
| Rat plasma | 19 → 19 | 1 → 1 |
| Rat spinal cord | 55 → 55 | 6 → 6 |
| Rat brain | 56 → 56 | 0 → 0 |
| Pex5 mouse | 65 → 65 | 3 → 3 |
| Total, 14 unique datasets | 971 → 971 | 32 → 32 |

## Reproduction and artifacts

The [companion CSV](local_pe_calibration_20261021.csv) contains all 120
dataset/mode/scope rows. The [provenance JSON](local_pe_calibration_20261021_provenance.json)
records frozen hashes and checks; the [calibration audit](local_pe_calibration_20261021_audit.csv)
shows which mass and RT fits passed their automatic-anchor checks in each folder.
Calibration decisions,
source fingerprints, additions and full per-entry comparisons are retained under
`outputs/local_pe_calibration_20261021/pilot_complete/` in the local workspace.
The documented research runner requires explicit trust for saved local pickles;
never load untrusted pickles. The frozen pilot's acceptance result is not an
improvement claim and is not an estimate of analytical false-discovery rate.

The scientific functions were moved unchanged to `tools/local_pe_calibration.py`,
with its calibrated candidate generator isolated there too. The app's original
candidate generator is restored byte for byte. Neither normal nor optional app
workflows call the research code, and it is excluded from the built package.
A second replay using `tools/evaluate_local_pe_calibration.py` reproduces every
selected feature and every metric for all 15 folders. The preserved original
pilot sources match their fingerprints, and every scientific function and copied
candidate helper has the same syntax tree after the move. This is a reproduction
check, not a separately tuned annotation experiment.

Final checks: 408 tests and 83 subtests pass; changed-file lint and whitespace
checks pass. The built package matches 158 source/data files and contains neither
the rejected calibration modules nor the development tools. App runtime is
unchanged from `3323003`; no install or restart was performed.

From the repository, run `python3 tools/evaluate_local_pe_calibration.py --help`.
Supply the trusted `full_runs` folder, the verified rollback folder, the manual
crosswalk folder, a fresh output folder and `--trust-local-pickles`. To verify
reproduction, also supply the frozen pilot through `--expected-output-dir`.
