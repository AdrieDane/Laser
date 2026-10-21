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

Correct-match percentages now use ALL manual annotations in the same dataset,
mode and class scope, including unresolved rows and explicitly marked standards.
The formula is 100 × correct original-peak matches / all manual annotations.
Counts and denominators accompany percentages; pooled percentages divide summed
counts rather than averaging dataset percentages. Unresolved mappings remain
in the denominator and are separately counted, not classified as wrong peaks.
The previous all-class mapped-only denominator was 45,712; the full manual list
has 46,255 entries across 14 unique datasets. Mapped-only percentages remain
clearly labelled secondary diagnostics in the CSV. This is a reporting change
to the SAME frozen experiment, not a change in annotations or trial acceptance.
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

**All classes, 14 unique datasets:** correct 74.3% (34,348/46,255)→74.3% (34,349/46,255); extras 3,652→3,657; different-peak calls 534→536; total calls 38,887→38,895.

**PE class, 14 unique datasets:** correct 95.4% (4,086/4,283)→95.4% (4,087/4,283); extras 486→491; different-peak calls 74→76; total calls 4,689→4,697.

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

| Dataset | Correct before → trial (% of manual list) | Extras before → trial |
| --- | ---: | ---: |
| Kidney summer | 76.8% (1,165/1,516) → 76.8% (1,165/1,516) | 76 → 77 |
| Heart | 80.8% (1,329/1,645) → 80.8% (1,329/1,645) | 75 → 75 |
| Worms | 82.9% (1,144/1,380) → 83.0% (1,145/1,380) | 127 → 127 |
| Fish larvae | 33.5% (680/2,032) → 33.5% (680/2,032) | 31 → 31 |
| Fish larvae (duplicate) | 33.5% (680/2,032) → 33.5% (680/2,032) | 31 → 31 |
| Serhii brain | 38.0% (586/1,544) → 38.0% (586/1,544) | 31 → 31 |
| EDT brain | 73.4% (1,338/1,824) → 73.4% (1,338/1,824) | 105 → 105 |
| Lengthening spinal cord | 80.5% (1,267/1,574) → 80.5% (1,267/1,574) | 151 → 152 |
| V2 | 82.5% (1,294/1,568) → 82.5% (1,294/1,568) | 60 → 60 |
| Cells | 68.8% (1,204/1,749) → 68.8% (1,204/1,749) | 65 → 65 |
| ALD brain | 71.1% (1,271/1,788) → 71.1% (1,271/1,788) | 88 → 91 |
| Rat plasma | 78.7% (959/1,219) → 78.7% (959/1,219) | 23 → 23 |
| Rat spinal cord | 84.6% (1,194/1,412) → 84.6% (1,194/1,412) | 107 → 107 |
| Rat brain | 85.7% (1,452/1,694) → 85.7% (1,452/1,694) | 66 → 66 |
| Pex5 mouse | 82.5% (949/1,151) → 82.5% (949/1,151) | 28 → 28 |
| Total, 14 unique datasets | 71.7% (15,832/22,096) → 71.7% (15,833/22,096) | 1,033 → 1,038 |

### RP negative

| Dataset | Correct before → trial (% of manual list) | Extras before → trial |
| --- | ---: | ---: |
| Kidney summer | 74.5% (353/474) → 74.5% (353/474) | 35 → 35 |
| Heart | 72.4% (328/453) → 72.4% (328/453) | 30 → 30 |
| Worms | 79.1% (400/506) → 79.1% (400/506) | 48 → 48 |
| Fish larvae | 45.6% (323/708) → 45.6% (323/708) | 31 → 31 |
| Fish larvae (duplicate) | 45.6% (323/708) → 45.6% (323/708) | 31 → 31 |
| Serhii brain | 74.9% (514/686) → 74.9% (514/686) | 24 → 24 |
| EDT brain | 74.5% (451/605) → 74.5% (451/605) | 50 → 50 |
| Lengthening spinal cord | 74.0% (527/712) → 74.0% (527/712) | 38 → 38 |
| V2 | 67.4% (262/389) → 67.4% (262/389) | 90 → 90 |
| Cells | 74.3% (500/673) → 74.3% (500/673) | 91 → 91 |
| ALD brain | 69.3% (441/636) → 69.3% (441/636) | 55 → 55 |
| Rat plasma | 62.8% (137/218) → 62.8% (137/218) | 8 → 8 |
| Rat spinal cord | 67.7% (422/623) → 67.7% (422/623) | 24 → 24 |
| Rat brain | 73.0% (496/679) → 73.0% (496/679) | 16 → 16 |
| Pex5 mouse | 73.4% (329/448) → 73.4% (329/448) | 59 → 59 |
| Total, 14 unique datasets | 70.2% (5,483/7,810) → 70.2% (5,483/7,810) | 599 → 599 |

### NP positive

| Dataset | Correct before → trial (% of manual list) | Extras before → trial |
| --- | ---: | ---: |
| Kidney summer | 78.4% (487/621) → 78.4% (487/621) | 142 → 142 |
| Heart | 83.7% (540/645) → 83.7% (540/645) | 135 → 135 |
| Worms | 86.2% (474/550) → 86.2% (474/550) | 92 → 92 |
| Fish larvae | 44.6% (383/859) → 44.6% (383/859) | 112 → 112 |
| Fish larvae (duplicate) | 44.6% (383/859) → 44.6% (383/859) | 112 → 112 |
| Serhii brain | 80.0% (509/636) → 80.0% (509/636) | 130 → 130 |
| EDT brain | 84.3% (683/810) → 84.3% (683/810) | 121 → 121 |
| Lengthening spinal cord | 85.5% (725/848) → 85.5% (725/848) | 142 → 142 |
| V2 | 85.5% (677/792) → 85.5% (677/792) | 59 → 59 |
| Cells | 79.2% (419/529) → 79.2% (419/529) | 79 → 79 |
| ALD brain | 75.2% (613/815) → 75.2% (613/815) | 44 → 44 |
| Rat plasma | 94.7% (378/399) → 94.7% (378/399) | 59 → 59 |
| Rat spinal cord | 83.7% (627/749) → 83.7% (627/749) | 82 → 82 |
| Rat brain | 90.1% (682/757) → 90.1% (682/757) | 72 → 72 |
| Pex5 mouse | 92.1% (340/369) → 92.1% (340/369) | 30 → 30 |
| Total, 14 unique datasets | 80.4% (7,537/9,379) → 80.4% (7,537/9,379) | 1,299 → 1,299 |

### NP negative

| Dataset | Correct before → trial (% of manual list) | Extras before → trial |
| --- | ---: | ---: |
| Kidney summer | 77.3% (357/462) → 77.3% (357/462) | 61 → 61 |
| Heart | 63.1% (303/480) → 63.1% (303/480) | 53 → 53 |
| Worms | 78.7% (435/553) → 78.7% (435/553) | 75 → 75 |
| Fish larvae | 83.0% (534/643) → 83.0% (534/643) | 102 → 102 |
| Fish larvae (duplicate) | 83.0% (534/643) → 83.0% (534/643) | 102 → 102 |
| Serhii brain | 83.7% (514/614) → 83.7% (514/614) | 58 → 58 |
| EDT brain | 67.6% (317/469) → 67.6% (317/469) | 19 → 19 |
| Lengthening spinal cord | 66.5% (381/573) → 66.5% (381/573) | 19 → 19 |
| V2 | 88.7% (408/460) → 88.7% (408/460) | 48 → 48 |
| Cells | 80.0% (440/550) → 80.0% (440/550) | 53 → 53 |
| ALD brain | 73.3% (396/540) → 73.3% (396/540) | 24 → 24 |
| Rat plasma | 90.2% (165/183) → 90.2% (165/183) | 71 → 71 |
| Rat spinal cord | 88.5% (461/521) → 88.5% (461/521) | 71 → 71 |
| Rat brain | 82.3% (443/538) → 82.3% (443/538) | 21 → 21 |
| Pex5 mouse | 89.1% (342/384) → 89.1% (342/384) | 46 → 46 |
| Total, 14 unique datasets | 78.9% (5,496/6,970) → 78.9% (5,496/6,970) | 721 → 721 |

## PE class only

### RP positive

| Dataset | Correct before → trial (% of manual list) | Extras before → trial |
| --- | ---: | ---: |
| Kidney summer | 91.8% (67/73) → 91.8% (67/73) | 4 → 5 |
| Heart | 94.5% (86/91) → 94.5% (86/91) | 9 → 9 |
| Worms | 97.0% (131/135) → 97.8% (132/135) | 3 → 3 |
| Fish larvae | 88.1% (89/101) → 88.1% (89/101) | 10 → 10 |
| Fish larvae (duplicate) | 88.1% (89/101) → 88.1% (89/101) | 10 → 10 |
| Serhii brain | 80.6% (58/72) → 80.6% (58/72) | 9 → 9 |
| EDT brain | 77.5% (55/71) → 77.5% (55/71) | 22 → 22 |
| Lengthening spinal cord | 88.7% (55/62) → 88.7% (55/62) | 33 → 34 |
| V2 | 93.4% (71/76) → 93.4% (71/76) | 14 → 14 |
| Cells | 90.6% (77/85) → 90.6% (77/85) | 19 → 19 |
| ALD brain | 90.4% (66/73) → 90.4% (66/73) | 29 → 32 |
| Rat plasma | 100.0% (20/20) → 100.0% (20/20) | 3 → 3 |
| Rat spinal cord | 95.5% (63/66) → 95.5% (63/66) | 13 → 13 |
| Rat brain | 98.7% (75/76) → 98.7% (75/76) | 15 → 15 |
| Pex5 mouse | 97.1% (68/70) → 97.1% (68/70) | 1 → 1 |
| Total, 14 unique datasets | 91.6% (981/1,071) → 91.7% (982/1,071) | 184 → 189 |

### RP negative

| Dataset | Correct before → trial (% of manual list) | Extras before → trial |
| --- | ---: | ---: |
| Kidney summer | 92.0% (80/87) → 92.0% (80/87) | 2 → 2 |
| Heart | 97.1% (99/102) → 97.1% (99/102) | 7 → 7 |
| Worms | 90.7% (127/140) → 90.7% (127/140) | 5 → 5 |
| Fish larvae | 91.6% (109/119) → 91.6% (109/119) | 8 → 8 |
| Fish larvae (duplicate) | 91.6% (109/119) → 91.6% (109/119) | 8 → 8 |
| Serhii brain | 97.6% (81/83) → 97.6% (81/83) | 7 → 7 |
| EDT brain | 93.2% (82/88) → 93.2% (82/88) | 11 → 11 |
| Lengthening spinal cord | 100.0% (84/84) → 100.0% (84/84) | 15 → 15 |
| V2 | 98.0% (96/98) → 98.0% (96/98) | 18 → 18 |
| Cells | 96.8% (90/93) → 96.8% (90/93) | 31 → 31 |
| ALD brain | 97.7% (86/88) → 97.7% (86/88) | 20 → 20 |
| Rat plasma | 100.0% (23/23) → 100.0% (23/23) | 2 → 2 |
| Rat spinal cord | 98.7% (76/77) → 98.7% (76/77) | 12 → 12 |
| Rat brain | 97.9% (95/97) → 97.9% (95/97) | 5 → 5 |
| Pex5 mouse | 98.9% (87/88) → 98.9% (87/88) | 1 → 1 |
| Total, 14 unique datasets | 95.9% (1,215/1,267) → 95.9% (1,215/1,267) | 144 → 144 |

### NP positive

| Dataset | Correct before → trial (% of manual list) | Extras before → trial |
| --- | ---: | ---: |
| Kidney summer | 97.4% (75/77) → 97.4% (75/77) | 5 → 5 |
| Heart | 95.1% (77/81) → 95.1% (77/81) | 6 → 6 |
| Worms | 100.0% (118/118) → 100.0% (118/118) | 16 → 16 |
| Fish larvae | 91.5% (75/82) → 91.5% (75/82) | 27 → 27 |
| Fish larvae (duplicate) | 91.5% (75/82) → 91.5% (75/82) | 27 → 27 |
| Serhii brain | 95.7% (66/69) → 95.7% (66/69) | 2 → 2 |
| EDT brain | 100.0% (51/51) → 100.0% (51/51) | 21 → 21 |
| Lengthening spinal cord | 100.0% (71/71) → 100.0% (71/71) | 14 → 14 |
| V2 | 98.9% (87/88) → 98.9% (87/88) | 4 → 4 |
| Cells | 97.0% (64/66) → 97.0% (64/66) | 0 → 0 |
| ALD brain | 95.3% (61/64) → 95.3% (61/64) | 4 → 4 |
| Rat plasma | 100.0% (19/19) → 100.0% (19/19) | 0 → 0 |
| Rat spinal cord | 98.2% (55/56) → 98.2% (55/56) | 12 → 12 |
| Rat brain | 96.3% (52/54) → 96.3% (52/54) | 12 → 12 |
| Pex5 mouse | 98.0% (48/49) → 98.0% (48/49) | 3 → 3 |
| Total, 14 unique datasets | 97.2% (919/945) → 97.2% (919/945) | 126 → 126 |

### NP negative

| Dataset | Correct before → trial (% of manual list) | Extras before → trial |
| --- | ---: | ---: |
| Kidney summer | 96.2% (76/79) → 96.2% (76/79) | 2 → 2 |
| Heart | 96.3% (79/82) → 96.3% (79/82) | 2 → 2 |
| Worms | 99.2% (118/119) → 99.2% (118/119) | 6 → 6 |
| Fish larvae | 91.2% (83/91) → 91.2% (83/91) | 5 → 5 |
| Fish larvae (duplicate) | 91.2% (83/91) → 91.2% (83/91) | 5 → 5 |
| Serhii brain | 98.6% (70/71) → 98.6% (70/71) | 1 → 1 |
| EDT brain | 98.3% (57/58) → 98.3% (57/58) | 1 → 1 |
| Lengthening spinal cord | 97.4% (75/77) → 97.4% (75/77) | 1 → 1 |
| V2 | 98.9% (87/88) → 98.9% (87/88) | 0 → 0 |
| Cells | 98.6% (69/70) → 98.6% (69/70) | 1 → 1 |
| ALD brain | 96.9% (62/64) → 96.9% (62/64) | 3 → 3 |
| Rat plasma | 100.0% (19/19) → 100.0% (19/19) | 1 → 1 |
| Rat spinal cord | 94.8% (55/58) → 94.8% (55/58) | 6 → 6 |
| Rat brain | 98.2% (56/57) → 98.2% (56/57) | 0 → 0 |
| Pex5 mouse | 97.0% (65/67) → 97.0% (65/67) | 3 → 3 |
| Total, 14 unique datasets | 97.1% (971/1,000) → 97.1% (971/1,000) | 32 → 32 |

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
