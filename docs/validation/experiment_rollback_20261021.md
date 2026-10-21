# Rejected RP positive approaches and rollback

All three approaches from the last RP-positive round are rejected at the user's
request. The first two were never deployed. The third, four-stream consensus,
is removed from runtime; its source/test modules remain recoverable in Git.
Historical reports are retained. The earlier optional PE correction from
`a06782f`, including its NP-negative improvement, remains. Cache version28.

Before is `02270eb`/`9e3fce9` with the experimental option on; after reproduces
`a06782f` with that same option on. All15 saved original-feature/core outputs
were replayed and exactly matched the earlier selected feature IDs. This is
not15 new full raw-input runs. Fish and Serhii original folders remain absent;
retained feature tables are used. These examples are development data.

Correct means the same lipid, adduct and ORIGINAL peak as manual curation.
Extras mean absent from the corresponding manual list, not proven false.
The duplicate fish folder is shown and excluded from every total. Exactly3
correct RP-positive calls are intentionally removed: Serhii PE38:1, V2 PE38:5,
Cells PE40:5. No other choices change. All-class correct34351→34348,
extras3652→3652, different-peak534→534. PE correct4089→4086, extras486→486,
different-peak74→74. This rollback is separate from the calibration experiment.

## All lipid classes

### RP positive

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 1,165 → 1,165 | 76 → 76 |
| Heart | 1,329 → 1,329 | 75 → 75 |
| Worms | 1,144 → 1,144 | 127 → 127 |
| Fish larvae | 680 → 680 | 31 → 31 |
| Fish larvae (duplicate) | 680 → 680 | 31 → 31 |
| Serhii brain | 587 → 586 | 31 → 31 |
| EDT brain | 1,338 → 1,338 | 105 → 105 |
| Lengthening spinal cord | 1,267 → 1,267 | 151 → 151 |
| V2 | 1,295 → 1,294 | 60 → 60 |
| Cells | 1,205 → 1,204 | 65 → 65 |
| ALD brain | 1,271 → 1,271 | 88 → 88 |
| Rat plasma | 959 → 959 | 23 → 23 |
| Rat spinal cord | 1,194 → 1,194 | 107 → 107 |
| Rat brain | 1,452 → 1,452 | 66 → 66 |
| Pex5 mouse | 949 → 949 | 28 → 28 |
| **Total, 14 unique datasets** | **15,835 → 15,832** | **1,033 → 1,033** |

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
| **Total, 14 unique datasets** | **5,483 → 5,483** | **599 → 599** |

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
| **Total, 14 unique datasets** | **7,537 → 7,537** | **1,299 → 1,299** |

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
| **Total, 14 unique datasets** | **5,496 → 5,496** | **721 → 721** |

## PE only

### RP positive

| Dataset | Correct before → after | Extras before → after |
| --- | ---: | ---: |
| Kidney summer | 67 → 67 | 4 → 4 |
| Heart | 86 → 86 | 9 → 9 |
| Worms | 131 → 131 | 3 → 3 |
| Fish larvae | 89 → 89 | 10 → 10 |
| Fish larvae (duplicate) | 89 → 89 | 10 → 10 |
| Serhii brain | 59 → 58 | 9 → 9 |
| EDT brain | 55 → 55 | 22 → 22 |
| Lengthening spinal cord | 55 → 55 | 33 → 33 |
| V2 | 72 → 71 | 14 → 14 |
| Cells | 78 → 77 | 19 → 19 |
| ALD brain | 66 → 66 | 29 → 29 |
| Rat plasma | 20 → 20 | 3 → 3 |
| Rat spinal cord | 63 → 63 | 13 → 13 |
| Rat brain | 75 → 75 | 15 → 15 |
| Pex5 mouse | 68 → 68 | 1 → 1 |
| **Total, 14 unique datasets** | **984 → 981** | **184 → 184** |

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
| **Total, 14 unique datasets** | **1,215 → 1,215** | **144 → 144** |

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
| **Total, 14 unique datasets** | **919 → 919** | **126 → 126** |

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
| **Total, 14 unique datasets** | **971 → 971** | **32 → 32** |

## Numerical details

The [full CSV](experiment_rollback_20261021.csv) records all-class and PE counts,
manual denominators, match percentages, wrong-peak choices, total calls and
manual entries not on the same peak. The latter includes unresolved mappings
and wrong-peak choices; categories must not be added as disjoint errors.
All-class percentages use reliably mapped entries. Original manual-row counts
are also retained. After restoration392 tests and83 subtests pass. No previous
TXT exports are regenerated. The calibration experiment starts from this
restored checkpoint, not the rejected approaches.
