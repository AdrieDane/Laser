# Stage 6: experimental joint MS1 hypotheses

Authorized 6 November 2026. Software/internal comparison complete; current defaults,
accepted annotations and statistical decisions are preserved. Pending native
SIRIUS/MZmine activation, Binner and independently verified isotope gold remain
open. Stage 7 and historical annotation optimization have not started.

## Workflow and fixed model

In Feature-evidence settings choose complete isotope patterns, lipid-aware
candidates, then **Joint MS1 hypotheses (experimental)**. Optional **Use msbuddy
alongside LASER** uses a separately configured interpreter. Detached jobs retain
progress/logs when the browser closes. Native failures are explicit; LASER results
remain saved with a completion warning. No account or automatic download is used.

New tables retain complete hypotheses, native cases/candidates, physical links,
supporting features, component budgets, resolved calls and cross-phase alternatives.
Old statistics/annotations/tables are unchanged. The viewer uses saved joint
tables, eight alternatives and 24 support records per query; clicking a point runs
no prediction. Margins/statuses are heuristic, never accuracy probabilities.

Candidate generation uses Stage 5 lipid backbones, public formulas and broader
de-novo CHNOPS. msbuddy 1.0.0 gets a separate case for **every allowed ion hypothesis**,
not a reference adduct or charge. All native cases/statuses remain saved. Formula
unions obey the same elemental, valence, mass and ion-composition checks. Prohibited
ions remain excluded. Thirty-two source isotope slots reserve two candidates per
ion, then fill by the structure/mass prescore. Unscored hypotheses are retained.
This budget allocation differs from Stage 5 and must be included in attribution.

The correlated spectral block averages complete ion isotope/mass cost (weight
0.75) and native within-adduct rank cost (0.25). Native cost is log2(rank), capped
at 3; an absent native candidate has cost 3. Unavailable native cases use the
complete-ion cost and are explicitly flagged. The original structure/oxidation
prior remains. A 0.35 chemistry cost uses method preferences and possible public
template classes, not reference or accepted lipid identities. Method-aware
preferences are local to joint scoring; acquisition metadata is not edited.

All original coordinates supply physical support, including non-significant and
accepted features. Initial mass-pair discovery uses twice the tolerance, then
each conditional formula requires every target inside the original tolerance.
Profiles require six uniquely paired biological specimens, Spearman >=0.6 and
within-group correlation >=0.2. QC/blanks and group shifts alone cannot support a
link. These are area profiles, not chromatographic shapes. Up to two candidates
per mode/adduct are isotope-scored; additional unscored targets or competing costs
within 0.25 cannot give unique support.

Complete coelution spans use existing peak-width/cap rules independently within
RP and NP. Water-loss assignments require a same-mode intact parent in the actual
family. No cross-chromatography RT or RT-library prediction is invented. Local
context reward is capped by mode (weight 0.75); other-phase support weight is 0.40.
Reciprocal query evidence is half-weighted. Homologous grids remain review context,
never compound identity unions. Mass errors are tolerance-normalized, not fitted
measurement uncertainties; no calibration evidence is fabricated.

Coupled local alternatives use three direct choices per root, explicit no-call
cost 3, beam width 64 and maximum 32 roots/component. Conflicting formula/ion labels
cannot coexist in a state. Oversize components abstain instead of being split.
Margins below 0.25 abstain; pruned searches with no retained competing state also
abstain. Margins refer only to retained states. All budgets/alternatives are saved.
Groups require bounded neutral/RT spans and one original ion per mode/adduct.
RP/NP groups are first independent; cross-phase unions require unique reciprocal,
formula-compatible/profile-supported families. One-to-many associations stay
separate. Formula equality alone never links compounds.

## Optional native setup

Register an existing supported Python 3.9–3.12 msbuddy 1.0.0 environment:

```sh
python tools/configure_msbuddy.py --python /path/to/msbuddy-env/bin/python \
  --data /path/to/msbuddy-data --libomp /path/to/libomp.dylib
```

Omit `--libomp` where unnecessary. Configuration belongs to per-user application
settings, or `LASER_MSBUDDY_CONFIG`, never source. Executable, package versions,
installed msbuddy source, worker, three pinned models and optional OpenMP hashes
are frozen at launch and verified around execution. The venv path is preserved.
Model hashes are checked before native joblib loading. Only the child receives
its loader/thread settings; no system library is installed.

## Frozen comparison

Retain all 3413 Stage 5 natural formula queries (RP positive 1592 / RP negative 585 / NP positive 724 / NP negative 512),
all 3668 reference rows and all 252 class/mode cells. Same ALD development acquisition,
not independent validation. No MS/MS exists. Freeze/push source/protocol before real
scoring. Predefined arms: sequential selector on the same query keys, joint LASER,
joint LASER+msbuddy. The immutable Stage 5 blind top-1 baseline is 1032/3413.

Separate forced top-1/top-5 from resolved recovery, conditional agreement and
abstentions. Report generated/scored/direct-option truth retention, library exposure,
wrong adducts, original-peak choices, phase links, purity, unconfirmed extras and
runtime. Audit actual JavaScript sample-bar representatives. Model timing is not
end-to-end Plotly/browser latency. Unsupported/failed cases remain in denominators.

The 3413-query replay is broader than the original workflow's 762 raw-p queries;
keep the saved-current replay separately and call all-manual readouts recovery,
not identification accuracy. Curated truth covers H+, NH4+, H- and negative charge
two only. Other adducts lack independently verified gold here. Ions absent from
curation are unconfirmed extras, not proven false positives.

Publish safe aggregates only. Keep all attempts, gains and losses; no post-hoc
weights/threshold/budget tuning or per-mode winner splicing. No default promotion
from this one development acquisition. Stop for review before Stage 7.


## Completed internal comparison — 6 November 2026

**Decision: keep CURRENT as the default.** Forced formula/adduct ranking improved,
but the resolved joint model recovered fewer reference counterparts and abstained
more often. The option remains experimental; it is not a validated identification
method or a replacement for the current grouping. Stage 7 has not started.

Inference source 8f276b0; display-only compatibility source 2a2e4fd;
persistence fix e6aeac8. One ALD development acquisition reused from earlier
stages, saved-result replays rather than new raw annotation/statistical runs.
No independent validation or analytical FDR estimate. All 63 curated class labels
× four modes are retained, including empty formula-query cells. Gold formulas,
adducts, charges and class names were not supplied to the prediction workers.

### Original-feature/adduct agreement

Percentages below use all manual rows in the mode, including unresolved mappings.
These are reference ion/adduct agreement rates at the original feature, not
confirmed lipid-identity accuracy. The matched sequential comparator uses the
same 3413 query keys as both joint arms; the historical 762-query default replay is
kept separately. Do not compare their coverages as though their scopes matched.

| Mode | Matched sequential | Joint LASER | Joint + msbuddy | Unconfirmed extras: sequential / LASER / native |
| --- | --- | --- | --- | --- |
| RP_Pos | 46.6% (790/1694) | 45.0% (763/1694) | 52.5% (889/1694) | 1201 / 828 / 897 |
| RP_Neg | 48.3% (328/679) | 28.7% (195/679) | 39.9% (271/679) | 1102 / 764 / 818 |
| NP_Pos | 66.8% (506/757) | 44.0% (333/757) | 51.7% (391/757) | 882 / 536 / 624 |
| NP_Neg | 67.7% (364/538) | 46.1% (248/538) | 61.3% (330/538) | 908 / 545 / 619 |

Both joint arms reduce unconfirmed assigned extras; this does not prove that the
removed ions were false positives. Matched sequential total original-ion/adduct
recovery is 1988/3668, joint LASER 1539/3668 and native-joint 1881/3668.
Native-joint gains 99 RP-positive matches but loses 57 RP-negative,115 NP-positive
and 34 NP-negative net matches. All per-query gains/losses remain in the private
report, including those hidden by a net change. Conditional agreement among the
remaining assignments improves while coverage falls; neither measure substitutes
for the full reference denominator.

### Forced formula/adduct ranking and resolved calls

| Mode | Natural queries | Stage 5 blind top 1 | Joint LASER top 1 | Joint + msbuddy top 1 |
| --- | --- | --- | --- | --- |
| RP_Pos | 1592 | 53.7% (855/1592) | 85.4% (1360/1592) | 87.4% (1391/1592) |
| RP_Neg | 585 | 12.3% (72/585) | 49.1% (287/585) | 60.2% (352/585) |
| NP_Pos | 724 | 6.8% (49/724) | 72.5% (525/724) | 75.7% (548/724) |
| NP_Neg | 512 | 10.9% (56/512) | 69.1% (354/512) | 78.3% (401/512) |

Forced local top 1 totals: Stage 5 baseline 1032/3413 (30.2%), joint LASER 2526/3413
(74.0%), native-joint 2692/3413 (78.9%). Joint top 5: 3280/3413 and 3332/3413.
Generated and isotope-ranked reference formula/adduct retention 3409/3413 in both
arms. No retained reference hypothesis was lost to the 32-source scoring budget.
However, reserving two slots per ion differs from Stage 5; these gains cannot be
attributed to joint context alone. The correct direct top-three option is present
for 3198/3413 and 3287/3413 queries. All 1,991,792 generated hypotheses per arm remain
saved;2953 sources reach the explicit scoring budget. No component exceeded the
32-root limit;4/6 components pruned beam states.

Resolved formula **and** adduct are correct for 1516/3413 (LASER) and 1865/3413
(native), with 1696/2022 assigned roots respectively. Forced ranking is therefore
not the same outcome as global assignment, and an abstention is not a successful
identification. Heuristic margins are not calibrated accuracy probabilities.
The existing LASER formula-library overlap and 127 library-absent cases are
reported separately; absence from LASER does not imply absence from native data.

### Counterparts, peaks and PC regression

Reference-pair recovery: sequential 446/960; LASER 153/960; native 217/960.
Conditional pair agreement:446/659,153/196,217/269. Discordant curated pairs:
213,43,52. Groups with multiple curated ions and mixed curator names:224,49,55.
These are curation-consistency diagnostics, not independent molecular validation.
One original peak per mode/adduct, neutral mass spans and complete phase-specific
RT bounds were verified independently on the saved outputs. All 254324 original
coordinates match exactly. Water losses retain same-mode intact-parent support.

The matched broad replay’s 14 expected positive PC counterparts fall from 9 to 2
(LASER) or 3 (native). In the normal 762-query workflow, the native-joint result is
4/14 versus the saved-current 10/14. Thus the regression is not explained solely
by the broader diagnostic scope. All 15 PC sources × four mode slots remain in
the private report. PC(50:0) lacks a reference formula and is outside the 3413-query
formula experiment; it has no expected other-positive counterpart and does not
change the 14-case primary denominator. The production replay includes it.
There is no negative-mode PC identity truth in that regression; negative PC links
remain unconfirmed. Unassigned adducts are reported separately from wrong ones.

### Native coverage, runtime and saved-project preservation

msbuddy 1.0.0 ran 18162 blind ion cases: 17766 completed, 396 outside its precursor
range. Every supported case completed; unsupported cases remain explicit, with
the LASER evidence fallback. The native candidate union adds no unique hypotheses
to this particular broad LASER pool, but its within-ion ranking changes costs.
Pinned models, package/source/binary hashes and Python 3.12.12 were verified.
A durable per-user installation passes 4/4 synthetic controls; no account or study
upload was needed. Synthetic controls are not study identification accuracy.

Native-joint evidence computation:691.36 seconds for 3413 queries. The normal
762-query evidence overlay took 239.75 seconds, including 4187 native cases.
LASER-only inference timing is unavailable because its original display path
failed before timing was saved. Model-only selection medians are 0.34/0.51 ms
(LASER/native); these exclude Plotly rendering and do not establish end-to-end
subsecond browser latency. No Safari automation was used.

All original statistical/legacy tables were exactly unchanged by the production
overlay, including 251431 statistical rows and 2915 previously saved compound rows.
The new typed project artifacts now preserve text feature IDs, empty labels and
full precision directly from the original frames. The real 762-query viewer
payload is exactly identical before saving and after reopening; the evidence
predictions were not rerun to fix serialization. CSV exports remain portable.
All 94 prepared inputs, including 68 raw files, IS and method metadata, retain
their hashes; the inputs-only project's result folders remain empty.

Full application checks:1086 passed,9 skipped,91 subtests passed. Initial failed
attempts are preserved: invalid water-loss composition, a preservation checker
that included new empty placeholders, and integer/string display-key joining.
The final presentation replay preserved scientific output hashes and used
identical inference files and selector JavaScript. No post-hoc weight, threshold,
budget or class/mode winner tuning was performed.

### Aggregate files and remaining work

- [All-class and class/mode before/after](all_class_mode_before_after.csv)
- [Coverage, conditional agreement and extras](all_class_mode_adduct_recovery.csv)
- [Forced formula/adduct ranks](all_class_mode_forced_ranking.csv)
- [Resolved formulas and abstentions](all_class_mode_resolved_formula.csv)
- [Truth retention](all_class_mode_truth_retention.csv)
- [Resolved gains/losses](all_class_mode_resolved_transitions.csv)
- [Forced-rank gains/losses](all_class_mode_forced_transitions.csv)
- [Viewer choices and different peaks](all_class_mode_viewer_choices.csv)
- [Budgets](candidate_component_budgets.csv), [native coverage](native_case_coverage.csv),
  [group consistency](curated_group_purity.csv), [timing](runtime.csv), [summary](summary.json)

Class-specific extras and total runtime class assignments remain unavailable:
runtime hypotheses do not establish a lipid-class identity for unconfirmed ions.
Extras here require a nonempty ion assignment. The old frozen saved-current
report also counted 219 unassigned unconfirmed display rows; its historical values
remain preserved separately. This distinction does not change matched sequential
extras, because its unassigned rows are all at reference-query keys.

Private report identifier: ALD_stage6_003. Detailed study predictions, feature
crosswalks, source/input hashes and all PC mode slots remain outside Git.
Verified isotope-member gold, independent acquisitions, Binner and native
SIRIUS/MZmine study comparisons remain open. Stage 7 raw chromatographic traces
are the next evidence route; do not start it automatically from this checkpoint.
