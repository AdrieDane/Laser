# Stage 6: experimental joint MS1 hypotheses

Authorized 6 November 2026. Implementation under validation; current defaults,
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

Retain all 3413 Stage 5 natural formula queries (RPpos1592/RPneg585/NPpos724/NPneg512),
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
