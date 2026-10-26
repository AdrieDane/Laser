Declared-standard mass-trend review — no annotation changes

Scope: all33active datasets,132dataset/mode cases,30project groups. Independent
known-spike measurements from027, not curator-picked endogenous annotations.
All original feature ppm values and declared actual labelled masses checked;
all scalar/vector coefficient and prediction checks passed.1364leave-neutral-
out predictions. No manual comparison tables, new annotation scores, candidate
windows, annotation cutoffs, calls or app defaults changed. Separate from048.

Question: does a fixed robust mass-dependent ppm line predict OTHER known
standards more accurately than an unconditional constant ppm median?
Each neutral votes once; all its ions are withheld together. Compare only
interior held-out masses when considering conservative future use:

Mode   | Usable cases | Trend better | Constant better | Median case improvement
RPpos  | 32/33        | 19           | 13              |  0.0307ppm
RPneg  | 31/33        | 18           | 13              |  0.0231ppm
NPpos  | 32/33        | 25           |  7              |  0.0514ppm
NPneg  | 31/33        | 12           | 19              | -0.0118ppm

Positive improvement means a lower median absolute OTHER-standard prediction
error, NOT more correct lipid annotations. Unsupported cases remain unavailable,
not zero error. The CSV files also retain all-LOO results including endpoints,
which are extrapolation diagnostics, not authorized calibration corrections.

Decision: defer a standalone annotation-changing rule. Some modes show a small
trend, but changes are small and inconsistent; NPnegative generally worsens.
These results alone do not justify wider mass windows or a claim of annotation
improvement. No favorable per-mode correction is spliced into the active048run.

Limitations: existing anchors were ALREADY selected with absppm<3 and reference
RT restrictions, so this is conditional, censored standard evidence. Neutral
group medians approximate multiple adducts. Standards at different RTs can mix
mass and chromatographic/adduct effects. These are cached original-feature
measurements, not fresh raw-scan processing or external validation. An eventual
new candidate experiment needs all363proper nested standard-reference masks;
outer-only027 caches cannot enter INNER scores.

Primary motivation, not evidence that our proposed slope works: Wenger etal2010
compared proportional and linear internal Orbitrap calibration and favoured
the proportional approach in their single-calibrant proteomics implementation:
https://pmc.ncbi.nlm.nih.gov/articles/PMC2871411/
The inference tested here is our own descriptive mass-trend hypothesis, not
a result established by that paper for these lipid peak lists.

Frozen source plan: docs/validation/next_standard_mass_trend_diagnostic.txt.
Original private receipt SHA256:
5e01ccac0d7964efd2596eaa9583a47b071e6a342acf8400bdcc386a0b105e8b
Only aggregate summaries are published; no neutral IDs, raw feature IDs,
per-standard prediction tables, private input paths or models are included.
