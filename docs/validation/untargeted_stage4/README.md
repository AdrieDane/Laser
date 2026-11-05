# Stage 4: complete isotope patterns and measurement quality

5 November 2026. Implementation under validation. Current isotope evidence and
current grouping remain the defaults. Stage 5 has not started. Binner's separate
Stage 3 native benchmark remains pending.

## Implementation and scope

**Feature-evidence search settings → Isotope model** offers an experimental
complete-ion model. It adds saved candidate scores, component observations and
quality summaries. Existing statistical values, accepted annotations, compound
assignments and legacy formula labels are preserved. The viewer reads the saved
summary; point selection does not run an isotope search.

[IsoSpecPy](https://github.com/MatteoLacki/IsoSpec) **2.6.1** generates fine natural
isotopologues to 99.999% cumulative probability. Every allowed ion includes its
complete elemental delta, signed electron correction and charge. Formate/acetate
carbon, ammonium nitrogen/hydrogen, chloride and potassium isotopes are included.
The two user-prohibited ions remain unsupported. Labelled formulas and known
internal-standard annotations are excluded from natural-abundance scoring.
Unrecognized labelled features cannot be identified reliably from areas alone.

The full calculation is retained in memory; components above 0.1% of the parent
are searched. A bounded two-FWHM span approximates centroid merging within each
nominal isotope. This is an explicit approximation, not an instrument line-shape
fit or proof of experimentally resolved fine structure. Unknown instrument laws
or resolution references use nominal aggregation and receive an uncertainty flag.

The [Q Exactive Plus specification](https://documents.thermofisher.com/TFS-Assets/CMD/Specification-Sheets/PS-63912-LC-MS-Q-Exactive-Plus-Orbitrap-PS63912-EN.pdf)
defines its resolving power at m/z 200. For that named model, a missing reference
is inferred from the manufacturer specification and recorded; project metadata
is not overwritten. The [Orbitrap resolution law](https://assets.thermofisher.com/TFS-Assets/CMD/Technical-Notes/tn-64287-hram-orbitrap-ms-tn64287-en.pdf)
scales local resolving power inversely with the square root of m/z. Generic
instruments do not inherit this assumption. OpenMS's
[fine/coarse distribution distinction](https://openms.de/documentation/html/classOpenMS_1_1IsotopePatternGenerator.html)
was reviewed; no native OpenMS performance benchmark is claimed.

Matching uses mass error, capped peak-width/coelution windows and per-sample
ratios. Biological specimens supply primary technical ratio evidence; QC provides
a separate diagnostic and fallback when fewer than three biological pairs exist.
Blanks and standard injections supply neither biological nor isotope-ratio fits.
At least three positive finite pairs and log-ratio MAD at most 0.5 are needed for
an intensity fit. Ratios within a factor of two are called compatible; this is a
heuristic, not a calibrated probability. Competing feature alternatives, shared
components, variable ratios, possible lower monoisotopic parents and missing peaks
remain recorded. Mass and intensity errors contribute jointly to the score.

The fifth percentile of positive sample feature areas is used only as a missingness
proxy. It is not a measured noise floor or LOD. Saturation is acted on only when
an explicit feature flag exists; otherwise it is recorded as unassessable from
integrated areas. Missing/noisy M+1 does not eliminate a hypothesis. Existing
candidate formulas are supplemented by a mass-only search of the existing public
formula library, with its SHA256 recorded. This is not a new de-novo formula
generator or the Stage 5 external-engine comparison. All candidates are retained;
the best additive score does not replace the legacy formula/adduct assignment.

## Evaluation contract

`tools/evaluate_complete_isotopes.py` freezes committed source and protocol before
comparing current and complete evidence. Workers receive no manual formulas or
identities. Every original statistical/compound value must remain identical, and
current compound assignments must reproduce the Stage 1 replay. Detailed study
rows and source/input fingerprints stay outside Git.

Formula candidate recall and joint top-1/top-5 are measured on the SAME mapped,
unambiguous, formula-bearing raw-p queries in both arms. All four modes and all
252 reference class/mode cells remain visible, including zero-query cells.
These are development measurements with prior library exposure, not external
validation. Expanded candidate pools and isotope reranking are reported together;
any gains cannot be attributed solely to the isotope calculator.

A separate all-reference readout supplies the curated formula/adduct to inspect
conditional envelope quality. It retains all 3,668 reference rows, with explicit
unmapped, formula-missing, labelled, unsupported and mass-incompatible outcomes.
Those reference hypotheses never enter the prediction workers. The raw check
examines apex centroid snapshots from the three QC injections per mode within
±0.10 min of each original feature RT. It records absent/ambiguous centroids and
raw ratios, without fitting a trace model or treating an absent centroid as a
demonstrated false prediction. This limited measurement check is not the Stage 7
trace-evidence workflow.

**No independently verified isotope-member gold set exists for this acquisition.**
Consequently real isotope-recognition precision/recall must remain unavailable.
Controlled cases, reference-conditioned fits and raw QC observations must not be
substituted for those metrics. The implementation can be published while the
verified-envelope validation requirement remains open. Formula recall on the
declared real queries is still reported. Do not promote the experimental model
or claim Stage 4 fully validated from synthetic tests alone.
