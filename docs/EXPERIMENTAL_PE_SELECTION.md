# Experimental PE peak correction

The optional PE correction compares competing original peaks using mass,
local retention time, sample consistency and possible related-ion evidence.
It improves NP-negative coverage on the development examples without changing
other lipid classes. It is **off by default**; the existing annotation workflow
remains available unchanged. See the [measured results](validation/experimental_pe_20261021.md)
before enabling it.

This concerns ordinary diacyl PE, not PE-O, PE-P or positional isomers. An
accepted annotation is a software selection, not proof of molecular structure.

## Enable the experiment

In Annotation settings, select `Experimental PE peak correction`. Keep the
post-regression and PE-completion settings enabled: they are prerequisites.
The option uses the MATLAB-equivalent engine and runs after its existing
phase models, PE completion and independent declared-standard safeguard.

For the command line:

```sh
laser-standard /path/to/project --engine matlab --phase both --experimental-pe --output /path/to/new/results
```

For the workflow API, set `use_experimental_pe_selector=True` in the request
settings. The cache key and saved settings include the option. Existing
projects without that setting remain off. Enabling it requires a new
annotation run; opening an old result does not silently change it.

The TXT export contract is unchanged: add only `laser_annotation` in column 2
and `adduct` in column 3, retain every original field and row, and use the
`_annotated.txt` suffix. This checkpoint does not regenerate previous exports.

## Evidence and selection

Candidate discovery reuses the bounded PE pool and user composition ranges.
It does not enlarge carbon or unsaturation limits. Discovery searches within
40 seconds by default, with a maximum configurable search of 60 seconds;
acceptance still requires corrected mass error strictly below the configured
tolerance, at most 3 ppm, and local RT error at most 10 seconds. At least five
other local anchors and four usable sample measurements are required.

New descriptive evidence includes robust median abundance, detection fraction,
sample outliers, biological-to-blank contrast and pooled-QC relative variation.
Explicit sample roles take priority; otherwise validated group labels identify
blanks, QC pools and standards. Unknown roles or missing measurements stay
unknown. QC variation requires at least three QC columns.

Possible singly charged carbon-isotope neighbours must have the expected mass
spacing, coelute within two seconds and have correlated intensity profiles
across at least four usable measurements. Positive-mode ammonium/sodium
relationships are also descriptive model inputs. These are possible ion
relationships, not confirmed isotope or adduct identities. The carbon-13
spacing follows [NIST isotope masses](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=C).
Combining mass, RT and intensity profiles has precedent in
[CAMERA](https://pmc.ncbi.nlm.nih.gov/articles/PMC3658281/); the implementation
here uses peak lists, not CAMERA or measured chromatographic shapes.

Separate portable pairwise models rank competing peaks for RP positive,
RP negative and NP negative. Feature IDs, dataset names, manual labels and
previous selection flags are not model inputs. Preference scores are neither
accuracy probabilities nor false-discovery rates. Runtime does not read manual
curation or require scikit-learn.

An existing automatic PE can change peaks only with preference at least 0.90,
overall rank at least 0.65, no mass deterioration beyond 0.2 ppm and no local
RT deterioration beyond one second. RP positive additionally requires both
a mass improvement of at least 0.3 ppm and an RT improvement of at least two
seconds. RP negative and NP negative require either improvement. Strong
evidence that the proposed peak is an isotope child blocks the change.

Only NP negative can gain new identities. These additionally require mass
error below 2.5 ppm, local RT error at most five seconds, detection in at least
half the usable samples, and either a plausible upper carbon-isotope envelope
or an already accepted NP-positive peak for the same PE within ten seconds.
When biological and blank roles are available, the ratio of their
one-plus-medians must be at least three. Missing blank information is recorded
as unknown, not a demonstrated blank pass. A model rank alone cannot admit
a new identity. Stricter user mass/RT tolerances remain binding.

Opposite-stream annotations are snapshotted before selection to avoid circular
additions. This is same-identity RT corroboration, not validated specimen
matching or a correlation of paired samples across modes. Missing opposite
mode evidence does not automatically reject an existing annotation.

## Protected choices and recorded decisions

Other classes and their occupied features, declared standards, names ending
in ` (IS)`, and explicitly marked manual choices are protected. The programmatic
manual-lock contract recognizes `manual_confirmed`, `manual_override` and
`annotation_source == "manual"`; callers must retain those fields. Multiple
existing choices for an identity are preserved rather than forcibly collapsed.
NP positive is unchanged. No peak is created to fill a grid position.

New choices are full accepted annotations with status
`pe_experimental_supported`, explicit independent-evidence approval and an
explicit grid-coherence exemption. Historical grid scores, rejection flags,
original m/z/RT and feature IDs remain available. The grid displays the
actually accepted original feature. Replaced automatic candidates lose their
accepted flags but stay in the candidate history.

Audit tables expose the model hash, proposed and previous feature, mass/RT
changes, supporting evidence, gate outcomes and final choice:
`matlab_experimental_pe_audit`, `matlab_experimental_pe_changes` and
`matlab_experimental_pe_decisions`. The UI audit panel and result tables expose
these records. An unavailable or invalid model leaves the preceding choices
unchanged for that stream and records `experiment_unavailable`.

## Calibration and remaining work

`tools/calibrate_experimental_pe.py` fits a requested stream from explicitly
chosen trusted candidate caches and manual crosswalks. It requires
`--trust-local-pickles` because pickle inputs can execute code. Install the
optional `calibration` dependencies to refit; they are not runtime dependencies.
Only reliably mapped ordinary PE identities with both a manual peak and
competing peaks supply training pairs. Identities absent from the manual list
are not treated as proven false annotations. Identity weights prevent large
candidate groups from dominating calibration. Source hashes and training
cohorts are recorded in each shipped model.

The first models use rat brain, plasma, spinal cord and Pex5 mouse. Four-fold
checks refit only this selector, not the preceding pipeline. All examples
informed development, including tightening the RP-positive guard after a
dataset-level regression. These are development comparisons, not untouched
external validation. Actual EIC shape/integration, broader charge grouping,
matched-specimen cross-mode evidence, a joint multi-class resolver and fully
refitted nested validation remain future work.
