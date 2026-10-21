# Manual annotation comparison

The evaluation-only comparison tool checks lipid name, adduct, measurement
stream, internal-standard identity and original XCMS feature. It reads existing
LASER annotations and an independently prepared manual-to-feature crosswalk.
It does not run annotation, fit models or change any selection settings.

## Compare saved outputs

Use matching dataset subfolders under the run and curation roots. Each run
contains an annotations CSV. Each curation folder contains
`curated_feature_crosswalk.csv`, with these required columns:

```text
stream,matlab_species,matlab_class,matlab_adduct,xcms_feature_id,crosswalk_status
```

`profile_match` means the manual entry has been reliably linked to an original
feature. Other mapping statuses remain unresolved, even if a possible feature
ID is present. An optional `matlab_is_internal_standard` flag distinguishes
declared standards from endogenous lipids with the same composition.

The LASER annotations CSV requires:

```text
stream,name,class,adduct,feat_feature_id
```

It can also include `is_is`, acceptance/status fields and recorded checks.
The supplied CSV represents final selection. Candidate checks never override
it during evaluation.

```bash
python tools/compare_manual_annotations.py \
  --runs-root /path/to/runs \
  --curation-root /path/to/manual_crosswalks \
  --annotations-name final_automatic_annotations.csv \
  --exclude-from-pooled duplicate_dataset_name \
  --output /path/to/new_comparison
```

`--dataset name` selects one dataset and can be repeated. Exclusions are
explicit and must name selected datasets. Excluded datasets retain their
individual reports but do not contribute to pooled results. Duplicate datasets
are not guessed from folder names. The output folder must be new or empty.

## Optional candidate evidence

`--candidates-name candidates.csv` supplies the final candidate table in every
selected run folder. `--pe-decisions-name full_candidate_decisions.csv` supplies
the optional PE discovery decisions; this is its default filename. Missing PE
files remain unknown. Candidate CSVs use the LASER identity columns above.

For existing trusted local LASER result dictionaries, candidate tables can
instead be read from a pickle:

```bash
python tools/compare_manual_annotations.py \
  --runs-root /path/to/full_runs \
  --curation-root /path/to/manual_crosswalks \
  --result-name full_annotation.pkl \
  --trust-local-pickles \
  --annotation-revision KNOWN_ANNOTATION_COMMIT \
  --exclude-from-pooled duplicate_dataset_name \
  --output /path/to/new_comparison
```

Never enable the pickle option for untrusted or downloaded files. Pickle can
execute code. The trust flag is mandatory, and no pickle is read without it.
CSV inputs do not require pickle. Do not supply both candidate-source options.

Presence in the final or PE discovery table is an observation. Absence is not
proof that the candidate was never considered earlier. Explicit failed checks
can coexist with a later accepted decision and do not prove why selection
failed. Unknown values are not passes. PE scores and local-support counts are
reported separately; a recorded `below_pe_score` label with fewer than five
supporting anchors must not be interpreted as a pure score-threshold failure.
The five-anchor field describes the current PE method, not a new universal rule.

## Read the results

`overview.html` gives a plain-language summary. The detailed exports are:

- `manual_comparison.csv`: every manual entry, selection outcome, original
  feature, conflicting choices and available candidate evidence.
- `manual_disagreements.csv`: entries not annotated on the same original peak,
  including unresolved manual mappings.
- `laser_comparison.csv`: every LASER call, including names absent from the
  manual list and identities whose manual peak is unresolved.
- `by_dataset_stream.csv` and `by_dataset_stream_class.csv`: individual results.
- `pooled_by_stream.csv` and `pooled_by_stream_class.csv`: explicitly included
  datasets only.
- `validation_manifest.json`: input hashes, source artifacts, excluded
  datasets, evaluation source revision/hashes and output hashes. A supplied
  annotation revision is recorded as operator-declared, not inferred from an
  old pickle or from the current checkout.

The two disagreement flags can overlap. LASER can put the manual identity on
another peak while putting another identity on the manual peak. Do not add
these two counts as though they represented separate missed entries.

`percent_mapped_manual_same_peak` uses unique, reliably mapped manual
annotations. Duplicate rows are retained and reported but cannot inflate this
percentage. `percent_all_manual_rows_same_peak` uses every original manual row
and counts unresolved mappings as misses. An absent denominator is unavailable,
not zero percent. Both measures preserve standard identities. Positional-lyso
spacing aliases and harmless numeric `.0` suffixes are normalized; leading
zeros, different isomers and ether/plasmalogen identities are not collapsed.

Names absent from the manual list are unconfirmed additions, not established
false lipids. Candidate availability is not an estimated identification accuracy.
Development examples cannot establish prospective external performance.

## Next instrumentation

This first implementation observes saved selection and candidate evidence.
Recording each actual discovery, gate and replacement event during future
annotation runs is a separate next step. Without those events, the report must
not invent a complete pipeline rejection history.
