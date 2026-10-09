# Editable lipid MS/MS rules

`lipid_msms_rules.json` is the machine-readable source of truth for the
class-specific annotation rules. Both the programmable engine and the GUI load
this file; do not duplicate scientific rules in Plotly or Streamlit code.

## Mandatory class rules

| Classes | Evidence type | Required value |
|---|---|---:|
| LPC, PC, SM, PC-O, PC-P, LPC-O, LPC-P | product ion: phosphocholine headgroup | m/z 184.0733 |
| CE | product ion: cholestenyl/cholestadienyl cation | m/z 369.3516 |
| PS | neutral loss: phosphoserine | 185.0089 Da |
| HexCer | neutral loss: hexose | 162.0528 Da |
| AC | product ion: acylcarnitine fragment | m/z 85.0284 |
| PE, LPE, PE-O, PE-P, LPE-O, LPE-P | neutral loss: phosphoethanolamine | 141.0191 Da |
| TG, DG `[M+NH4]+` | composition-compatible fatty acid + ammonia loss | dynamic |

Negative mode focuses on `[M-H]-` and `[M+HCOO]-`:

| Classes/adduct | Evidence type | Required value |
|---|---|---:|
| PI `[M-H]-` | product ion: inositol phosphate | m/z 241.0119 |
| Sulfatide, ST, SHexCer, SM4 `[M-H]-` | product ion: HSO4- | m/z 96.9601 |
| PG `[M-H]-` | product ion: glycerophosphoglycerol backbone | m/z 152.9958 |
| PC/LPC forms and SM `[M+HCOO]-` | neutral loss: methyl formate/demethylation | 60.0211 Da |
| HexCer/Hex2Cer/Hex3Cer `[M-H]-` or `[M+HCOO]-` | neutral loss: hexose | 162.0528 Da |

The CE target is the commonly reported C27H45 diagnostic ion at m/z 369.3516.
The other exact targets are the project rules specified for this platform.

Negative-mode PS and PE/LPE deliberately have **no** class rule. Both were
removed after measuring them on the bundled negative example:

* PS required the serine neutral loss to the PA anion (87.0320 Da). None of the
  128 strict-filter failures had that product within 0.1 Da, and only 22 of 150
  PS spectra passed, while all 128 showed strong fatty-acyl carboxylate anions.
  At this collision energy PS does not fragment through serine loss here.
* PE/LPE required the ethanolamine phosphate product ion (m/z 140.0118), which
  is below the recorded mass range of half the negative spectra. 66% of the PE
  failures (61 of 93) came from spectra that never recorded any signal at or
  below that m/z, so the ion was unobservable rather than absent.

Both classes still receive fatty-acyl evidence from
`NEG_GLYCEROPHOSPHOLIPID_FA_ANIONS`, and remain subject to m/z, RT and CCS
matching. Restoring either rule requires re-measuring it against data whose MS2
range and collision energy actually produce the target ion.

## Evaluation contract

- `product_ion`: compare every exported fragment m/z directly with `target_mz`.
- `neutral_loss`: calculate `precursor_mz - fragment_mz` for every exported
  fragment, then compare that loss with `target_mz`.
- `fatty_acyl`: use the compatible dynamic evidence rule described below.
- The GUI's absolute fragment tolerance applies to both evidence types and
  defaults to 0.1 Da.
- A separate fatty-acyl intensity criterion removes compatible FA peaks below
  the configured percentage of that spectrum's base peak. It defaults to 1.0%.
- An applicable rule with a matching peak is `supported` and `class_rule_pass`
  is true.
- An applicable rule with a linked spectrum but no matching peak is
  `not_detected` and `class_rule_pass` is false. Strict mode excludes it.
- An applicable rule without a linked spectrum is `no_msms`; it is retained as
  provisional and `class_rule_pass` is blank, not false.
- A class without a configured rule is `not_evaluated`; it is retained and
  `class_rule_pass` is blank.

## Fatty-acyl and molecular-species evidence

Fatty-acyl evidence is configured by `fatty_acyl_rules` in the same JSON file.
The top-level `default_fatty_acyl_min_relative_intensity_percent` controls the
default base-peak-relative threshold; both the annotation run and live review
GUI expose it without changing the raw spectrum display.

- Positive TG/DG `[M+NH4]+`: scan every composition-compatible fatty acid from
  C4 through C40 and 0 through 12 double bonds. Match the neutral loss
  `RCOOH + NH3` using exact calculated mass. At least one compatible loss is the
  mandatory class rule when an MS/MS spectrum exists.
- Positive protonated glycerophospholipids (including PC and PE) and non-ether
  lysophospholipids: scan composition-compatible neutral losses of `RCOOH` and
  the corresponding ketene (`RCOOH - H2O`). These detections are supplementary
  molecular-species evidence only. They do not replace the positive-mode
  class/headgroup rule and do not exclude a lipid when absent. Negative-mode
  carboxylate ions remain the preferred chain evidence when a paired spectrum
  is available.
- Negative glycerophospholipids: scan direct `[FA-H]-` carboxylate ions for the
  same composition-compatible range. These are supplementary molecular-species
  evidence; the class-specific headgroup rule remains the mandatory class test.
- Negative sphingolipids: scan candidate non-hydroxylated and mono-hydroxylated
  N-acyl carboxylate ions. Infer the complementary long-chain base from the total
  species composition. Negative `[M-H]-` is preferred for structural ceramide
  work; formate-adducted reference entries are supported but can be less
  structurally informative.
- The bundled reference supplies formate-adducted Cer/HexCer entries. At load
  time the platform adds theoretical `[M-H]-` aliases by subtracting exact formic
  acid mass. The aliases reuse molecular RT but intentionally clear CCS because
  CCS cannot be transferred between adducts without evidence.
- Diacyl lipids resolve a detected acyl plus its complementary chain. TG retains
  every composition-compatible three-chain interpretation. Sphingolipids report
  N-acyl/long-chain-base candidates. Cardiolipins with insufficient evidence are
  explicitly marked with unresolved remaining chains rather than over-resolved.
- Do not infer sn-position or double-bond position. A merged MZmine spectrum may
  contain multiple isomers or co-isolated species, so multiple molecular-species
  candidates are expected and retained.

Export fields include the detected groups, detailed mass/intensity evidence,
all compatible molecular species, and a JSON mapping from each detected fatty
acyl group to its compatible molecular-species candidates. The applied minimum
relative-intensity threshold is exported as provenance.
The reconciled CSV keeps every compatible identity in
`molecular_species_candidates`, but caps the redundant detailed
`molecular_species_ranking_json` to an Excel-safe highest-ranked prefix.

## AI editing instructions

Add or edit rules only in `lipid_msms_rules.json`, preserve valid JSON, and bump
`schema_version` when the structure changes. A rule requires:

```json
{
  "id": "UNIQUE_STABLE_ID",
  "lipid_classes": ["Class", "Class-O"],
  "polarity": "pos",
  "description": "Human-readable scientific meaning.",
  "evidence": {
    "type": "product_ion",
    "name": "short evidence label",
    "target_mz": 123.4567
  }
}
```

Use `type: "neutral_loss"` for precursor-minus-fragment rules. Add an optional
`adducts` array only when the rule is experimentally valid for specific adducts;
without it, the rule applies to all adducts of the listed class. Keep stable IDs
because exports use them as provenance. After a rule change, update
`tests/test_msms.py`, this table, and `docs/ANNOTATION_PIPELINE.md`.

For fatty-acyl behavior edit `fatty_acyl_rules`, not Python mass constants. Each
entry defines polarity, applicable classes/adducts, family, chain count, evidence
type, carbon/double-bond range, and supported additional oxygens. Stable rule IDs
are exported as provenance. Update `tests/test_fatty_acyl.py` after any change.

## Scientific implementation references

- Positive ammoniated TG/DG fatty-acid plus ammonia losses:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2034497/
- Negative phospholipid fatty-acyl carboxylate ions:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3199966/
- Negative ceramide fatty-acyl and long-chain-base fragmentation:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5086283/
