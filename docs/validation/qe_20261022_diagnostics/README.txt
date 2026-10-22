Read-only diagnostics on the sole33-folder QE collection,22 October2026.
NOT annotation experiments,not new correct matches,not enabled in the app.

potential_by_mode.csv:
Among903 mass-window gaps identified in007,standards-only median offsets place
522 inside3ppm;475 also fit the EXISTING atlas10s window. Offsets were saved
before loading evaluation labels. >=6 distinct standard masses,MAD<=.6ppm,
p90<=1.5ppm,abs center<=2ppm. Automatic009 standard calls were used,not curator-
confirmed standards. No slope/RT fit or actual new candidate selections. Future
calibration must validate explicit positive-quantity declared chemical spikes
and distinct neutral identities. A window opportunity is not identification.

failures_by_class_phase.csv:
Baseline annotation audit aggregated failed definition/phase/folder steps.
304 such steps,NOT304 failed datasets. Some non-CDB isoprenoid failures recur
in every folder; several sphingolipid/low-support classes also show empty-array
or assertion failures. Manual class labels and chemistry must not be guessed.

zero_match_class_diagnostics.csv:
Classes/modes with zero correct matches in009,including all manual entries and
number of project groups. Catalogue formulas/reference presence/mapping errors
and number of outer folds with a class RTmodel are independent diagnostics,
not proof that a single issue is the sole cause. Native X4 chemistry is blank;
its2 manual entries remain in the denominator pending a chemical definition.
GM2[t]/GA2[d]have native formulas,so catalogue-based enumeration may help without
inventing chemistry. Rare one-project classes cannot train a group-held-out
class-specific RTmodel from those same project labels. No entries were removed.

Executed local runners under workspace/work/,portable copies under
tools/qe_optimization/. Source workbooks/raw feature lists were not modified.
Private per-feature diagnostics,source profiles,crosswalks and modelpickles
remain outside Git. These aggregate tables contain no raw sample profiles or
original feature IDs.

native_catalogue_mass_parity.csv (added after016):
The UNUSED catalogue-target generator was compared with the union of surviving
chemical references from all33 baseline runs. Frozen fold0 envelope supplies
enumeration bounds.15,070 overlapping targets;0 differences>1microDa,maximum
4.55e-13Da. Current read-only run native_catalogue_parity_v2 follows an equivalent-
charge spike-protection guard fix; V1 remains historical. No annotations were
selected and this is NOT an annotation-accuracy result. It does not validate
unseen formula definitions,missing RTmodels or unknown X4 chemistry.
