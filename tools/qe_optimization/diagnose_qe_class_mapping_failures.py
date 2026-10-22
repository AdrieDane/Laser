"""Read-only baseline failure/omitted-class audit, NOT annotation performance."""

import json
import sys
from types import SimpleNamespace

import pandas as pd

from context import REPO, WORKSPACE as ROOT

QE = ROOT / "outputs/qe_optimization_20261022"
OUT = QE / "class_mapping_failure_audit_v1"
sys.path.insert(0, str(REPO / "app"))
from annotator.composition_ranges import canonical_lipid_class
from matlab_annotation.decode import lipid_class_from_template
from matlab_annotation.xcms_project import load_bundled_compound_catalog
from matlab_annotation.vendor.naming import compound_display_template


def main():
    OUT.mkdir(exist_ok=False)
    catalogue = load_bundled_compound_catalog(REPO / "reference_data/matlab_pipeline")
    project = SimpleNamespace(compounds=catalogue)
    definitions = []
    for i in range(catalogue.count):
        template = compound_display_template(project, i)
        definitions.append(
            dict(
                definition=template,
                lipid_class=canonical_lipid_class(lipid_class_from_template(template)),
                catalogue_formula=str(catalogue.legacy_fields["chem"][i]),
                catalogue_measurements=str(catalogue.legacy_fields["measurement"][i]),
            )
        )
    catalogue_frame = pd.DataFrame(definitions).drop_duplicates("definition")
    catalogue_frame.to_csv(OUT / "catalogue_definition_audit.csv", index=False)
    audits = []
    counts = []
    for p in sorted((QE / "baseline/runs").glob("*/full_annotation.pkl")):
        r = pd.read_pickle(p)
        a = r["matlab_compound_audit"].copy()
        a["dataset"] = p.parent.name
        a["lipid_class"] = a.definition.map(lipid_class_from_template).map(
            canonical_lipid_class
        )
        audits.append(a)
        for stream, payload in r["streams"].items():
            for cls, g in payload["reference"].groupby("class"):
                counts.append(
                    dict(
                        dataset=p.parent.name,
                        stream=stream,
                        lipid_class=canonical_lipid_class(cls),
                        reference_targets=len(g),
                    )
                )
    a = pd.concat(audits, ignore_index=True)
    a.to_csv(OUT / "private_dataset_mapping_audits.csv", index=False)
    failures = (
        a[a.status.eq("failed")]
        .groupby(["phase", "definition", "lipid_class", "detail"], as_index=False)
        .agg(affected_folders=("dataset", "nunique"))
    )
    failures.to_csv(OUT / "failures_by_class_phase.csv", index=False)
    manual = pd.read_pickle(QE / "experiment_009/reports_v2/all_manual_comparison.pkl")
    manual["canonical"] = manual.lipid_class.map(canonical_lipid_class)
    manual["cohort"] = manual.dataset.str[:7]
    totals = manual.groupby(["stream", "canonical"], as_index=False).agg(
        manual_entries=("species", "size"),
        correct_same_peak=("same_manual_peak", "sum"),
        project_groups=("cohort", "nunique"),
    )
    zero = (
        totals[totals.correct_same_peak.eq(0)]
        .rename(columns={"canonical": "lipid_class"})
        .copy()
    )
    inventory = pd.DataFrame(counts)
    records = []
    for row in zero.to_dict("records"):
        cls, stream = row["lipid_class"], row["stream"]
        entries = catalogue_frame[catalogue_frame.lipid_class.eq(cls)]
        ref = inventory[inventory.lipid_class.eq(cls) & inventory.stream.eq(stream)]
        failed = a[
            a.lipid_class.eq(cls) & a.phase.eq(stream[:2]) & a.status.eq("failed")
        ]
        models = 0
        for fold in range(5):
            atlas = json.loads(
                (QE / f"rt_atlas_development/fold_{fold}_atlas.json").read_text()
            )["atlas"]
            models += int(stream + "|" + cls in atlas["models"])
        records.append(
            dict(
                **row,
                catalogue_entries=len(entries),
                nonempty_chemical_formulas=int(
                    entries.catalogue_formula.astype(str).str.strip().ne("").sum()
                ),
                folders_with_any_baseline_reference=int(ref.dataset.nunique()),
                phase_mapping_failed_folders=int(failed.dataset.nunique()),
                folds_with_class_RT_model=models,
                note="Diagnostic only; phase failures may affect both polarities. One missing step is not proof it is the sole cause.",
            )
        )
    result = pd.DataFrame(records)
    result.to_csv(OUT / "zero_match_class_diagnostics.csv", index=False)
    print(result.to_string(index=False))
    print("BASELINE MAPPING FAILURES", len(a[a.status.eq("failed")]), flush=True)


if __name__ == "__main__":
    main()
