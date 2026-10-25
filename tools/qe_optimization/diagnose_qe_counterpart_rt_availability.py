"""Numerical declared-spike RT availability only; NEVER opens manual curation."""

from pathlib import Path

import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import run_qe_add_only_class_experiment as common

QE = common.QE
OUT = QE / "raw_counterpart_RT_availability_v1"


def main():
    configure_frozen_resources(REPO)
    from matlab_annotation.raw_counterpart_rt import (
        independent_spike_pairs,
        fit_independent_rt_links,
    )

    datasets, folds = common.inventory()
    files = {
        Path(__file__),
        Path(common.__file__),
        Path(common.common.__file__),
        Path(__file__).with_name("frozen_resources.py"),
        REPO / "docs/validation/next_raw_counterpart_evidence.txt",
    }
    files.update(
        p
        for name in ("app", "reference_data", "methods")
        for p in (REPO / name).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
    )
    for d in datasets:
        name = f"exclude_{folds[d]}"
        files.update(
            [
                QE / "baseline/runs" / d / "full_annotation.pkl",
                common.common.prepared_path(d, {folds[d]}),
                common.common.PREPARED / "measurements" / name / (d + ".pkl"),
                QE / "nested_standard_development_v1" / (name + ".json"),
                QE / "nested_standard_development_v1/automatic_inputs" / (d + ".pkl"),
            ]
        )
    hashes = {str(p): common.sha(p) for p in sorted(files)}
    OUT.mkdir(exist_ok=False)
    (OUT / "private").mkdir()
    rows = []
    for d in datasets:
        raw = common.common.load_predictors(d, {folds[d]})
        baseline = pd.read_pickle(QE / "baseline/runs" / d / "full_annotation.pkl")
        measured = pd.read_pickle(
            common.common.PREPARED
            / "measurements"
            / f"exclude_{folds[d]}"
            / (d + ".pkl")
        )
        for phase in ("RP", "NP"):
            # Check matching query-excluded reference provenance BEFORE using
            # its independently detected standards; never reads manual labels.
            common.common.physical_protection(d, {folds[d]}, phase + "_Pos", raw)
            pairs = independent_spike_pairs(
                measured,
                baseline["streams"][phase + "_Neg"]["features"],
                REPO / "reference_data/matlab_pipeline",
                source_stream=phase + "_Pos",
            )
            pairs.to_pickle(OUT / "private" / (d + "_" + phase + "_pairs.pkl"))
            positive = fit_independent_rt_links(pairs, source_stream=phase + "_Pos")
            inverse = pairs.copy()
            for a, b in (
                ("source_rt_seconds", "target_rt_seconds"),
                ("source_feature_ids", "target_feature_ids"),
            ):
                inverse[a], inverse[b] = pairs[b].copy(), pairs[a].copy()
            negative = fit_independent_rt_links(inverse, source_stream=phase + "_Neg")
            common.dump(
                OUT / "private" / (d + "_" + phase + "_links.json"),
                dict(positive=positive, negative=negative),
            )
            for stream, links in (
                (phase + "_Pos", positive),
                (phase + "_Neg", negative),
            ):
                rows.append(
                    dict(
                        dataset=d,
                        stream=stream,
                        existing_independent_pairs=int(
                            pairs.origin.eq("existing_independent_declared_pair").sum()
                        ),
                        augmented_alternate_ion_pairs=int(
                            pairs.origin.eq(
                                "unique_native_alternate_ion_evidence_only"
                            ).sum()
                        ),
                        total_distinct_pairs=len(pairs),
                        supported_global_RP_map=links["global_map"] is not None,
                        supported_exact_class_NP_maps=len(links["class_maps"]),
                    )
                )
        print("RAW DECLARED-ION NUMERICAL RT AVAILABILITY", d, flush=True)
    table = pd.DataFrame(rows)
    assert len(table) == 132 and table.dataset.nunique() == 33
    table.to_csv(OUT / "by_dataset_mode.csv", index=False)
    assert all(common.sha(Path(p)) == h for p, h in hashes.items())
    common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            annotation_changes=False,
            classifier_fits=0,
            manual_files_opened=0,
            source_input_sha256=hashes,
            valid_global_RP_directional_maps=int(table.supported_global_RP_map.sum()),
            valid_NP_directional_exact_class_maps=int(
                table.supported_exact_class_NP_maps.sum()
            ),
            note="Numerical availability ofindependentlyobserveddeclared-spike pairs,notannotationperformance/chemicalidentity/FDR or deploymentapproval. InversefitusesSAMEobservedpairs,notadditionalindependentdata. Reviewonlyionsareevidenceonly,nonepromoted.",
        ),
    )
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
