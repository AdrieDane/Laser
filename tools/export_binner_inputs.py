"""Prepare private per-mode aligned feature inputs for native Binner 1.0.0.

No external program is installed, launched, or emulated by this converter.
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def export(project, output):
    if output.resolve().is_relative_to(ROOT):
        raise ValueError("Study exports must stay outside Git.")
    output.mkdir(parents=True)
    sys.path.insert(0, str(ROOT / "app"))
    from standard_lipidomics.project_store import load_untargeted_workflow
    from standard_lipidomics.full_feature_evidence import FeatureIndex
    from standard_lipidomics.untargeted_adducts import ADDUCTS

    result, _ = load_untargeted_workflow(project)
    audit = []
    for mode, item in sorted(result["streams"].items()):
        index = FeatureIndex(mode, item["features"])
        frame = pd.DataFrame(
            {
                "Feature": [f"{mode}__{i:07d}" for i in range(len(index.ids))],
                "MZ": index.mz,
                "RT": index.rt,
            }
        )
        for number, sample in enumerate(index.samples, 1):
            frame[f"sample_{number:03d}"] = index.frame[sample].to_numpy()
        frame.to_csv(output / (mode + ".txt"), sep="\t", index=False)
        pd.DataFrame(
            {"Feature": frame.Feature, "original_feature_id": index.ids, "mode": mode}
        ).to_csv(output / (mode + "_feature_mapping.csv"), index=False)
        pd.DataFrame(
            {
                "column": [f"sample_{i:03d}" for i in range(1, len(index.samples) + 1)],
                "original_sample": index.samples,
            }
        ).to_csv(output / (mode + "_sample_mapping.csv"), index=False)
        audit.append(
            {
                "mode": mode,
                "original_features": len(index.ids),
                "exported_features": len(frame),
                "biological_samples": len(index.samples),
                "RT_unit": "minutes",
                "intensities": "original areas; no imputation/log transform",
                "invalid_coordinates": int((~index.valid).sum()),
            }
        )
    # Canonical library is inspectable; native column-value mappings must be
    # verified against the actual trusted Binner distribution before execution.
    pd.DataFrame(
        [
            {
                "Annotation": ion.name,
                "Mass": ion.offset,
                "Mode": ion.polarity,
                "Charge": ion.charge * (1 if ion.polarity == "pos" else -1),
                "Tier": 1 if ion.name in {"[M+H]+", "[M-H]-"} else 2,
            }
            for ion in ADDUCTS
        ]
    ).to_csv(output / "canonical_ion_library.tsv", sep="\t", index=False)
    record = {
        "tool": "Binner",
        "version": "1.0.0",
        "status": "inputs_prepared_native_execution_unavailable",
        "source": "https://binner.med.umich.edu/",
        "artifact_url": "https://metscape.med.umich.edu/jar/binner-1.0.0.jar",
        "blocker": "Verified TLS download fails; presented server certificate expired 17 February 2026. No unverified executable run.",
        "planned_comparison": {
            "mode": "four separate polarity/chromatography feature lists",
            "correlation": "Spearman",
            "RT_binning_gap_min": 0.03,
            "native_preprocessing": "Must record any native imputation/filtering in Summary output",
            "native_annotation_mapping": "Not yet validated; canonical library is not claimed to be native-ready",
        },
        "rows": audit,
        "QC_blank_samples_exported": False,
        "manual_labels_exported": False,
        "benchmark_performed": False,
    }
    (output / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.project, args.output), indent=2))
