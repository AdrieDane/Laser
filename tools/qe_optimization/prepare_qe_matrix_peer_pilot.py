"""All33 blind numeric sample-context descriptors, no annotation changes."""

import json
import shutil

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import matrix_peer_pilot_data as data
import matrix_peer_composition as peer


def main():
    datasets, folds = data.inventory()
    assert json.loads((data.REFERENCE / "integrity_receipt.json").read_text())[
        "all_passed"
    ]
    assert json.loads(
        (data.QE / "paired_isomer_gap_diagnostic_041_v1/receipt.json").read_text()
    )["all_passed"]
    hashes = data.inputs()
    data.OUT.mkdir(exist_ok=False)
    snapshot = data.OUT / "source_snapshot"
    snapshot.mkdir()
    for folder in ("app", "methods", "reference_data"):
        shutil.copytree(
            REPO / folder,
            snapshot / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    resources = configure_frozen_resources(snapshot)
    data.dump(
        data.OUT / "protocol.json",
        dict(
            source_input_sha256=hashes,
            numeric_runtime=data.environment(),
            resources=resources,
            diagnostic_only=True,
            classifier_fits=0,
            annotation_changes=False,
            numerical_workers=1,
            new_numeric_descriptors=8,
            policy="OTHERFAMILYnativehistograms;5nearestreviewedclassprojectsincludingALLboundaryties;2globalpseudovotes;nohardlimits. Explicitotherprojectscopes BEFOREtrainingcurationdecode. All33freezeBEFOREheldoutcurationcomparison.",
        ),
    )
    signatures = {d: peer.signature(data.native_inputs(d)) for d in datasets}
    data.dump(data.OUT / "native_signatures.json", signatures)
    (data.OUT / "models").mkdir()
    for outer in range(5):
        rows, allowed, excluded, reads = data.training_rows(outer)
        permitted = {d: s for d, s in signatures.items() if folds[d] != outer}
        model = peer.fit_reference(
            rows, permitted, training_groups=allowed, excluded_groups=excluded
        )
        folder = data.OUT / "models" / f"fold_{outer}"
        folder.mkdir()
        data.dump(folder / "model.json", model)
        rows.to_pickle(folder / "authorized_composition_rows.pkl")
        pd.DataFrame(reads).to_csv(folder / "training_role_audit.csv", index=False)
        print(
            "OTHER-PROJECT MATRIX PEER MODEL",
            outer,
            len(reads),
            len(model["classes"]),
            flush=True,
        )
    (data.OUT / "private").mkdir()
    availability = []
    for d in datasets:
        model_path = data.OUT / "models" / f"fold_{folds[d]}" / "model.json"
        model = json.loads(model_path.read_text())
        original = data.load_pool(d, folds[d])
        pool = original["pool"]
        evidence = peer.predict(pool, signatures[d], model, query_group=d[:7])
        assert evidence.index.equals(pool.index) and list(evidence) == [
            peer.PREFIX + f for f in peer.FIELDS
        ]
        numeric = evidence.to_numpy(float)
        assert np.isfinite(numeric[~np.isnan(numeric)]).all()
        evidence.to_pickle(data.OUT / "private" / (d + ".pkl"))
        for stream in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"):
            q = evidence.loc[pool.stream.eq(stream)]
            availability.append(
                dict(
                    dataset=d,
                    stream=stream,
                    candidate_rows=len(q),
                    supported_options=int(q.iloc[:, 0].notna().sum()),
                )
            )
        print(
            "BLIND MATRIX PEER DESCRIPTORS",
            d,
            len(evidence),
            int(evidence.iloc[:, 0].notna().sum()),
            flush=True,
        )
    assert len(availability) == 132 and hashes == data.inputs()
    pd.DataFrame(availability).to_csv(
        data.OUT / "by_dataset_mode_availability.csv", index=False
    )
    artifacts = {
        str(p): data.sha(p)
        for p in data.OUT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    data.dump(
        data.OUT / "blind_receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            dataset_modes=132,
            numeric_runtime=data.environment(),
            source_input_sha256=hashes,
            artifact_sha256=artifacts,
            authorized_training_table_uses=132,
            query_gold_validation_tables_opened=0,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
        ),
    )
    print(
        "ALL33 MATRIX PEER NUMERIC OUTPUTS FROZEN; INDEPENDENT CHECK REQUIRED",
        flush=True,
    )


if __name__ == "__main__":
    main()
