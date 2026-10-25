"""All25explicitly nested empirical RT references; NO annotation decisions."""

import json
from pathlib import Path
import shutil

import pandas as pd

from context import WORKSPACE, REPO
from frozen_resources import configure_frozen_resources
from run_qe_standard_mode_experiment import sha, dump

QE = WORKSPACE / "outputs/qe_optimization_20261022"
OUT = QE / "nested_empirical_RT_references_v1"
PARENTS = QE / "nested_reference_development_v2"
DIAGNOSTIC = QE / "empirical_RT_reference_diagnostic_v1"


def main():
    paths = sorted(PARENTS.glob("exclude_*.json"))
    assert len(paths) == 25
    if not (OUT / "protocol.json").exists():
        OUT.mkdir(exist_ok=False)
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for name in ("app", "methods", "reference_data"):
            shutil.copytree(
                REPO / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
        inputs = [
            *paths,
            QE / "rt_atlas_development/observations.pkl",
            QE / "rt_atlas_development/fold_assignment.json",
            DIAGNOSTIC / "protocol.json",
            DIAGNOSTIC / "summary.json",
            Path(__file__).with_name("frozen_resources.py"),
            Path(__file__).with_name("run_qe_standard_mode_experiment.py"),
            Path(__file__).with_name("run_qe_atlas_experiment.py"),
        ]
        inputs.extend(
            DIAGNOSTIC / f"fold_{f}_numerical_reference.json" for f in range(5)
        )
        dump(
            OUT / "protocol.json",
            dict(
                purpose="Reference fitting ONLY,no query labels/annotation decisions. Same25masks as029;5singlefold models must exactly match numerical diagnostic. Exactclass/CDB,oneprojectvote,n>=3/MAD<=5s/absmedian<=30s/shrinkn/(n+3).",
                source_sha256={
                    p.relative_to(snapshot).as_posix(): sha(p)
                    for p in snapshot.rglob("*")
                    if p.is_file()
                },
                input_sha256={str(p): sha(p) for p in inputs},
                resource_receipt=configure_frozen_resources(snapshot),
            ),
        )
        shutil.copy2(Path(__file__), OUT / "runner_source.py")
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (OUT / "runner_source.py").read_bytes() == Path(__file__).read_bytes()
    for p, digest in protocol["source_sha256"].items():
        assert sha(OUT / "source_snapshot" / p) == digest
    for p, digest in protocol["input_sha256"].items():
        assert sha(Path(p)) == digest
    configure_frozen_resources(OUT / "source_snapshot")
    from matlab_annotation.empirical_retention import fit_empirical_retention

    observations = pd.read_pickle(QE / "rt_atlas_development/observations.pkl")
    assert observations.cohort.eq(observations.dataset.str[:7]).all()
    rows = []
    for path in paths:
        parent = json.loads(path.read_text())
        train = observations[
            observations.dataset.isin(parent["training_datasets"])
            & observations.mapped.astype(bool)
        ].copy()
        assert set(train.cohort) == set(parent["training_groups"])
        assert not set(train.cohort) & set(parent["excluded_groups"])
        model = fit_empirical_retention(
            train,
            parent["atlas"],
            training_groups=parent["training_groups"],
            excluded_groups=parent["excluded_groups"],
        )
        if len(parent["excluded_folds"]) == 1:
            f = parent["excluded_folds"][0]
            assert model == json.loads(
                (DIAGNOSTIC / f"fold_{f}_numerical_reference.json").read_text()
            )
        record = dict(
            format="laser-nested-empirical-RT-v1",
            excluded_folds=parent["excluded_folds"],
            training_groups=parent["training_groups"],
            excluded_groups=parent["excluded_groups"],
            training_datasets=parent["training_datasets"],
            excluded_datasets=parent["excluded_datasets"],
            parent_reference_sha256=sha(path),
            empirical_reference=model,
        )
        destination = OUT / path.name
        if destination.exists():
            assert record == json.loads(destination.read_text())
        else:
            dump(destination, record)
        rows.append(
            dict(
                excluded_folds="_".join(map(str, parent["excluded_folds"])),
                training_projects=len(parent["training_groups"]),
                reference_compositions=len(model["models"]),
            )
        )
        print("EMPIRICAL NESTED REFERENCE", path.stem, len(model["models"]), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "by_exclusion_mask.csv", index=False)
    dump(
        OUT / "summary.json",
        dict(
            all_complete=True,
            reference_bundles=25,
            diagnostic_singlefold_exact_parity_checks=5,
            annotation_changes=False,
            model_sha256={p.name: sha(p) for p in OUT.glob("exclude_*.json")},
        ),
    )


if __name__ == "__main__":
    main()
