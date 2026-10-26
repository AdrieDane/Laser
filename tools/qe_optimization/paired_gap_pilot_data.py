"""Explicit paired-gap pilot inputs; curated rows only in authorized TRAINING.

Query prediction loads frozen automatic041 rows and native chemical coordinates
only. Training RT is the matched ORIGINAL feature measurement, never saved fits.
"""

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from context import WORKSPACE
import paired_retention_gap as gap

QE = WORKSPACE / "outputs/qe_optimization_20261022"
BASE = QE / "baseline"
REFERENCE = QE / "experiment_041"
OUT = QE / "paired_isomer_gap_pilot_v1"
COMPOSITION = re.compile(r"\((?:[OP]-)?(\d+):(\d+)\)")
CHEMICAL = ["stream", "name", "class", "adduct", "carbon", "db", "ref_mz"]
OBSERVATION = [
    "dataset",
    "cohort",
    "stream",
    "lipid_class",
    "carbon",
    "db",
    "adduct",
    "feature_id",
    "rt_seconds",
    "reference_mz",
    "observed_mz",
]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    return dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__)


def inventory():
    datasets = json.loads((BASE / "run_manifest.json").read_text())["datasets"]
    folds = json.loads((QE / "rt_atlas_development/fold_assignment.json").read_text())
    assert len(datasets) == 33 and len({d[:7] for d in datasets}) == 30
    assert all(
        len({folds[d] for d in datasets if d[:7] == group}) == 1
        for group in {d[:7] for d in datasets}
    )
    return datasets, folds


def chemical_reference(baseline):
    result = pd.concat(
        [
            p["reference"].assign(stream=s)[CHEMICAL]
            for s, p in baseline["streams"].items()
        ],
        ignore_index=True,
    )
    assert not {"ref_rt", "rt_expected", "atlas_rt_seconds"} & set(result)
    result = result.loc[
        ~result["name"].astype(str).str.endswith(" (IS)")
        & ~result["name"].astype(str).str.startswith("IS: ")
    ].copy()
    from matlab_annotation.standard_rt_library import ion_key
    from standard_lipidomics.validation import _species

    result["_species"] = result["name"].map(_species)
    result["adduct"] = result.adduct.map(ion_key)
    return result


def training_rows(outer):
    """Check dataset authorization BEFORE opening its manual comparison table."""
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key
    from standard_lipidomics.validation import canonical_feature_id

    datasets, folds = inventory()
    training = {d[:7] for d in datasets if folds[d] != outer}
    excluded = {d[:7] for d in datasets if folds[d] == outer}
    rows, support, reads = [], [], []
    for d in datasets:
        if folds[d] == outer:
            continue
        assert d[:7] in training and d[:7] not in excluded
        path = BASE / "evaluation_v2" / d / "manual_comparison.pkl"
        manual = pd.read_pickle(path)
        # Only identity/mapping flags enter this RT-training role. Never copy
        # same_manual_peak/outcome/candidate-evidence performance columns.
        manual = manual[
            [
                "stream",
                "species",
                "lipid_class",
                "adduct",
                "feature_id",
                "internal_standard",
                "mapped",
                "duplicate_manual_entry",
                "conflicting_manual_feature",
            ]
        ].copy()
        assert manual.internal_standard.isin([True, False]).all()
        manual = manual.loc[~manual.internal_standard].copy()
        manual["lipid_class"] = manual.lipid_class.map(canonical_lipid_class)
        manual["adduct"] = manual.adduct.map(ion_key)
        for (s, cls), n in manual.groupby(["stream", "lipid_class"]).size().items():
            support.append(
                dict(cohort=d[:7], stream=s, lipid_class=cls, manual_rows=int(n))
            )
        baseline = pd.read_pickle(BASE / "runs" / d / "full_annotation.pkl")
        chemical = chemical_reference(baseline)
        mass_groups = {
            key: part.ref_mz.to_numpy(float)
            for key, part in chemical.groupby(
                ["stream", "_species", "adduct"], sort=False
            )
        }
        originals = {}
        for s, payload in baseline["streams"].items():
            feature = payload["features"]
            ids = feature.feature_id.map(canonical_feature_id)
            assert ids.is_unique
            originals[s] = feature.set_index(ids, drop=False)
        primary = {
            (s, cls, ion)
            for s, ion, a, b, _, _ in gap.FAMILIES.values()
            for cls in (a, b)
        }
        family_rows = pd.Series(
            [
                key in primary
                for key in manual[["stream", "lipid_class", "adduct"]].itertuples(
                    index=False, name=None
                )
            ],
            index=manual.index,
        )
        eligible = manual.loc[
            manual.mapped
            & ~manual.duplicate_manual_entry
            & ~manual.conflicting_manual_feature
            & family_rows
        ]
        for r in eligible.itertuples():
            match = COMPOSITION.search(r.species)
            masses = mass_groups.get((r.stream, r.species, r.adduct))
            if match is None or masses is None or not len(masses):
                continue
            if not np.isfinite(masses).all() or np.max(masses) - np.min(masses) > 1e-8:
                continue
            carbon, db = map(int, match.groups())
            measured = originals[r.stream].loc[canonical_feature_id(r.feature_id)]
            rows.append(
                dict(
                    dataset=d,
                    cohort=d[:7],
                    stream=r.stream,
                    lipid_class=r.lipid_class,
                    carbon=carbon,
                    db=db,
                    adduct=r.adduct,
                    feature_id=canonical_feature_id(r.feature_id),
                    rt_seconds=float(measured.rt) * 60,
                    reference_mz=float(masses[0]),
                    observed_mz=float(measured.mz),
                )
            )
        reads.append(
            dict(
                dataset=d,
                outer=outer,
                cohort=d[:7],
                authorized_training=True,
                manual_sha256=sha(path),
            )
        )
    return (
        pd.DataFrame(rows, columns=OBSERVATION),
        pd.DataFrame(support),
        training,
        excluded,
        reads,
    )


def query_inputs(dataset, outer):
    from matlab_annotation.declared_standards import declared_standard_targets
    from standard_lipidomics.validation import canonical_feature_id
    from resource_paths import REFERENCE_DATA_DIRECTORY

    datasets, folds = inventory()
    assert dataset in datasets and folds[dataset] == outer
    baseline = pd.read_pickle(BASE / "runs" / dataset / "full_annotation.pkl")
    accepted = pd.read_csv(
        REFERENCE / "runs" / dataset / "automatic_annotations.csv",
        dtype={"feat_feature_id": str},
        float_precision="round_trip",
        low_memory=False,
    )
    columns = [
        "stream",
        "name",
        "class",
        "adduct",
        "carbon",
        "db",
        "feat_feature_id",
        "feat_mz",
        "feat_rt",
        "is_is",
        "class_selection_score",
    ]
    anchors = accepted[columns].copy()
    assert anchors.is_is.isin([True, False]).all()
    chemical = chemical_reference(baseline)[CHEMICAL].copy()
    declared = declared_standard_targets(
        REFERENCE_DATA_DIRECTORY / "matlab_pipeline",
        baseline["internal_standards_expected"],
    )
    masses = {}
    for s in ("RP_Pos", "RP_Neg"):
        part = declared.loc[
            declared.stream.eq(s) & declared.internal_standard_quantity.gt(0)
        ]
        masses[s] = part.ref_mz.to_numpy(float)
    for s, part in anchors.groupby("stream", sort=False):
        feature = baseline["streams"][s]["features"]
        original = feature.set_index(feature.feature_id.map(canonical_feature_id))
        raw = original.loc[part.feat_feature_id.map(canonical_feature_id)]
        np.testing.assert_allclose(
            part.feat_mz.to_numpy(float),
            raw.mz.to_numpy(float),
            rtol=8 * np.finfo(float).eps,
            atol=0,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            part.feat_rt.to_numpy(float),
            raw.rt.to_numpy(float),
            rtol=8 * np.finfo(float).eps,
            atol=0,
            equal_nan=True,
        )
    return anchors, chemical, masses, baseline, accepted
