"""Read-only029: distinguish choosing a peak from accepting a lipid identity.

Counts are retrospective opportunities, NOT a ranker fit or accuracy gain.
Manual identities/coordinates enter diagnosis only, never predictor generation.
"""

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from context import WORKSPACE
from run_qe_atlas_experiment import species, adduct
from run_qe_standard_mode_experiment import sha, dump

QE = WORKSPACE / "outputs/qe_optimization_20261022"
CORE = QE / "experiment_029"
OUT = QE / "peak_competition_diagnostic_029_v1"


def inspect(pool, manual):
    pool = pool.copy()
    pool["_species"] = pool.name.map(species)
    pool["_adduct"] = pool.adduct.map(adduct)
    pool["_fid"] = pool.feat_feature_id.astype(str)
    grouped = {
        key: part
        for key, part in pool.groupby(["stream", "_species", "_adduct"], sort=False)
    }
    records = []
    # Structural diagnosis only. These exclusions NEVER redefine performance N.
    for row in manual[manual.mapped & ~manual.internal_standard].itertuples():
        key = (row.stream, row.species, adduct(row.adduct))
        if key not in grouped:
            continue
        part = grouped[key]
        right = part[part._fid.eq(str(row.feature_id))]
        if right.empty:
            continue
        wrong = part[~part._fid.eq(str(row.feature_id))].drop_duplicates("_fid")
        if wrong.empty:
            continue
        records.append(
            dict(
                stream=row.stream,
                lipid_class=row.lipid_class,
                manual_identities_with_correct_and_other_candidate_peaks=1,
                wrong_peak_alternatives=len(wrong),
                currently_wrong_peak_calls=int(row.same_lipid_different_peak),
                currently_missing_same_peak=int(not row.same_manual_peak),
            )
        )
    return records


def main():
    assert json.loads((CORE / "integrity_receipt.json").read_text())["all_passed"]
    projects = json.loads((CORE / "run_manifest.json").read_text())["datasets"]
    fp = QE / "rt_atlas_development/fold_assignment.json"
    folds = json.loads(fp.read_text())
    inputs = {str(fp): sha(fp)}
    rows, per_mode = [], []
    OUT.mkdir(exist_ok=False)
    for dataset in projects:
        pp = (
            QE
            / "standard_context_preparation_v1/prepared"
            / f"exclude_{folds[dataset]}"
            / (dataset + ".pkl")
        )
        mp = CORE / "evaluation_v2" / dataset / "manual_comparison.pkl"
        for path in (pp, mp):
            inputs[str(path)] = sha(path)
        data = pd.read_pickle(pp)
        assert (
            data["query_dataset"] == dataset
            and dataset[:7] not in data["reference_training_groups"]
        )
        result = inspect(data["pool"], pd.read_pickle(mp))
        rows.extend(dict(dataset=dataset, **r) for r in result)
        for mode in ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg"):
            values = Counter()
            for row in result:
                if row["stream"] == mode:
                    values.update({k: v for k, v in row.items() if isinstance(v, int)})
            per_mode.append(dict(dataset=dataset, stream=mode, **dict(values)))
        print("READ-ONLY PEAK COMPETITION", dataset, flush=True)
    rows = pd.DataFrame(rows)
    numeric = list(rows.select_dtypes("number"))
    table = pd.DataFrame(per_mode).fillna(0)
    assert len(table) == 132 and table.dataset.nunique() == 33
    table.to_csv(OUT / "by_dataset_mode.csv", index=False)
    rows.groupby(["dataset", "stream", "lipid_class"], as_index=False)[
        numeric
    ].sum().to_csv(OUT / "by_dataset_mode_class.csv", index=False)
    rows.groupby(["stream", "lipid_class"], as_index=False)[numeric].sum().to_csv(
        OUT / "by_mode_class.csv", index=False
    )
    totals = table.groupby("stream")[numeric].sum().astype(int)
    assert all(sha(Path(p)) == h for p, h in inputs.items())
    receipt = dict(
        all_passed=True,
        annotation_changes=False,
        fits_or_cutoff_changes=False,
        datasets=33,
        dataset_modes=132,
        by_mode=totals.to_dict(orient="index"),
        source_sha256=sha(Path(__file__)),
        input_sha256=inputs,
        note="Retrospective029structuralopportunities,NOTvalidatedgains. Onlymappedendogenousmanualidentities whosecorrectoriginalpeak ANDanotherdistinctfeature arealreadyincorepool. Standards/unresolved/not-in-pool identities excludedONLYfromstructuraldiagnosis,NOTbenchmarkdenominator. Alternatives/identities/missing/wrongcalls areoverlappingquantities,notadditiveerrors. Notuntouchedvalidation/FDR. Noannotation/score/labelchanges.",
    )
    dump(OUT / "summary.json", receipt)
    print(
        json.dumps({k: v for k, v in receipt.items() if k != "input_sha256"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
