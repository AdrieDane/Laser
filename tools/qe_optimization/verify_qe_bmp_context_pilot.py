"""Independent scalar candidate graph/charge/NumPy-profile reconstruction.

Does not call the production measurement, eligibility, template_mass, _profiles
or _profile_result. Shares native catalogue, metadata parser and schema ONLY.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources
import prepare_qe_bmp_context_pilot as prep
import verify_qe_raw_counterpart_pilot as numeric
from bmp_context_measurements import PREFIX, FIELDS, PROFILE_FIELDS

OUT = prep.common.QE / "BMP_candidate_context_independent_check_v1"


def eligibility(pool, protections):
    from matlab_annotation.automatic import matlab_annotation_adducts
    from matlab_annotation.standard_rt_library import ion_key

    values = []
    for r in pool.itertuples():
        if r.stream not in protections:
            values.append(False)
            continue
        protection = protections[r.stream]
        masses = np.asarray(protection["declared_masses"], float)
        shielded = str(r.feat_feature_id) in set(
            map(str, protection["protected_features"])
        )
        for m in (r.feat_mz, r.ref_mz):
            shielded |= bool(np.any(np.abs(m - masses) <= masses * 5.01e-6))
        coords = np.array([r.ref_mz, r.feat_mz, r.feat_rt, r.feat_intensity], float)
        ion = ion_key(r.adduct)
        native = {ion_key(a) for a in matlab_annotation_adducts(r.stream[:2])}
        ref_rts = []
        for col, scale in (
            ("ref_rt", 60),
            ("rt_expected", 60),
            ("atlas_rt_seconds", 1),
        ):
            if col in pool:
                ref_rts.append(
                    float(pd.to_numeric(getattr(r, col), errors="coerce")) * scale
                )
        deviations = [abs(float(r.feat_rt) * 60 - v) for v in ref_rts if np.isfinite(v)]
        assert ref_rts
        isotope = getattr(r, "isotopologue", "")
        isotope = "" if pd.isna(isotope) else str(isotope)
        valid = (
            not shielded
            and not r.is_is
            and not str(r.name).startswith("IS: ")
            and not str(r.name).endswith(" (IS)")
        )
        valid &= (
            ion in native
            and ion.endswith("+" if r.stream.endswith("Pos") else "-")
            and isotope == ""
        )
        valid &= bool(np.isfinite(coords).all() and (coords > 0).all())
        valid &= abs((r.feat_mz - r.ref_mz) / r.ref_mz * 1e6) < 3
        valid &= bool(deviations) and min(deviations) <= 10 + 1e-9
        values.append(bool(valid))
    return pd.Series(values, index=pool.index)


def independent_measure(pool, features, maps, families, physical):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.vendor.compound_model import PROTON_MASS

    p = pool.copy()
    p["canonical"] = p["class"].map(canonical_lipid_class)
    p["ion"] = p.adduct.map(ion_key)
    p["fid"] = p.feat_feature_id.astype(str)
    query = p.loc[
        p.stream.isin(["NP_Pos", "NP_Neg"])
        & p.canonical.eq("BMP")
        & ~p.is_is.astype(bool)
    ]
    expected = pd.DataFrame(
        np.nan, index=query.index, columns=[PREFIX + f for f in FIELDS]
    )
    expected[PREFIX + "source_physical"] = physical.loc[query.index].astype(float)
    matrices = {
        s: numeric.profiles(features[s], maps.get(s, {})) for s in ("NP_Pos", "NP_Neg")
    }
    for s in matrices:
        original = features[s].set_index(features[s].feature_id.astype(str))
        queries = p.loc[p.stream.eq(s)].set_index("fid")
        for q, r in (("feat_mz", "mz"), ("feat_rt", "rt")):
            np.testing.assert_allclose(
                queries[q].to_numpy(float),
                original.loc[queries.index, r].to_numpy(float),
                rtol=8 * np.finfo(float).eps,
                atol=0,
            )

    def profile(source, target):
        a, ac, al = matrices[source.stream]
        b, bc, bl = matrices[target.stream]
        shared = sorted(set(al) & set(bl))
        assert all(al[x] == bl[x] for x in shared)
        if len(shared) < 8:
            return None
        return numeric.comparison(
            a.loc[source.fid, shared].to_numpy(float),
            b.loc[target.fid, shared].to_numpy(float),
            np.array([ac[x] for x in shared]),
            np.array([bc[x] for x in shared]),
            np.array([al[x] for x in shared]),
        )

    def summarize(source, hits, kind):
        # Scalar de-duplication of physical partners, independent of producer
        # DataFrame sorting/group membership implementation.
        unique = {}
        for target, ppm in hits:
            k = (target.stream, target.fid)
            rank = (ppm, -float(target.feat_intensity))
            if k not in unique or rank < unique[k][0]:
                unique[k] = (rank, target, ppm)
        measured = []
        for _, target, ppm in unique.values():
            result = profile(source, target)
            if result is not None:
                measured.append(
                    (result["raw"], -ppm, float(target.feat_intensity), target, result)
                )
        values = {kind + "_profiles": float(len(measured))}
        best = max(measured, key=lambda x: x[:3]) if measured else None
        if best is not None:
            for f in PROFILE_FIELDS:
                values[kind + "_" + f] = best[4][f]
            values[kind + "_abs_ppm"] = -best[1]
            if kind == "PA":
                values["PA_RT_seconds"] = abs(
                    float(best[3].feat_rt) * 60 - float(source.feat_rt) * 60
                )
        return values, None if best is None else best[3]

    def mass_from(mass, from_ion, to_ion, family):
        if from_ion not in family or to_ion not in family:
            return None
        addition, charge = family[from_ion]
        neutral = abs(charge) * mass - charge * PROTON_MASS - addition
        addition, charge = family[to_ion]
        return (neutral + addition) / abs(charge) + np.sign(charge) * PROTON_MASS

    eligible = p.loc[physical]
    pa_candidates = eligible.loc[
        eligible.stream.eq("NP_Neg")
        & eligible.canonical.eq("PA")
        & eligible.ion.eq("[M-H]-")
    ]
    negative = {}
    for idx, source in query.loc[query.stream.eq("NP_Neg")].iterrows():
        mz = mass_from(
            float(source.ref_mz), "[M+CHO2]-", "[M-H]-", families.get(("NP", "PA"), {})
        )
        values = {"PA_candidates": np.nan, "PA_profiles": np.nan}
        if mz is not None:
            hits = []
            for _, target in pa_candidates.iterrows():
                ppm = abs(float(target.feat_mz) - mz) / mz * 1e6
                theory = abs(float(target.ref_mz) - mz) / mz * 1e6
                seconds = abs(float(target.feat_rt) - float(source.feat_rt)) * 60
                if ppm < 3 and theory < 3 and seconds <= 2 and target.fid != source.fid:
                    hits.append((target, ppm))
            values["PA_candidates"] = float(len({t.fid for t, _ in hits}))
            measured, _ = summarize(source, hits, "PA")
            values.update(measured)
        negative[idx] = values
        for f, value in values.items():
            expected.loc[idx, PREFIX + f] = value
    bmp_candidates = eligible.loc[eligible.canonical.eq("BMP")]
    for idx, source in query.iterrows():
        family = families.get(("NP", "BMP"), {})
        if source.ion not in family:
            continue
        opposite = "NP_Neg" if source.stream == "NP_Pos" else "NP_Pos"
        hits = []
        for _, target in bmp_candidates.iterrows():
            if (
                target.stream != opposite
                or target["name"] != source["name"]
                or target.ion not in family
            ):
                continue
            mz = mass_from(float(source.ref_mz), source.ion, target.ion, family)
            ppm = abs(float(target.feat_mz) - mz) / mz * 1e6
            theory = abs(float(target.ref_mz) - mz) / mz * 1e6
            if ppm < 3 and theory < 3:
                hits.append((target, ppm))
        expected.loc[idx, PREFIX + "candidate_counterparts"] = float(
            len({t.fid for t, _ in hits})
        )
        values, best = summarize(source, hits, "counterpart")
        for f, value in values.items():
            expected.loc[idx, PREFIX + f] = value
        if best is not None and source.stream == "NP_Pos" and best.name in negative:
            for f in ("candidates", "profiles", "raw", "log_ratio_location"):
                expected.loc[idx, PREFIX + "counterpart_PA_" + f] = negative[
                    best.name
                ].get("PA_" + f, np.nan)
    return expected


def one(item):
    d, fold = item
    configure_frozen_resources(REPO)
    from matlab_annotation.adduct_family_evidence import native_adduct_families

    raw, features, maps = prep.load(d, fold)
    saved = pd.read_pickle(prep.OUT / "private" / (d + ".pkl"))
    assert saved["query_dataset"] == d and saved["excluded_folds"] == [fold]
    assert saved["original_prepared_sha256"] == prep.common.sha(
        prep.common.common.prepared_path(d, {fold})
    )
    protections = {
        s: prep.common.common.physical_protection(d, {fold}, s, raw)
        for s in ("NP_Pos", "NP_Neg")
    }
    physical = eligibility(raw["pool"], protections)
    pd.testing.assert_series_equal(
        physical, saved["physical"], check_names=False, check_exact=True
    )
    families = native_adduct_families(str(REPO / "reference_data/matlab_pipeline"))
    expected = independent_measure(raw["pool"], features, maps, families, physical)
    pd.testing.assert_frame_equal(
        expected, saved["evidence"], check_exact=False, atol=1e-6, rtol=1e-8
    )
    print("INDEPENDENT BMP CONTEXT ALL CELLS CHECKED", d, expected.size, flush=True)
    return expected.size


def main():
    datasets, folds = prep.common.inventory()
    receipt_path = prep.OUT / "blind_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["all_passed"] and receipt["all33_physical_tables_frozen"]
    assert receipt["manual_files_opened"] == receipt["classifier_fits"] == 0
    hashes = {
        **receipt["source_input_sha256"],
        **receipt["physical_sha256"],
        str(receipt_path): prep.common.sha(receipt_path),
        str(Path(__file__)): prep.common.sha(Path(__file__)),
        str(Path(numeric.__file__)): prep.common.sha(Path(numeric.__file__)),
    }
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    OUT.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as executor:
        cells = list(executor.map(one, [(d, folds[d]) for d in datasets]))
    assert len(cells) == 33
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    prep.common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            numeric_cells_checked=sum(cells),
            classifier_fits=0,
            annotation_changes=False,
            manual_files_opened=0,
            blind_receipt_sha256=prep.common.sha(receipt_path),
            source_input_sha256=hashes,
        ),
    )


if __name__ == "__main__":
    main()
