"""Pure unused measurements: native NP BMP candidate links and PA interference.

No labels, fits, annotation decisions, file I/O or RP/NP RT transfer. Candidate
membership must be rebuilt under EACH authorized reference mask by the caller.
"""

import numpy as np
import pandas as pd

PREFIX = "q_BMP_context_"
PROFILE_FIELDS = (
    "raw",
    "normalized",
    "within",
    "leave_one_out",
    "normalized_leave_one_out",
    "log_ratio_location",
    "normalized_log_ratio_location",
    "mad",
    "n",
    "dof",
)
FIELDS = (
    "source_physical",
    "candidate_counterparts",
    "counterpart_profiles",
    *("counterpart_" + f for f in PROFILE_FIELDS),
    "counterpart_abs_ppm",
    "PA_candidates",
    "PA_profiles",
    *("PA_" + f for f in PROFILE_FIELDS),
    "PA_abs_ppm",
    "PA_RT_seconds",
    "counterpart_PA_candidates",
    "counterpart_PA_profiles",
    "counterpart_PA_raw",
    "counterpart_PA_log_ratio_location",
)


def measure(pool, features, specimen_mapping, families, physical):
    """NP BMP only, target's OWN physical/RT eligibility, strict3ppm/2sPA.

    physical is the complete mask-matched original mass/RT/native-ion/STD
    eligibility series, not an acceptance flag. No minimum profile correlation
    is imposed here. Unknown evidence stays NaN. Duplicate hypothesis rows for
    the same physical partner count once. Biological groups must agree.
    """
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.raw_counterpart_evidence import (
        _profile_result,
        _validate_original_queries,
    )
    from matlab_annotation.paired_sample_evidence import _profiles
    from matlab_annotation.raw_counterpart_rt import template_mass, SUPERVISION
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.vendor.compound_model import PROTON_MASS

    if SUPERVISION & set(pool):
        raise ValueError("BMP context refuses supervision")
    if (
        not pool.index.is_unique
        or not pool.index.equals(physical.index)
        or not physical.isin([False, True]).all()
    ):
        raise ValueError("Aligned explicit physical eligibility required")
    required = {
        "stream",
        "class",
        "name",
        "adduct",
        "ref_mz",
        "feat_mz",
        "feat_rt",
        "feat_feature_id",
        "feat_intensity",
        "is_is",
    }
    if not required.issubset(pool):
        raise ValueError("Original physical candidates required")
    p = pool.assign(
        _cls=pool["class"].map(canonical_lipid_class),
        _ion=pool.adduct.map(ion_key),
        _fid=pool.feat_feature_id.astype(str),
    )
    np_pool = p.loc[p.stream.isin(["NP_Pos", "NP_Neg"])]
    for s in ("NP_Pos", "NP_Neg"):
        _validate_original_queries(
            np_pool.loc[np_pool.stream.eq(s)],
            features,
            s,
            "NP_Neg" if s.endswith("Pos") else "NP_Pos",
        )
    query = np_pool.loc[np_pool._cls.eq("BMP") & ~np_pool.is_is.astype(bool)]
    output = pd.DataFrame(
        np.nan, index=query.index, columns=[PREFIX + c for c in FIELDS]
    )
    output[PREFIX + "source_physical"] = physical.loc[query.index].astype(float)
    eligible = np_pool.loc[physical.loc[np_pool.index]].copy()
    bmp = eligible.loc[eligible._cls.eq("BMP")]
    pa = eligible.loc[
        eligible.stream.eq("NP_Neg")
        & eligible._cls.eq("PA")
        & eligible._ion.eq("[M-H]-")
    ]
    bmp_family, pa_family = (
        families.get(("NP", "BMP"), {}),
        families.get(("NP", "PA"), {}),
    )
    profiles, centers, labels = {}, {}, {}
    for s in ("NP_Pos", "NP_Neg"):
        requested = set(query.loc[query.stream.eq(s), "_fid"]) | set(
            eligible.loc[eligible.stream.eq(s), "_fid"]
        )
        profiles[s], centers[s], labels[s] = _profiles(
            features[s], requested, specimen_mapping.get(s, {})
        )
    memo = {}

    def compare(sa, fa, sb, fb):
        key = (sa, fa, sb, fb)
        if key not in memo:
            shared = sorted(set(labels[sa]) & set(labels[sb]))
            if any(labels[sa][x] != labels[sb][x] for x in shared):
                raise ValueError("Paired biological groups disagree")
            if len(shared) < 8:
                memo[key] = None
            else:
                memo[key] = _profile_result(
                    profiles[sa].loc[fa, shared].to_numpy(float),
                    profiles[sb].loc[fb, shared].to_numpy(float),
                    centers[sa].reindex(shared).to_numpy(float),
                    centers[sb].reindex(shared).to_numpy(float),
                    [labels[sa][x] for x in shared],
                )
        return memo[key]

    def summary(source, partners, kind):
        # Ranking uses measurements only; no class labels from curation, old
        # automatic calls or fitted agreement scores can choose a partner.
        unique = partners.sort_values(
            ["_ppm", "feat_intensity"], ascending=[True, False], kind="stable"
        ).drop_duplicates(["stream", "_fid"])
        result = {kind + "_profiles": 0.0}
        measured = []
        for target in unique.itertuples():
            # Leading-underscore columns are deliberately read by index below,
            # not pandas itertuples' renamed positional attributes.
            tr = unique.loc[target.Index]
            comp = compare(source.stream, source["_fid"], tr.stream, tr["_fid"])
            if comp is not None:
                measured.append(
                    (
                        comp["raw"],
                        -float(tr["_ppm"]),
                        float(tr.feat_intensity),
                        comp,
                        tr,
                    )
                )
        result[kind + "_profiles"] = float(len(measured))
        best = max(measured, key=lambda x: x[:3]) if measured else None
        if best is not None:
            comp, tr = best[3:]
            for f in PROFILE_FIELDS:
                result[kind + "_" + f] = comp[f]
            result[kind + "_abs_ppm"] = float(tr["_ppm"])
            if kind == "PA":
                result["PA_RT_seconds"] = (
                    abs(float(tr.feat_rt) - float(source.feat_rt)) * 60
                )
        return result, None if best is None else best[4]

    pa_evidence = {}
    for idx, source in query.loc[query.stream.eq("NP_Neg")].iterrows():
        row = {"PA_candidates": np.nan, "PA_profiles": np.nan}
        formate = template_mass(
            float(source.ref_mz), "[M+CHO2]-", pa_family, positive=False
        )
        primary = pa_family.get("[M-H]-")
        if formate is not None and primary is not None:
            offset, charge = primary
            expected = abs((formate + offset) / charge + PROTON_MASS)
            ppm = ((pa.feat_mz / expected - 1) * 1e6).abs()
            theoretical = ((pa.ref_mz / expected - 1) * 1e6).abs()
            valid = (
                ppm.lt(3)
                & theoretical.lt(3)
                & ((pa.feat_rt - float(source.feat_rt)) * 60).abs().le(2)
                & pa._fid.ne(source["_fid"])
            )
            partners = pa.loc[valid].assign(_ppm=ppm.loc[valid])
            row["PA_candidates"] = float(partners._fid.nunique())
            values, _ = summary(source, partners, "PA")
            row.update(values)
        pa_evidence[idx] = row
        for f, value in row.items():
            output.loc[idx, PREFIX + f] = value

    for idx, source in query.iterrows():
        opposite = "NP_Neg" if source.stream == "NP_Pos" else "NP_Pos"
        backbone = template_mass(
            float(source.ref_mz),
            source["_ion"],
            bmp_family,
            positive=source.stream.endswith("Pos"),
        )
        if backbone is None:
            continue
        partners = bmp.loc[bmp.stream.eq(opposite) & bmp.name.eq(source["name"])].copy()
        valid = pd.Series(False, index=partners.index)
        ppm = pd.Series(np.nan, index=partners.index)
        for ion, (offset, charge) in bmp_family.items():
            if not charge or (charge > 0) != opposite.endswith("Pos"):
                continue
            expected = abs((backbone + offset) / charge + PROTON_MASS)
            error = ((partners.feat_mz / expected - 1) * 1e6).abs()
            ref_error = ((partners.ref_mz / expected - 1) * 1e6).abs()
            belongs = partners._ion.eq(ion) & error.lt(3) & ref_error.lt(3)
            valid |= belongs
            ppm.loc[belongs] = error.loc[belongs]
        partners = partners.loc[valid].assign(_ppm=ppm.loc[valid])
        output.loc[idx, PREFIX + "candidate_counterparts"] = float(
            partners._fid.nunique()
        )
        values, best = summary(source, partners, "counterpart")
        for f, value in values.items():
            output.loc[idx, PREFIX + f] = value
        if best is not None and source.stream == "NP_Pos":
            negative = pa_evidence.get(best.name)
            # best.name is the original candidate index, NOT its lipid name.
            if negative is not None:
                for f in ("candidates", "profiles", "raw", "log_ratio_location"):
                    output.loc[idx, PREFIX + "counterpart_PA_" + f] = negative.get(
                        "PA_" + f, np.nan
                    )
    return output.replace([np.inf, -np.inf], np.nan)
