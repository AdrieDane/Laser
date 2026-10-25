"""Pure class-RT-valid cross-PHASE graph/profile measurements, not annotations."""

import numpy as np
import pandas as pd

PREFIX = "q_cross_phase_class_"
MODES = ("RP_Neg", "NP_Neg")
FIELDS = (
    "source_physical",
    "counterphase_configured",
    "candidate_counterparts",
    "counterpart_profiles",
    "best_raw",
    "best_normalized",
    "best_within",
    "best_spearman",
    "best_leave_one_out",
    "best_normalized_leave_one_out",
    "best_log_ratio_location",
    "best_mad",
    "best_n",
    "best_dof",
    "profile_margin",
    "best_theory_neutral_ppm",
    "best_observed_neutral_delta_ppm",
)


def measure(pool, features, specimen_mapping, families, physical):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.ion_footprints import neutral_ion_mass
    from matlab_annotation.paired_sample_evidence import _profiles
    from matlab_annotation.raw_counterpart_evidence import (
        _profile_result,
        _validate_original_queries,
        _family,
    )
    from matlab_annotation.raw_counterpart_rt import SUPERVISION
    from matlab_annotation.standard_rt_library import ion_key

    if SUPERVISION & set(pool):
        raise ValueError("Cross-phase measurements refuse supervision")
    if (
        not pool.index.is_unique
        or not pool.index.equals(physical.index)
        or not physical.isin([False, True]).all()
    ):
        raise ValueError("Aligned explicit scoped physical eligibility required")
    if not {
        "stream",
        "class",
        "name",
        "adduct",
        "ref_mz",
        "feat_mz",
        "feat_rt",
        "feat_intensity",
        "feat_feature_id",
        "is_is",
    }.issubset(pool):
        raise ValueError("Complete original physical hypotheses required")
    if not pool.is_is.isin([False, True]).all():
        raise ValueError("Explicit original standard flags required")
    classes = {canonical_lipid_class(c) for c in ("FA", "FA-OH")}
    p = pool.assign(
        _cls=pool["class"].map(canonical_lipid_class),
        _fid=pool.feat_feature_id.astype(str),
        _ion=pool.adduct.map(ion_key),
    )
    query = p.loc[
        p.stream.isin(MODES) & p._cls.isin(classes) & ~p.is_is.astype(bool)
    ].copy()
    for s in MODES:
        other = MODES[1] if s == MODES[0] else MODES[0]
        part = query.loc[query.stream.eq(s)]
        _validate_original_queries(part, features, s, other)
        original = (
            features[s].set_index(features[s].feature_id.astype(str)).loc[part._fid]
        )
        np.testing.assert_allclose(
            part.feat_intensity.to_numpy(float),
            original.intensity.to_numpy(float),
            rtol=8 * np.finfo(float).eps,
            atol=0,
            equal_nan=True,
        )
    query["_neutral"] = [
        neutral_ion_mass(m, a)
        for m, a in query[["ref_mz", "adduct"]].itertuples(index=False, name=None)
    ]
    query["_observed_neutral"] = [
        neutral_ion_mass(m, a)
        for m, a in query[["feat_mz", "adduct"]].itertuples(index=False, name=None)
    ]
    eligible = query.loc[
        physical.loc[query.index] & np.isfinite(query._neutral) & query._neutral.gt(0)
    ]
    output = pd.DataFrame(
        np.nan, index=query.index, columns=[PREFIX + c for c in FIELDS]
    )
    output[PREFIX + "source_physical"] = physical.loc[query.index].astype(float)
    profiles, centers, labels = {}, {}, {}
    for s in MODES:
        ids = set(query.loc[query.stream.eq(s), "_fid"])
        profiles[s], centers[s], labels[s] = _profiles(
            features[s], ids, specimen_mapping.get(s, {})
        )
    memo = {}

    def compare(sa, fa, sb, fb):
        key = (sa, fa, sb, fb)
        if key not in memo:
            shared = sorted(set(labels[sa]) & set(labels[sb]))
            if any(labels[sa][x] != labels[sb][x] for x in shared):
                raise ValueError("Cross-phase BIO group metadata disagree")
            memo[key] = (
                None
                if len(shared) < 8
                else _profile_result(
                    profiles[sa].loc[fa, shared].to_numpy(float),
                    profiles[sb].loc[fb, shared].to_numpy(float),
                    centers[sa].reindex(shared).to_numpy(float),
                    centers[sb].reindex(shared).to_numpy(float),
                    [labels[sa][x] for x in shared],
                )
            )
        return memo[key]

    links = []
    for idx, source in query.iterrows():
        other = MODES[1] if source.stream == MODES[0] else MODES[0]
        own_family = _family(families, source.stream[:2], source["_cls"])
        other_family = _family(families, other[:2], source["_cls"])
        configured = bool(source["_ion"] in own_family and other_family)
        output.loc[idx, PREFIX + "counterphase_configured"] = float(configured)
        if (
            not configured
            or not bool(physical.loc[idx])
            or not np.isfinite(source["_neutral"])
        ):
            continue
        candidates = eligible.loc[
            eligible.stream.eq(other)
            & eligible._cls.eq(source["_cls"])
            & eligible._ion.isin(other_family)
        ].copy()
        ppm = (
            (candidates._neutral - float(source["_neutral"])).abs()
            / np.minimum(candidates._neutral, float(source["_neutral"]))
            * 1e6
        )
        matching = ppm.le(0.05 + 1e-12)
        candidates = candidates.loc[matching].assign(
            _theory_ppm=ppm.loc[matching], _theory_rank=ppm.loc[matching].round(8)
        )
        candidates = candidates.sort_values(
            ["_theory_rank", "feat_intensity"], ascending=[True, False], kind="stable"
        ).drop_duplicates(["stream", "_fid"])
        output.loc[idx, PREFIX + "candidate_counterparts"] = float(len(candidates))
        measured = []
        for target_idx, target in candidates.iterrows():
            comparison = compare(source.stream, source["_fid"], other, target["_fid"])
            links.append(
                dict(
                    source_index=idx,
                    target_index=target_idx,
                    source_stream=source.stream,
                    target_stream=other,
                    source_feature=source["_fid"],
                    target_feature=target["_fid"],
                    theory_neutral_ppm=float(target["_theory_ppm"]),
                    profile_measured=comparison is not None,
                )
            )
            if comparison is not None:
                measured.append(
                    (
                        comparison["raw"],
                        -float(target["_theory_ppm"]),
                        float(target.feat_intensity),
                        comparison,
                        target,
                    )
                )
        output.loc[idx, PREFIX + "counterpart_profiles"] = float(len(measured))
        if not measured:
            continue
        # Stable scientific ties: only ranking keys are rounded, never raw
        # intensities, observed coordinates or reported profile measurements.
        measured.sort(
            key=lambda x: (round(x[0], 12), round(x[1], 8), x[2]), reverse=True
        )
        comp, target = measured[0][3:]
        for field in (
            "raw",
            "normalized",
            "within",
            "spearman",
            "leave_one_out",
            "normalized_leave_one_out",
            "log_ratio_location",
            "mad",
            "n",
            "dof",
        ):
            output.loc[idx, PREFIX + "best_" + field] = comp[field]
        output.loc[idx, PREFIX + "profile_margin"] = (
            max(0.0, measured[0][0] - measured[1][0]) if len(measured) > 1 else np.nan
        )
        output.loc[idx, PREFIX + "best_theory_neutral_ppm"] = float(
            target["_theory_ppm"]
        )
        output.loc[idx, PREFIX + "best_observed_neutral_delta_ppm"] = (
            float(target["_observed_neutral"]) / float(source["_observed_neutral"]) - 1
        ) * 1e6
    return output, pd.DataFrame(
        links,
        columns=[
            "source_index",
            "target_index",
            "source_stream",
            "target_stream",
            "source_feature",
            "target_feature",
            "theory_neutral_ppm",
            "profile_measured",
        ],
    )
