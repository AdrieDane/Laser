"""Scalar graph/native-charge and original-area NumPy profile reconstruction.

Does NOT call producer, neutral_ion_mass, _family, _profiles or _profile_result.
Shares native catalogue, metadata parser, output schema and independently
implemented profile mathematics from the previously checked raw pilot.
"""

import numpy as np
import pandas as pd

from cross_phase_candidate_measurements import PREFIX, FIELDS, MODES


def independent_measure(pool, features, maps, families, physical):
    from annotator.composition_ranges import canonical_lipid_class
    from matlab_annotation.standard_rt_library import ion_key
    from matlab_annotation.vendor.composition_initializer import _ADDUCTS
    from matlab_annotation.vendor.compound_model import PROTON_MASS
    import verify_qe_raw_counterpart_pilot as numeric

    ions = {ion_key(label): (float(add), int(z)) for label, add, z in _ADDUCTS.values()}
    classes = {canonical_lipid_class(c) for c in ("FA", "FA-OH")}
    p = pool.copy()
    p["canonical"] = p["class"].map(canonical_lipid_class)
    p["fid"] = p.feat_feature_id.astype(str)
    p["ion"] = p.adduct.map(ion_key)
    query = p.loc[
        p.stream.isin(MODES) & p.canonical.isin(classes) & ~p.is_is.astype(bool)
    ]
    out = pd.DataFrame(np.nan, index=query.index, columns=[PREFIX + c for c in FIELDS])
    out[PREFIX + "source_physical"] = physical.loc[query.index].astype(float)
    matrices = {s: numeric.profiles(features[s], maps.get(s, {})) for s in MODES}

    def neutral(mz, ion):
        if ion not in ions or not np.isfinite(float(mz)) or float(mz) <= 0:
            return np.nan
        add, z = ions[ion]
        return abs(z) * float(mz) - z * PROTON_MASS - add

    def profile(a, b):
        am, ac, al = matrices[a.stream]
        bm, bc, bl = matrices[b.stream]
        shared = sorted(set(al) & set(bl))
        assert all(al[s] == bl[s] for s in shared)
        if len(shared) < 8:
            return None
        return numeric.comparison(
            am.loc[a.fid, shared].to_numpy(float),
            bm.loc[b.fid, shared].to_numpy(float),
            np.array([ac[s] for s in shared]),
            np.array([bc[s] for s in shared]),
            np.array([al[s] for s in shared]),
        )

    links = []
    for source in query.itertuples():
        other = MODES[1] if source.stream == MODES[0] else MODES[0]
        own = families.get((source.stream[:2], source.canonical), {})
        target_family = families.get((other[:2], source.canonical), {})
        configured = bool(source.ion in own and target_family)
        out.loc[source.Index, PREFIX + "counterphase_configured"] = float(configured)
        mass = neutral(source.ref_mz, source.ion)
        if not configured or not physical.loc[source.Index] or not np.isfinite(mass):
            continue
        unique = {}
        for target in query.itertuples():
            if (
                target.stream != other
                or target.canonical != source.canonical
                or not physical.loc[target.Index]
                or target.ion not in target_family
            ):
                continue
            tm = neutral(target.ref_mz, target.ion)
            if not np.isfinite(tm) or tm <= 0:
                continue
            ppm = abs(tm - mass) / min(tm, mass) * 1e6
            if ppm > 0.05 + 1e-12:
                continue
            rank = (round(ppm, 8), -float(target.feat_intensity))
            if target.fid not in unique or rank < unique[target.fid][0]:
                unique[target.fid] = (rank, target, ppm)
        hits = sorted(unique.values(), key=lambda x: x[0])
        out.loc[source.Index, PREFIX + "candidate_counterparts"] = float(len(hits))
        measured = []
        for _rank, target, ppm in hits:
            comp = profile(source, target)
            links.append(
                dict(
                    source_index=source.Index,
                    target_index=target.Index,
                    source_stream=source.stream,
                    target_stream=target.stream,
                    source_feature=source.fid,
                    target_feature=target.fid,
                    theory_neutral_ppm=ppm,
                    profile_measured=comp is not None,
                )
            )
            if comp is not None:
                measured.append(
                    (comp["raw"], -ppm, float(target.feat_intensity), target, comp)
                )
        out.loc[source.Index, PREFIX + "counterpart_profiles"] = float(len(measured))
        if not measured:
            continue
        ordered = sorted(
            measured,
            key=lambda x: (round(x[0], 12), round(x[1], 8), x[2]),
            reverse=True,
        )
        best = ordered[0]
        target, comp = best[3:]
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
            out.loc[source.Index, PREFIX + "best_" + field] = comp[field]
        out.loc[source.Index, PREFIX + "profile_margin"] = (
            max(0.0, best[0] - ordered[1][0]) if len(ordered) > 1 else np.nan
        )
        out.loc[source.Index, PREFIX + "best_theory_neutral_ppm"] = abs(best[1])
        a, b = neutral(source.feat_mz, source.ion), neutral(target.feat_mz, target.ion)
        out.loc[source.Index, PREFIX + "best_observed_neutral_delta_ppm"] = (
            b / a - 1
        ) * 1e6
    return out, pd.DataFrame(
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
