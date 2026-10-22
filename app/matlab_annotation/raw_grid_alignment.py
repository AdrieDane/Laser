"""Research-only raw mass-series RT alignment, independent of curation.

This produces RT hypotheses, NOT annotations. A broad search locates coherent
class offsets; any later feature admission must still pass its declared strict
mass/RT and ambiguity criteria. Own composition cannot calibrate its RT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from annotator.composition_ranges import canonical_lipid_class
from .automatic import matlab_annotation_adducts


def _cluster(rows, center, tolerance):
    part = rows[rows.delta.sub(center).abs().le(tolerance)]
    return part.sort_values(
        ["relative_intensity", "absolute_ppm"], ascending=[False, True], kind="stable"
    ).drop_duplicates("composition")


def _candidate_centers(votes, tolerance):
    """Exact support prefilter; avoids sorting centers with fewer than9 votes."""
    centers = np.arange(-90.0, 90.001, 2.0)
    codes, compositions = pd.factorize(votes.composition, sort=False)
    inside = np.abs(votes.delta.to_numpy(float)[:, None] - centers) <= tolerance
    support = np.zeros((len(compositions), len(centers)), dtype=bool)
    np.logical_or.at(support, codes, inside)
    return centers[support.sum(axis=0) >= 9]


def infer_raw_grid_alignment(
    targets, features, priors, *, stream, declared_standard_masses=()
):
    """Find a clearly dominant OTHER-composition RT-shift hypothesis per target.

    Anchor masses <1.5ppm, within90s of prior RT, >=10% of that composition's
    strongest mass-matched feature. Only compositions seen in >=25% and >=3
    OTHER training groups may anchor (max80 most prevalent/class). At least8
    OTHER compositions with C span>=4 and >=2 DB levels must agree. Competing
    clusters within2 compositions of the best are ambiguous. No label/feature
    selection is performed, and no curated RT is a runtime input.
    """
    required = {"name", "class", "carbon", "db", "adduct", "ref_mz", "atlas_rt_seconds"}
    if not required.issubset(targets):
        raise ValueError("Raw-grid targets need independent RT, mass and composition")
    if not {"feature_id", "mz", "rt", "intensity"}.issubset(features):
        raise ValueError("Raw grid requires original feature coordinates/intensities")
    out = pd.DataFrame(index=targets.index)
    out["raw_grid_status"] = "insufficient_support"
    for col in [
        "raw_grid_rt_seconds",
        "raw_grid_shift_seconds",
        "raw_grid_support",
        "raw_grid_competing_support",
        "raw_grid_mad_seconds",
    ]:
        out[col] = np.nan
    if targets.empty or features.empty:
        return out
    masses = pd.to_numeric(features.mz, errors="coerce").to_numpy(float)
    times = pd.to_numeric(features.rt, errors="coerce").to_numpy(float) * 60
    intensity = pd.to_numeric(features.intensity, errors="coerce").to_numpy(float)
    good = np.isfinite(masses) & (masses > 0) & np.isfinite(times) & (intensity > 0)
    spikes = np.sort(np.asarray(list(declared_standard_masses), dtype=float))
    spikes = spikes[np.isfinite(spikes) & (spikes > 0)]
    if len(spikes):
        nearest = np.clip(np.searchsorted(spikes, masses), 0, len(spikes) - 1)
        previous = np.clip(nearest - 1, 0, len(spikes) - 1)
        distance = np.minimum(
            np.abs(masses - spikes[nearest]) / spikes[nearest],
            np.abs(masses - spikes[previous]) / spikes[previous],
        )
        good &= distance >= 3e-6
    order = np.flatnonzero(good)
    order = order[np.argsort(masses[order], kind="stable")]
    sorted_mass = masses[order]
    canonical = targets["class"].map(canonical_lipid_class)
    tolerance = 3.0 if stream.startswith("NP") else 5.0
    for cls, indices in canonical.groupby(canonical, sort=False).groups.items():
        prior = priors.get(stream + "|" + cls, {})
        groups = int(prior.get("cohorts", 0))
        if groups < 3:
            continue
        refs = targets.loc[indices].copy()
        refs = refs[
            ~refs.name.astype(str).str.endswith(" (IS)")
            & refs.adduct.isin(matlab_annotation_adducts(stream[:2]))
        ]
        refs = refs[
            np.isfinite(refs[["carbon", "db", "ref_mz", "atlas_rt_seconds"]]).all(
                axis=1
            )
        ]
        refs["composition"] = list(zip(refs.carbon, refs.db))
        refs["prevalence"] = [
            prior.get("joint", {}).get(f"{int(c)}:{int(d)}", 0)
            for c, d in refs.composition
        ]
        refs = refs[refs.prevalence.ge(max(3, 0.25 * groups))]
        identities = (
            refs.sort_values("prevalence", ascending=False, kind="stable")
            .composition.drop_duplicates()
            .head(80)
        )
        refs = refs[refs.composition.isin(identities)].drop_duplicates(
            ["composition", "adduct", "ref_mz"]
        )
        rows = []
        for ref in refs.itertuples():
            mz, predicted = float(ref.ref_mz), float(ref.atlas_rt_seconds)
            if mz <= 0 or predicted <= 0:
                continue
            lo, hi = np.searchsorted(
                sorted_mass, [mz * (1 - 1.5e-6), mz * (1 + 1.5e-6)]
            )
            for pos in order[lo:hi]:
                ppm = abs((masses[pos] - mz) / mz * 1e6)
                delta = times[pos] - predicted
                if ppm < 1.5 and abs(delta) <= 90:
                    rows.append(
                        dict(
                            composition=ref.composition,
                            carbon=float(ref.carbon),
                            db=float(ref.db),
                            delta=delta,
                            intensity=intensity[pos],
                            absolute_ppm=ppm,
                            position=int(pos),
                        )
                    )
        if not rows:
            continue
        votes = pd.DataFrame(rows).drop_duplicates(["composition", "position"])
        # A mass-degenerate original feature is not independent evidence for
        # several distinct compositions. Withhold every such ambiguous vote.
        alternatives = votes.groupby("position").composition.transform("nunique")
        votes = votes[alternatives.eq(1)].copy()
        if votes.empty:
            continue
        maximum = votes.groupby("composition").intensity.transform("max")
        votes["relative_intensity"] = votes.intensity / maximum
        votes = votes[votes.relative_intensity.ge(0.1)]
        if votes.composition.nunique() < 9:
            continue
        hypotheses = []
        for center in _candidate_centers(votes, tolerance):
            part = _cluster(votes, center, tolerance)
            if len(part) < 9:
                continue
            center = float(part.delta.median())
            part = _cluster(votes, center, tolerance)
            hypotheses.append(
                (
                    len(part),
                    float(part.relative_intensity.sum()),
                    -abs(center),
                    center,
                    part,
                )
            )
        if not hypotheses:
            continue
        best = max(hypotheses, key=lambda x: x[:3])
        count, _, _, center, anchors = best
        runner = max(
            [h[0] for h in hypotheses if abs(h[3] - center) > 2 * tolerance] or [0]
        )
        if count - 1 < runner + 2:
            out.loc[indices, "raw_grid_status"] = "ambiguous_competing_grid"
            continue
        for (carbon, db), query in targets.loc[indices].groupby(
            ["carbon", "db"], sort=False
        ):
            other = anchors[~(anchors.carbon.eq(carbon) & anchors.db.eq(db))]
            if (
                len(other) < 8
                or other.carbon.max() - other.carbon.min() < 4
                or other.db.nunique() < 2
            ):
                continue
            shift = float(other.delta.median())
            error = np.abs(other.delta.to_numpy(float) - shift)
            if np.median(error) > (2 if stream.startswith("NP") else 3) or np.quantile(
                error, 0.9
            ) > (4 if stream.startswith("NP") else 6):
                continue
            if not (
                other.carbon.min() - 4 <= carbon <= other.carbon.max() + 4
                and other.db.min() - 2 <= db <= other.db.max() + 2
            ):
                continue
            idx = query.index
            out.loc[idx, "raw_grid_support"] = len(other)
            out.loc[idx, "raw_grid_competing_support"] = runner
            out.loc[idx, "raw_grid_mad_seconds"] = float(np.median(error))
            out.loc[idx, "raw_grid_shift_seconds"] = shift
            if abs(shift) < 5:
                out.loc[idx, "raw_grid_status"] = "near_original_no_extra_shift"
                continue
            out.loc[idx, "raw_grid_rt_seconds"] = query.atlas_rt_seconds + shift
            out.loc[idx, "raw_grid_status"] = "independent_mass_series_shift"
    return out
