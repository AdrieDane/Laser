"""Connected CH2/H2 lattice families from mass and retention time alone.

No manual lipid labels or class reference identities enter this search. A
candidate grid is a homologous-family hypothesis, not a compound assignment.
"""

from collections import deque

import numpy as np
import pandas as pd

CH2 = 14.015650064
H2 = 2.01565006446
GRID_VERSION = 1
GRID_COLUMNS = (
    "stream",
    "source_feature_id",
    "target_feature_id",
    "carbon_steps",
    "hydrogen_pair_steps",
    "charge_hypothesis",
    "source_mz",
    "source_rt",
    "target_mz",
    "target_rt",
    "expected_rt",
    "rt_residual_seconds",
    "mass_error_ppm",
    "grid_size",
    "carbon_rt_slope",
    "hydrogen_pair_rt_slope",
)


class LipidGridIndex:
    def __init__(self, features, ppm=4.0, edge_rt=0.5, adduct_alias_rt=None):
        self.mz = features.mz.to_numpy(float)
        self.rt = features.rt.to_numpy(float)
        self.ids = features.feature_id.astype(str).to_numpy()
        self.lookup = {fid: i for i, fid in enumerate(self.ids)}
        self.ppm, self.edge_rt = float(ppm), float(edge_rt)
        self.adduct_alias_rt = adduct_alias_rt

    def ambiguous(self, left, right, charge):
        return (self.adduct_alias_rt is not None and charge == 1
                and abs(self.rt[left] - self.rt[right]) <= self.adduct_alias_rt + 1e-10
                and abs(abs(self.mz[left] - self.mz[right]) - CH2)
                <= max(self.mz[left], self.mz[right]) * self.ppm * 1e-6)

    def candidates(self, mass):
        tolerance = mass * self.ppm * 1e-6
        return range(
            np.searchsorted(self.mz, mass - tolerance),
            np.searchsorted(self.mz, mass + tolerance, side="right"),
        )

    def family(self, feature_id, charge=1, phase="RP"):
        root = self.lookup.get(str(feature_id))
        if root is None:
            return []
        mz, rt = self.mz[root], self.rt[root]
        slopes = [[], []]
        for axis, step_mass in enumerate((CH2, H2)):
            for direction in (-1, 1):
                expected = mz + direction * step_mass / charge
                choices = [
                    i
                    for i in self.candidates(expected)
                    if abs(self.rt[i] - rt) <= self.edge_rt
                    and not self.ambiguous(root, i, charge)
                ]
                if phase == "RP":
                    choices = [
                        i for i in choices if (self.rt[i] - rt) * direction >= -0.03
                    ]
                if not choices:
                    continue
                best = min(
                    choices,
                    key=lambda i: (
                        abs(self.rt[i] - rt) / self.edge_rt
                        + abs(self.mz[i] - expected) / expected * 1e6 / self.ppm
                    ),
                )
                slopes[axis].append((self.rt[best] - rt) / direction)
        # A lone mass-spaced pair cannot establish a two-dimensional family.
        if not all(slopes) or sum(map(len, slopes)) < 3:
            return []
        c_slope, h_slope = (float(np.median(values)) for values in slopes)
        if phase == "RP" and (c_slope < 0.02 or h_slope < 0.02):
            return []
        residual_tolerance = min(0.20, self.edge_rt)
        cells = {(0, 0): (root, rt, 0.0)}
        for carbon in range(-12, 13):
            for hydrogen in range(-8, 9):
                if carbon == hydrogen == 0:
                    continue
                expected = mz + (carbon * CH2 + hydrogen * H2) / charge
                if expected <= 0:
                    continue
                predicted_rt = rt + carbon * c_slope + hydrogen * h_slope
                choices = [
                    i
                    for i in self.candidates(expected)
                    if abs(self.rt[i] - predicted_rt) <= residual_tolerance
                ]
                if not choices:
                    continue
                best = min(
                    choices,
                    key=lambda i: (
                        abs(self.rt[i] - predicted_rt) / residual_tolerance
                        + abs(self.mz[i] - expected) / expected * 1e6 / self.ppm * 0.2
                    ),
                )
                cells[(carbon, hydrogen)] = (
                    best,
                    predicted_rt,
                    (self.mz[best] - expected) / expected * 1e6,
                )
        # Only occupied adjacent cells can connect. Never hop over an empty
        # grid cell or across a retention-time gap to a disconnected RT family.
        queue, visited = deque([(0, 0)]), {(0, 0)}
        while queue:
            carbon, hydrogen = queue.popleft()
            current = cells[(carbon, hydrogen)][0]
            for step_c, step_h in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbour = (carbon + step_c, hydrogen + step_h)
                if neighbour in visited or neighbour not in cells:
                    continue
                if abs(self.rt[cells[neighbour][0]] - self.rt[current]) > self.edge_rt:
                    continue
                if self.ambiguous(current, cells[neighbour][0], charge):
                    continue
                visited.add(neighbour)
                queue.append(neighbour)
        if (
            len(visited) < 5
            or len({c for c, _ in visited}) < 2
            or len({h for _, h in visited}) < 2
        ):
            return []
        records = []
        for carbon, hydrogen in sorted(visited - {(0, 0)}):
            index, expected_rt, mass_error = cells[(carbon, hydrogen)]
            records.append(
                {
                    "source_feature_id": str(feature_id),
                    "target_feature_id": self.ids[index],
                    "carbon_steps": carbon,
                    "hydrogen_pair_steps": hydrogen,
                    "charge_hypothesis": charge,
                    "source_mz": mz,
                    "source_rt": rt,
                    "target_mz": self.mz[index],
                    "target_rt": self.rt[index],
                    "expected_rt": expected_rt,
                    "rt_residual_seconds": (self.rt[index] - expected_rt) * 60,
                    "mass_error_ppm": mass_error,
                    "grid_size": len(visited),
                    "carbon_rt_slope": c_slope,
                    "hydrogen_pair_rt_slope": h_slope,
                }
            )
        return records


def search_lipid_grids(by_stream, sources, criteria, method=None):
    from standard_lipidomics.acquisition_method import ion_state
    records = []
    for stream, table in by_stream.items():
        source = sources.get(stream, pd.DataFrame())
        if source.empty or table.empty:
            continue
        alias_rt = None
        if (method is not None and method['adduct_scoring'] == 'method_aware' and stream.endswith('Neg')
                and all(ion_state(method, stream, ion) != 'disabled' for ion in ('[M+HCOO]-', '[M+CH3COO]-'))):
            alias_rt = criteria.same_mode_adduct_rt_tolerance_min
        index = LipidGridIndex(source, criteria.cross_mode_ppm_tolerance,
                               criteria.ch2_rt_tolerance_min, alias_rt)
        mask = table.get(
            "raw_p_significant",
            table.get("significant", pd.Series(False, index=table.index)),
        ).fillna(False)
        for hit in table[mask].itertuples(index=False):
            charge = getattr(hit, "isotope_charge_hypothesis", np.nan)
            charge = int(charge) if pd.notna(charge) else 1
            records.extend(
                {"stream": stream, **row}
                for row in index.family(
                    str(hit.feature_id), charge, stream.split("_")[0]
                )
            )
    return pd.DataFrame(records, columns=GRID_COLUMNS)
