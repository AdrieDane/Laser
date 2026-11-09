"""Native mapping and first-pass calibration of MATLAB RTDB observations."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from scipy.interpolate import interp1d

from .candidate_engine import match_targets
from .compound_model import DecodedCode, encode_dense_code, phase_composition, theoretical_mz
from .project import LegacyProject
from .reference import ReferenceData, RetentionTimeObservation
from .rt_model import (
    FA_PREPROCESSING_PENALTIES,
    R_GROUP_RANGES,
    fa_preprocessing_matrix,
)


_TOTAL_TRAIT = re.compile(
    r"^(?P<prefix>[^()]*)\((?P<marker>[^\d()]*)"
    r"(?P<carbon>\d+):(?P<double_bonds>\d+)\)(?: \(IS\))?$"
)
_ISOPRENE_TRAIT = re.compile(r"^(?P<prefix>[^()]*)\((?P<count>\d+)\)$")
_R_GROUP_RANGES = {**R_GROUP_RANGES, "isoprene": np.asarray([[10.0, 25.0]])}


def _canonical_trait_key(value: str) -> str:
    """Return the catalogue key form used for RTDB trait matching.

    The bundled catalogue renders positional lyso species as ``1-acylLPC``
    while the SRD5A3 RTDB writes the same trait as ``1-acyl LPC``.  Whitespace
    is not part of the chemical key, so normalise it at the mapping boundary.
    This also makes the mapping robust to harmless spacing differences in
    future RTDB/catalogue revisions without changing the displayed trait.
    """

    return re.sub(r"\s+", "", str(value))


@dataclass(frozen=True)
class MappedRetentionTime:
    trait: str
    phase: str
    study: str
    retention_time_seconds: float
    compound: int
    composition_index_matlab: int
    carbon: int
    double_bonds: int


@dataclass(frozen=True)
class StudyPreprocessingFit:
    study: str
    global_beta: np.ndarray
    compound_beta: np.ndarray
    compounds: np.ndarray
    predicted_rt_seconds: np.ndarray
    observed_rt_seconds: np.ndarray


@dataclass(frozen=True)
class RetentionTimePrediction:
    observation: MappedRetentionTime
    predicted_rt_seconds: float


@dataclass(frozen=True)
class RtdbMassTarget:
    observation: MappedRetentionTime
    code: int
    theoretical_mz: float
    predicted_rt_seconds: float
    abundant: bool


@dataclass(frozen=True)
class CorrectedRetentionTimePrediction:
    observation: MappedRetentionTime
    predicted_rt_seconds: float
    corrected_rt_seconds: float


@dataclass(frozen=True)
class ProjectFaCalibration:
    study_order: tuple[str, ...]
    correction_coefficients: np.ndarray
    correction_scale: float
    preprocessing_beta: np.ndarray
    isoprene_preprocessing_beta: np.ndarray


def _string(value) -> str:
    array = np.asarray(value)
    return "" if array.size == 0 else str(value)


def _r_group_labels(project: LegacyProject) -> np.ndarray:
    return np.asarray([
        _string(value)
        for value in np.asarray(
            project.compounds.legacy_fields["R_group"], dtype=object
        ).reshape(-1)
    ])


def _catalog_trait_keys(project: LegacyProject) -> dict[str, int]:
    keys: dict[str, int] = {}
    duplicates: set[str] = set()
    composition = np.asarray(
        project.compounds.legacy_fields["composition"], dtype=object
    )
    for compound, (label, parts) in enumerate(
        zip(project.compounds.classes, composition, strict=True), start=1
    ):
        key = _canonical_trait_key(
            f"{_string(parts[0])}{label}{_string(parts[1])}_{_string(parts[2])}"
        )
        if key in keys:
            duplicates.add(key)
        else:
            keys[key] = compound
    for key in duplicates:
        keys.pop(key, None)
    return keys


def map_retention_time_observations(
    project: LegacyProject,
    reference: ReferenceData,
) -> tuple[MappedRetentionTime, ...]:
    """Port the total-composition branch of MATLAB ``get_compound_index``."""
    keys = _catalog_trait_keys(project)
    r_groups = _r_group_labels(project)
    n_r = np.asarray(
        project.compounds.legacy_fields["nR"], dtype=object
    ).reshape(-1)
    mapped: list[MappedRetentionTime] = []
    for observation in reference.retention_times:
        normalized = re.sub(r" \(IS\)$", "", observation.trait.strip())
        match = _TOTAL_TRAIT.fullmatch(normalized)
        isoprene = _ISOPRENE_TRAIT.fullmatch(normalized)
        if match is not None:
            key = f"{match.group('prefix')}_{match.group('marker')}"
            carbon = int(match.group("carbon"))
            double_bonds = int(match.group("double_bonds"))
        elif isoprene is not None:
            key = f"{isoprene.group('prefix')}_"
            carbon = int(isoprene.group("count"))
            double_bonds = 0
        else:
            key = f"{normalized}_"
            carbon = 0
            double_bonds = 0
        compound = keys.get(_canonical_trait_key(key))
        if compound is None:
            continue
        r_group = str(r_groups[compound - 1])
        if r_group not in _R_GROUP_RANGES and r_group != "":
            continue
        ranges = _R_GROUP_RANGES.get(r_group, np.empty((0, 2)))
        count = int(n_r[compound - 1])
        values = np.asarray([carbon, double_bonds], dtype=int)[: len(ranges)]
        minimum = ranges[:, 0].astype(int) * count
        sizes = (ranges[:, 1] - ranges[:, 0]).astype(int) * count + 1
        coordinates = values - minimum
        if np.any(coordinates < 0) or np.any(coordinates >= sizes):
            continue
        composition_index = 1
        for coordinate, size in zip(coordinates, sizes, strict=True):
            composition_index = (composition_index - 1) * int(size) + int(coordinate) + 1
        mapped.append(MappedRetentionTime(
            trait=observation.trait,
            phase=observation.phase,
            study=observation.study,
            retention_time_seconds=observation.retention_time_seconds,
            compound=compound,
            composition_index_matlab=composition_index,
            carbon=carbon,
            double_bonds=double_bonds,
        ))
    return tuple(mapped)


def _combined_leaders(project: LegacyProject) -> np.ndarray:
    leaders = np.arange(1, project.compounds.count + 1, dtype=np.int64)
    first: dict[str, int] = {}
    for compound, label in enumerate(project.compounds.combined, start=1):
        if not label:
            continue
        leaders[compound - 1] = first.setdefault(label, compound)
    return leaders


def _fa_rows(
    project: LegacyProject,
    rows: tuple[MappedRetentionTime, ...] | list[MappedRetentionTime],
    *,
    phase: str,
) -> np.ndarray:
    n_r = np.asarray(
        project.compounds.legacy_fields["nR"], dtype=float
    ).reshape(-1)
    composition = np.asarray(
        project.compounds.legacy_fields["composition"], dtype=object
    )
    carbon = np.asarray([item.carbon for item in rows], dtype=float)
    double_bonds = np.asarray([item.double_bonds for item in rows], dtype=float)
    compounds = np.asarray([item.compound for item in rows], dtype=np.int64)
    count = n_r[compounds - 1]
    sphingolipid = np.asarray([
        len(composition[item - 1]) >= 3 and str(composition[item - 1][2]) == "d"
        for item in compounds
    ])
    return fa_preprocessing_matrix(
        carbon, double_bonds, count, sphingolipid, phase=phase
    )


def fit_fa_study_preprocessing(
    project: LegacyProject,
    mapped: tuple[MappedRetentionTime, ...] | list[MappedRetentionTime],
    *,
    phase: str,
    study: str,
) -> StudyPreprocessingFit:
    """Port the first per-study FA regression in ``correct_RT_database_new``."""
    r_groups = _r_group_labels(project)
    rows = tuple(
        item for item in mapped
        if item.phase == phase and item.study == study
        and str(r_groups[item.compound - 1]) == "FA"
    )
    if not rows:
        raise ValueError(f"no mapped FA observations for {phase}/{study}")
    raw = _fa_rows(project, rows, phase=phase)
    y = np.asarray([item.retention_time_seconds for item in rows], dtype=float)
    compounds = np.asarray([item.compound for item in rows], dtype=np.int64)
    leaders = _combined_leaders(project)
    penalties = FA_PREPROCESSING_PENALTIES[phase]
    width = len(penalties)
    columns: dict[tuple[str, int, int], int] = {}
    triplets: list[tuple[int, int, float]] = []
    diagonal: list[float] = []

    def column(key: tuple[str, int, int], penalty: float) -> int:
        if key not in columns:
            columns[key] = len(columns)
            diagonal.append(penalty)
        return columns[key]

    for row in range(len(raw)):
        for feature in np.flatnonzero(raw[row] != 0):
            value = float(raw[row, feature])
            if penalties[feature] > 0:
                triplets.append((
                    row, column(("global", 0, int(feature)), 0.0), value
                ))
            owner = (
                int(leaders[compounds[row] - 1])
                if penalties[feature] != 0
                else int(compounds[row])
            )
            triplets.append((
                row,
                column(("compound", owner, int(feature)), abs(penalties[feature])),
                value,
            ))
    design = np.zeros((len(raw), len(columns)), dtype=float)
    for row, col, value in triplets:
        design[row, col] += value
    normal = design.T @ design + np.diag(diagonal)
    coefficients = np.linalg.solve(normal, design.T @ y)
    global_beta = np.zeros(width, dtype=float)
    compound_beta = np.full((width, project.compounds.count), np.nan, dtype=float)
    for feature in range(width):
        global_beta[feature] = coefficients[
            columns[("global", 0, feature)]
        ] if ("global", 0, feature) in columns else 0.0
    for compound in np.flatnonzero(r_groups == "FA") + 1:
        for feature in range(width):
            owner = int(leaders[compound - 1]) if penalties[feature] != 0 else int(compound)
            key = ("compound", owner, feature)
            local = coefficients[columns[key]] if key in columns else 0.0
            compound_beta[feature, compound - 1] = local + global_beta[feature]
    return StudyPreprocessingFit(
        study=study,
        global_beta=global_beta,
        compound_beta=compound_beta,
        compounds=compounds,
        predicted_rt_seconds=design @ coefficients,
        observed_rt_seconds=y,
    )


def predict_fa_rtdb_studies(
    project: LegacyProject,
    mapped: tuple[MappedRetentionTime, ...] | list[MappedRetentionTime],
    *,
    phase: str,
) -> tuple[RetentionTimePrediction, ...]:
    """Return MATLAB's first corrected RT for every mapped FA study row."""
    studies = []
    seen = set()
    for item in mapped:
        if item.phase == phase and item.study not in seen:
            studies.append(item.study)
            seen.add(item.study)
    result: list[RetentionTimePrediction] = []
    r_groups = _r_group_labels(project)
    for study in studies:
        rows = tuple(
            item for item in mapped
            if item.phase == phase and item.study == study
            and r_groups[item.compound - 1] == "FA"
        )
        if not rows:
            continue
        fit = fit_fa_study_preprocessing(
            project, rows, phase=phase, study=study
        )
        result.extend(
            RetentionTimePrediction(item, float(predicted))
            for item, predicted in zip(
                rows, fit.predicted_rt_seconds, strict=True
            )
        )
    return tuple(result)


def matlab_rtdb_study_order(
    mapped: tuple[MappedRetentionTime, ...] | list[MappedRetentionTime],
) -> tuple[str, ...]:
    """Return MATLAB's descending per-file trait-count study order."""
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for index, item in enumerate(mapped):
        key = (item.study, item.trait)
        if key in first:
            continue
        first[key] = index
        counts[item.study] = counts.get(item.study, 0) + 1
    return tuple(sorted(counts, key=lambda study: (-counts[study], min(
        position for (name, _), position in first.items() if name == study
    ))))


def _fit_simple_study_preprocessing(
    project: LegacyProject,
    rows: tuple[MappedRetentionTime, ...],
    *,
    study: str,
    r_group: str,
) -> StudyPreprocessingFit:
    compounds = np.asarray([item.compound for item in rows], dtype=np.int64)
    y = np.asarray([item.retention_time_seconds for item in rows], dtype=float)
    if r_group == "isoprene":
        # MATLAB's one-dimensional branch passes the one-based composition
        # index directly to the configured preprocessing function.
        n_r = np.asarray(project.compounds.legacy_fields["nR"], dtype=float).reshape(-1)
        scaled = np.asarray([
            item.composition_index_matlab for item in rows
        ], dtype=float) / n_r[compounds - 1] / 25.0
        raw = np.column_stack((np.ones(len(rows)), scaled - 0.5, scaled ** 0.5))
        penalties = np.asarray([0.0, -0.01, 0.01])
    elif r_group == "":
        raw = np.asarray([[item.composition_index_matlab] for item in rows], dtype=float)
        penalties = np.asarray([0.0])
    else:
        raise ValueError(f"unsupported RTDB preprocessing group {r_group!r}")
    leaders = _combined_leaders(project)
    columns: dict[tuple[str, int, int], int] = {}
    diagonal: list[float] = []
    triplets: list[tuple[int, int, float]] = []

    def column(key: tuple[str, int, int], penalty: float) -> int:
        if key not in columns:
            columns[key] = len(columns)
            diagonal.append(penalty)
        return columns[key]

    for row in range(len(raw)):
        for feature in np.flatnonzero(raw[row] != 0):
            value = float(raw[row, feature])
            if penalties[feature] > 0:
                triplets.append((row, column(("global", 0, int(feature)), 0.0), value))
            owner = int(leaders[compounds[row] - 1]) if penalties[feature] != 0 else int(compounds[row])
            triplets.append((row, column(("compound", owner, int(feature)), abs(penalties[feature])), value))
    design = np.zeros((len(raw), len(columns)), dtype=float)
    for row, col, value in triplets:
        design[row, col] += value
    coefficients = np.linalg.solve(
        design.T @ design + np.diag(diagonal), design.T @ y
    )
    return StudyPreprocessingFit(
        study=study,
        global_beta=np.asarray([
            coefficients[columns[("global", 0, feature)]]
            if ("global", 0, feature) in columns else 0.0
            for feature in range(len(penalties))
        ]),
        compound_beta=np.empty((len(penalties), 0)),
        compounds=compounds,
        predicted_rt_seconds=design @ coefficients,
        observed_rt_seconds=y,
    )


def predict_supported_rtdb_studies(
    project: LegacyProject,
    mapped: tuple[MappedRetentionTime, ...] | list[MappedRetentionTime],
    *,
    phase: str,
) -> tuple[RetentionTimePrediction, ...]:
    """Predict all natively supported RTDB rows used by project alignment."""
    r_groups = _r_group_labels(project)
    result: list[RetentionTimePrediction] = []
    for study in matlab_rtdb_study_order(mapped):
        for r_group in ("FA", "isoprene", ""):
            rows = tuple(item for item in mapped if (
                item.phase == phase and item.study == study
                and r_groups[item.compound - 1] == r_group
            ))
            if not rows:
                continue
            fit = (
                fit_fa_study_preprocessing(project, rows, phase=phase, study=study)
                if r_group == "FA"
                else _fit_simple_study_preprocessing(
                    project, rows, study=study, r_group=r_group
                )
            )
            result.extend(
                RetentionTimePrediction(item, float(predicted))
                for item, predicted in zip(rows, fit.predicted_rt_seconds, strict=True)
            )
    return tuple(result)


def build_fa_rtdb_mass_targets(
    project: LegacyProject,
    predictions: tuple[RetentionTimePrediction, ...] | list[RetentionTimePrediction],
    *,
    phase: str,
) -> tuple[RtdbMassTarget, ...]:
    """Expand first-pass FA predictions over each phase adduct as MATLAB does."""
    result: list[RtdbMassTarget] = []
    study_order = matlab_rtdb_study_order([item.observation for item in predictions])
    studies = {name: index + 1 for index, name in enumerate(study_order)}
    leaders = _combined_leaders(project)
    models = {}
    ordered = sorted(predictions, key=lambda item: (
        int(leaders[item.observation.compound - 1]),
        item.observation.compound,
        studies[item.observation.study],
        item.observation.composition_index_matlab,
    ))
    for prediction in ordered:
        observation = prediction.observation
        attr = project.compounds.composition_attr[observation.compound - 1]
        try:
            model = models.get(observation.compound)
            if model is None:
                model = phase_composition(attr, phase)
                models[observation.compound] = model
        except ValueError:
            continue
        carbon = model.text[3].index(str(observation.carbon))
        double_bonds = model.text[4].index(str(observation.double_bonds))
        abundance = np.asarray(
            attr.get("R_group_abundant", []), dtype=float
        ).reshape(-1)
        abundance_index = carbon * model.bases[4] + double_bonds
        abundant = bool(
            abundance_index < len(abundance) and abundance[abundance_index] != 0
        )
        for adduct in range(model.bases[0]):
            code = encode_dense_code(DecodedCode(
                digits=(adduct, 0, 0, carbon, double_bonds),
                bases=model.bases,
                adduct_index=adduct + 1,
                isotope_index=0,
                is_internal_standard=False,
            ))
            result.append(RtdbMassTarget(
                observation=observation,
                code=code,
                theoretical_mz=theoretical_mz(attr, phase, code, model=model),
                predicted_rt_seconds=prediction.predicted_rt_seconds,
                abundant=abundant,
            ))
    result.sort(key=lambda item: item.theoretical_mz)
    return tuple(result)


def filter_project_alignment_peaks(
    project: LegacyProject,
    reference: ReferenceData,
    *,
    phase: str,
    ppm_tolerance: float = 3.0,
    rt_tolerance_seconds: float = 1.2,
) -> np.ndarray:
    """Port MATLAB's C13-C12 filter before project RTDB alignment."""
    data = project.phases[phase]
    peaks = np.column_stack((data.mz_grouped, data.rt_grouped)).astype(float)
    isotope_mass = {item.atom: item.mass for item in reference.stable_isotopes}
    delta = isotope_mass["13C"] - isotope_mass["12C"]
    shifted = np.column_stack((peaks[:, 0] - delta, peaks[:, 1]))
    pairs = match_targets(
        peaks,
        shifted,
        np.asarray([ppm_tolerance, rt_tolerance_seconds], dtype=float),
    )
    if len(pairs):
        negative = peaks[pairs[:, 0], 0] < 0
        pairs[negative] = pairs[negative, ::-1]
    first = np.bincount(pairs[:, 0], minlength=len(peaks)) if len(pairs) else np.zeros(len(peaks), dtype=int)
    second = np.bincount(pairs[:, 1], minlength=len(peaks)) if len(pairs) else np.zeros(len(peaks), dtype=int)
    return peaks[(first > 0) & (second == 0)]


def match_project_alignment_targets(
    peaks: np.ndarray,
    targets: tuple[RtdbMassTarget, ...] | list[RtdbMassTarget],
    *,
    ppm_tolerance: float = 3.0,
    rt_tolerance_seconds: float = 300.0,
) -> np.ndarray:
    """Map filtered project peaks to RTDB mass/RT targets."""
    target_array = np.asarray([
        (item.theoretical_mz, item.predicted_rt_seconds) for item in targets
    ], dtype=float)
    return match_targets(
        np.asarray(peaks, dtype=float),
        target_array,
        np.asarray([ppm_tolerance, rt_tolerance_seconds], dtype=float),
    )


def fit_project_rt_correction(
    peaks: np.ndarray,
    alignment_targets: np.ndarray,
    pairs: np.ndarray,
    *,
    study_count: int,
) -> tuple[np.ndarray, float]:
    """Port the study/polarity correction inside ``correct_RT_database_new``."""
    x = np.asarray(peaks, dtype=float)
    y = np.asarray(alignment_targets, dtype=float)
    mapping = np.asarray(pairs, dtype=np.int64)
    if mapping.ndim != 2 or mapping.shape[1] != 2:
        raise ValueError("alignment pairs must contain peak and target indices")
    if not len(mapping):
        # A single blank or a weak run has no 13C-supported peak near an RT
        # database lipid; MATLAB would fail here too. Say so instead of
        # numpy's "zero-size array to reduction operation maximum".
        raise ValueError(
            "no RT alignment anchors: no 13C-supported peak matched the RT "
            "database (3 ppm, 300 s), so this phase cannot be annotated"
        )
    target = mapping[:, 1]
    peak = mapping[:, 0]
    groups = y[target, 2].astype(np.int64) * 2 - (x[peak, 0] < 0)
    residual = x[peak, 1] - y[target, 1]
    # MATLAB scales by the largest *matched* RTDB prediction (y(I(:,2),2)),
    # not by an unmatched target at the edge of the database.
    scale = float(np.max(y[target, 1]))

    ordered = np.column_stack((groups, residual))
    ordered = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]
    duplicate = (
        (np.diff(ordered[:, 1]) < 1e-4)
        & (np.diff(ordered[:, 0]) == 0)
    )
    ordered = ordered[~np.r_[duplicate, False]]
    initial = np.zeros(study_count * 2, dtype=float)
    for group in range(1, len(initial) + 1):
        values = ordered[ordered[:, 0] == group, 1]
        if not len(values):
            continue
        if len(values) == 1:
            initial[group - 1] = values[0]
            continue
        coordinates = np.arange(1, len(values) + 1, dtype=float)
        interpolation = interp1d(
            values,
            coordinates,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )(np.arange(-300.0, 301.0))
        difference = interpolation[10:] - interpolation[:-10]
        # MATLAB's max selects the first maximum.
        position = int(np.nanargmax(difference))
        initial[group - 1] = position + 1 - (len(difference) + 1) / 2

    normalized = y[target, 1] / scale
    base = np.column_stack((
        normalized,
        np.sin(np.pi * normalized),
        np.sin(2 * np.pi * normalized),
    ))
    design = np.zeros((len(mapping), 3 * study_count * 2), dtype=float)
    for row, group in enumerate(groups):
        start = (int(group) - 1) * 3
        design[row, start : start + 3] = base[row]
    penalty = np.tile(np.asarray([1e-6, 1e-3, 1e-3]), study_count * 2)
    active = np.any(design != 0, axis=0)
    active_design = design[:, active]
    active_penalty = penalty[active]
    seed_rows = np.abs(residual - initial[groups - 1]) < 45.0
    seed_normal = (
        active_design[seed_rows].T @ active_design[seed_rows]
        + np.diag(active_penalty)
    )
    coefficients = np.linalg.solve(
        seed_normal, active_design[seed_rows].T @ residual[seed_rows]
    )
    included = np.isfinite(residual)
    previous = ~included
    use_seed = True
    repeat = True
    iteration = 0
    while (iteration < 20 and not np.array_equal(included, previous)) or repeat:
        repeat = False
        previous = included.copy()
        if use_seed:
            use_seed = False
            repeat = True
        else:
            count = int(np.sum(included))
            normal = (
                active_design[included].T @ active_design[included]
                + count * np.diag(active_penalty)
            )
            coefficients = np.linalg.solve(
                normal,
                active_design[included].T @ residual[included],
            )
        error = residual - active_design @ coefficients
        included = (np.abs(error) <= 15.0) & np.isfinite(residual)
        iteration += 1
    result = np.zeros(design.shape[1], dtype=float)
    result[active] = coefficients
    return result, scale


def apply_project_rt_correction(
    project: LegacyProject,
    predictions: tuple[RetentionTimePrediction, ...] | list[RetentionTimePrediction],
    coefficients: np.ndarray,
    scale: float,
    *,
    phase: str,
    study_order: tuple[str, ...] | list[str] | np.ndarray,
) -> tuple[CorrectedRetentionTimePrediction, ...]:
    """Apply MATLAB's fitted study/polarity correction to RTDB rows."""
    studies = {str(name): index + 1 for index, name in enumerate(study_order)}
    beta = np.asarray(coefficients, dtype=float).reshape(3, -1, order="F")
    result: list[CorrectedRetentionTimePrediction] = []
    modes = {}
    for item in predictions:
        study = studies[item.observation.study]
        compound = item.observation.compound
        mode = modes.get(compound)
        if mode is None:
            attr = project.compounds.composition_attr[compound - 1]
            try:
                charges = np.asarray(phase_composition(attr, phase).charges, dtype=float)
                nonzero = charges[charges != 0]
                mode = int(np.sign(nonzero[0])) if len(nonzero) else 0
            except ValueError:
                mode = 0
            modes[compound] = mode
        group = study * 2 + (mode - 1) / 2
        normalized = item.predicted_rt_seconds / float(scale)
        design = np.asarray([
            normalized,
            np.sin(np.pi * normalized),
            np.sin(2 * np.pi * normalized),
        ])
        lower = int(np.floor(group)) - 1
        upper = int(np.ceil(group)) - 1
        delta = float(
            (design @ beta[:, lower] + design @ beta[:, upper]) / 2.0
        )
        result.append(CorrectedRetentionTimePrediction(
            observation=item.observation,
            predicted_rt_seconds=item.predicted_rt_seconds,
            corrected_rt_seconds=item.predicted_rt_seconds + delta,
        ))
    return tuple(result)


def fit_project_fa_preprocessing(
    project: LegacyProject,
    corrected: tuple[CorrectedRetentionTimePrediction, ...]
    | list[CorrectedRetentionTimePrediction],
    study_fits: tuple[StudyPreprocessingFit, ...] | list[StudyPreprocessingFit],
    *,
    study_order: tuple[str, ...] | list[str] | np.ndarray,
    phase: str = "RP",
) -> np.ndarray:
    """Fit MATLAB's synthetic ``all`` FA preprocessing coefficient plane."""
    studies = {str(name): index + 1 for index, name in enumerate(study_order)}
    baseline = np.mean(
        np.column_stack([item.global_beta for item in study_fits]), axis=1
    )
    penalties = FA_PREPROCESSING_PENALTIES[phase]
    leaders = _combined_leaders(project)
    r_groups = _r_group_labels(project)
    output = np.full((len(penalties), project.compounds.count), np.nan, dtype=float)
    output[:, r_groups == "FA"] = baseline[:, None]
    groups: dict[int, list[CorrectedRetentionTimePrediction]] = {}
    for item in corrected:
        compound = item.observation.compound
        if r_groups[compound - 1] != "FA":
            continue
        groups.setdefault(int(leaders[compound - 1]), []).append(item)

    for leader, rows in groups.items():
        # MATLAB constructs ic1 from the complete MS.combined group, not only
        # compounds represented by RTDB rows in the active phase.  This is
        # observable in NP, where the global group leader can be an RP-only
        # O/P definition while the combined NP definition owns all rows.
        compounds = (
            np.flatnonzero((leaders == leader) & (r_groups == "FA")) + 1
        ).astype(int).tolist()
        positions = {compound: index for index, compound in enumerate(compounds)}
        mapped_rows = tuple(item.observation for item in rows)
        raw = _fa_rows(project, mapped_rows, phase=phase)
        y = np.asarray([item.corrected_rt_seconds for item in rows], dtype=float)
        y0 = raw @ baseline
        width = len(penalties) * len(compounds)
        study_count = len(studies)
        design = np.zeros((len(rows), width * (study_count + 1)), dtype=float)
        diagonal = np.concatenate((
            np.tile(np.abs(penalties), len(compounds)),
            np.ones(width * study_count),
        ))
        for row, item in enumerate(rows):
            compound = item.observation.compound
            study = studies[item.observation.study]
            for feature in np.flatnonzero(raw[row] != 0):
                owner = leader if penalties[feature] != 0 else compound
                owner_position = positions[owner]
                base_column = owner_position * len(penalties) + int(feature)
                design[row, base_column] += raw[row, feature]
                study_column = base_column + study * width
                design[row, study_column] += raw[row, feature]
        active = np.any(design != 0, axis=0)
        coefficients = np.zeros(design.shape[1], dtype=float)
        normal = (
            design[:, active].T @ design[:, active]
            + np.diag(diagonal[active])
        )
        coefficients[active] = np.linalg.solve(
            normal, design[:, active].T @ (y - y0)
        )
        beta = coefficients[:width].reshape(
            len(penalties), len(compounds), order="F"
        )
        if len(compounds) > 1:
            leader_position = positions[leader]
            shared = penalties != 0
            beta[shared, :] = beta[shared, leader_position, None]
        beta += baseline[:, None]
        output[:, np.asarray(compounds) - 1] = beta
    return output


def fit_project_isoprene_preprocessing(
    project: LegacyProject,
    corrected: tuple[CorrectedRetentionTimePrediction, ...]
    | list[CorrectedRetentionTimePrediction],
    study_fits: tuple[StudyPreprocessingFit, ...] | list[StudyPreprocessingFit],
    *,
    study_order: tuple[str, ...] | list[str] | np.ndarray,
) -> np.ndarray:
    """Fit MATLAB's synthetic ``all`` one-dimensional isoprene plane."""
    studies = {str(name): index + 1 for index, name in enumerate(study_order)}
    baseline = np.mean(
        np.column_stack([item.global_beta for item in study_fits]), axis=1
    )
    penalties = np.asarray([0.0, -0.01, 0.01])
    r_groups = _r_group_labels(project)
    n_r = np.asarray(project.compounds.legacy_fields["nR"], dtype=float).reshape(-1)
    output = np.full((3, project.compounds.count), np.nan, dtype=float)
    output[:, r_groups == "isoprene"] = baseline[:, None]
    for compound in np.flatnonzero(r_groups == "isoprene") + 1:
        rows = [item for item in corrected if item.observation.compound == compound]
        if not rows:
            continue
        scaled = np.asarray([
            item.observation.composition_index_matlab for item in rows
        ], dtype=float) / n_r[compound - 1] / 25.0
        raw = np.column_stack((np.ones(len(rows)), scaled - 0.5, scaled ** 0.5))
        y = np.asarray([item.corrected_rt_seconds for item in rows], dtype=float)
        y0 = raw @ baseline
        width = len(penalties)
        design = np.zeros((len(rows), width * (len(studies) + 1)), dtype=float)
        diagonal = np.concatenate((
            np.abs(penalties), np.ones(width * len(studies))
        ))
        for row, item in enumerate(rows):
            study = studies[item.observation.study]
            for feature in np.flatnonzero(raw[row] != 0):
                design[row, feature] += raw[row, feature]
                design[row, feature + study * width] += raw[row, feature]
        active = np.any(design != 0, axis=0)
        coefficients = np.zeros(design.shape[1], dtype=float)
        coefficients[active] = np.linalg.solve(
            design[:, active].T @ design[:, active]
            + np.diag(diagonal[active]),
            design[:, active].T @ (y - y0),
        )
        output[:, compound - 1] = coefficients[:width] + baseline
    return output


def calibrate_project_fa_preprocessing(
    project: LegacyProject,
    reference: ReferenceData,
    *,
    phase: str,
) -> ProjectFaCalibration:
    """Run MATLAB's complete project-specific FA RTDB calibration path."""
    mapped = map_retention_time_observations(project, reference)
    study_order = matlab_rtdb_study_order(mapped)
    predictions = predict_supported_rtdb_studies(
        project, mapped, phase=phase
    )
    targets = build_fa_rtdb_mass_targets(project, predictions, phase=phase)
    peaks = filter_project_alignment_peaks(
        project, reference, phase=phase
    )
    pairs = match_project_alignment_targets(peaks, targets)
    studies = {name: index + 1 for index, name in enumerate(study_order)}
    alignment_targets = np.asarray([
        (
            item.theoretical_mz,
            item.predicted_rt_seconds,
            studies[item.observation.study],
            item.observation.composition_index_matlab,
        )
        for item in targets
    ], dtype=float)
    coefficients, scale = fit_project_rt_correction(
        peaks,
        alignment_targets,
        pairs,
        study_count=len(study_order),
    )
    r_groups = _r_group_labels(project)
    fa_predictions = tuple(
        item for item in predictions
        if r_groups[item.observation.compound - 1] == "FA"
    )
    corrected = apply_project_rt_correction(
        project,
        fa_predictions,
        coefficients,
        scale,
        phase=phase,
        study_order=study_order,
    )
    study_fits = tuple(
        fit_fa_study_preprocessing(
            project, mapped, phase=phase, study=study
        )
        for study in study_order
    )
    beta = fit_project_fa_preprocessing(
        project,
        corrected,
        study_fits,
        phase=phase,
        study_order=study_order,
    )
    isoprene_predictions = tuple(
        item for item in predictions
        if r_groups[item.observation.compound - 1] == "isoprene"
    )
    corrected_isoprene = apply_project_rt_correction(
        project,
        isoprene_predictions,
        coefficients,
        scale,
        phase=phase,
        study_order=study_order,
    )
    isoprene_fits = tuple(
        _fit_simple_study_preprocessing(
            project,
            tuple(item for item in mapped if (
                item.phase == phase and item.study == study
                and r_groups[item.compound - 1] == "isoprene"
            )),
            study=study,
            r_group="isoprene",
        )
        for study in study_order
        if any(
            item.phase == phase and item.study == study
            and r_groups[item.compound - 1] == "isoprene"
            for item in mapped
        )
    )
    if isoprene_fits:
        isoprene_beta = fit_project_isoprene_preprocessing(
            project,
            corrected_isoprene,
            isoprene_fits,
            study_order=study_order,
        )
    else:
        isoprene_beta = np.full(
            (3, project.compounds.count), np.nan, dtype=float
        )
    return ProjectFaCalibration(
        study_order=study_order,
        correction_coefficients=coefficients,
        correction_scale=scale,
        preprocessing_beta=beta,
        isoprene_preprocessing_beta=isoprene_beta,
    )
