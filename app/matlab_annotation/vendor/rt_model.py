"""Construction of MATLAB annotation RT preprocessing/model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compound_model import phase_composition
from .model_types import AnnotationModel
from .project import LegacyProject
from .spline import (
    precalculate_one_dimensional_spline,
    precalculate_two_dimensional_spline,
)


@dataclass(frozen=True)
class RetentionPreprocessingDesign:
    carbon: np.ndarray
    double_bonds: np.ndarray
    matrix: np.ndarray


R_GROUP_RANGES = {
    "FA": np.asarray([[8.0, 42.0], [0.0, 8.0]]),
    "FAfull": np.asarray([[1.0, 42.0], [0.0, 8.0]]),
    "ST": np.asarray([[17.0, 42.0], [0.0, 8.0]]),
}

FA_PREPROCESSING_PENALTIES = {
    "RP": np.asarray([0, -0.01, 0.01, -0.01, 0.01, -0.01, 20, -0.001]),
    "NP": np.asarray([0, -0.01, 0.01, 0.01, 1, -0.01]),
}


def fa_preprocessing_matrix(
    carbon: np.ndarray,
    double_bonds: np.ndarray,
    n_r: np.ndarray | float,
    sphingolipid: np.ndarray | bool,
    *,
    phase: str,
) -> np.ndarray:
    """Evaluate the phase-specific MATLAB ``R_group_FA`` function."""
    carbon = np.asarray(carbon, dtype=float)
    double_bonds = np.asarray(double_bonds, dtype=float)
    count = np.asarray(n_r, dtype=float)
    sphingo = np.asarray(sphingolipid, dtype=bool)
    scaled_carbon = (carbon / count) / R_GROUP_RANGES["FA"][0, 1]
    scaled_db = (double_bonds / count) / R_GROUP_RANGES["FA"][1, 1]
    indicator = ((double_bonds == 0) & sphingo).astype(float)
    if phase == "RP":
        return np.column_stack((
            np.ones(len(carbon)),
            scaled_carbon,
            (scaled_carbon + 0.01) ** -0.5,
            scaled_carbon ** 2,
            scaled_db ** 0.9,
            scaled_db ** 2,
            indicator,
            (scaled_carbon ** 0.8) * (scaled_db ** 0.8),
        ))
    if phase == "NP":
        return np.column_stack((
            np.ones(len(carbon)),
            scaled_carbon,
            (scaled_carbon + 0.01) ** -0.5,
            scaled_db ** 0.9,
            indicator,
            (carbon * double_bonds / count)
            / R_GROUP_RANGES["FA"][0, 1]
            / R_GROUP_RANGES["FA"][1, 1],
        ))
    raise ValueError(f"unsupported FA preprocessing phase {phase!r}")


def build_fa_preprocessing_design(
    project: LegacyProject,
    phase: str,
    compound: int,
) -> RetentionPreprocessingDesign:
    """Evaluate MATLAB's configured FA/FAfull preprocessing function."""
    index = int(compound) - 1
    r_group = str(np.asarray(
        project.compounds.legacy_fields["R_group"], dtype=object
    ).reshape(-1)[index])
    if r_group not in {"FA", "FAfull"}:
        raise ValueError(f"compound {compound} uses unsupported R-group {r_group!r}")
    n_r = float(np.asarray(
        project.compounds.legacy_fields["nR"], dtype=object
    ).reshape(-1)[index])
    if n_r <= 0:
        raise ValueError("FA preprocessing requires a positive nR")
    model = phase_composition(project.compounds.composition_attr[index], phase)
    carbon_values = np.asarray(model.text[3], dtype=float)
    double_bond_values = np.asarray(model.text[4], dtype=float)
    carbon = np.repeat(carbon_values, len(double_bond_values))
    double_bonds = np.tile(double_bond_values, len(carbon_values))
    ranges = R_GROUP_RANGES[r_group]
    composition = np.asarray(
        project.compounds.legacy_fields["composition"], dtype=object
    )[index]
    sphingolipid = bool(len(composition) >= 3 and str(composition[2]) == "d")
    matrix = fa_preprocessing_matrix(
        carbon, double_bonds, n_r, sphingolipid, phase=phase
    )
    return RetentionPreprocessingDesign(carbon, double_bonds, matrix)


def build_fa_annotation_model(
    project: LegacyProject,
    *,
    phase: str,
    compounds: tuple[int, ...] | list[int] | np.ndarray,
    preprocessing_beta: np.ndarray,
    integration_steps_per_unit: int = 1000,
) -> AnnotationModel:
    """Build the fixed annotation model from calibrated FA coefficients."""
    active = tuple(int(value) for value in np.asarray(compounds).reshape(-1))
    if not active:
        raise ValueError("at least one compound is required")
    designs = [build_fa_preprocessing_design(project, phase, item) for item in active]
    first_shape = designs[0].matrix.shape
    if any(item.matrix.shape != first_shape for item in designs[1:]):
        raise ValueError("active siblings have incompatible preprocessing grids")
    beta = np.asarray(preprocessing_beta, dtype=float)
    if beta.ndim == 1:
        beta = beta[:, None]
    if beta.shape != (first_shape[1], len(active)):
        raise ValueError(
            f"preprocessing beta has shape {beta.shape}; "
            f"expected {(first_shape[1], len(active))}"
        )
    baseline = np.column_stack([
        design.matrix @ beta[:, sibling]
        for sibling, design in enumerate(designs)
    ])
    # MATLAB repeats the composition baseline for each adduct/label block and
    # keeps sibling baselines as columns.
    bases = phase_composition(
        project.compounds.composition_attr[active[0] - 1], phase
    ).bases
    baseline = np.tile(baseline, (bases[0] * bases[1], 1))
    if baseline.shape[1] == 1:
        baseline = baseline[:, 0]
    thresholds = {
        "RP": np.asarray([30.0, 20.0, 15.0]),
        "NP": np.asarray([10.0, 7.0, 4.0]),
        "HILIC": np.asarray([10.0, 7.0, 4.0]),
    }
    try:
        phase_thresholds = thresholds[phase]
    except KeyError as exc:
        raise ValueError(f"unknown annotation phase {phase!r}") from exc
    term = precalculate_two_dimensional_spline(
        (bases[3], bases[4]),
        (6.0, 3.0),
        dimensions_matlab=(4, 5),
        integration_steps_per_unit=integration_steps_per_unit,
    )
    return AnnotationModel(
        ranges=np.asarray([len(active), *bases], dtype=np.int64),
        thresholds_seconds=phase_thresholds,
        rt_preprocessing=np.asarray(baseline),
        term=term,
    )


def build_isoprene_preprocessing_design(
    project: LegacyProject,
    phase: str,
    compound: int,
) -> RetentionPreprocessingDesign:
    """Evaluate MATLAB's one-dimensional isoprene preprocessing function."""
    index = int(compound) - 1
    r_group = str(np.asarray(
        project.compounds.legacy_fields["R_group"], dtype=object
    ).reshape(-1)[index])
    if r_group != "isoprene":
        raise ValueError(f"compound {compound} uses unsupported R-group {r_group!r}")
    n_r = float(np.asarray(
        project.compounds.legacy_fields["nR"], dtype=object
    ).reshape(-1)[index])
    model = phase_composition(project.compounds.composition_attr[index], phase)
    composition_index = np.arange(1, model.bases[3] + 1, dtype=float)
    scaled = composition_index / n_r / 25.0
    matrix = np.column_stack((
        np.ones(len(scaled)),
        scaled - 0.5,
        scaled ** 0.5,
    ))
    values = np.asarray(model.text[3], dtype=float)
    return RetentionPreprocessingDesign(
        carbon=values,
        double_bonds=np.zeros(len(values), dtype=float),
        matrix=matrix,
    )


def build_isoprene_annotation_model(
    project: LegacyProject,
    *,
    phase: str,
    compound: int,
    preprocessing_beta: np.ndarray,
    integration_steps_per_unit: int = 1000,
) -> AnnotationModel:
    """Build MATLAB's fixed one-dimensional isoprenoid annotation model."""
    design = build_isoprene_preprocessing_design(project, phase, compound)
    beta = np.asarray(preprocessing_beta, dtype=float).reshape(-1)
    if beta.shape != (design.matrix.shape[1],):
        raise ValueError(
            f"preprocessing beta has shape {beta.shape}; "
            f"expected {(design.matrix.shape[1],)}"
        )
    model = phase_composition(
        project.compounds.composition_attr[int(compound) - 1], phase
    )
    baseline = np.tile(
        design.matrix @ beta, model.bases[0] * model.bases[1]
    )
    thresholds = {
        "RP": np.asarray([30.0, 20.0, 15.0]),
        "NP": np.asarray([10.0, 7.0, 4.0]),
        "HILIC": np.asarray([10.0, 7.0, 4.0]),
    }
    term = precalculate_one_dimensional_spline(
        model.bases[3],
        2.0,
        dimension_matlab=4,
        integration_steps_per_unit=integration_steps_per_unit,
    )
    return AnnotationModel(
        ranges=np.asarray([1, *model.bases], dtype=np.int64),
        thresholds_seconds=thresholds[phase],
        rt_preprocessing=baseline,
        term=term,
    )


def build_empty_r_annotation_model(
    project: LegacyProject,
    *,
    phase: str,
    compounds: tuple[int, ...] | list[int] | np.ndarray,
) -> AnnotationModel:
    """Build MATLAB's no-retention-fit model for an empty ``R_group``.

    The legacy path uses ``spline_model={1,2;0,0}``, an empty R-group range,
    and scalar-zero RT preprocessing.  In the only phase-capable ProjectData2
    case (RP cholesterol 24), there are no stored inliers, so MATLAB exposes
    mass candidates but never produces an RT fit or selection.
    """
    active = tuple(int(value) for value in np.asarray(compounds).reshape(-1))
    if not active:
        raise ValueError("at least one compound is required")
    r_groups = np.asarray(
        project.compounds.legacy_fields["R_group"], dtype=object
    ).reshape(-1)
    models = []
    for compound in active:
        value = np.asarray(r_groups[compound - 1], dtype=object).reshape(-1)
        r_group = "" if value.size == 0 else str(value[0])
        if r_group:
            raise ValueError(
                f"compound {compound} uses non-empty R-group {r_group!r}"
            )
        models.append(phase_composition(
            project.compounds.composition_attr[compound - 1], phase
        ))
    bases = models[0].bases
    if any(model.bases != bases for model in models[1:]):
        raise ValueError("active siblings have incompatible composition radices")
    thresholds = {
        "RP": np.asarray([30.0, 20.0, 15.0]),
        "NP": np.asarray([10.0, 7.0, 4.0]),
        "HILIC": np.asarray([10.0, 7.0, 4.0]),
    }
    try:
        phase_thresholds = thresholds[phase]
    except KeyError as exc:
        raise ValueError(f"unknown annotation phase {phase!r}") from exc
    return AnnotationModel(
        ranges=np.asarray([len(active), *bases], dtype=np.int64),
        thresholds_seconds=phase_thresholds,
        rt_preprocessing=np.asarray(0.0),
        term=None,
    )
