"""Run the MATLAB annotation logic and shape it as a LASER standard result.

Standard mode's own engine matches an XCMS peak-group list against LASER's
chemical reference with a ppm window and a guarded class RT grid. This module
does something different: it drives the vendored reverse-engineering of the
Amsterdam UMC MATLAB application, which enumerates theoretical targets per lipid
definition, predicts retention time from the RTDB spline calibrated on the
project itself, and runs MATLAB's staged 6 ppm to 3 ppm regression loop with
sibling resolution.

The two engines answer the same question with different science, so the results
are deliberately shaped alike: this module returns the dictionary standard mode
already renders, so the run summary, the exports and the grid browser work
unchanged.

The automatic standard-mode entry point constructs the required domain model
from XCMS peak-group lists.  A legacy replay entry point remains available for
validation, but production annotation no longer reads MATLAB ``.mat`` files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from scientific_defaults import SCIENTIFIC_DEFAULTS

from annotator.composition_ranges import (
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
    CompositionRangeSettings,
)
from annotator.np_rt import (
    DEFAULT_NP_RT_REFERENCE_PATH,
    combine_np_ether_plasmalogen_labels,
    load_np_rt_reference,
)
from annotator.reference import load_reference
from standard_lipidomics.workflow import (
    annotate_standard_batch,
    discover_standard_streams,
)
from standard_lipidomics.policy import (
    DEFAULT_MATLAB_POLICY_PATH,
    load_matlab_lipid_policy,
)
from standard_lipidomics.mass_calibration import (
    attach_matlab_mass_calibration_result,
    mass_calibration_provenance,
    normalize_mass_calibrations,
)
from annotator.grid_coherence import score_grid_coherence

from .composition_filter import filter_compound_set_targets
from .confidence import (
    PostRegressionCompletionSettings,
    complete_post_regression_annotations,
)
from .fitted_grid_completion import (
    FittedGridCompletionSettings, refine_fitted_grid_annotations,
)
from .decode import CodeDecoder
from .isomer_resolver import (
    PositionalIsomerSettings,
    resolve_positional_lyso_isomers,
)
from .vendor.annotation_engine import AnnotationEngine
from .vendor.engine import resolve_annotation_siblings
from .vendor.legacy_loader import LegacyProjectError, load_legacy_project
from .vendor.internal_standard_model import build_simple_internal_standard_targets
from .vendor.model_dispatch import (
    UnsupportedPhaseDefinition,
    build_calibrated_annotation_model,
)
from .vendor.naming import compound_display_template
from .vendor.reference_loader import ReferenceDataError, load_reference_data
from .vendor.target_model import (
    build_compound_set_target_table,
    predict_compound_set_target_rt,
)
from .automatic import (
    AutomaticAnchorSettings,
    ITERATIVE_POSITIONAL_REFIT,
    POSITIONAL_ISOMER_WORKFLOW_MODES,
    derive_automatic_memberships,
    matlab_annotation_adducts,
)
from .xcms_project import build_xcms_project, with_memberships
from .result_assembly import assemble_result, concat_result_tables
from resource_paths import REFERENCE_DATA_DIRECTORY

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MATLAB_PIPELINE_REFERENCE = REFERENCE_DATA_DIRECTORY / "matlab_pipeline"
DEFAULT_LASER_CHEMICAL_REFERENCE = REFERENCE_DATA_DIRECTORY / "4d_lipid_reference.csv"
REQUIRED_PROJECT_FILES = ("MS.mat", "MS_data.mat", "MS_metadata.mat")

# A definition without a double-bond axis has no carbon x double-bond surface
# to draw, so it contributes no grid nodes.
_DOUBLE_BOND_AXIS_INDEX = 4

# MATLAB's mapper opens at 6 ppm and closes at 3 ppm. The narrow pass is the
# tolerance a surviving candidate actually had to meet, so it is what the
# browser reviews against.
MATLAB_BROAD_PPM = 6.0
MATLAB_NARROW_PPM = 3.0

_POSITIONAL_LYSO_CLASSES = frozenset(
    {
        "1-acylLPC",
        "2-acylLPC",
        "1-acylLPE",
        "2-acylLPE",
    }
)


class MatlabAnnotationError(ValueError):
    """Raised when a project or the bundled MATLAB reference cannot be used."""


@dataclass(frozen=True)
class CompoundOutcome:
    """What happened to one lipid definition in one phase."""

    phase: str
    compound: int
    template: str
    status: str
    annotations: int
    detail: str = ""


def discover_matlab_annotation_project(directory: str | Path) -> Path | None:
    """Find the ``Processing/Annotation`` folder holding the MATLAB project.

    Accepts the project root or the annotation folder itself, matching how
    standard mode already resolves XCMS peak-group lists.
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        return None
    candidates = [root, root / "Processing" / "Annotation"]
    for parent in root.parents:
        candidate = parent / "Processing" / "Annotation"
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / name).is_file() for name in REQUIRED_PROJECT_FILES
        ):
            return candidate
    return None


def _sample_columns(project, phase: str) -> list[str]:
    fields = project.metadata.fields
    names = np.asarray(fields.get("sample_name", []), dtype=object)
    columns = np.asarray(fields.get("column", []), dtype=object)
    if names.ndim == 2 and columns.shape == names.shape:
        matches = np.flatnonzero(columns[0] == phase)
        if len(matches):
            return [str(value) for value in names[:, matches[0]]]
    return [f"sample_{index + 1}" for index in range(project.metadata.sample_count)]


def _phase_features(project, phase: str, sample_columns: list[str]) -> pd.DataFrame:
    """Every grouped peak in one phase, in LASER's feature-table shape."""
    data = project.phases[phase]
    features = pd.DataFrame(
        {
            "feature_id": np.arange(1, data.peak_count + 1).astype(str),
            "mz": np.asarray(data.mz_grouped, dtype=float),
            "rt": np.asarray(data.rt_grouped, dtype=float) / 60.0,
            "intensity": np.asarray(data.representative_intensity, dtype=float),
        }
    )
    intensities = np.asarray(data.sample_intensity, dtype=float)
    for index, column in enumerate(sample_columns):
        features[column] = intensities[:, index]
    features.attrs["sample_columns"] = list(sample_columns)
    return features


def _sibling_groups(project, phase: str, *, only_accepted: bool = True) -> list[int]:
    """One representative definition per sibling set that this phase can map.

    MATLAB annotates a whole sibling set in one run, so running each member
    separately would repeat the same work and emit the same rows. Sibling
    resolution and phase selectability both come from the vendored rule rather
    than from reading ``combined``/``priority`` again here.

    ``only_accepted`` restricts the sweep to the lipid definitions the project
    itself marks as accepted for this phase — the set the analyst actually
    annotated in MATLAB. That is the default because it reproduces the project's
    own coverage; the full catalogue is available for discovery but proposes
    classes the analyst deliberately left out.
    """
    accepted = None
    if only_accepted and phase in project.compounds.accepted:
        flags = np.asarray(project.compounds.accepted[phase]).reshape(-1)
        accepted = {
            int(index) + 1 for index in np.flatnonzero(np.nan_to_num(flags) != 0)
        }
    seen: set[frozenset[int]] = set()
    out: list[int] = []
    for compound in range(1, project.compounds.count + 1):
        if accepted is not None and compound not in accepted:
            continue
        active = tuple(
            resolve_annotation_siblings(
                compound,
                project.compounds.combined,
                project.compounds.priority[phase],
            )
        )
        if not active:
            continue
        # Each member resolves to the same set with itself in front, so the set
        # — not the ordering — is what identifies an already-covered group.
        members = frozenset(active)
        if members in seen:
            continue
        seen.add(members)
        out.append(compound)
    return out


def _sibling_lookup(result) -> dict[tuple[int, int], int]:
    """Map ``(viewer row, code)`` to the sibling that proposed it."""
    lookup: dict[tuple[int, int], int] = {}
    for stage in result.stages:
        candidates = stage.candidates
        for viewer, code, sibling in zip(
            candidates.viewer_rows,
            candidates.codes,
            candidates.sibling_positions,
        ):
            lookup[(int(viewer), int(code))] = int(sibling)
    return lookup


def _frame(
    project,
    result,
    phase: str,
    *,
    viewer_rows,
    codes,
    theoretical_mz,
    predicted_rt_seconds,
    sibling_lookup: dict[tuple[int, int], int],
    decoders: dict,
) -> pd.DataFrame:
    """Build LASER annotation rows from one set of MATLAB mapper rows."""
    viewer_rows = np.asarray(viewer_rows, dtype=np.int64)
    if not len(viewer_rows):
        return pd.DataFrame()
    codes = np.asarray(codes, dtype=np.int64)
    rows = np.asarray(result.source_rows, dtype=np.int64)[viewer_rows - 1]
    data = project.phases[phase]
    siblings = [
        sibling_lookup.get((int(viewer), int(code)), 1)
        for viewer, code in zip(viewer_rows, codes)
    ]
    compounds = [
        result.compounds[min(position - 1, len(result.compounds) - 1)]
        for position in siblings
    ]
    decoded = [
        decoders[compound].decode(int(code)) for compound, code in zip(compounds, codes)
    ]
    frame = pd.DataFrame(
        {
            "feat_feature_id": (rows + 1).astype(str),
            "name": [item.species for item in decoded],
            "display_name": [item.display_name for item in decoded],
            "class": [item.lipid_class for item in decoded],
            "base": "",
            "adduct": [item.adduct for item in decoded],
            "isotopologue": [item.isotopologue for item in decoded],
            "carbon": [item.carbon for item in decoded],
            "db": [item.double_bonds for item in decoded],
            "is_is": [item.internal_standard for item in decoded],
            "feat_mz": np.asarray(data.mz_grouped, dtype=float)[rows],
            "feat_rt": np.asarray(data.rt_grouped, dtype=float)[rows] / 60.0,
            "feat_intensity": np.asarray(data.representative_intensity, dtype=float)[
                rows
            ],
            "ref_mz": np.asarray(theoretical_mz, dtype=float),
            "rt_expected": np.asarray(predicted_rt_seconds, dtype=float) / 60.0,
            "matlab_code": codes,
            "matlab_compound": compounds,
        }
    )
    frame["ref_rt"] = frame["rt_expected"]
    frame["ppm_error"] = (frame["feat_mz"] - frame["ref_mz"]) / frame["ref_mz"] * 1e6
    frame["rt_error"] = frame["feat_rt"] - frame["rt_expected"]
    return frame


def _attach_sample_intensities(
    frame: pd.DataFrame, project, phase: str, sample_columns: list[str]
) -> pd.DataFrame:
    """Add ``feat_<sample>`` columns so the browser can plot intensities."""
    if frame.empty or not sample_columns:
        return frame
    rows = frame["feat_feature_id"].astype(np.int64).to_numpy() - 1
    intensities = np.asarray(project.phases[phase].sample_intensity, dtype=float)[rows]
    for index, column in enumerate(sample_columns):
        frame[f"feat_{column}"] = intensities[:, index]
    return frame


def _monoisotopic_codes(decoder) -> np.ndarray:
    """Every adduct x carbon x double-bond code with no isotope label.

    Enumerating the node codes and looking them up is what keeps this cheap. A
    lipid definition can carry hundreds of thousands of theoretical targets
    across all isotopologues, but the grid surface is only the monoisotopic
    corner of that space.
    """
    isotopes = [
        index for index, value in enumerate(decoder.axis(2)) if value == ""
    ] or [0]
    axes = (
        range(len(decoder.axis(0))),  # adduct
        range(len(decoder.axis(1))),  # unused axis, base 1 in practice
        isotopes,
        range(len(decoder.axis(3))),  # carbon
        range(len(decoder.axis(4))),  # double bonds
    )
    mesh = np.meshgrid(
        *[np.asarray(values, dtype=np.int64) for values in axes], indexing="ij"
    )
    codes = np.ones(mesh[0].size, dtype=np.int64)
    for index, grid in enumerate(mesh):
        codes = codes + grid.reshape(-1) * decoder.weights[index]
    return codes


def _reference_grid(project, result, phase: str, decoders: dict) -> pd.DataFrame:
    """Monoisotopic theoretical nodes, for the browser's reference mesh.

    Their m/z and RT come straight from MATLAB's own target table and RTDB
    prediction, so the grid the reviewer sees is the grid the mapper used.
    """
    codes = np.asarray(result.targets.codes, dtype=np.int64)
    mz = np.asarray(result.targets.theoretical_mz, dtype=float)
    siblings = np.asarray(result.targets.sibling_positions, dtype=np.int64)
    rt_seconds = np.asarray(result.database_rt_seconds, dtype=float)
    frames = []
    for position, compound in enumerate(result.compounds, start=1):
        decoder = decoders[compound]
        if len(decoder.bases) <= _DOUBLE_BOND_AXIS_INDEX:
            continue
        selected = (siblings == position) & (codes > 0)
        if not selected.any():
            continue
        available = pd.DataFrame(
            {
                "matlab_code": codes[selected],
                "ref_mz": mz[selected],
                "ref_rt": rt_seconds[selected] / 60.0,
            }
        ).drop_duplicates("matlab_code", keep="first")
        wanted = pd.DataFrame({"matlab_code": _monoisotopic_codes(decoder)})
        nodes = wanted.merge(available, on="matlab_code", how="inner")
        if nodes.empty:
            continue
        decoded = [decoder.decode(int(code)) for code in nodes["matlab_code"]]
        nodes["name"] = [item.species for item in decoded]
        nodes["class"] = decoder.lipid_class
        nodes["base"] = ""
        nodes["adduct"] = [item.adduct for item in decoded]
        nodes["carbon"] = [item.carbon for item in decoded]
        nodes["db"] = [item.double_bonds for item in decoded]
        nodes["matlab_compound"] = int(compound)
        frames.append(nodes.dropna(subset=["carbon", "db"]))
    if not frames:
        return pd.DataFrame()
    grid = pd.concat(frames, ignore_index=True)
    return grid.drop_duplicates(["name", "adduct"], keep="first")[
        [
            "name",
            "class",
            "base",
            "adduct",
            "carbon",
            "db",
            "ref_mz",
            "ref_rt",
            "matlab_code",
            "matlab_compound",
        ]
    ]


def _map_compound(
    engine: AnnotationEngine,
    project: Any,
    reference: Any,
    phase: str,
    compound: int,
    composition_ranges: CompositionRangeSettings,
) -> tuple[Any, Any, Any]:
    active = engine.siblings(phase, compound)
    model = build_calibrated_annotation_model(
        project, engine.calibration(phase), phase=phase, compounds=active
    )
    targets = build_compound_set_target_table(project, reference, phase, active)
    targets, audit = filter_compound_set_targets(
        project, targets, active, phase=phase, settings=composition_ranges
    )
    if not len(targets.codes):
        raise ValueError("composition limits removed every target")
    result = engine.run(
        phase=phase, compound=compound, model=model, targets=targets
    )
    return result, model, audit


def _compound_frames(
    project: Any, result: Any, model: Any, phase: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookup = _sibling_lookup(result)
    decoders = {
        member: CodeDecoder(project, member, phase) for member in result.compounds
    }
    history = result.final_history
    fitted_stage = next(
        (
            stage_result
            for stage_result in reversed(result.stages)
            if stage_result.primary is not None
        ),
        None,
    )
    final_target_rt = (
        predict_compound_set_target_rt(
            result.targets,
            primary_coefficients=fitted_stage.primary.coefficients,
            adduct_offsets=fitted_stage.complete.adduct_offsets,
            ranges=model.ranges,
            rt_preprocessing=model.rt_preprocessing,
            term=model.term,
        )
        if fitted_stage is not None and model.term is not None
        else np.full(len(result.targets.codes), np.nan, dtype=float)
    )
    accepted = _frame(
        project, result, phase,
        viewer_rows=history.viewer_rows,
        codes=history.codes,
        theoretical_mz=history.theoretical_mz,
        predicted_rt_seconds=history.predicted_rt_seconds,
        sibling_lookup=lookup,
        decoders=decoders,
    )
    proposals = []
    for stage_result in result.stages:
        stage = stage_result.candidates
        frame = _frame(
            project, result, phase,
            viewer_rows=stage.viewer_rows,
            codes=stage.codes,
            theoretical_mz=stage.theoretical_mz,
            predicted_rt_seconds=np.asarray(final_target_rt, dtype=float)[
                stage.target_indices
            ],
            sibling_lookup=lookup,
            decoders=decoders,
        )
        if not frame.empty:
            frame["matlab_candidate_stage"] = int(stage_result.stage)
            proposals.append(frame)
    proposed = pd.concat(proposals, ignore_index=True) if proposals else pd.DataFrame()
    grid = _reference_grid(project, result, phase, decoders)
    return accepted, proposed, grid


def _phase_stream(
    project: Any,
    phase: str,
    sample_columns: list[str],
    annotation_frames: list[pd.DataFrame],
    candidate_frames: list[pd.DataFrame],
    reference_frames: list[pd.DataFrame],
) -> dict:
    annotations = (
        pd.concat(annotation_frames, ignore_index=True)
        if annotation_frames
        else pd.DataFrame()
    )
    all_pairs = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame()
    )
    if not all_pairs.empty:
        all_pairs = all_pairs.drop_duplicates(
            ["feat_feature_id", "name", "adduct", "matlab_code"], keep="first"
        ).reset_index(drop=True)
    grid_reference = (
        pd.concat(reference_frames, ignore_index=True).drop_duplicates(
            ["name", "adduct"], keep="first"
        )
        if reference_frames
        else pd.DataFrame()
    )

    if not annotations.empty:
        annotations = _attach_sample_intensities(
            annotations, project, phase, sample_columns
        )
        # MATLAB's mapper is the decision: a row that survived the final stage
        # is an accepted annotation, not a candidate awaiting a LASER rule.
        annotations["automatic_accepted"] = True
        annotations["automatic_candidate"] = True
        annotations["auto_annotation_status"] = np.where(
            annotations["isotopologue"].astype(str).eq(""),
            "matlab_mapper_accepted",
            "matlab_mapper_accepted_isotopologue",
        )
        annotations["rank_for_feature"] = annotations.groupby(
            "feat_feature_id"
        ).cumcount()
        annotations["standard_stream"] = phase
        annotations["chromatography"] = phase
        annotations["rt_evidence_mode"] = "matlab_rtdb_spline"
    if not all_pairs.empty:
        all_pairs = _attach_sample_intensities(
            all_pairs, project, phase, sample_columns
        )

    return {
        "stream": phase,
        "features": _phase_features(project, phase, sample_columns),
        "reference": grid_reference,
        "annotations": annotations,
        "all_pairs": all_pairs,
        "rt_models": pd.DataFrame(),
        "np_rt_alignment": pd.DataFrame(),
        "np_rt_audit": pd.DataFrame(),
        "policy_audit": pd.DataFrame(),
        "criteria": {
            "ppm_tolerance": MATLAB_NARROW_PPM,
            # MATLAB's acceptance is the regression loop, not a fixed RT window.
            # The browser needs a finite window to review deviations against;
            # this is the widest |ΔRT| the mapper actually kept.
            "rt_tolerance_min": _observed_rt_window(annotations),
            "uses_rt_grid": True,
        },
    }


def _failed_compound_outcome(
    phase: str, compound: int, template: str, error: Exception
) -> CompoundOutcome:
    skipped = isinstance(error, (UnsupportedPhaseDefinition, ValueError))
    return CompoundOutcome(
        phase=phase,
        compound=compound,
        template=template,
        status="skipped" if skipped else "failed",
        annotations=0,
        detail=str(error) if skipped else f"{type(error).__name__}: {error}",
    )


def _run_phase(
    project: Any,
    reference: Any,
    phase: str,
    *,
    progress: Callable[[float, str], None] | None,
    phase_index: int,
    phase_count: int,
    only_accepted: bool,
    composition_ranges: CompositionRangeSettings,
) -> tuple[dict, list[CompoundOutcome]]:
    """Annotate every mappable lipid definition in one chromatography phase."""
    engine = AnnotationEngine(project=project, reference=reference)
    if progress:
        progress(phase_index / phase_count, f"{phase}: calibrating RTDB model")
    engine.calibration(phase)
    sample_columns = _sample_columns(project, phase)
    compounds = _sibling_groups(project, phase, only_accepted=only_accepted)
    annotations: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    references: list[pd.DataFrame] = []
    outcomes: list[CompoundOutcome] = []
    for index, compound in enumerate(compounds, start=1):
        template = compound_display_template(project, compound - 1)
        if progress:
            done = (phase_index + index / max(len(compounds), 1)) / phase_count
            progress(done, f"{phase}: {template} ({index}/{len(compounds)})")
        try:
            result, model, audit = _map_compound(
                engine, project, reference, phase, compound, composition_ranges
            )
            accepted, proposed, grid = _compound_frames(
                project, result, model, phase
            )
        except Exception as error:  # noqa: BLE001
            # A vendor edge case is isolated to this definition and surfaced
            # in the audit so the rest of the phase remains useful.
            outcomes.append(_failed_compound_outcome(
                phase, compound, template, error
            ))
            continue
        if not accepted.empty:
            annotations.append(accepted)
        if not proposed.empty or not accepted.empty:
            candidates.append(pd.concat(
                [item for item in (proposed, accepted) if not item.empty],
                ignore_index=True,
            ))
        if not grid.empty:
            references.append(grid)
        outcomes.append(CompoundOutcome(
            phase=phase,
            compound=compound,
            template=template,
            status="annotated" if not accepted.empty else "no_annotations",
            annotations=int(len(accepted)),
            detail=(f"composition limits removed {audit.endogenous_filtered} "
                    f"of {audit.targets_before} target(s)"),
        ))
    return (
        _phase_stream(
            project, phase, sample_columns, annotations, candidates, references
        ),
        outcomes,
    )


def _observed_rt_window(annotations: pd.DataFrame) -> float:
    if annotations.empty or "rt_error" not in annotations:
        return 0.5
    errors = pd.to_numeric(annotations["rt_error"], errors="coerce").abs()
    errors = errors[np.isfinite(errors)]
    return float(max(errors.max(), 0.05)) if len(errors) else 0.5


def run_legacy_matlab_annotation(
    project_dir: str | Path,
    *,
    reference_dir: str | Path = DEFAULT_MATLAB_PIPELINE_REFERENCE,
    phases: tuple[str, ...] | None = None,
    only_accepted: bool = True,
    progress: Callable[[float, str], None] | None = None,
    composition_ranges: CompositionRangeSettings = (DEFAULT_COMPOSITION_RANGE_SETTINGS),
) -> dict:
    """Annotate a MATLAB project and return a LASER standard-mode result.

    ``only_accepted`` sweeps the lipid definitions the project marks as accepted
    for each phase, reproducing the coverage the analyst chose in MATLAB. Set it
    to ``False`` to sweep the whole catalogue.
    """
    annotation_dir = discover_matlab_annotation_project(project_dir)
    if annotation_dir is None:
        raise MatlabAnnotationError(
            "No MATLAB project found. The MATLAB engine needs MS.mat, "
            "MS_data.mat and MS_metadata.mat in the project's "
            f"Processing/Annotation folder; searched from {project_dir}."
        )
    reference_root = Path(reference_dir).expanduser()
    try:
        project = load_legacy_project(annotation_dir)
        reference = load_reference_data(
            code_dir=reference_root / "Code",
            database_dir=reference_root / "DataBase",
            rtdb_dir=reference_root / "RTDB",
            project_dir=annotation_dir,
        )
    except (LegacyProjectError, ReferenceDataError) as error:
        raise MatlabAnnotationError(str(error)) from error

    available = tuple(project.phases)
    selected = tuple(phase for phase in (phases or available) if phase in available)
    if not selected:
        raise MatlabAnnotationError(
            f"Project has no requested phase; it provides {available}."
        )

    streams: dict[str, dict] = {}
    outcomes: list[CompoundOutcome] = []
    for index, phase in enumerate(selected):
        stream, phase_outcomes = _run_phase(
            project,
            reference,
            phase,
            progress=progress,
            phase_index=index,
            phase_count=len(selected),
            only_accepted=only_accepted,
            composition_ranges=composition_ranges,
        )
        streams[phase] = stream
        outcomes.extend(phase_outcomes)

    result = assemble_result(streams, outcomes, annotation_dir, reference_root)
    result["composition_range_settings"] = composition_ranges.to_dict()
    return result


def _restore_xcms_frame(frame: pd.DataFrame, bundle, stream_name: str) -> pd.DataFrame:
    """Split a MATLAB phase frame and restore XCMS ids/intensities."""
    if frame.empty:
        return frame.copy()
    global_rows = pd.to_numeric(frame["feat_feature_id"], errors="coerce") - 1
    valid = global_rows.notna()
    rows = global_rows[valid].astype(np.int64).to_numpy()
    selected = np.asarray(bundle.row_stream, dtype=object)[rows] == stream_name
    output = frame.loc[valid].iloc[np.flatnonzero(selected)].copy()
    if output.empty:
        return output
    rows = rows[selected]
    local = bundle.row_local_index[rows]
    output["feat_feature_id"] = [bundle.row_feature_id[row] for row in rows]
    for column in ("feat_mz", "ref_mz"):
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce").abs()
    # The internal combined project uses generic sample slots.  Replace them
    # with the exact sample columns from this polarity's XCMS table.
    output = output.drop(
        columns=[column for column in output if column.startswith("feat_sample_")],
        errors="ignore",
    )
    source = bundle.features[stream_name]
    if "mz_raw" in source:
        output["feat_mz_raw"] = pd.to_numeric(
            source["mz_raw"], errors="coerce"
        ).to_numpy(float)[local]
    for column, value in mass_calibration_provenance(
        source.attrs.get("mass_calibration")
    ).items():
        output[column] = value
    for column in source.attrs.get("sample_columns", []):
        values = pd.to_numeric(source[column], errors="coerce").to_numpy(float)
        output[f"feat_{column}"] = values[local]
    # Carry XCMS feature bounds/peak-quality fields when the source export
    # provides them.  They are review metadata only; the MATLAB matcher still
    # decides annotation acceptance from m/z and RT evidence.
    for column in ("mzmin", "mzmax", "rtmin", "rtmax", "into", "intb", "maxo", "sn"):
        target = f"feat_{column}"
        if target in output or column not in source:
            continue
        output[target] = pd.to_numeric(source[column], errors="coerce").to_numpy(float)[
            local
        ]
    output["standard_stream"] = stream_name
    output["chromatography"] = stream_name.split("_", 1)[0]
    return output.reset_index(drop=True)


def _restore_xcms_anchor_frame(
    anchors: pd.DataFrame,
    bundle,
    stream_name: str,
) -> pd.DataFrame:
    """Attach lossless stream/XCMS provenance to retained MATLAB anchors."""
    if anchors is None or anchors.empty or "row" not in anchors:
        return pd.DataFrame(
            columns=[
                *(anchors.columns if isinstance(anchors, pd.DataFrame) else []),
                "standard_stream",
                "feat_feature_id",
            ]
        )
    global_rows = pd.to_numeric(anchors["row"], errors="coerce")
    valid = global_rows.notna()
    rows = global_rows[valid].astype(np.int64).to_numpy()
    in_bounds = (rows >= 0) & (rows < len(bundle.row_stream))
    source = anchors.loc[valid].iloc[np.flatnonzero(in_bounds)].copy()
    rows = rows[in_bounds]
    selected = np.asarray(bundle.row_stream, dtype=object)[rows] == stream_name
    output = source.iloc[np.flatnonzero(selected)].copy()
    rows = rows[selected]
    if output.empty:
        output["standard_stream"] = pd.Series(dtype=object)
        output["feat_feature_id"] = pd.Series(dtype=object)
        return output
    output["standard_stream"] = stream_name
    output["feat_feature_id"] = [bundle.row_feature_id[row] for row in rows]
    return output.reset_index(drop=True)


def _positional_anchor_seeds(
    anchors: pd.DataFrame, excluded: set[str]
) -> tuple[dict[tuple[str, str, str, int, int], str], set[tuple[int, int, str]]]:
    # A feature can legitimately be a mass-isobaric candidate for several
    # classes.  Seed identity therefore includes the complete lyso family and
    # composition.  Feature ID + adduct alone caused LPE(18:x) anchors to leak
    # into isobaric LPC(15:x) groups and shift a coherent grid to the wrong
    # chromatographic peak.
    seeds: dict[tuple[str, str, str, int, int], str] = {}
    lpc_groups: set[tuple[int, int, str]] = set()
    for row, species, adduct in anchors[["row", "species", "adduct"]].itertuples(
        index=False, name=None
    ):
        if pd.isna(row):
            continue
        species = str(species)
        adduct = str(adduct)
        lpc = re.match(r"^[12]-acylLPC\((\d+):(\d+)\)$", species)
        if lpc:
            one_species = f"1-acylLPC({lpc.group(1)}:{lpc.group(2)})"
            if one_species not in excluded:
                lpc_groups.add((int(lpc.group(1)), int(lpc.group(2)), adduct))
            continue
        lpe = re.match(r"^([12])-acylLPE\((\d+):(\d+)\)$", species)
        if lpe and species not in excluded:
            seeds[
                (
                    str(int(row) + 1),
                    adduct,
                    "LPE",
                    int(lpe.group(2)),
                    int(lpe.group(3)),
                )
            ] = species
    return seeds, lpc_groups


def _strongest_lpc_seed(
    candidates: pd.DataFrame,
    carbon: int,
    db: int,
    adduct: str,
    maximum_rt_error_seconds: float,
) -> tuple[tuple[str, str, str, int, int], str] | None:
    group = candidates[
        candidates["class"].astype(str).isin(("1-acylLPC", "2-acylLPC"))
        & pd.to_numeric(candidates["carbon"], errors="coerce").eq(carbon)
        & pd.to_numeric(candidates["db"], errors="coerce").eq(db)
        & candidates["adduct"].astype(str).eq(adduct)
        & pd.to_numeric(candidates["ppm_error"], errors="coerce")
        .abs()
        .lt(MATLAB_NARROW_PPM)
    ].copy()
    if group.empty:
        return None
    group["_seed_intensity"] = pd.to_numeric(
        group["feat_intensity"], errors="coerce"
    ).fillna(0.0)
    group["_seed_ppm"] = pd.to_numeric(group["ppm_error"], errors="coerce").abs()
    if {"feat_rt", "rt_expected"}.issubset(group.columns):
        group["_seed_rt_error_seconds"] = (
            pd.to_numeric(group["feat_rt"], errors="coerce")
            - pd.to_numeric(group["rt_expected"], errors="coerce")
        ).abs() * 60.0
        group = group[group["_seed_rt_error_seconds"].le(maximum_rt_error_seconds)]
    else:
        group["_seed_rt_error_seconds"] = 0.0
    if group.empty:
        return None
    strongest = (
        group.sort_values(
            ["_seed_intensity", "_seed_rt_error_seconds", "_seed_ppm"],
            ascending=[False, True, True],
            kind="stable",
        )
        .drop_duplicates("feat_feature_id", keep="first")
        .iloc[0]
    )
    key = (str(strongest["feat_feature_id"]), adduct, "LPC", carbon, db)
    return key, f"1-acylLPC({carbon}:{db})"


def _mark_positional_seed_anchors(
    candidates: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    excluded_species: tuple[str, ...] = (),
    maximum_rt_error_seconds: float = 10.0,
) -> pd.DataFrame:
    """Promote abundant LPC seed features to the primary 1-acyl identity."""
    output = candidates.copy()
    output["positional_isomer_seed_anchor"] = False
    output["positional_isomer_seed_species"] = ""
    output["positional_isomer_seed_identity_valid"] = False
    required = {
        "feat_feature_id", "name", "class", "adduct", "carbon", "db",
        "ppm_error", "feat_intensity",
    }
    if (
        output.empty or anchors is None or anchors.empty
        or not {"row", "species", "adduct"}.issubset(anchors.columns)
        or not required.issubset(output.columns)
    ):
        return output
    excluded = set(map(str, excluded_species))
    seeds, lpc_groups = _positional_anchor_seeds(anchors, excluded)

    for carbon, db, adduct in lpc_groups:
        seed = _strongest_lpc_seed(
            output, carbon, db, adduct, float(maximum_rt_error_seconds)
        )
        if seed is not None:
            seeds[seed[0]] = seed[1]
    output["positional_isomer_seed_species"] = [
        seeds.get(
            (
                str(feature_id),
                str(adduct),
                "LPC"
                if "LPC" in str(lipid_class).upper()
                else "LPE"
                if "LPE" in str(lipid_class).upper()
                else "",
                int(carbon) if pd.notna(carbon) else -1,
                int(db) if pd.notna(db) else -1,
            ),
            "",
        )
        for feature_id, adduct, lipid_class, carbon, db in output[
            ["feat_feature_id", "adduct", "class", "carbon", "db"]
        ].itertuples(index=False, name=None)
    ]
    output["positional_isomer_seed_identity_valid"] = (
        output["positional_isomer_seed_species"].astype(str).ne("")
    )
    output["positional_isomer_seed_anchor"] = [
        bool(seed_species) and str(name) == str(seed_species)
        for name, seed_species in output[
            ["name", "positional_isomer_seed_species"]
        ].itertuples(index=False, name=None)
    ]
    return output


def _reconcile_positional_review_anchors(
    seed_anchors: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    excluded_species: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Show resolved lyso identities instead of provisional RTDB seed labels.

    Positional LPC/LPE seeds are generated before the early/late assignment.
    Reviewer anchors must use the resolver-selected feature/species identity,
    but resolver acceptance alone is not anchor evidence. Only a selected row
    that carries the original ``positional_isomer_seed_anchor`` flag may enter
    the automatic-anchor table. Derived primary annotations and optional pair
    partners remain annotations, not anchors. The original seed table remains
    unchanged in the top-level MATLAB audit.
    """
    output = (
        seed_anchors.copy()
        if isinstance(seed_anchors, pd.DataFrame)
        else pd.DataFrame()
    )
    if not output.empty and "species" in output:
        positional_seed = (
            output["species"]
            .astype(str)
            .str.match(r"^[12]-acyl(?:LPC|LPE)\(", na=False)
        )
        output = output.loc[~positional_seed].copy()
    if (
        candidates is None
        or candidates.empty
        or "positional_isomer_selected" not in candidates
    ):
        return output.reset_index(drop=True)
    statuses = candidates.get(
        "positional_isomer_status", pd.Series("", index=candidates.index)
    ).astype(str)
    resolved_status = statuses.eq("paired_rt_order_supported") | (
        statuses.str.startswith("one_acyl_primary_only_")
    )
    selected = candidates[
        candidates["positional_isomer_selected"].fillna(False).astype(bool)
        & resolved_status
        & candidates.get(
            "positional_isomer_seed_anchor",
            pd.Series(False, index=candidates.index),
        )
        .fillna(False)
        .astype(bool)
    ].copy()
    if excluded_species:
        selected = selected.loc[
            ~selected["name"].astype(str).isin(excluded_species)
        ].copy()
    if selected.empty:
        return output.reset_index(drop=True)
    selected["species"] = selected["name"].astype(str)
    selected["anchor_role"] = "resolved_positional_seed"
    if not output.empty:
        output["anchor_role"] = output.get(
            "anchor_role", pd.Series("matlab_rt_seed", index=output.index)
        ).fillna("matlab_rt_seed")
    reconciled = pd.concat([output, selected], ignore_index=True, sort=False)
    return reconciled.drop_duplicates(
        ["standard_stream", "feat_feature_id", "species", "adduct"],
        keep="first",
    ).reset_index(drop=True)


def _split_xcms_streams(combined: dict, bundle) -> dict[str, dict]:
    streams: dict[str, dict] = {}
    for stream_name, features in bundle.features.items():
        reference = combined["reference"].copy()
        if not reference.empty and "ref_mz" in reference:
            signed = pd.to_numeric(reference["ref_mz"], errors="coerce")
            keep = signed.lt(0) if stream_name.endswith("Neg") else signed.gt(0)
            reference = reference.loc[keep].copy()
            reference["ref_mz"] = signed.loc[keep].abs()
        criteria = dict(combined.get("criteria", {}))
        criteria["ppm_tolerance"] = 3.0
        criteria["acceptance_ppm_rule"] = "strictly_less_than_3"
        criteria["mass_calibration"] = bundle.features[stream_name].attrs.get(
            "mass_calibration", {"mode": "none", "on_failure": "disable"}
        )
        streams[stream_name] = {
            **combined,
            "stream": stream_name,
            "features": features.copy(),
            "reference": reference.reset_index(drop=True),
            "annotations": _restore_xcms_frame(
                combined["annotations"], bundle, stream_name
            ),
            "all_pairs": _restore_xcms_frame(
                combined["all_pairs"], bundle, stream_name
            ),
            "criteria": criteria,
        }
    return streams


def _trusted_reference_anchors(
    project_dir: str | Path,
    bundle,
    *,
    phase: str,
    composition_ranges: CompositionRangeSettings,
    mass_calibrations: Mapping[str, object] | None = None,
):
    """High-confidence independent reference matches for seed consensus.

    These rows do not become annotations directly.  They can only initialize a
    MATLAB target carrying the same XCMS feature, species and preferred adduct;
    the staged MATLAB loop must still accept that target independently.
    """
    sources = {
        name: path
        for name, path in discover_standard_streams(project_dir).items()
        if name.startswith(f"{phase}_")
    }
    policy = load_matlab_lipid_policy(DEFAULT_MATLAB_POLICY_PATH)
    result = annotate_standard_batch(
        sources,
        load_reference(DEFAULT_LASER_CHEMICAL_REFERENCE),
        ppm_tolerance=3.0,
        rp_rt_tolerance_min=SCIENTIFIC_DEFAULTS.rt_tolerance_min,
        align_rp_rt=False,
        adapt_class_rt=True,
        matlab_policy=policy,
        composition_ranges=composition_ranges,
        np_rt_reference=(
            load_np_rt_reference(DEFAULT_NP_RT_REFERENCE_PATH)
            if phase == "NP"
            else None
        ),
        mass_calibrations=mass_calibrations,
    )
    row_lookup = {
        (stream, feature_id): row
        for row, (stream, feature_id) in enumerate(
            zip(bundle.row_stream, bundle.row_feature_id, strict=True)
        )
    }
    trusted: set[tuple[int, str, str]] = set()
    audit_rows = []
    for stream_name, stream in result["streams"].items():
        annotations = stream["annotations"]
        accepted_status = (
            "high_confidence_np_grid" if phase == "NP" else "high_confidence_auto"
        )
        required = {"automatic_accepted", "auto_annotation_status", "adduct"}
        selected = (
            annotations[
                annotations["automatic_accepted"].fillna(False).astype(bool)
                & annotations["auto_annotation_status"].eq(accepted_status)
                & annotations["adduct"].isin(matlab_annotation_adducts(phase))
            ]
            if required.issubset(annotations.columns)
            else annotations.iloc[0:0]
        )
        mapped = 0
        for row in selected.itertuples(index=False):
            feature_row = row_lookup.get(
                (stream_name, str(row.feat_feature_id))
            )
            if feature_row is None:
                continue
            name = str(row.name)
            trusted.add((feature_row, name, str(row.adduct)))
            if name.endswith(" (IS)"):
                # MATLAB encodes internal standards by a negative composition
                # code; decoding that code yields the trait without the visual
                # ``(IS)`` suffix used by LASER's chemical reference.
                trusted.add((feature_row, name[:-5], str(row.adduct)))
            mapped += 1
        audit_rows.append(
            {
                "stream": stream_name,
                "high_confidence_reference_matches": int(len(selected)),
                "mapped_to_xcms_rows": int(mapped),
            }
        )
    return trusted, pd.DataFrame(audit_rows)


def _internal_standard_project_dir(
    project_dir: str | Path, reference_root: Path
) -> Path:
    """Use study-specific spike metadata, falling back to the bundled panel."""
    source = Path(project_dir).expanduser().resolve()
    candidates = [source if source.is_dir() else source.parent]
    candidates.append(source / "Processing" / "Annotation")
    for path in discover_standard_streams(source).values():
        candidates.append(Path(path).resolve().parent)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "internal_standards.tsv").is_file():
            return candidate
    return reference_root / "Project"


def _label_internal_standards(
    frame: pd.DataFrame,
    project,
    reference,
    *,
    phase: str = "RP",
) -> pd.DataFrame:
    """Expose MATLAB's negative-code standards as explicit ``(IS)`` rows."""
    if frame.empty or "is_is" not in frame:
        return frame
    output = frame.copy()
    targets = build_simple_internal_standard_targets(
        project,
        reference,
        phase,
        np.arange(1, project.compounds.count + 1, dtype=np.int64),
    )
    requested_by_target = {
        (int(compound), int(code)): str(name)
        for compound, code, name in zip(
            targets.compound_ids, targets.codes, targets.names, strict=True
        )
    }
    quantity_by_name = {
        item.name: (item.quantity, item.unit) for item in reference.internal_standards
    }
    requested = [
        requested_by_target.get((int(compound), int(code)), "")
        for compound, code in output[["matlab_compound", "matlab_code"]].itertuples(
            index=False, name=None
        )
    ]
    output["internal_standard_name"] = requested
    output["internal_standard_quantity"] = [
        quantity_by_name.get(name, (np.nan, ""))[0] if name else np.nan
        for name in requested
    ]
    output["internal_standard_unit"] = [
        quantity_by_name.get(name, (np.nan, ""))[1] if name else ""
        for name in requested
    ]
    is_standard = output["is_is"].fillna(False).astype(bool)
    output.loc[is_standard, "name"] = output.loc[is_standard, "name"].map(
        lambda name: name if str(name).endswith(" (IS)") else f"{name} (IS)"
    )
    return output


def _internal_standard_audit(
    project,
    reference,
    anchors: pd.DataFrame,
    annotations: pd.DataFrame,
    all_pairs: pd.DataFrame,
    *,
    phase: str = "RP",
) -> pd.DataFrame:
    """Summarize every declared spike, including standards not recovered."""
    targets = build_simple_internal_standard_targets(
        project,
        reference,
        phase,
        np.arange(1, project.compounds.count + 1, dtype=np.int64),
    )
    target_names: dict[tuple[int, int], str] = {}
    traits: dict[str, set[str]] = {}
    adducts: dict[str, set[str]] = {}
    for compound, code, requested in zip(
        targets.compound_ids, targets.codes, targets.names, strict=True
    ):
        key = (int(compound), int(code))
        requested = str(requested)
        target_names[key] = requested
        decoded = CodeDecoder(project, int(compound), phase).decode(int(code))
        traits.setdefault(requested, set()).add(decoded.species)
        adducts.setdefault(requested, set()).add(decoded.adduct)

    anchor_counts: dict[str, int] = {}
    if anchors is not None and not anchors.empty:
        for compound, code in anchors[["compound", "code"]].itertuples(
            index=False, name=None
        ):
            matched_name = target_names.get((int(compound), int(code)))
            if matched_name:
                anchor_counts[matched_name] = anchor_counts.get(matched_name, 0) + 1

    def counts(frame: pd.DataFrame) -> dict[str, int]:
        if frame is None or frame.empty or "internal_standard_name" not in frame:
            return {}
        values = frame.loc[
            frame["internal_standard_name"].astype(str).ne(""),
            "internal_standard_name",
        ].value_counts()
        return {str(name): int(count) for name, count in values.items()}

    candidate_counts = counts(all_pairs)
    accepted_counts = counts(annotations)
    rows = []
    for item in reference.internal_standards:
        accepted = accepted_counts.get(item.name, 0)
        candidates = candidate_counts.get(item.name, 0)
        anchors_found = anchor_counts.get(item.name, 0)
        status = (
            "accepted"
            if accepted
            else "review_candidate"
            if candidates
            else "anchor_only"
            if anchors_found
            else "not_recovered"
        )
        rows.append(
            {
                "internal_standard_name": item.name,
                "reported_traits": "; ".join(sorted(traits.get(item.name, set()))),
                "target_adducts": "; ".join(sorted(adducts.get(item.name, set()))),
                "quantity": item.quantity,
                "unit": item.unit,
                "anchor_matches": anchors_found,
                "candidate_matches": candidates,
                "accepted_matches": accepted,
                "status": status,
            }
        )
    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit.insert(0, "phase", phase)
    return audit


def _matlab_grid_coherence_audit(
    scored: pd.DataFrame,
    settings: AutomaticAnchorSettings,
    threshold: float,
) -> pd.DataFrame:
    assessed = scored["grid_coherence_assessed"].fillna(False).astype(bool)
    outliers = scored["grid_coherence_outlier"].fillna(False).astype(bool)
    values = pd.to_numeric(
        scored.loc[assessed, "grid_coherence_score"], errors="coerce"
    ).dropna()
    return pd.DataFrame([{
        "status": "calculated",
        "threshold": threshold,
        "ppm_scale": MATLAB_NARROW_PPM,
        "rt_scale_seconds": float(settings.grid_coherence_rt_tolerance_seconds),
        "candidates": int(len(scored)),
        "annotations": int(len(scored)),
        "assessed": int(assessed.sum()),
        "unassessed": int((~assessed).sum()),
        "diagnostic_outliers": int(outliers.sum()),
        "median_score": float(values.median()) if len(values) else np.nan,
        "used_for_annotation": bool(settings.use_grid_coherence),
        "filtered_annotations": 0,
        "orphan_two_acyl_removed": 0,
    }])


def _calculate_matlab_grid_coherence(
    candidates: pd.DataFrame,
    *,
    settings: AutomaticAnchorSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate local m/z/RT grid coherence for the staged candidate pool.

    The scorer is intentionally run on the complete staged candidate pool, not
    only on the final accepted annotations.  The MATLAB mapper keeps broad
    early-stage proposals in ``all_pairs`` for review; using only the final
    rows leaves accepted/anchor points with no neighbouring nodes and makes the
    browser display ``n/a`` even when the grid has enough evidence.  The
    optional ``use_grid_coherence`` setting applies the resulting pass/fail
    status to the final accepted table; unassessed edge nodes remain passing.
    """
    threshold = float(settings.grid_coherence_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Grid-coherence threshold must be between 0 and 1")
    if candidates.empty:
        return candidates, pd.DataFrame(
            [
                {
                    "status": "no_candidates",
                    "threshold": threshold,
                    "used_for_annotation": bool(settings.use_grid_coherence),
                    "filtered_annotations": 0,
                }
            ]
        )
    scored = score_grid_coherence(
        candidates,
        ppm_tolerance=MATLAB_NARROW_PPM,
        rt_tolerance=(float(settings.grid_coherence_rt_tolerance_seconds) / 60.0),
        ccs_tolerance_percent=1.0,
        use_ccs=False,
        threshold=threshold,
    )
    # Positional lyso candidates contain both identity alternatives for the
    # same precursor-mass features.  Letting those provisional alternatives
    # choose each node representative can build a highly coherent but wrongly
    # labelled 1-acyl grid from the early 2-acyl peaks.  Once the joint resolver
    # has selected the primary 1-acyl feature and any optional earlier 2-acyl
    # partner, recalculate their scores from that resolved pool. Grouping
    # remains acyl-specific through the ordinary class/adduct grouping in
    # ``score_grid_coherence``; a 1-acyl-only group can never create a 2-acyl
    # annotation.
    selected = (
        scored.get("positional_isomer_selected", pd.Series(False, index=scored.index))
        .fillna(False)
        .astype(bool)
    )
    positional_status = scored.get(
        "positional_isomer_status", pd.Series("", index=scored.index)
    ).astype(str)
    selected &= positional_status.eq(
        "paired_rt_order_supported"
    ) | positional_status.str.startswith("one_acyl_primary_only_")
    if selected.any():
        resolved = score_grid_coherence(
            scored.loc[selected].copy(),
            ppm_tolerance=MATLAB_NARROW_PPM,
            rt_tolerance=(float(settings.grid_coherence_rt_tolerance_seconds) / 60.0),
            ccs_tolerance_percent=1.0,
            use_ccs=False,
            threshold=threshold,
        )
        coherence_columns = [
            "grid_coherence_score",
            "grid_coherence_neighbors",
            "grid_coherence_dimensions",
            "grid_coherence_assessed",
            "grid_coherence_outlier",
            "grid_coherence_pass",
        ]
        scored.loc[resolved.index, coherence_columns] = resolved[coherence_columns]
    scored["grid_coherence_threshold"] = threshold
    return scored, _matlab_grid_coherence_audit(scored, settings, threshold)


def _copy_grid_coherence(
    target: pd.DataFrame,
    scored_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Copy candidate-pool coherence values onto accepted annotation rows.

    Coherence is evaluated in the full candidate neighbourhood.  Accepted
    rows are a filtered view of that pool, so joining by feature/species/adduct
    (and MATLAB code where available) preserves the per-row score without
    recalculating it against an artificially sparse accepted-only grid.
    """
    if target.empty or scored_candidates.empty:
        return target
    key_columns = [
        column
        for column in ("feat_feature_id", "name", "adduct", "matlab_code")
        if column in target.columns and column in scored_candidates.columns
    ]
    if not key_columns:
        return target
    coherence_columns = [
        "grid_coherence_score",
        "grid_coherence_neighbors",
        "grid_coherence_dimensions",
        "grid_coherence_assessed",
        "grid_coherence_outlier",
        "grid_coherence_pass",
        "grid_coherence_threshold",
    ]
    available = [column for column in coherence_columns if column in scored_candidates]
    if not available:
        return target
    lookup = scored_candidates.drop_duplicates(key_columns, keep="first")
    target = target.copy()
    left_keys = pd.MultiIndex.from_frame(target[key_columns].astype(str))
    right_keys = pd.MultiIndex.from_frame(lookup[key_columns].astype(str))
    for column in available:
        values = (
            pd.Series(lookup[column].to_numpy(), index=right_keys)
            .reindex(left_keys)
            .to_numpy()
        )
        target[column] = values
    return target


def _drop_orphan_two_acyl_lyso(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove 2-acyl LPC/LPE rows whose 1-acyl primary was filtered."""
    required = {"class", "carbon", "db", "adduct"}
    if annotations.empty or not required.issubset(annotations.columns):
        return annotations, 0
    key_columns = ["carbon", "db", "adduct"]
    if "isotopologue" in annotations:
        key_columns.append("isotopologue")
    orphan_indices: list[object] = []
    for family in ("LPC", "LPE"):
        one = annotations[annotations["class"].astype(str).eq(f"1-acyl{family}")]
        two_mask = annotations["class"].astype(str).eq(f"2-acyl{family}")
        if not two_mask.any():
            continue
        one_keys = set(
            one[key_columns].fillna("").astype(str).itertuples(index=False, name=None)
        )
        two_keys = annotations.loc[two_mask, key_columns].fillna("").astype(str)
        orphan_indices.extend(
            [
                index
                for index, key in zip(
                    two_keys.index,
                    two_keys.itertuples(index=False, name=None),
                    strict=True,
                )
                if key not in one_keys
            ]
        )
    if len(orphan_indices) == 0:
        return annotations, 0
    return (
        annotations.drop(index=orphan_indices).reset_index(drop=True),
        int(len(orphan_indices)),
    )


def _drop_orphan_two_acyl_lpc(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Backward-compatible alias for the now family-complete lyso guard."""
    return _drop_orphan_two_acyl_lyso(annotations)


def _finalize_positional_isomer_audit(
    audit: pd.DataFrame,
    annotations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Attach post-grid retention outcomes to the resolver's group audit."""
    if audit is None or audit.empty or "pair_id" not in audit:
        return audit
    out = audit.copy()
    retained = annotations if isinstance(annotations, pd.DataFrame) else pd.DataFrame()
    staged = candidates if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    rows = []
    for pair_id in out["pair_id"].astype(str):
        kept = (
            retained[
                retained.get(
                    "positional_isomer_pair_id",
                    pd.Series("", index=retained.index),
                )
                .astype(str)
                .eq(pair_id)
            ]
            if not retained.empty
            else pd.DataFrame()
        )
        one_count = int(
            kept.get("class", pd.Series(dtype=object))
            .astype(str)
            .str.startswith("1-acyl")
            .sum()
        )
        two_count = int(
            kept.get("class", pd.Series(dtype=object))
            .astype(str)
            .str.startswith("2-acyl")
            .sum()
        )
        selected = (
            staged[
                staged.get(
                    "positional_isomer_pair_id",
                    pd.Series("", index=staged.index),
                )
                .astype(str)
                .eq(pair_id)
                & staged.get(
                    "positional_isomer_selected",
                    pd.Series(False, index=staged.index),
                )
                .fillna(False)
                .astype(bool)
            ]
            if not staged.empty
            else pd.DataFrame()
        )
        grid_outliers = (
            int(
                selected.get(
                    "grid_coherence_outlier",
                    pd.Series(False, index=selected.index),
                )
                .fillna(False)
                .astype(bool)
                .sum()
            )
            if not selected.empty
            else 0
        )
        rows.append(
            {
                "retained_after_grid": int(len(kept)),
                "one_acyl_retained_after_grid": one_count,
                "two_acyl_retained_after_grid": two_count,
                "selected_grid_outliers": grid_outliers,
                "post_grid_status": (
                    "paired_retained"
                    if one_count and two_count
                    else "one_acyl_retained"
                    if one_count
                    else "removed"
                ),
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _positional_resolver_settings(
    anchor_settings: AutomaticAnchorSettings,
    *,
    maximum_rt_error_seconds: float | None = None,
) -> PositionalIsomerSettings:
    """Translate run settings into one positional-resolver pass."""
    return PositionalIsomerSettings(
        maximum_rt_error_seconds=float(
            anchor_settings.positional_isomer_rt_tolerance_seconds
            if maximum_rt_error_seconds is None
            else maximum_rt_error_seconds
        ),
        maximum_gap_error_seconds=(
            anchor_settings.positional_isomer_gap_tolerance_seconds
        ),
        minimum_peak_separation_seconds=(
            anchor_settings.positional_isomer_minimum_separation_seconds
        ),
        minimum_assignment_margin=(anchor_settings.positional_isomer_score_margin),
        ppm_tolerance=anchor_settings.ppm_tolerance,
        maximum_local_rt_adjustment_seconds=(
            anchor_settings.positional_isomer_local_rt_adjustment_seconds
        ),
        minimum_local_support_groups=(
            anchor_settings.positional_isomer_local_support_groups
        ),
        maximum_local_composition_distance=(
            anchor_settings.positional_isomer_local_composition_distance
        ),
    )


def _prepare_mapper_output(
    combined: dict,
    project,
    reference,
    *,
    phase: str,
) -> dict:
    """Apply invariant narrow-mass and internal-standard preparation."""
    if not combined["annotations"].empty:
        combined["annotations"] = (
            combined["annotations"]
            .loc[
                pd.to_numeric(combined["annotations"]["ppm_error"], errors="coerce")
                .abs()
                .lt(MATLAB_NARROW_PPM)
                & combined["annotations"]["adduct"].isin(
                    matlab_annotation_adducts(phase)
                )
            ]
            .reset_index(drop=True)
        )
    combined["annotations"] = _label_internal_standards(
        combined["annotations"], project, reference, phase=phase
    )
    combined["all_pairs"] = _label_internal_standards(
        combined["all_pairs"], project, reference, phase=phase
    )
    return combined


def _resolved_positional_memberships(
    resolved_annotations: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "class", "matlab_compound", "matlab_code", "feat_feature_id",
        "feat_mz", "feat_rt", "name", "adduct",
    }
    if not (
        isinstance(resolved_annotations, pd.DataFrame)
        and required.issubset(resolved_annotations.columns)
    ):
        return pd.DataFrame(columns=sorted(required))
    selected = resolved_annotations[
        resolved_annotations["class"].astype(str).isin(_POSITIONAL_LYSO_CLASSES)
    ].copy()
    if "isotopologue" in selected:
        selected = selected[selected["isotopologue"].fillna("").astype(str).eq("")]
    return selected.drop_duplicates(
        ["matlab_compound", "matlab_code", "feat_feature_id"], keep="first"
    )


def _append_refit_memberships(
    memberships: list[list[list[Any]]], selected: pd.DataFrame, phase: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audit_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        compound = int(row.matlab_compound)
        memberships[compound - 1].append([
            phase, int(row.matlab_code), float(row.feat_mz), float(row.feat_rt) * 60.0,
        ])
        audit_rows.append({
            "phase": phase,
            "status": "refit_membership",
            "compound": compound,
            "matlab_code": int(row.matlab_code),
            "species": str(row.name),
            "adduct": str(row.adduct),
            "feature_id": str(row.feat_feature_id),
            "mz": float(row.feat_mz),
            "rt_seconds": float(row.feat_rt) * 60.0,
        })
        feature_number = pd.to_numeric(
            pd.Series([row.feat_feature_id]), errors="coerce"
        ).iloc[0]
        if pd.notna(feature_number):
            anchor_rows.append({
                "row": int(feature_number) - 1,
                "species": str(row.name),
                "adduct": str(row.adduct),
            })
    return audit_rows, anchor_rows


def _build_positional_refit_project(
    seeded_project: Any,
    resolved_annotations: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    *,
    phase: str,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Replace provisional positional memberships with resolved identities."""
    memberships = [[list(row) for row in compound_rows]
                   for compound_rows in seeded_project.compounds.inliers]
    positional_compounds: set[int] = set()
    if not candidate_rows.empty and {
        "class",
        "matlab_compound",
    }.issubset(candidate_rows.columns):
        values = pd.to_numeric(
            candidate_rows.loc[
                candidate_rows["class"].astype(str).isin(_POSITIONAL_LYSO_CLASSES),
                "matlab_compound",
            ],
            errors="coerce",
        ).dropna()
        positional_compounds = {int(value) for value in values}

    for compound in positional_compounds:
        memberships[compound - 1] = [
            row
            for row in memberships[compound - 1]
            if not row or str(row[0]).upper() != str(phase).upper()
        ]

    selected = _resolved_positional_memberships(resolved_annotations)
    audit_rows, anchor_rows = _append_refit_memberships(
        memberships, selected, phase
    )
    audit = pd.DataFrame(audit_rows)
    anchors = (
        pd.DataFrame(
            anchor_rows, columns=["row", "species", "adduct"]
        ).drop_duplicates()
        if anchor_rows
        else pd.DataFrame(columns=["row", "species", "adduct"])
    )
    if not audit.empty:
        return with_memberships(seeded_project, tuple(memberships)), audit, anchors
    return (
        seeded_project,
        pd.DataFrame(
            [
                {
                    "phase": phase,
                    "status": "no_resolved_memberships",
                    "detail": "The provisional resolver supplied no positional anchors",
                }
            ]
        ),
        anchors,
    )


def _load_automatic_phase(
    project_dir: str | Path,
    phase: str,
    reference_root: Path,
    only_accepted: bool,
    progress: Callable[[float, str], None] | None,
    anchor_settings: AutomaticAnchorSettings,
    composition_ranges: CompositionRangeSettings,
    mass_calibrations: Mapping[str, object] | None = None,
) -> tuple[Any, Path, Any, Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle = build_xcms_project(
        project_dir,
        phase=phase,
        reference_root=reference_root,
        mass_calibrations=mass_calibrations,
    )
    internal_dir = _internal_standard_project_dir(project_dir, reference_root)
    reference = load_reference_data(
        code_dir=reference_root / "Code",
        database_dir=reference_root / "DataBase",
        rtdb_dir=reference_root / "RTDB",
        project_dir=internal_dir,
    )
    groups = _sibling_groups(bundle.project, phase, only_accepted=only_accepted)
    if progress:
        progress(0.03, f"{phase}: building independent high-confidence seed consensus")
    trusted, reference_audit = _trusted_reference_anchors(
        project_dir,
        bundle,
        phase=phase,
        composition_ranges=composition_ranges,
        mass_calibrations=mass_calibrations,
    )
    seeded, seed_audit, anchors = derive_automatic_memberships(
        bundle.project,
        reference,
        groups,
        phase=phase,
        settings=anchor_settings,
        composition_ranges=composition_ranges,
        trusted_anchors=trusted,
        progress=progress,
    )
    return bundle, internal_dir, reference, seeded, seed_audit, anchors, reference_audit


def _collapse_np_mapping(
    combined: dict, anchors: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    for table_name in ("reference", "annotations", "all_pairs"):
        table = combine_np_ether_plasmalogen_labels(
            combined.get(table_name, pd.DataFrame())
        )
        candidates = (
            ("name", "adduct") if table_name == "reference"
            else ("feat_feature_id", "name", "adduct", "isotopologue")
        )
        keys = [column for column in candidates if column in table]
        combined[table_name] = (
            table.drop_duplicates(keys, keep="first").reset_index(drop=True)
            if not table.empty and keys else table
        )
    return combined, combine_np_ether_plasmalogen_labels(
        anchors, species_column="species"
    )


def _initial_automatic_mapping(
    seeded: Any,
    reference: Any,
    phase: str,
    only_accepted: bool,
    progress: Callable[[float, str], None] | None,
    iterative_refit: bool,
    composition_ranges: CompositionRangeSettings,
) -> tuple[dict, list[CompoundOutcome]]:
    mapper_progress = None
    if progress:
        def mapper_progress(fraction: float, message: str) -> None:
            progress(
            0.4 + (0.3 if iterative_refit else 0.6) * fraction, message
            )
    return _run_phase(
        seeded, reference, phase,
        progress=mapper_progress,
        phase_index=0,
        phase_count=1,
        only_accepted=only_accepted,
        composition_ranges=composition_ranges,
    )


def _provisional_positional_refit(
    combined: dict,
    outcomes: list[CompoundOutcome],
    seeded: Any,
    reference: Any,
    project: Any,
    anchors: pd.DataFrame,
    settings: AutomaticAnchorSettings,
    only_accepted: bool,
    composition_ranges: CompositionRangeSettings,
    progress: Callable[[float, str], None] | None,
) -> tuple[dict, list[CompoundOutcome], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    refit_audit = pd.DataFrame([{
        "phase": "RP", "status": "disabled",
        "workflow_mode": settings.positional_isomer_workflow_mode,
    }])
    if settings.positional_isomer_workflow_mode != ITERATIVE_POSITIONAL_REFIT:
        return combined, outcomes, refit_audit, pd.DataFrame(), anchors
    tolerance = max(
        float(settings.rt_tolerance_seconds),
        float(settings.positional_isomer_rt_tolerance_seconds),
    )
    provisional_pairs = _mark_positional_seed_anchors(
        combined["all_pairs"], anchors,
        excluded_species=settings.excluded_anchor_species,
        maximum_rt_error_seconds=tolerance,
    )
    provisional_annotations, provisional_pairs, provisional_audit = (
        resolve_positional_lyso_isomers(
            combined["annotations"], provisional_pairs,
            reference=combined["reference"],
            settings=_positional_resolver_settings(
                settings, maximum_rt_error_seconds=tolerance
            ),
        )
    )
    refit_project, refit_audit, refit_anchors = _build_positional_refit_project(
        seeded, provisional_annotations, provisional_pairs, phase="RP"
    )
    refit_audit["workflow_mode"] = settings.positional_isomer_workflow_mode
    if not refit_audit["status"].astype(str).eq("refit_membership").any():
        return combined, outcomes, refit_audit, provisional_audit, anchors
    if progress:
        progress(0.7, "RP: refitting RT model with resolved LPC/LPE identities")
    combined, outcomes = _run_phase(
        refit_project, reference, "RP",
        progress=(lambda f, message: progress(0.7 + 0.3 * f, message))
        if progress else None,
        phase_index=0,
        phase_count=1,
        only_accepted=only_accepted,
        composition_ranges=composition_ranges,
    )
    combined = _prepare_mapper_output(combined, project, reference, phase="RP")
    return combined, outcomes, refit_audit, provisional_audit, (
        refit_anchors if not refit_anchors.empty else anchors
    )


def _merge_provisional_positional_audit(
    final: pd.DataFrame, provisional: pd.DataFrame
) -> pd.DataFrame:
    if final.empty or provisional.empty or "pair_id" not in final or "pair_id" not in provisional:
        return final
    columns = [column for column in (
        "pair_id", "status", "selected_one_feature_id",
        "selected_two_feature_id", "assignment_score",
    ) if column in provisional]
    earlier = provisional[columns].copy().rename(columns={
        column: f"provisional_{column}" for column in columns if column != "pair_id"
    })
    return final.merge(earlier, on="pair_id", how="left", validate="one_to_one")


def _resolve_final_positional(
    combined: dict,
    anchors: pd.DataFrame,
    settings: AutomaticAnchorSettings,
    provisional_audit: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    combined["all_pairs"] = _mark_positional_seed_anchors(
        combined["all_pairs"], anchors,
        excluded_species=settings.excluded_anchor_species,
        maximum_rt_error_seconds=settings.positional_isomer_rt_tolerance_seconds,
    )
    combined["annotations"], combined["all_pairs"], audit = (
        resolve_positional_lyso_isomers(
            combined["annotations"], combined["all_pairs"],
            reference=combined["reference"],
            settings=_positional_resolver_settings(settings),
        )
    )
    audit["workflow_mode"] = settings.positional_isomer_workflow_mode
    return combined, _merge_provisional_positional_audit(audit, provisional_audit)


def _apply_matlab_grid_policy(
    combined: dict,
    phase: str,
    settings: AutomaticAnchorSettings,
) -> tuple[dict, pd.DataFrame]:
    combined["all_pairs"], audit = _calculate_matlab_grid_coherence(
        combined["all_pairs"], settings=settings
    )
    combined["annotations"] = _copy_grid_coherence(
        combined["annotations"], combined["all_pairs"]
    )
    if not settings.use_grid_coherence or combined["annotations"].empty:
        return combined, audit
    outliers = combined["annotations"]["grid_coherence_outlier"].fillna(False).astype(bool)
    filtered = int(outliers.sum())
    combined["annotations"] = combined["annotations"].loc[~outliers].reset_index(drop=True)
    orphaned = 0
    if phase == "RP":
        combined["annotations"], orphaned = _drop_orphan_two_acyl_lyso(
            combined["annotations"]
        )
    audit.loc[0, "filtered_annotations"] = filtered + orphaned
    audit.loc[0, "orphan_two_acyl_removed"] = orphaned
    return combined, audit


def _assemble_automatic_phase_result(
    project_dir: str | Path,
    reference_root: Path,
    internal_standard_dir: Path,
    phase: str,
    bundle: Any,
    reference: Any,
    combined: dict,
    outcomes: list[CompoundOutcome],
    anchors: pd.DataFrame,
    seed_audit: pd.DataFrame,
    reference_seed_audit: pd.DataFrame,
    grid_audit: pd.DataFrame,
    positional_audit: pd.DataFrame,
    refit_audit: pd.DataFrame,
    settings: AutomaticAnchorSettings,
    composition_ranges: CompositionRangeSettings,
    mass_calibrations: Mapping[str, object] | None = None,
) -> dict:
    streams = _split_xcms_streams(combined, bundle)
    restored_anchors = []
    completion_audits = []
    fitted_grid_audits = []
    for stream_name, stream in streams.items():
        stream["criteria"]["rt_model_source"] = f"matlab_{phase.lower()}_rtdb"
        stream["criteria"]["use_grid_coherence"] = bool(
            settings.use_grid_coherence
        )
        stream["criteria"]["grid_coherence_threshold"] = float(
            settings.grid_coherence_threshold
        )
        seed_stream_anchors = _restore_xcms_anchor_frame(
            anchors, bundle, stream_name
        )
        stream["automatic_anchors"] = (
            _reconcile_positional_review_anchors(
                seed_stream_anchors,
                stream["all_pairs"],
                excluded_species=settings.excluded_anchor_species,
            )
            if phase == "RP"
            else seed_stream_anchors.copy()
        )
        if not seed_stream_anchors.empty:
            restored_anchors.append(seed_stream_anchors)
        (
            stream["annotations"],
            stream["all_pairs"],
            completion_audit,
        ) = complete_post_regression_annotations(
            stream["annotations"],
            stream["all_pairs"],
            stream=stream_name,
            settings=PostRegressionCompletionSettings(
                enabled=settings.use_post_regression_completion,
                confidence_threshold=(
                    settings.post_regression_confidence_threshold
                ),
                ambiguity_margin=settings.post_regression_ambiguity_margin,
            ),
        )
        completion_audits.append(completion_audit)
        (
            stream["annotations"], stream["all_pairs"], fitted_grid_audit,
        ) = refine_fitted_grid_annotations(
            stream["annotations"], stream["all_pairs"],
            stream["reference"], stream["features"], stream=stream_name,
            settings=FittedGridCompletionSettings(
                enabled=(settings.use_post_regression_completion
                         and settings.use_fitted_grid_completion),
                confidence_threshold=settings.post_regression_confidence_threshold,
                ambiguity_margin=settings.post_regression_ambiguity_margin,
                feature_ambiguity_margin=settings.fitted_grid_feature_ambiguity_margin,
                grid_threshold=settings.grid_coherence_threshold,
            ),
        )
        fitted_grid_audits.append(fitted_grid_audit)
    result = assemble_result(
        streams,
        outcomes,
        Path(project_dir).expanduser().resolve(),
        reference_root,
    )
    result["matlab_seed_audit"] = seed_audit
    result["matlab_reference_seed_audit"] = reference_seed_audit
    result["matlab_automatic_anchors"] = (
        pd.concat(restored_anchors, ignore_index=True)
        if restored_anchors
        else _restore_xcms_anchor_frame(anchors, bundle, "")
    )
    result["matlab_internal_standard_audit"] = _internal_standard_audit(
        bundle.project,
        reference,
        anchors,
        combined["annotations"],
        combined["all_pairs"],
        phase=phase,
    )
    result["matlab_grid_coherence_audit"] = grid_audit
    result["matlab_post_regression_audit"] = pd.concat(
        completion_audits, ignore_index=True, sort=False
    )
    result["matlab_fitted_grid_audit"] = pd.concat(
        fitted_grid_audits, ignore_index=True, sort=False,
    )
    result["matlab_positional_isomer_audit"] = positional_audit
    result["matlab_positional_refit_audit"] = refit_audit
    result["matlab_xcms_only"] = True
    result["matlab_phase"] = phase
    result["matlab_rtdb_phase"] = phase
    result["matlab_project_dir"] = None
    result["xcms_input_dir"] = str(Path(project_dir).expanduser().resolve())
    result["anchor_settings"] = {
        key: value for key, value in vars(settings).items()
    }
    result["composition_range_settings"] = composition_ranges.to_dict()
    result["internal_standard_source"] = str(
        internal_standard_dir / "internal_standards.tsv"
    )
    result["internal_standards_expected"] = [
        {
            "name": item.name,
            "quantity": item.quantity,
            "unit": item.unit,
        }
        for item in reference.internal_standards
    ]
    attach_matlab_mass_calibration_result(
        result, phase=phase, calibrations=mass_calibrations
    )
    return result


def _run_matlab_annotation_phase(
    project_dir: str | Path,
    *,
    phase: str,
    reference_dir: str | Path = DEFAULT_MATLAB_PIPELINE_REFERENCE,
    only_accepted: bool = True,
    progress: Callable[[float, str], None] | None = None,
    anchor_settings: AutomaticAnchorSettings = AutomaticAnchorSettings(),
    composition_ranges: CompositionRangeSettings = DEFAULT_COMPOSITION_RANGE_SETTINGS,
    mass_calibrations: Mapping[str, object] | None = None,
) -> dict:
    """Automatically annotate one isolated NP or RP XCMS phase."""
    phase = str(phase).upper()
    if phase not in {"NP", "RP"}:
        raise MatlabAnnotationError(
            f"Automatic MATLAB-engine mode requires NP or RP, not {phase!r}."
        )
    mode = anchor_settings.positional_isomer_workflow_mode
    if mode not in POSITIONAL_ISOMER_WORKFLOW_MODES:
        raise MatlabAnnotationError(
            f"unknown positional-isomer workflow mode: {mode!r}; "
            f"expected one of {POSITIONAL_ISOMER_WORKFLOW_MODES}"
        )
    reference_root = Path(reference_dir).expanduser().resolve()
    try:
        bundle, internal_dir, reference, seeded, seed_audit, anchors, ref_audit = (
            _load_automatic_phase(
                project_dir, phase, reference_root, only_accepted, progress,
                anchor_settings, composition_ranges,
                mass_calibrations,
            )
        )
        iterative = phase == "RP" and mode == ITERATIVE_POSITIONAL_REFIT
        combined, outcomes = _initial_automatic_mapping(
            seeded, reference, phase, only_accepted, progress, iterative,
            composition_ranges,
        )
        reporting_anchors = anchors
        if phase == "NP":
            combined, reporting_anchors = _collapse_np_mapping(combined, anchors)
        combined = _prepare_mapper_output(
            combined, bundle.project, reference, phase=phase
        )
        refit_audit = pd.DataFrame([{
            "phase": phase,
            "status": "disabled" if phase == "RP" else "not_applicable",
            "workflow_mode": mode,
        }])
        positional_audit = pd.DataFrame([{
            "phase": "NP", "status": "not_applicable", "workflow_mode": mode,
            "detail": "NP uses its own combined phase identities",
        }])
        if phase == "RP":
            combined, outcomes, refit_audit, provisional, final_anchors = (
                _provisional_positional_refit(
                    combined, outcomes, seeded, reference, bundle.project, anchors,
                    anchor_settings, only_accepted, composition_ranges, progress,
                )
            )
            combined, positional_audit = _resolve_final_positional(
                combined, final_anchors, anchor_settings, provisional
            )
        combined, grid_audit = _apply_matlab_grid_policy(
            combined, phase, anchor_settings
        )
        if phase == "RP":
            positional_audit = _finalize_positional_isomer_audit(
                positional_audit, combined["annotations"], combined["all_pairs"]
            )
    except (ValueError, ReferenceDataError, np.linalg.LinAlgError) as error:
        raise MatlabAnnotationError(str(error)) from error
    return _assemble_automatic_phase_result(
        project_dir, reference_root, internal_dir, phase, bundle, reference,
        combined, outcomes, reporting_anchors, seed_audit, ref_audit, grid_audit,
        positional_audit, refit_audit, anchor_settings, composition_ranges,
        mass_calibrations,
    )


def _merge_matlab_phase_results(
    results: list[dict],
    *,
    requested_phases: tuple[str, ...],
) -> dict:
    """Merge phase-local exports without merging their models or grids."""
    if not results:
        raise MatlabAnnotationError("No requested NP/RP XCMS streams were found.")
    streams = {
        name: stream for result in results for name, stream in result["streams"].items()
    }
    first = results[0]
    merged = assemble_result(
        streams,
        [],
        Path(first["xcms_input_dir"]),
        Path(first["matlab_reference_dir"]),
    )
    table_keys = (
        "matlab_compound_audit",
        "matlab_seed_audit",
        "matlab_reference_seed_audit",
        "matlab_automatic_anchors",
        "matlab_internal_standard_audit",
        "matlab_grid_coherence_audit",
        "matlab_post_regression_audit",
        "matlab_fitted_grid_audit",
        "matlab_positional_isomer_audit",
        "matlab_positional_refit_audit",
        "matlab_mass_calibration_audit",
        "matlab_mass_calibration_anchors",
    )
    for key in table_keys:
        merged[key] = concat_result_tables(results, key)
    completed = tuple(result["matlab_phase"] for result in results)
    merged.update(
        {
            "matlab_xcms_only": True,
            "matlab_phase": "+".join(completed),
            "matlab_phases": completed,
            "matlab_rtdb_phase": "+".join(completed),
            "matlab_rtdb_phases": completed,
            "matlab_requested_phases": requested_phases,
            "matlab_skipped_phases": tuple(
                phase for phase in requested_phases if phase not in completed
            ),
            "matlab_project_dir": None,
            "xcms_input_dir": first["xcms_input_dir"],
            "anchor_settings": first["anchor_settings"],
            "composition_range_settings": first.get(
                "composition_range_settings",
                DEFAULT_COMPOSITION_RANGE_SETTINGS.to_dict(),
            ),
            "internal_standard_source": first["internal_standard_source"],
            "internal_standards_expected": first["internal_standards_expected"],
            "mass_calibrations": {
                stream: calibration
                for result in results
                for stream, calibration in result.get(
                    "mass_calibrations", {}
                ).items()
            },
        }
    )
    return merged


def run_matlab_annotation(
    project_dir: str | Path,
    *,
    reference_dir: str | Path = DEFAULT_MATLAB_PIPELINE_REFERENCE,
    phases: tuple[str, ...] | None = None,
    only_accepted: bool = True,
    progress: Callable[[float, str], None] | None = None,
    anchor_settings: AutomaticAnchorSettings = AutomaticAnchorSettings(),
    composition_ranges: CompositionRangeSettings = (DEFAULT_COMPOSITION_RANGE_SETTINGS),
    mass_calibrations: Mapping[str, object] | None = None,
) -> dict:
    """Annotate RP and NP by default as fully isolated phase-local runs.

    A caller can still request only RP or only NP. When both are requested,
    RTDB calibration, anchors, sibling regression and grid coherence are run
    independently and only the finished result tables are combined.
    """
    raw_phases = phases or ("RP", "NP")
    requested = tuple(dict.fromkeys(str(item).upper() for item in raw_phases))
    invalid = tuple(phase for phase in requested if phase not in {"NP", "RP"})
    if not requested or invalid:
        detail = f": {', '.join(invalid)}" if invalid else ""
        raise MatlabAnnotationError(
            f"Automatic MATLAB-engine phases must be NP and/or RP{detail}."
        )
    calibrations = normalize_mass_calibrations(mass_calibrations)
    sources = discover_standard_streams(project_dir)
    available = tuple(
        phase
        for phase in requested
        if any(name.startswith(f"{phase}_") for name in sources)
    )
    if not available:
        expected = ", ".join(f"peakgrouplist_{phase}_Pos/Neg" for phase in requested)
        raise MatlabAnnotationError(
            f"No requested XCMS phase files were found; expected {expected}."
        )

    results: list[dict] = []
    total = len(available)
    for index, phase in enumerate(available):
        phase_progress = None
        if progress:

            def phase_progress(fraction, message, index=index):
                progress(
                    (index + min(max(fraction, 0.0), 1.0)) / total,
                    message,
                )

        results.append(
            _run_matlab_annotation_phase(
                project_dir,
                phase=phase,
                reference_dir=reference_dir,
                only_accepted=only_accepted,
                progress=phase_progress,
                anchor_settings=anchor_settings,
                composition_ranges=composition_ranges,
                mass_calibrations=calibrations,
            )
        )
    if len(results) == 1:
        result = results[0]
        result["matlab_phases"] = available
        result["matlab_rtdb_phases"] = available
        result["matlab_requested_phases"] = requested
        result["matlab_skipped_phases"] = tuple(
            phase for phase in requested if phase not in available
        )
        return result
    return _merge_matlab_phase_results(results, requested_phases=requested)
