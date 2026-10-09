"""MATLAB-equivalent annotation engine for LASER's standard-lipidomics mode.

``vendor`` is a verbatim subset of AI_Lipidomics_Pipeline's
``headless_annotation`` package — the reverse-engineered Amsterdam UMC MATLAB
annotation application. Everything outside ``vendor`` is LASER's adapter: it
decodes MATLAB annotation codes into LASER's fields and shapes the run as the
result dictionary standard mode already renders.
"""

from .normalization import (
    DecodedAnnotation,
    composition_dimensions,
    decode_annotation_code,
    lipid_class_from_template,
    resolve_positional_lyso_isomers,
)
from .configuration import (
    DEFAULT_MATLAB_PIPELINE_REFERENCE,
    MATLAB_BROAD_PPM,
    MATLAB_NARROW_PPM,
    AUTOMATIC_ANCHOR_SELECTION_MODES,
    AUTOMATIC_REGRESSION_INLIERS,
    BIOLOGICAL_ABUNDANCE_PRIOR,
    ITERATIVE_POSITIONAL_REFIT,
    POSITIONAL_ISOMER_WORKFLOW_MODES,
    POST_REGRESSION_POSITIONAL_RESOLUTION,
    UNIFIED_EVIDENCE_ANCHORS,
    AutomaticAnchorSettings,
    PositionalIsomerSettings,
)
from .execution import (
    CompoundOutcome,
    MatlabAnnotationError,
    run_legacy_matlab_annotation,
    run_matlab_annotation,
)
from .preparation import discover_matlab_annotation_project

__all__ = [
    "DEFAULT_MATLAB_PIPELINE_REFERENCE",
    "MATLAB_BROAD_PPM",
    "MATLAB_NARROW_PPM",
    "CompoundOutcome",
    "AUTOMATIC_ANCHOR_SELECTION_MODES",
    "AUTOMATIC_REGRESSION_INLIERS",
    "BIOLOGICAL_ABUNDANCE_PRIOR",
    "ITERATIVE_POSITIONAL_REFIT",
    "POSITIONAL_ISOMER_WORKFLOW_MODES",
    "POST_REGRESSION_POSITIONAL_RESOLUTION",
    "UNIFIED_EVIDENCE_ANCHORS",
    "AutomaticAnchorSettings",
    "DecodedAnnotation",
    "MatlabAnnotationError",
    "PositionalIsomerSettings",
    "composition_dimensions",
    "decode_annotation_code",
    "discover_matlab_annotation_project",
    "lipid_class_from_template",
    "run_legacy_matlab_annotation",
    "run_matlab_annotation",
    "resolve_positional_lyso_isomers",
]
