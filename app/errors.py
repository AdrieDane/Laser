"""Typed public errors for LASER workflow boundaries."""


class LASERError(Exception):
    """Base class for errors that can be presented directly to a user."""


class InputSchemaError(LASERError, ValueError):
    """An input table does not satisfy its versioned data contract."""


class ManifestSchemaError(LASERError, ValueError):
    """A run or portable-input manifest is invalid or unsupported."""


class MZmineExecutionError(LASERError, RuntimeError):
    """MZmine did not complete its requested processing stage."""


class AnnotationStageError(LASERError, RuntimeError):
    """A named scientific annotation stage could not be completed."""


class WorkflowConfigurationError(LASERError, ValueError):
    """Resolved workflow settings are missing, invalid, or inconsistent."""


class ArtifactContractError(LASERError, ValueError):
    """A declared saved-project artifact is missing or ambiguous."""


class OutputWriteError(LASERError, OSError):
    """A durable workflow output could not be written atomically."""
