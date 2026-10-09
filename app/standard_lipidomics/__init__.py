"""Coherent, lazily loaded domain API for conventional XCMS lipidomics."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "DEFAULT_FA_CARBON_RANGE": "policy",
    "DEFAULT_FA_DB_RANGE": "policy",
    "DEFAULT_MATLAB_POLICY_PATH": "policy",
    "apply_composition_range_filter": "policy",
    "apply_matlab_lipid_policy": "policy",
    "discover_matlab_fa_ranges": "policy",
    "discover_matlab_policy_file": "policy",
    "load_matlab_lipid_policy": "policy",
    "build_standard_report_data": "report",
    "render_standard_lipidomics_report": "report",
    "write_standard_lipidomics_report": "report",
    "UntargetedComparisonCriteria": "untargeted",
    "analyse_standard_untargeted": "untargeted",
    "attach_excel_sample_metadata": "untargeted",
    "default_biological_groups": "untargeted",
    "group_catalog": "untargeted",
    "DEFAULT_STANDARD_PPM_TOLERANCE": "workflow",
    "MAX_STANDARD_PPM_TOLERANCE": "workflow",
    "StandardStream": "workflow",
    "annotate_standard_batch": "workflow",
    "annotate_standard_stream": "workflow",
    "discover_standard_streams": "workflow",
    "fit_automatic_class_rt_models": "workflow",
    "load_standard_feature_list": "workflow",
    "parse_standard_stream": "workflow",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load one public owner only when its symbol is requested."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
