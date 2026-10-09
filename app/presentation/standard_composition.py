"""Editable composition-range criteria for standard lipidomics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from annotator.composition_ranges import (
    DEFAULT_COMPOSITION_RANGE_SETTINGS,
    CompositionRangeSettings,
    composition_range_settings_from_dict,
)


def _range_columns() -> dict:
    return {
        "carbon_min": st.column_config.NumberColumn("C min", min_value=0, step=1, format="%d"),
        "carbon_max": st.column_config.NumberColumn("C max", min_value=0, step=1, format="%d"),
        "db_min": st.column_config.NumberColumn("DB min", min_value=0, step=1, format="%d"),
        "db_max": st.column_config.NumberColumn("DB max", min_value=0, step=1, format="%d"),
    }


def _edit_groups(key_prefix: str) -> pd.DataFrame:
    defaults = DEFAULT_COMPOSITION_RANGE_SETTINGS
    rows = [{"major_group": name, **limits.to_dict()} for name, limits in defaults.group_ranges]
    return st.data_editor(
        pd.DataFrame(rows), key=f"{key_prefix}_composition_group_editor",
        hide_index=True, width="stretch", disabled=["major_group"],
        column_config={"major_group": st.column_config.TextColumn("Major lipid group"), **_range_columns()},
    )


def _edit_classes(key_prefix: str) -> pd.DataFrame:
    rows = [{
        "lipid_class": rule.lipid_class, "major_group": rule.major_group,
        "filter_enabled": rule.enabled, "use_group": rule.use_group_range,
        **rule.limits.to_dict(),
    } for rule in DEFAULT_COMPOSITION_RANGE_SETTINGS.class_ranges]
    return st.data_editor(
        pd.DataFrame(rows), key=f"{key_prefix}_composition_class_editor",
        hide_index=True, width="stretch", height=420,
        disabled=["lipid_class", "major_group"],
        column_config={
            "lipid_class": st.column_config.TextColumn("Lipid class"),
            "major_group": st.column_config.TextColumn("Major group"),
            "filter_enabled": st.column_config.CheckboxColumn("Filter enabled"),
            "use_group": st.column_config.CheckboxColumn("Use group"),
            **_range_columns(),
        },
    )


def _settings(enabled: bool, use_groups: bool, groups: pd.DataFrame, classes: pd.DataFrame) -> CompositionRangeSettings:
    keys = ("carbon_min", "carbon_max", "db_min", "db_max")
    group_config = {
        str(row["major_group"]): {key: int(row[key]) for key in keys}
        for _, row in groups.iterrows()
    }
    class_config = {
        str(row["lipid_class"]): {
            "major_group": str(row["major_group"]),
            "enabled": bool(row["filter_enabled"]),
            "use_group_range": bool(row["use_group"]),
            **{key: int(row[key]) for key in keys},
        }
        for _, row in classes.iterrows()
    }
    return composition_range_settings_from_dict({
        "enabled": bool(enabled), "use_group_ranges": bool(use_groups),
        "groups": group_config, "classes": class_config,
    })


def render_composition_range_settings(key_prefix: str) -> CompositionRangeSettings:
    """Render group/class carbon and double-bond criteria."""
    defaults = DEFAULT_COMPOSITION_RANGE_SETTINGS
    enabled = st.checkbox(
        "Enable carbon and double-bond limits",
        value=defaults.enabled,
        key=f"{key_prefix}_composition_ranges_enabled",
        help=(
            "Use inclusive total-carbon and double-bond limits. Bundled class "
            "defaults equal the extrema in the example workbooks; internal "
            "standards are always protected. Turn this off to disable all "
            "composition-range filtering."
        ),
    )
    use_groups = st.checkbox(
        "Apply major lipid-group limits",
        value=defaults.use_group_ranges,
        key=f"{key_prefix}_composition_group_ranges_enabled",
        help=(
            "Both group and class limits apply when enabled. Turn this off to "
            "keep only per-class ranges. In the class table, clear Filter "
            "enabled to remove all limits for one class or clear Use group to "
            "retain only that class's own editable range."
        ),
    )
    groups = _edit_groups(key_prefix)
    classes = _edit_classes(key_prefix)
    try:
        return _settings(enabled, use_groups, groups, classes)
    except (TypeError, ValueError) as error:
        st.error(f"Invalid carbon/double-bond limits: {error}")
        st.stop()
        raise RuntimeError(
            "Streamlit failed to stop after invalid composition limits"
        ) from error
