"""Project method editor for the untargeted launcher."""
from copy import deepcopy

import pandas as pd
import streamlit as st

from standard_lipidomics.acquisition_method import (
    ADDITIVES, UNITS, ion_state, method_id, normalize_acquisition_method,
    save_acquisition_method,
)
from standard_lipidomics.untargeted_adducts import ADDUCTS


def _rows(frame):
    clean = frame.dropna(how="all").astype(object)
    return clean.where(pd.notna(clean), None).to_dict("records")


def render_acquisition_settings(result, project):
    method = normalize_acquisition_method(result.get("acquisition_method"))
    prefix = "acq-" + str(project) + "-" + method_id(method)[:12]
    edited = deepcopy(method)
    with st.expander("Acquisition and adduct settings", expanded=False):
        mode = st.selectbox("Adduct scoring", ("legacy", "method_aware"),
            index=(0 if method["adduct_scoring"] == "legacy" else 1),
            format_func=lambda value: "Current scoring" if value == "legacy" else "Method-aware (experimental)",
            key=prefix + "-scoring")
        edited["adduct_scoring"] = mode
        edited["provenance"] = st.text_input("Method source", value=method["provenance"], key=prefix + "-source")
        edited["lc_system"] = st.text_input("LC system", value=method["lc_system"], key=prefix + "-lc")
        for phase, tab in zip(("RP", "NP"), st.tabs(("RP method", "NP method"))):
            with tab:
                settings = edited["methods"][phase]
                settings["column"] = st.text_input("Column", value=settings["column"], key=prefix + phase + "-column")
                settings["polarities"] = st.multiselect("Acquired polarities", ("pos", "neg"),
                    default=settings["polarities"], key=prefix + phase + "-polarities")
                for solvent in ("A", "B"):
                    recipe = settings["solvents"][solvent]
                    recipe["composition"] = st.text_input(f"Solvent {solvent}", value=recipe["composition"],
                        key=prefix + phase + solvent + "-composition")
                    table = pd.DataFrame(recipe["additives"], columns=["name", "concentration", "unit"])
                    rows = st.data_editor(table, num_rows="dynamic", hide_index=True,
                        key=prefix + phase + solvent + "-additives", width="stretch",
                        column_config={"name": st.column_config.SelectboxColumn("Additive", options=ADDITIVES, required=True),
                            "concentration": st.column_config.NumberColumn("Concentration", min_value=0.0),
                            "unit": st.column_config.SelectboxColumn("Unit", options=UNITS, required=True)})
                    recipe["additives"] = _rows(rows)
                cols = st.columns(2)
                settings["temperature_c"] = cols[0].number_input("Column temperature (°C)",
                    value=settings["temperature_c"], key=prefix + phase + "-temperature")
                settings["flow_ml_min"] = cols[1].number_input("Flow (mL/min)", min_value=0.001,
                    value=settings["flow_ml_min"], format="%.3f", key=prefix + phase + "-flow")
                gradient = pd.DataFrame(settings["gradient"], columns=["time_min", "percent_a"])
                settings["gradient"] = _rows(st.data_editor(gradient, hide_index=True, num_rows="dynamic",
                    key=prefix + phase + "-gradient", width="stretch", column_config={
                        "time_min": st.column_config.NumberColumn("Time (min)", min_value=0.0),
                        "percent_a": st.column_config.NumberColumn("Solvent A (%)", min_value=0.0, max_value=100.0)}))
                instrument = settings["instrument"]
                instrument["model"] = st.text_input("Instrument", value=instrument["model"], key=prefix + phase + "-instrument")
                instrument["spray_voltage_v"] = st.number_input("Spray voltage (V)", min_value=1.0,
                    value=float(instrument["spray_voltage_v"]) if instrument["spray_voltage_v"] else None,
                    key=prefix + phase + "-spray")
                cols = st.columns(2)
                instrument["resolution"] = cols[0].number_input("Resolving power", min_value=1,
                    value=int(instrument["resolution"]) if instrument["resolution"] else None, key=prefix + phase + "-resolution")
                instrument["resolution_reference_mz"] = cols[1].number_input("Resolution reference m/z", min_value=1.0,
                    value=float(instrument["resolution_reference_mz"]) if instrument["resolution_reference_mz"] else None, key=prefix + phase + "-reference-mz")
                cols = st.columns(2)
                bounds = instrument["mass_range_mz"] or (None, None)
                lower = cols[0].number_input("Acquisition lower m/z", min_value=1.0, value=float(bounds[0]) if bounds[0] else None, key=prefix + phase + "-lower")
                upper = cols[1].number_input("Acquisition upper m/z", min_value=1.0, value=float(bounds[1]) if bounds[1] else None, key=prefix + phase + "-upper")
                instrument["mass_range_mz"] = None if lower is None and upper is None else [lower, upper]
                for polarity in ("pos", "neg"):
                    st.markdown(f"**{polarity.upper()} ions**")
                    records = [{"ion": ion.name, "setting": settings["adducts"][polarity].get(ion.name, "automatic"),
                                "recipe_state": ion_state(edited, f"{phase}_{polarity.title()}", ion.name)}
                               for ion in ADDUCTS if ion.polarity == polarity]
                    ions = st.data_editor(pd.DataFrame(records), hide_index=True, width="stretch",
                        disabled=["ion", "recipe_state"], key=prefix + phase + polarity + "-ions",
                        column_config={"ion": "Ion", "setting": st.column_config.SelectboxColumn("Use",
                            options=("automatic", "common", "possible", "disabled"), required=True),
                            "recipe_state": "Effective state"})
                    settings["adducts"][polarity] = {r.ion: r.setting for r in ions.itertuples() if r.setting != "automatic"}
                    calibration = settings["calibration"][polarity]
                    cols = st.columns(2)
                    for column, kind in zip(cols, ("mz", "rt")):
                        calibration[kind] = column.text_input(f"{polarity.upper()} {kind} input calibration record",
                            value=calibration[kind] or "", key=prefix + phase + polarity + kind + "-calibration",
                            help="Record the calibration identifier used for these input features. This field does not apply a calibration.") or None
        try:
            edited = normalize_acquisition_method(edited)
        except ValueError as exc:
            st.error(str(exc))
            return None
        if st.button("Save method settings", key=prefix + "-save"):
            save_acquisition_method(project, edited)
            st.success("Method settings saved.")
    return edited
