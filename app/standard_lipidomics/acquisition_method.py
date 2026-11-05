"""Versioned project chemistry; preferences are soft evidence, not identities.

Literature rationale: doi:10.1194/jlr.D001149; doi:10.1038/s41467-021-24984-y.
No study reference labels or fitted class frequencies enter these preferences.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

from output_io import atomic_write_json
from standard_lipidomics.project_layout import xcms_project_root

METHOD_FILE = "acquisition_method.json"
POLICY_VERSION = 1
EXCLUDED_IONS = {"[M+2H]2+", "[M+Na-2H]-"}
STATES = ("common", "possible", "disabled")
ADDITIVES = ("formic acid", "ammonia", "ammonium formate", "ammonium acetate", "acetic acid", "other")
UNITS = ("mM", "µM", "M", "%", "% v/v", "% w/v", "unknown")


def default_acquisition_method():
    return {"schema_version": 1, "adduct_scoring": "legacy", "provenance": "Method not supplied", "lc_system": "unknown",
        "methods": {phase: {"polarities": ["pos", "neg"], "column": "unknown", "solvents": {
            name: {"composition": "unknown", "additives": []} for name in ("A", "B")},
            "gradient": [], "temperature_c": None, "flow_ml_min": None,
            "instrument": {"model": "unknown", "resolution": None,
                           "spray_voltage_v": None,
                           "resolution_reference_mz": None, "mass_range_mz": None},
            "calibration": {pol: {"mz": None, "rt": None} for pol in ("pos", "neg")},
            "adducts": {"pos": {}, "neg": {}}} for phase in ("RP", "NP")}}


def _number(value, label, *, positive=False):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{label} must be a finite {'positive ' if positive else ''}number or null.")


def _validate_gradient(gradient):
    if not isinstance(gradient, list):
        raise ValueError("Gradient must be a list.")
    previous = -1
    for point in gradient:
        if not isinstance(point, dict) or set(point) != {"time_min", "percent_a"}:
            raise ValueError("Gradient points need time_min and percent_a.")
        for key in ("time_min", "percent_a"):
            _number(point[key], key)
            if point[key] is None:
                raise ValueError("Gradient values cannot be null.")
        if point["time_min"] < 0 or point["time_min"] <= previous or not 0 <= point["percent_a"] <= 100:
            raise ValueError("Gradient times must increase and A must be between 0 and 100%.")
        previous = point["time_min"]


def normalize_acquisition_method(value=None):
    if value is None:
        return default_acquisition_method()
    if not isinstance(value, dict):
        raise ValueError("Acquisition method must be a JSON object.")
    default = default_acquisition_method()
    if set(value) != set(default) or type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise ValueError("Unsupported acquisition method schema or missing/unknown top-level fields.")
    if (value["adduct_scoring"] not in ("legacy", "method_aware") or not isinstance(value["provenance"], str)
            or not isinstance(value["lc_system"], str)):
        raise ValueError("Choose legacy or method_aware adduct scoring and a text provenance.")
    if not isinstance(value["methods"], dict) or set(value["methods"]) != {"RP", "NP"}:
        raise ValueError("Acquisition method must define RP and NP separately.")
    from standard_lipidomics.untargeted_adducts import ADDUCTS
    for phase, method in value["methods"].items():
        if not isinstance(method, dict) or set(method) != set(default["methods"][phase]):
            raise ValueError(f"Invalid fields in {phase} method.")
        if not isinstance(method["column"], str):
            raise ValueError("Column must be text.")
        if not isinstance(method["polarities"], list) or not method["polarities"] or not set(method["polarities"]) <= {"pos", "neg"}:
            raise ValueError(f"Invalid {phase} polarities.")
        if not isinstance(method["solvents"], dict) or set(method["solvents"]) != {"A", "B"}:
            raise ValueError(f"{phase} needs separate A/B solvent recipes.")
        for solvent in method["solvents"].values():
            if not isinstance(solvent, dict) or set(solvent) != {"composition", "additives"} or not isinstance(solvent["composition"], str):
                raise ValueError("Solvent needs composition text and additive records.")
            if not isinstance(solvent["additives"], list):
                raise ValueError("Additives must be a list.")
            for additive in solvent["additives"]:
                if not isinstance(additive, dict) or set(additive) != {"name", "concentration", "unit"}:
                    raise ValueError("Each additive needs name, concentration and unit.")
                if additive["name"] not in ADDITIVES or additive["unit"] not in UNITS:
                    raise ValueError("Unknown additive name or concentration unit.")
                _number(additive["concentration"], "Additive concentration", positive=True)
                if additive["unit"].startswith("%") and additive["concentration"] is not None and additive["concentration"] > 100:
                    raise ValueError("Percentage concentrations cannot exceed 100.")
        _number(method["temperature_c"], "Column temperature")
        _number(method["flow_ml_min"], "Flow rate", positive=True)
        _validate_gradient(method["gradient"])
        instrument = method["instrument"]
        if not isinstance(instrument, dict) or set(instrument) != set(default["methods"][phase]["instrument"]) or not isinstance(instrument["model"], str):
            raise ValueError("Invalid instrument settings.")
        for key in ("resolution", "resolution_reference_mz", "spray_voltage_v"):
            _number(instrument[key], key, positive=True)
        mass_range = instrument["mass_range_mz"]
        if mass_range is not None:
            if not isinstance(mass_range, list) or len(mass_range) != 2:
                raise ValueError("Mass range needs two values or null.")
            for x in mass_range:
                _number(x, "Mass range", positive=True)
            if None in mass_range or mass_range[0] >= mass_range[1]:
                raise ValueError("Mass range lower bound must be below upper bound.")
        for section in ("calibration", "adducts"):
            if not isinstance(method[section], dict) or set(method[section]) != {"pos", "neg"}:
                raise ValueError(f"{section} needs positive and negative settings.")
        for pol in ("pos", "neg"):
            calibration = method["calibration"][pol]
            if not isinstance(calibration, dict) or set(calibration) != {"mz", "rt"} or any(v is not None and not isinstance(v, str) for v in calibration.values()):
                raise ValueError("Calibration selections must be identifiers/text or null.")
            choices = method["adducts"][pol]
            if not isinstance(choices, dict):
                raise ValueError("Adduct choices must be an object.")
            known = {ion.name for ion in ADDUCTS if ion.polarity == pol}
            for name, state in choices.items():
                if name in EXCLUDED_IONS or name not in known or state not in STATES:
                    raise ValueError(f"Unsupported/excluded ion or invalid state: {name}")
    return deepcopy(value)


def method_id(method):
    return hashlib.sha256(json.dumps(method, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_acquisition_method(project):
    path = xcms_project_root(project) / "metadata" / METHOD_FILE
    return normalize_acquisition_method(json.loads(path.read_text()) if path.is_file() else None)


def save_acquisition_method(project, value):
    method = normalize_acquisition_method(value)
    return atomic_write_json(xcms_project_root(project) / "metadata" / METHOD_FILE, method)


def ion_state(method, stream, adduct):
    phase, polarity = stream.split("_")
    polarity = polarity.casefold()
    settings = method["methods"][phase]
    if adduct in EXCLUDED_IONS or polarity not in settings["polarities"]:
        return "disabled"
    explicit = settings["adducts"][polarity].get(adduct)
    if explicit:
        return explicit
    additives = {item["name"] for solvent in settings["solvents"].values() for item in solvent["additives"]}
    if adduct in {"[M+H]+", "[M-H]-"}:
        return "common"
    if adduct == "[M+NH4]+" and additives & {"ammonia", "ammonium formate", "ammonium acetate"}:
        return "common"
    if adduct == "[M+HCOO]-" and additives & {"formic acid", "ammonium formate"}:
        return "common"
    if adduct == "[M+CH3COO]-" and additives & {"acetic acid", "ammonium acetate"}:
        return "common"
    return "possible"  # Absence from a recipe is not an absolute prohibition.


def chemistry_penalty(method, stream, adduct, lipid_class=""):
    """Small, documented preferences. Unknown classes retain generic chemistry."""
    state = ion_state(method, stream, adduct)
    if state == "disabled":
        return 1000.0
    if method["adduct_scoring"] == "legacy":
        return 0.0
    penalty = 0.0 if state == "common" else 1.0
    cls = re.sub(r"^(?:1-acyl|2-acyl)\s+", "", str(lipid_class)).split("(")[0].split("[")[0].strip()
    positive = stream.endswith("Pos")
    preferred = set()
    if cls in {"PC", "LPC", "SM"}:
        preferred = {"[M+H]+"} if positive else {"[M+HCOO]-", "[M+CH3COO]-", "[M+Cl]-"}
    elif cls in {"TG", "DG", "CE"} and positive:
        preferred = {"[M+NH4]+"}
    elif cls in {"PE", "LPE", "PS", "PI", "PG", "PA", "LPA", "LPG", "LPS", "LPI", "FA", "CL", "MLCL", "SM4"}:
        preferred = {"[M+H]+"} if positive else {"[M-H]-"}
        if cls in {"CL", "MLCL"} and not positive:
            preferred.add("[M-2H]2-")
    return penalty + (1.0 if preferred and adduct not in preferred else 0.0)


def stamp_method(frame, method):
    output = frame.copy()
    output["acquisition_method_id"] = method_id(method)
    output["adduct_policy_version"] = POLICY_VERSION
    return output


def guard_homologous_edges(frame, method, same_mode_rt):
    """Retain mass-isobaric adduct alternatives as uncertainty, not lipid edges.

    Formate/acetate differ by exactly one CH2. Without independent ion evidence
    a coeluting negative pair is ambiguous. Never resolve it using p-values.
    """
    if frame.empty:
        output = frame.copy()
        output["adduct_alias_ambiguous"] = False
        return stamp_method(output, method)
    values = []
    for row in frame.itertuples(index=False):
        carbon = getattr(row, "ch2_steps", getattr(row, "carbon_steps", 0))
        hydrogen = getattr(row, "hydrogen_pair_steps", 0)
        charge = getattr(row, "charge_hypothesis", 1)
        delta = getattr(row, "rt_delta_seconds", None)
        if delta is None:
            delta = (row.target_rt - row.source_rt) * 60
        ambiguous = (str(row.stream).endswith("Neg") and abs(carbon) == 1 and hydrogen == 0
                     and charge == 1 and abs(delta) <= same_mode_rt * 60 + 1e-8
                     and all(ion_state(method, row.stream, name) != "disabled" for name in
                             ("[M+HCOO]-", "[M+CH3COO]-")))
        values.append(bool(ambiguous))
    output = frame.copy()
    output["adduct_alias_ambiguous"] = values
    return stamp_method(output, method)


def record_method_on_analysis(analysis, method):
    output = {**analysis, "acquisition_method": normalize_acquisition_method(method)}
    for name, frame in analysis.items():
        if hasattr(frame, "columns") and "stream" in frame.columns:
            output[name] = stamp_method(frame, method)
    return output


def configure_adduct_matches(matches, method, result=None):
    """Only existing accepted runtime identities may provide a class hint."""
    class_map = {}
    annotations = (result or {}).get("automatic_annotations")
    if annotations is not None and not annotations.empty:
        stream_col = next((c for c in ("stream", "standard_stream") if c in annotations), None)
        fid_col = next((c for c in ("feature_id", "feat_feature_id") if c in annotations), None)
        class_col = next((c for c in ("lipid_class", "class") if c in annotations), None)
        if all((stream_col, fid_col, class_col)):
            for key, rows in annotations.groupby([stream_col, fid_col]):
                names = set(rows[class_col].dropna().astype(str)) - {""}
                if len(names) == 1:
                    class_map[(str(key[0]), str(key[1]))] = next(iter(names))
    output = matches.copy()
    for side in ("source", "target"):
        states, penalties, classes = [], [], []
        for row in output.itertuples(index=False):
            stream, fid = str(getattr(row, side + "_stream")), str(getattr(row, side + "_feature_id"))
            adduct = str(getattr(row, side + "_adduct"))
            cls = class_map.get((stream, fid), "")
            classes.append(cls)
            states.append(ion_state(method, stream, adduct))
            penalties.append(chemistry_penalty(method, stream, adduct, cls))
        output[side + "_ion_state"] = states
        output[side + "_chemistry_penalty"] = penalties
        output[side + "_class_hint"] = classes
    if not output.empty:
        output = output[(output.source_ion_state != "disabled") & (output.target_ion_state != "disabled")].copy()
    return stamp_method(output, method)
