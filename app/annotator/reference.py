"""Load and parse the reference lipid database.

The reference file (``rp_lipids_mz.csv``) has one row per (lipid species, adduct)
with a theoretical ``mz`` and a historical median retention time
(``RT_RP_median_min``, in minutes).  Each species name encodes a total carbon
number and total number of double bonds, e.g. ``PC(34:1)`` or ``SM(d30:1)``.
Those two integers are what make each class fall on a regular RT-vs-m/z grid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FORMIC_ACID_MONOISOTOPIC_MASS = 46.0054793036

# Matches the (carbon:doublebond) token, optionally prefixed by a sphingoid /
# ether / plasmalogen marker: d, t, m, o, O-, P-, e-.
_CHAIN_RE = re.compile(r"\(\s*([dtmoeOP])?-?(\d+):(\d+)\)")


@dataclass(frozen=True)
class ParsedName:
    carbon: int | None
    db: int | None
    base: str | None  # sphingoid/ether marker if present (d, t, O, P, ...)
    is_is: bool  # internal standard


def parse_lipid_name(name: str) -> ParsedName:
    """Extract total carbon number and double-bond count from a species name."""
    is_is = "(IS)" in name
    m = _CHAIN_RE.search(name)
    if not m:
        return ParsedName(None, None, None, is_is)
    base = m.group(1)
    carbon = int(m.group(2))
    db = int(m.group(3))
    return ParsedName(carbon, db, base, is_is)


def _polarity_of(charge) -> str:
    try:
        c = int(float(charge))
    except (TypeError, ValueError):
        return "unknown"
    if c > 0:
        return "pos"
    if c < 0:
        return "neg"
    return "unknown"


def load_reference(path: str | Path) -> pd.DataFrame:
    """Load the reference database and enrich it with parsed grid coordinates.

    Returns a DataFrame with the original columns plus: ``carbon``, ``db``,
    ``base``, ``is_is``, ``polarity``, and a numeric ``n_RT`` confidence weight.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    # A self-contained 4D reference may carry CCS aggregates extracted from an
    # MSP atlas. Normalise their descriptive file columns to internal names.
    ccs_aliases = {
        "CCS_N2_median_A2": "ref_ccs",
        "CCS_N2_min_A2": "ref_ccs_min",
        "CCS_N2_max_A2": "ref_ccs_max",
        "CCS_N2_n": "ref_ccs_n",
        "CCS_example_molecular_species": "ref_ccs_names",
        "CCS_mapping": "ref_ccs_mapping",
    }
    df = df.rename(columns={old: new for old, new in ccs_aliases.items()
                            if old in df and new not in df})

    # Normalise the column we rely on for reference RT.
    rt_col = _find_col(df, ["RT_RP_median_min", "RT_RP_min", "RT", "rt"])
    if rt_col is None:
        raise ValueError(
            f"Reference file {path} has no recognisable RT column "
            f"(looked for RT_RP_median_min / RT_RP_min / RT)."
        )
    df = df.rename(columns={rt_col: "ref_rt"})

    mz_col = _find_col(df, ["mz", "m/z", "mass_to_charge"])
    if mz_col is None:
        raise ValueError(f"Reference file {path} has no 'mz' column.")
    df = df.rename(columns={mz_col: "ref_mz"})

    if "n_RT" not in df.columns:
        df["n_RT"] = 1
    df["n_RT"] = pd.to_numeric(df["n_RT"], errors="coerce").fillna(1)

    df["ref_mz"] = pd.to_numeric(df["ref_mz"], errors="coerce")
    df["ref_rt"] = pd.to_numeric(df["ref_rt"], errors="coerce")
    df = df.dropna(subset=["ref_mz", "ref_rt"]).reset_index(drop=True)

    parsed = df["name"].map(parse_lipid_name)
    df["carbon"] = [p.carbon for p in parsed]
    df["db"] = [p.db for p in parsed]
    df["base"] = [p.base for p in parsed]
    df["is_is"] = [p.is_is for p in parsed]

    if "charge" in df.columns:
        df["polarity"] = df["charge"].map(_polarity_of)
    else:
        df["polarity"] = "unknown"

    if "class" not in df.columns:
        df["class"] = df["name"].str.extract(r"^([A-Za-z0-9-]+)")[0]

    # The bundled sphingolipid reference currently supplies negative Cer and
    # glycosphingolipid entries as formate adducts. Add transparent theoretical
    # [M-H]- aliases for structural negative-mode MS/MS. RT is molecule-specific
    # and can be reused; CCS is ion/adduct-specific and must not be copied.
    sphingolipid_classes = {"Cer", "HexCer", "Hex2Cer", "Hex3Cer"}
    formate_mask = (
        df["class"].astype(str).isin(sphingolipid_classes)
        & df["adduct"].astype(str).eq("[M+HCOO]-")
    )
    existing_deprotonated = set(
        df.loc[df["adduct"].astype(str).eq("[M-H]-"), "name"].astype(str)
    )
    aliases = df[
        formate_mask & ~df["name"].astype(str).isin(existing_deprotonated)
    ].copy()
    if not aliases.empty:
        aliases["adduct"] = "[M-H]-"
        aliases["ref_mz"] = (
            aliases["ref_mz"] - FORMIC_ACID_MONOISOTOPIC_MASS
        )
        aliases["polarity"] = "neg"
        for ccs_column in (
            "ref_ccs", "ref_ccs_min", "ref_ccs_max", "ref_ccs_n",
            "ref_ccs_names", "ref_ccs_mapping",
        ):
            if ccs_column in aliases:
                aliases[ccs_column] = float("nan")
        df = pd.concat([df, aliases], ignore_index=True)

    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None
