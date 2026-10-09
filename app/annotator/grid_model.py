"""Predicted-RT grid ("mesh") for a lipid class.

Within a class+adduct every species sits at a node indexed by total carbon
number ``C`` and total double-bond count ``D``:

* **m/z** is essentially deterministic - each extra CH2 adds ~14.0157, each extra
  double bond removes ~2.0157 - so the node's y-coordinate is computed exactly
  from one anchor species.
* **RT** is a smooth but non-linear function of ``C`` and ``D``.  We fit that
  surface from the reference retention times, then evaluate it on a regular
  ``C x D`` lattice.  Plotted in the RT-vs-m/z plane the lattice becomes the
  characteristic warped purple mesh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CH2 = 14.015650   # mass of a CH2 unit
H2 = 2.015650     # mass lost per double bond


def fit_rt_surface(carbon, db, rt):
    """Least-squares fit of RT ~ f(C, D).

    Uses a quadratic surface when enough points are available, else a plane,
    else a constant.  Returns coefficients usable by :func:`predict_rt`.
    """
    C = np.asarray(carbon, float)
    D = np.asarray(db, float)
    RT = np.asarray(rt, float)
    ok = np.isfinite(C) & np.isfinite(D) & np.isfinite(RT)
    C, D, RT = C[ok], D[ok], RT[ok]
    n = len(RT)
    if n >= 8 and len(np.unique(C)) >= 3:
        A = np.column_stack([np.ones(n), C, D, C * C, C * D, D * D])
        kind = "quad"
    elif n >= 3:
        A = np.column_stack([np.ones(n), C, D])
        kind = "plane"
    elif n >= 1:
        return ("const", float(np.mean(RT)))
    else:
        return None
    coef, *_ = np.linalg.lstsq(A, RT, rcond=None)
    return (kind, coef)


def predict_rt(model, carbon, db):
    """Evaluate the fitted surface (vectorised)."""
    C = np.asarray(carbon, float)
    D = np.asarray(db, float)
    kind = model[0]
    if kind == "const":
        return np.full(np.broadcast(C, D).shape, model[1])
    coef = model[1]
    if kind == "quad":
        return (coef[0] + coef[1] * C + coef[2] * D
                + coef[3] * C * C + coef[4] * C * D + coef[5] * D * D)
    return coef[0] + coef[1] * C + coef[2] * D


def fit_ccs_anchor_correction(anchor_rows: pd.DataFrame) -> dict:
    """Fit a low-order percentage correction for the displayed CCS grid.

    This is deliberately a grid-placement model, not a validated instrumental
    CCS calibration. Confirmed anchors move the reference CCS surface toward
    their measured CCS values while the original CCS evidence remains intact:

    * 0 usable anchors: no correction;
    * 1 anchor: one multiplicative scale factor;
    * 2-3 anchors: percentage offset plus a trend along predicted CCS;
    * 4+ anchors: percentage offset plus mild carbon/DB terms.
    """
    cols = ["carbon", "db", "ccs_pred_db", "feat_ccs"]
    if anchor_rows.empty or any(col not in anchor_rows for col in cols):
        return {"kind": "none", "n": 0, "label": "CCS database grid"}

    d = anchor_rows.dropna(subset=cols).copy()
    d = d[(d["ccs_pred_db"] > 0) & (d["feat_ccs"] > 0)]
    if d.empty:
        return {"kind": "none", "n": 0, "label": "CCS database grid"}

    predicted = d["ccs_pred_db"].to_numpy(float)
    percent_resid = 100.0 * (d["feat_ccs"].to_numpy(float) / predicted - 1.0)
    n = len(d)
    if n == 1:
        scale = float(d["feat_ccs"].iloc[0] / d["ccs_pred_db"].iloc[0])
        return {
            "kind": "scale", "n": n, "scale_factor": scale,
            "label": f"CCS anchor {100 * (scale - 1):+.2f}%",
        }

    ccs0 = float(np.mean(predicted))
    c0 = float(d["carbon"].mean())
    db0 = float(d["db"].mean())
    if n <= 3:
        # Normalising by ccs0 makes the slope numerically stable and gives it
        # the intuitive meaning of a percent trend across the CCS range.
        x = (predicted - ccs0) / ccs0
        A = np.column_stack([np.ones(n), x])
        names = ["offset", "ccs"]
        centre = {"ccs0": ccs0}
    else:
        c_scale = max(float(d["carbon"].std(ddof=0)), 1.0)
        db_scale = max(float(d["db"].std(ddof=0)), 1.0)
        A = np.column_stack([
            np.ones(n),
            (d["carbon"].to_numpy(float) - c0) / c_scale,
            (d["db"].to_numpy(float) - db0) / db_scale,
        ])
        names = ["offset", "carbon", "db"]
        centre = {"c0": c0, "db0": db0,
                  "c_scale": c_scale, "db_scale": db_scale}

    coef, *_ = np.linalg.lstsq(A, percent_resid, rcond=None)
    fitted = A @ coef
    rmse = float(np.sqrt(np.mean((percent_resid - fitted) ** 2)))
    return {
        "kind": "percent_model", "n": n,
        "coef": dict(zip(names, map(float, coef))),
        "centre": centre, "rmse_percent": rmse,
        "label": f"CCS anchored ({n}, RMSE {rmse:.2f}%)",
    }


def apply_ccs_anchor_correction(correction: dict, ccs_pred, carbon, db):
    """Apply :func:`fit_ccs_anchor_correction` without changing raw CCS data."""
    ccs = np.asarray(ccs_pred, float)
    if correction["kind"] == "none":
        return ccs
    if correction["kind"] == "scale":
        return ccs * correction["scale_factor"]

    coef = correction["coef"]
    centre = correction["centre"]
    if "ccs" in coef:
        percent = (coef["offset"]
                   + coef["ccs"] * ((ccs - centre["ccs0"]) / centre["ccs0"]))
    else:
        c = np.asarray(carbon, float)
        d = np.asarray(db, float)
        percent = (
            coef["offset"]
            + coef.get("carbon", 0.0)
            * (c - centre["c0"]) / centre["c_scale"]
            + coef.get("db", 0.0)
            * (d - centre["db0"]) / centre["db_scale"]
        )
    corrected = ccs * (1.0 + percent / 100.0)
    return np.where(corrected > 0, corrected, np.nan)


def mz_at(anchor_mz, anchor_c, anchor_d, carbon, db):
    """Exact theoretical m/z for node (carbon, db) from one anchor species."""
    return anchor_mz + CH2 * (np.asarray(carbon, float) - anchor_c) \
        - H2 * (np.asarray(db, float) - anchor_d)


def build_mesh(ref_ca: pd.DataFrame, rt_col: str = "rt",
               c_pad: int = 2, d_pad: int = 1,
               draw_c: tuple | None = None, draw_d: tuple | None = None):
    """Build the predicted-RT mesh for one class+adduct.

    ``ref_ca`` must have columns ``carbon``, ``db``, ``ref_mz`` and ``rt_col``
    (the RT already aligned to the observed frame).  The surface is fitted on all
    of ``ref_ca``; the lattice is *drawn* over ``draw_c`` / ``draw_d`` when given
    (e.g. the annotated range) otherwise over the reference data range, both
    padded.  Returns a dict with the two line families (x/y arrays with ``None``
    separators) plus the fitted model and node grid, or ``None`` if the class
    can't be gridded.
    """
    d = ref_ca.dropna(subset=["carbon", "db", "ref_mz", rt_col])
    if len(d) < 3:
        return None

    model = fit_rt_surface(d["carbon"], d["db"], d[rt_col])
    if model is None:
        return None

    # Anchor for the exact m/z relationship.
    a = d.iloc[0]
    anchor_mz, anchor_c, anchor_d = a["ref_mz"], a["carbon"], a["db"]

    cr = draw_c if draw_c else (d["carbon"].min(), d["carbon"].max())
    dr = draw_d if draw_d else (d["db"].min(), d["db"].max())
    c_lo, c_hi = int(cr[0]) - c_pad, int(cr[1]) + c_pad
    d_lo, d_hi = max(0, int(dr[0]) - d_pad), int(dr[1]) + d_pad
    carbons = np.arange(c_lo, c_hi + 1)
    dbs = np.arange(d_lo, d_hi + 1)

    # Lines of constant double-bond count (vary carbon).
    cx, cy, cC, cD = [], [], [], []
    for db in dbs:
        rt = predict_rt(model, carbons, db)
        mz = mz_at(anchor_mz, anchor_c, anchor_d, carbons, db)
        cx.extend(list(rt) + [None])
        cy.extend(list(mz) + [None])
        cC.extend(list(carbons) + [None])
        cD.extend([db] * len(carbons) + [None])

    # Lines of constant carbon number (vary double bonds).
    ccx, ccy, ccC, ccD = [], [], [], []
    for c in carbons:
        rt = predict_rt(model, c, dbs)
        mz = mz_at(anchor_mz, anchor_c, anchor_d, c, dbs)
        ccx.extend(list(rt) + [None])
        ccy.extend(list(mz) + [None])
        ccC.extend([c] * len(dbs) + [None])
        ccD.extend(list(dbs) + [None])

    return {
        "model": model,
        "anchor": (anchor_mz, anchor_c, anchor_d),
        "carbons": carbons,
        "dbs": dbs,
        "db_lines": (cx, cy),      # constant-D family
        "c_lines": (ccx, ccy),     # constant-C family
        "db_line_coords": (cC, cD),
        "c_line_coords": (ccC, ccD),
    }
