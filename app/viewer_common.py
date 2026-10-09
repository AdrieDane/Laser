"""UI primitives shared by the 4D and standard-lipidomics grid browsers.

Both viewers draw the same Plotly panels and rely on the same zero-height click
bridge, so the styling helpers and the component declaration live here instead
of inside either page module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from annotator.rp_isomers import rp_grid_class


def stable_color_map(
    values: Iterable[object],
    palette: Sequence[str],
) -> dict[str, str]:
    """Assign palette colours to unique string values in first-seen order."""
    if not palette:
        raise ValueError("a viewer colour palette cannot be empty")
    ordered = dict.fromkeys(str(value) for value in values)
    return {
        value: palette[index % len(palette)]
        for index, value in enumerate(ordered)
    }


_PLOTLY_CLICK_BRIDGE_JS = """
    export default function(component) {
      const { data, parentElement, setTriggerValue } = component;
      const host = parentElement instanceof ShadowRoot
        ? parentElement.host
        : parentElement;
      let plot = null;
      let observer = null;
      let retryTimer = null;
      let wheelCanvas = null;
      let wheelScene = null;
      let wheelFrame = null;
      let pendingWheelDelta = 0;
      let latestWheelPoint = null;

      if (data?.selectionFeedback) {
        document.body.classList.remove("laser-selection-pending");
      }

      const clone = (value) => value == null
        ? null
        : JSON.parse(JSON.stringify(value));

      const sceneCamera = () => {
        const scene = plot?._fullLayout?.scene?._scene;
        if (scene && typeof scene.getCamera === "function") {
          try {
            return clone(scene.getCamera());
          } catch (_) {
            // Fall back to Plotly's last published layout camera.
          }
        }
        return clone(plot?._fullLayout?.scene?.camera);
      };

      const captureView = () => {
        const fullLayout = plot?._fullLayout;
        return {
          xaxis: clone(fullLayout?.xaxis?.range),
          yaxis: clone(fullLayout?.yaxis?.range),
          sceneCamera: sceneCamera(),
        };
      };

      const multiplyMat4 = (a, b) => {
        const out = new Array(16).fill(0);
        for (let column = 0; column < 4; column += 1) {
          for (let row = 0; row < 4; row += 1) {
            for (let k = 0; k < 4; k += 1) {
              out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
            }
          }
        }
        return out;
      };

      // Column-major 4x4 inverse, matching the matrices used by WebGL/Plotly.
      const invertMat4 = (a) => {
        const out = new Array(16);
        const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
        const b00 = a00 * a11 - a01 * a10;
        const b01 = a00 * a12 - a02 * a10;
        const b02 = a00 * a13 - a03 * a10;
        const b03 = a01 * a12 - a02 * a11;
        const b04 = a01 * a13 - a03 * a11;
        const b05 = a02 * a13 - a03 * a12;
        const b06 = a20 * a31 - a21 * a30;
        const b07 = a20 * a32 - a22 * a30;
        const b08 = a20 * a33 - a23 * a30;
        const b09 = a21 * a32 - a22 * a31;
        const b10 = a21 * a33 - a23 * a31;
        const b11 = a22 * a33 - a23 * a32;
        let det = b00 * b11 - b01 * b10 + b02 * b09
          + b03 * b08 - b04 * b07 + b05 * b06;
        if (!det || !Number.isFinite(det)) return null;
        det = 1.0 / det;
        out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
        out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
        out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
        out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
        out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
        out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
        out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
        out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
        out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
        out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
        out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
        out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
        out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
        out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
        out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
        out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
        return out;
      };

      const transformPoint = (matrix, point) => {
        const result = [0, 0, 0, 0];
        for (let row = 0; row < 4; row += 1) {
          result[row] = matrix[row] * point[0]
            + matrix[4 + row] * point[1]
            + matrix[8 + row] * point[2]
            + matrix[12 + row] * point[3];
        }
        if (!result.every(Number.isFinite) || Math.abs(result[3]) < 1e-12) return null;
        return result.slice(0, 3).map((value) => value / result[3]);
      };

      const norm = (vector) => Math.hypot(...vector);
      const subtract = (a, b) => a.map((value, index) => value - b[index]);
      const dot = (a, b) => a.reduce(
        (sum, value, index) => sum + value * b[index], 0
      );

      const pointerTarget = (scene, pointer, camera) => {
        const glplot = scene?.glplot;
        const canvas = glplot?.canvas;
        const model = glplot?.cameraParams?.model;

        // Prefer Plotly's picked data point when the pointer is over one. This
        // makes a dense lipid cluster a precise and natural zoom target.
        const picked = glplot?.selection?.dataCoordinate;
        if (picked?.length >= 3 && model?.length === 16) {
          const target = transformPoint(model, [picked[0], picked[1], picked[2], 1]);
          if (target) return target;
        }

        // Otherwise intersect the cursor ray with the plane through the current
        // focal point. This gives cursor-centred zoom even over empty plot space.
        const view = glplot?.cameraParams?.view;
        const projection = glplot?.cameraParams?.projection;
        if (!canvas || !pointer || view?.length !== 16 || projection?.length !== 16) {
          return camera.center;
        }
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return camera.center;
        const x = 2 * (pointer.clientX - rect.left) / rect.width - 1;
        const y = 1 - 2 * (pointer.clientY - rect.top) / rect.height;
        const inverse = invertMat4(multiplyMat4(projection, view));
        if (!inverse) return camera.center;
        const near = transformPoint(inverse, [x, y, -1, 1]);
        const far = transformPoint(inverse, [x, y, 1, 1]);
        if (!near || !far) return camera.center;
        const ray = subtract(far, near);
        const normalRaw = subtract(camera.center, camera.eye);
        const normalLength = norm(normalRaw);
        if (!normalLength) return camera.center;
        const normal = normalRaw.map((value) => value / normalLength);
        const denominator = dot(ray, normal);
        if (Math.abs(denominator) < 1e-9) return camera.center;
        const distance = dot(subtract(camera.center, near), normal) / denominator;
        const target = near.map((value, index) => value + distance * ray[index]);

        // Keep an extreme pointer position from moving the camera outside the
        // useful scene in a single wheel step.
        const offset = subtract(target, camera.center);
        const offsetLength = norm(offset);
        const maximumOffset = Math.max(0.5, norm(subtract(camera.eye, camera.center)));
        if (offsetLength > maximumOffset) {
          return camera.center.map(
            (value, index) => value + offset[index] * maximumOffset / offsetLength
          );
        }
        return target;
      };

      const apply3DWheel = () => {
        wheelFrame = null;
        const scene = wheelScene;
        const delta = Math.max(-240, Math.min(240, pendingWheelDelta));
        pendingWheelDelta = 0;
        if (!scene || !delta || typeof scene.getCamera !== "function") return;
        const camera = scene.getCamera();
        const eye = [camera.eye.x, camera.eye.y, camera.eye.z];
        const center = [camera.center.x, camera.center.y, camera.center.z];
        const up = [camera.up.x, camera.up.y, camera.up.z];
        const currentDistance = norm(subtract(eye, center));
        if (!Number.isFinite(currentDistance) || currentDistance <= 0) return;
        const requestedFactor = Math.exp(delta * 0.00125);
        const nextDistance = Math.max(0.16, Math.min(12, currentDistance * requestedFactor));
        const factor = nextDistance / currentDistance;
        const target = pointerTarget(scene, latestWheelPoint, { eye, center });
        const nextEye = target.map(
          (value, index) => value + factor * (eye[index] - value)
        );
        const nextCenter = target.map(
          (value, index) => value + factor * (center[index] - value)
        );
        scene.camera.lookAt(nextEye, nextCenter, up);
        scene.glplot.redraw();
        if (typeof scene.saveLayout === "function" && plot?.layout) {
          scene.saveLayout(plot.layout);
        }
        if (pendingWheelDelta && wheelFrame == null) {
          wheelFrame = window.requestAnimationFrame(apply3DWheel);
        }
      };

      const wheelHandler = (event) => {
        if (!data?.enable3dFocusZoom || !wheelScene) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        let delta = event.deltaY;
        if (event.deltaMode === 1) delta *= 16;
        if (event.deltaMode === 2) delta *= Math.max(wheelCanvas?.clientHeight ?? 1, 1);
        pendingWheelDelta += Math.max(-240, Math.min(240, delta));
        latestWheelPoint = { clientX: event.clientX, clientY: event.clientY };
        if (wheelFrame == null) {
          wheelFrame = window.requestAnimationFrame(apply3DWheel);
        }
      };

      const detach3DWheel = () => {
        if (wheelCanvas) {
          wheelCanvas.removeEventListener("wheel", wheelHandler, true);
        }
        if (wheelScene?.camera) wheelScene.camera.enableWheel = true;
        if (wheelFrame != null) window.cancelAnimationFrame(wheelFrame);
        wheelCanvas = null;
        wheelScene = null;
        wheelFrame = null;
        pendingWheelDelta = 0;
        latestWheelPoint = null;
      };

      const attach3DWheel = () => {
        if (!data?.enable3dFocusZoom) {
          detach3DWheel();
          return;
        }
        const scene = plot?._fullLayout?.scene?._scene;
        const canvas = scene?.glplot?.canvas;
        if (!scene || !canvas) return;
        if (wheelCanvas !== canvas) {
          detach3DWheel();
          wheelCanvas = canvas;
          wheelScene = scene;
          wheelCanvas.addEventListener(
            "wheel", wheelHandler, { capture: true, passive: false }
          );
        }
        // Plotly may restore this flag during a React update; keep its native
        // centre-anchored wheel handler disabled while our bridge is active.
        scene.camera.enableWheel = false;
      };

      const clickHandler = (event) => {
        const point = event?.points?.[0];
        if (!point) return;
        const payload = point.customdata ?? null;
        if (data?.selectionFeedback) {
          // Plotly's event+select mode has already highlighted the 2D point;
          // keep that immediate feedback visible during the authoritative rerun.
          document.body.classList.add("laser-selection-pending");
          window.setTimeout(
            () => document.body.classList.remove("laser-selection-pending"),
            30000,
          );
        }
        setTriggerValue("point", {
          customdata: payload,
          curveNumber: point.curveNumber ?? null,
          pointNumber: point.pointNumber ?? point.pointIndex ?? null,
          view: data?.captureView === false ? null : captureView(),
          timestamp: Date.now(),
        });
      };

      const selectionHandler = (event) => {
        const points = event?.points ?? [];
        setTriggerValue("selection", {
          customdata: points.map((point) => point.customdata ?? null),
          view: data?.captureView === false ? null : captureView(),
          timestamp: Date.now(),
        });
      };

      const findPrecedingPlot = () => {
        const plots = Array.from(document.querySelectorAll(
          '[data-testid="stPlotlyChart"] .js-plotly-plot'
        ));
        return plots.filter((candidate) =>
          candidate.compareDocumentPosition(host) & Node.DOCUMENT_POSITION_FOLLOWING
        ).at(-1) ?? null;
      };

      const attach = () => {
        const candidate = findPrecedingPlot();
        if (!candidate || typeof candidate.on !== "function") return false;
        if (plot === candidate) {
          attach3DWheel();
          return true;
        }
        detach3DWheel();
        if (plot && typeof plot.removeListener === "function") {
          plot.removeListener("plotly_click", clickHandler);
          plot.removeListener("plotly_selected", selectionHandler);
          plot.removeListener("plotly_deselect", selectionHandler);
        }
        plot = candidate;
        if (data?.captureClicks !== false) plot.on("plotly_click", clickHandler);
        if (data?.captureSelections) {
          plot.on("plotly_selected", selectionHandler);
          plot.on("plotly_deselect", selectionHandler);
        }
        attach3DWheel();
        return true;
      };

      // Keep observing after the initial attachment. Streamlit may replace the
      // Plotly graph div during a rerun; without this, only the first point
      // click reaches Python because the listener remains on the detached div.
      observer = new MutationObserver(() => attach());
      observer.observe(document.body, { childList: true, subtree: true });
      if (!attach()) {
        retryTimer = window.setInterval(() => {
          if (attach() && retryTimer) {
            window.clearInterval(retryTimer);
            retryTimer = null;
          }
        }, 100);
      }

      return () => {
        if (observer) observer.disconnect();
        if (retryTimer) window.clearInterval(retryTimer);
        detach3DWheel();
        if (plot && typeof plot.removeListener === "function") {
          plot.removeListener("plotly_click", clickHandler);
          plot.removeListener("plotly_selected", selectionHandler);
          plot.removeListener("plotly_deselect", selectionHandler);
        }
      };
    }
        """


def plotly_click_bridge():
    """Declare the shared ``plotly_click`` component and return the handle.

    Streamlit registers components per script run, so the declaration must be
    repeated on every rerun rather than memoised in a module global: a cached
    handle from an earlier run points at a component the current run never
    registered.
    """
    return st.components.v2.component(
        "plotly_click_bridge",
        css=":host { display: block; height: 0; overflow: hidden; }",
        js=_PLOTLY_CLICK_BRIDGE_JS,
    )



GROUP_COLORS = {
    "WT": "#1f77b4",
    "KO_PEX": "#e4572e",
    "WT_ATAD1": "#4e79a7",
    "KO_PEX_ATAD1": "#f28e2b",
    "QC_pool": "#8e44ad",
    "BlancoIS": "#d6a821",
    "Blanco": "#7f7f7f",
}

# Multi-word family names are quoted, and Segoe UI sits in the stack so Windows
# resolves on a real font rather than walking to the generic fallback. Plotly
# measures hover text with one font and paints it with another when the family
# does not resolve, which detaches the label text from its background box.
MATLAB_FONT = dict(
    family=(
        '-apple-system, BlinkMacSystemFont, "SF Pro Text", '
        '"Helvetica Neue", "Segoe UI", Arial, sans-serif'
    ),
    size=11,
    color="#1d1d1f",
)

# Hover labels get a plain, universally present family. Plotly sizes the label
# background from its own text measurement, so anything that measures
# differently from how it paints leaves the text outside the box.
HOVER_FONT = dict(family="Arial, Helvetica, sans-serif", size=11)
HOVER_LABEL = dict(
    font=HOVER_FONT,
    bgcolor="rgba(255,255,255,0.96)",
    bordercolor="#c7c7cc",
    align="left",
    # Plotly truncates the trace name at 15 characters by default and sizes the
    # box for the truncated string.
    namelength=-1,
)


def matlab_axis_config(grid: bool = True, **overrides) -> dict:
    """Return the shared axis style without issuing a Plotly update per axis."""
    config = dict(
        showline=True, linecolor="#aeaeb2", linewidth=1, mirror=True,
        ticks="outside", tickfont=dict(size=10, color="#3a3a3c"),
        title_font=dict(size=11, color="#1d1d1f"),
        gridcolor="#e5e5ea" if grid else "rgba(0,0,0,0)",
        zerolinecolor="#c7c7cc",
    )
    config.update(overrides)
    return config


def apply_matlab_plot_style(fig: go.Figure, grid: bool = True):
    """Give Plotly figures a clean macOS-style content panel and compact font."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=MATLAB_FONT,
        title_font=dict(
            family=MATLAB_FONT["family"], size=12, color="#1d1d1f"
        ),
        legend_font=dict(
            family=MATLAB_FONT["family"], size=10, color="#1d1d1f"
        ),
        hoverlabel=HOVER_LABEL,
    )
    fig.update_xaxes(**matlab_axis_config(grid))
    fig.update_yaxes(**matlab_axis_config(grid))
    return fig


def consensus_shift(resid, halfwidth):
    """Consensus RT deviation = the shift where the MOST species agree.

    Finds the densest window of width 2·halfwidth over the residuals (a robust
    1-D mode / mini-RANSAC) and returns its median plus the number of residuals
    inside it. Robust to a scattered background of wrong assignments and to a
    bimodal distribution (it locks onto the denser cluster) - unlike a mean or
    median, which can land in an empty gap between clusters.
    """
    r = np.sort(np.asarray(resid, dtype=float))
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0, 0
    if len(r) < 3:
        return float(np.median(r)), len(r)
    best_count, best_lo = 0, r[0]
    for i in range(len(r)):
        hi = r[i] + 2 * halfwidth
        count = int(np.searchsorted(r, hi, side="right") - i)
        if count > best_count:
            best_count, best_lo = count, r[i]
    in_band = r[(r >= best_lo) & (r <= best_lo + 2 * halfwidth)]
    return float(np.median(in_band)), best_count


def fit_anchor_correction(anchor_rows: pd.DataFrame) -> dict:
    """Fit a low-order RT correction from user-confirmed anchors.

    The database surface supplies the grid shape. Confirmed anchors only correct
    where that surface sits in the current batch:
      * 0 anchors: no correction;
      * 1 anchor: a pure RT shift;
      * 2-3 anchors: shift plus a gentle slope along predicted RT;
      * 4+ anchors: add mild carbon/db terms.
    """
    cols = ["carbon", "db", "rt_pred_db", "feat_rt"]
    if anchor_rows.empty or any(c not in anchor_rows for c in cols):
        return {"kind": "none", "n": 0, "label": "database grid"}

    d = anchor_rows.dropna(subset=cols).copy()
    if d.empty:
        return {"kind": "none", "n": 0, "label": "database grid"}

    resid = (d["feat_rt"] - d["rt_pred_db"]).to_numpy(float)
    n = len(d)
    if n == 1:
        shift = float(resid[0])
        return {
            "kind": "shift", "n": n, "shift": shift,
            "label": f"anchored shift {shift:+.3f} min",
        }

    rt0 = float(d["rt_pred_db"].mean())
    c0 = float(d["carbon"].mean())
    db0 = float(d["db"].mean())

    if n <= 3:
        x = (d["rt_pred_db"].to_numpy(float) - rt0)
        A = np.column_stack([np.ones(n), x])
        names = ["offset", "rt"]
        scale = {"rt0": rt0}
    else:
        # Mild, low-order terms only: keep the database curvature intact.
        c = d["carbon"].to_numpy(float) - c0
        db = d["db"].to_numpy(float) - db0
        A = np.column_stack([np.ones(n), c, db])
        names = ["offset", "carbon", "db"]
        scale = {"c0": c0, "db0": db0}

    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    pred = A @ coef
    rmse = float(np.sqrt(np.mean((resid - pred) ** 2))) if n else 0.0
    return {
        "kind": "linear", "n": n, "coef": dict(zip(names, map(float, coef))),
        "scale": scale, "rmse": rmse,
        "label": f"anchored grid ({n} anchors, RMSE {rmse:.3f} min)",
    }


def apply_anchor_correction(correction: dict, rt_pred, carbon, db):
    """Apply a correction returned by fit_anchor_correction."""
    rt = np.asarray(rt_pred, float)
    if correction["kind"] == "none":
        return rt
    if correction["kind"] == "shift":
        return rt + correction["shift"]

    coef = correction["coef"]
    if "rt" in coef:
        delta = coef["offset"] + coef["rt"] * (rt - correction["scale"]["rt0"])
    else:
        c = np.asarray(carbon, float)
        d = np.asarray(db, float)
        delta = (coef["offset"]
                 + coef.get("carbon", 0.0) * (c - correction["scale"]["c0"])
                 + coef.get("db", 0.0) * (d - correction["scale"]["db0"]))
    return rt + delta


def apply_anchor_correction_to_line(correction: dict, rt_vals, carbon_vals, db_vals):
    """Apply an anchor correction to a mesh line, preserving None separators."""
    out: list[float | None] = []
    for rt, c, d in zip(rt_vals, carbon_vals, db_vals):
        if rt is None or c is None or d is None:
            out.append(None)
        else:
            out.append(float(apply_anchor_correction(correction, [rt], [c], [d])[0]))
    return out


def ann_key(row) -> str:
    return f"{row['feat_feature_id']}|{row['name']}|{row['adduct']}"


def grid_class_series(cls, base, names=None):
    """Return the resolved class identity shown by RP grid viewers.

    Besides ether/plasmalogen and sphingoid-base subclasses, positional LPC/LPE
    identities must remain separate after the shared RP isomer resolver has
    selected them.  The optional ``names`` argument keeps older callers
    compatible while allowing the viewer to preserve ``1-acyl``/``2-acyl``.
    """
    classes = cls.copy() if isinstance(cls, pd.Series) else pd.Series(cls)
    bases = (
        base.reindex(classes.index)
        if isinstance(base, pd.Series)
        else pd.Series("", index=classes.index)
        if base is None
        else pd.Series(base, index=classes.index)
    )
    species = (
        names.reindex(classes.index)
        if isinstance(names, pd.Series)
        else pd.Series("", index=classes.index)
        if names is None
        else pd.Series(names, index=classes.index)
    )
    return rp_grid_class(classes, bases, species).tolist()


def row_sample_values(row: pd.Series, sample_cols: list[str] | tuple[str, ...]) -> np.ndarray:
    vals = []
    for c in sample_cols:
        key = f"feat_{c}"
        vals.append(row[key] if key in row else np.nan)
    return pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)


def species_label(df: pd.DataFrame) -> pd.Series:
    return (df["carbon"].astype(int).astype(str) + ":"
            + df["db"].astype(int).astype(str))
