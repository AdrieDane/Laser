"""LASER theme adapted from LASER-4D a188f72 (2026-11-04)."""

from __future__ import annotations

from collections.abc import Callable
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

THEMES = ("Minimal", "Midnight")
THEME_STATE_KEY = "laser_appearance_theme"
THEME_WIDGET_KEY = "laser_appearance_choice"


def settings_panel(title: str | None, *, key: str, grouped: bool = False) -> DeltaGenerator:
    """Place a heading above standalone fields or one related settings surface."""
    if title:
        st.subheader(title)
    prefix = "settings-surface" if grouped else "settings-group"
    return st.container(border=True, key=f"{prefix}-{key}")


def page_header(title: str, *, key: str, back_label: str = "Back to start",
                help: str | None = None, actions: Callable | None = None) -> bool:
    """Keep the page title and a content-sized home action aligned and responsive."""
    with st.container(horizontal=True, vertical_alignment="center", key=f"page-header-{key}"):
        st.title(title, help=help)
        if actions is not None:
            actions()
        return st.button(back_label, key=key, width="content")


_HORIZONTAL_SETTINGS_ROW = r'''[data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
        > [data-testid="stElementContainer"] > :is(
          [data-testid="stTextInput"], [data-testid="stNumberInput"],
          [data-testid="stTextArea"], [data-testid="stSelectbox"],
          [data-testid="stMultiSelect"], [data-testid="stRadio"],
          [data-testid="stCheckbox"], [data-testid="stToggle"],
          [data-testid="stSlider"], [data-testid="stSelectSlider"],
          [data-testid="stDateInput"], [data-testid="stTimeInput"],
          [data-testid="stColorPicker"], [data-testid="stFileUploader"],
          [data-testid="stSegmentedControl"], [data-testid="stPills"],
          [data-testid="stButtonGroup"]
        )
      )'''

_TITLED_SETTING = r'''
[data-testid="stElementContainer"]:not(:where(
        [data-testid="stSidebar"],
        .st-key-review-controls-panel, .st-key-review-controls-panel-hidden,
        .st-key-standard-review-controls-panel, .st-key-standard-review-controls-panel-hidden,
        .st-key-standard-grid-stream-row,
        [class*="st-key-settings-surface-"],
        __HORIZONTAL_SETTINGS_ROW__
      ) *) > :not(:is(
        [data-testid="stCheckbox"], [data-testid="stToggle"],
        [data-testid="stColorPicker"]
      )):has(> [data-testid="stWidgetLabel"])
'''.strip()

_ACTION_BUTTON = r''':is(
        .stButton button, .stDownloadButton button,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stPopoverButton"], [data-testid="stFileUploader"] button,
        [data-testid="stNumberInputContainer"] button
      )'''

# The viewer never uses the full-page veil. Keep these global relational
# selectors out of its stylesheet entirely: even hidden overlays cause Chrome
# to invalidate styles across Plotly's SVG tree on every Streamlit update.
_PAGE_LOADING_RULES = {
    '__PAGE_LOADING_VISIBILITY__': r"""
      body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"])::before,
      body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"])::after {
        visibility: hidden;
        animation: laser-loading-reveal 0s linear var(--laser-loading-delay) forwards;
      }
""",
    '__PAGE_LOADING_VEIL__': r"""
      body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"])::before {
        content: "";
        position: fixed;
        z-index: 2147483645;
        inset: 0;
        background: var(--app-loading-veil);
        pointer-events: all;
        cursor: progress;
      }
""",
    '__PAGE_LOADING_MESSAGE__': r"""
      body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"])::after {
        content: "LASER is loading…";
        position: fixed;
        z-index: 2147483646;
        left: 50%;
        top: 50%;
        width: min(22rem, calc(100vw - 3rem));
        box-sizing: border-box;
        transform: translate(-50%, -50%);
        padding: 0.95rem 1.1rem 1.15rem;
        overflow: hidden;
        border: 1px solid var(--app-border-strong);
        border-radius: 13px;
        background: var(--app-surface-material);
        color: var(--app-text);
        font-family: var(--app-font);
        font-size: 0.84rem;
        font-weight: 650;
        text-align: center;
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.16);
        pointer-events: all;
        cursor: progress;
        animation: laser-loading-reveal 0s linear var(--laser-loading-delay) forwards;
      }
""",
    '__PAGE_LOADING_PROGRESS__': r"""
      [data-testid="stElementContainer"]:has(.laser-loading-progress) {
        display: contents;
      }
      .laser-loading-progress {
        position: fixed;
        z-index: 2147483647;
        left: 50%;
        top: calc(50% + 22px);
        width: min(22rem, calc(100vw - 3rem));
        height: 4px;
        transform: translateX(-50%);
        overflow: hidden;
        contain: strict;
        visibility: hidden;
        pointer-events: none;
        border-radius: 0 0 13px 13px;
      }
      .laser-loading-progress::after {
        content: "";
        display: block;
        width: 42%;
        height: 100%;
        background: linear-gradient(90deg, transparent, var(--app-accent), transparent);
        transform: translate3d(-100%, 0, 0);
      }
      body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"]) .laser-loading-progress {
        animation: laser-loading-reveal 0s linear var(--laser-loading-delay) forwards;
      }
      body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"]) .laser-loading-progress::after {
        will-change: transform;
        animation: laser-loading-bar 1.05s linear infinite;
      }
      body.laser-selection-pending .laser-loading-progress,
      body:has(:is(.annotation-viewer-active, .sequence-live-monitor-active,
                   #targeted-inspector, #raw-data-inspector, #targeted-peak-integration)) .laser-loading-progress {
        display: none !important;
      }
""",
    '__PAGE_LOADING_VIEWER_SUPPRESSION__': r"""
      body:has(.annotation-viewer-active):has([data-stale="true"])::before,
      body:has(.annotation-viewer-active):has([data-stale="true"])::after {
        display: none !important;
      }
""",
    '__PAGE_LOADING_MONITOR_SUPPRESSION__': r"""
      body:has(.sequence-live-monitor-active):has([data-stale="true"])::before,
      body:has(.sequence-live-monitor-active):has([data-stale="true"])::after {
        display: none !important;
      }
""",
    '__PAGE_LOADING_REDUCED_MOTION__': r"""
        body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"]) .laser-loading-progress,
        body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"])::before,
        body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"])::after {
          animation: laser-loading-reveal 0s linear var(--laser-loading-delay) forwards !important;
        }
        body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"]) .laser-loading-progress::after {
          animation: none !important;
          transform: translate3d(70%, 0, 0);
          will-change: auto;
        }
""",
    '__PAGE_LOADING_MIDNIGHT_VEIL__': r"""
  body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"])::before {
    background: var(--app-loading-veil);
  }
""",
    '__PAGE_LOADING_MIDNIGHT_MESSAGE__': r"""
  body:has(.stApp[data-test-script-state="running"][data-test-connection-state="CONNECTED"]):has([data-stale="true"])::after {
    background: var(--app-surface-material);
  }
""",
}


def _loading_styles(template: str, *, enabled: bool) -> str:
    for token, rules in _PAGE_LOADING_RULES.items():
        template = template.replace(token, rules if enabled else "")
    return template


_APP_THEME_TEMPLATE = r"""
    <style>
      :root {
        --app-canvas: #ffffff;
        --app-surface: #ffffff;
        --app-surface-material: rgba(255, 255, 255, 0.97);
        --app-surface-soft: #f6f6f7;
        --app-group: #f6f6f7;
        --app-sidebar: #eeeef0;
        --app-selection-ink: #ffffff;
        --app-section-header: transparent;
        --app-control: #ffffff;
        --app-border: rgba(60, 60, 67, 0.12);
        --app-border-strong: rgba(60, 60, 67, 0.22);
        --app-divider: rgba(60, 60, 67, 0.08);
        --app-text: #1d1d1f;
        --app-muted: #6e6e73;
        --app-tertiary: #8e8e93;
        --app-accent: #0068d9;
        --app-accent-pressed: #1a72d8;
        --app-focus: rgba(0, 104, 217, 0.22);
        --app-shadow: none;
        --app-font: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                    "Segoe UI Variable Text", "Segoe UI",
                    system-ui, "Helvetica Neue", Arial, sans-serif;
        --app-font-mono: ui-monospace, "SFMono-Regular", Consolas,
                         "Liberation Mono", monospace;
        --app-type-body: 0.875rem;
        --app-type-control: 0.875rem;
        --app-type-data: 0.8125rem;
        --app-type-small: 0.75rem;
        --laser-loading-delay: 1.5s;
        --app-loading-veil: #ffffff;
      }
      html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: var(--app-canvas) !important;
        color: var(--app-text) !important;
        font-family: var(--app-font) !important;
        line-height: 1.45;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
      }
      body, .stApp { font-size: var(--app-type-control); }
      .block-container {
        max-width: 1440px;
        margin-inline: auto;
        padding-top: 0.9rem !important;
        padding-left: clamp(0.75rem, 1.8vw, 1.5rem) !important;
        padding-right: clamp(0.75rem, 1.8vw, 1.5rem) !important;
        padding-bottom: 1.5rem !important;
      }
      [data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
      }
      [data-testid="stMain"] {
        padding-top: 0 !important;
        background: var(--app-canvas) !important;
      }
      body:has(.annotation-viewer-active) [data-testid="stMain"],
      body:has(.annotation-viewer-active) [data-testid="stMainBlockContainer"] {
        background: var(--app-canvas) !important;
      }
      [data-testid="stMainBlockContainer"] {
        padding-top: 0.9rem !important;
      }
      /* Stylesheet-only Markdown blocks should not add empty form rows. */
      [data-testid="stElementContainer"]:has(> .stMarkdown [data-testid="stMarkdownContainer"] > style:only-child) {
        display: none;
      }
      /* Setup gets breathing room; scientific reviewers keep their full width. */
      body:has(.annotation-viewer-active) [data-testid="stMainBlockContainer"] {
        max-width: 100%;
        padding: 0.45rem 0.6rem 0.75rem !important;
      }
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      #MainMenu {
        display: none !important;
      }
      /* Keep Streamlit's native running indicator as a cross-platform
         fallback.  Older Windows browser builds may not support the :has()
         selector used by LASER's larger loading overlay. */
      [data-testid="stStatusWidget"] {
        position: fixed !important;
        z-index: 2147483644 !important;
        top: 0.45rem !important;
        right: 0.65rem !important;
      }
      /* Stale DOM can survive an interrupted page transition. Only a live,
         connected script run may show the blocking overlay; stale elements
         alone are not evidence that Python is still working. */
      [data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
      }
      /* Never leave old start-page navigation visible during route changes. */
      .st-key-home-module-buttons[data-stale="true"],
      [data-stale="true"] .st-key-home-module-buttons {
        display: none !important;
        pointer-events: none !important;
      }
      /* Quick reruns should neither flash a popup nor intercept clicks.
         Hidden elements cannot receive pointer events. Removing the busy
         selector/spinner cancels the reveal, so each update gets a fresh wait. */
__PAGE_LOADING_VISIBILITY__
      [class*="st-key-review-loading-"] [data-testid="stSpinner"] {
        visibility: hidden;
        animation: laser-loading-reveal 0s linear var(--laser-loading-delay) forwards;
      }
      @keyframes laser-loading-reveal {
        to { visibility: visible; }
      }
__PAGE_LOADING_VEIL__
__PAGE_LOADING_MESSAGE__
__PAGE_LOADING_PROGRESS__
      /* Main-plot clicks already show an immediate marker highlight. Keep the
         completed reviewer visible while its linked evidence is refreshed. */
      body.laser-selection-pending::before,
      body.laser-selection-pending::after {
        display: none !important;
      }
      /* Inspection pages open as lightweight forms. Their searches already
         have stage-specific spinners and the native running indicator.
         Never let leftover home-page elements cover the search controls. */
      body:has(#targeted-inspector)::before,
      body:has(#targeted-inspector)::after,
      body:has(#raw-data-inspector)::before,
      body:has(#raw-data-inspector)::after,
      body:has(#targeted-peak-integration)::before,
      body:has(#targeted-peak-integration)::after {
        display: none !important;
      }
      /* Class/adduct changes redraw the reviewer but leave the existing view
         useful. Avoid covering it with a modal while the new grid arrives. */
__PAGE_LOADING_VIEWER_SUPPRESSION__

      /* Explicit project opening survives Python reruns, then is removed once
         the browser has painted the initial viewer. No busy/stale selectors. */
      .laser-project-opening {
        position: fixed; inset: 0; z-index: 2147483647;
        display: grid; place-items: center; cursor: progress;
        background: var(--app-loading-veil); visibility: hidden;
        animation: laser-loading-reveal 0s linear var(--laser-loading-delay) forwards;
        font-family: var(--app-font);
      }
      .laser-project-opening-card {
        position: relative; overflow: hidden; box-sizing: border-box;
        width: min(22rem, calc(100vw - 3rem)); padding: 0.95rem 1.1rem 1.15rem;
        border: 1px solid var(--app-border-strong); border-radius: 13px;
        background: var(--app-surface-material); color: var(--app-text);
        font-size: 0.84rem; font-weight: 650; text-align: center;
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.16);
      }
      .laser-project-opening-progress {
        position: absolute; bottom: 0; left: 0; right: 0; height: 4px;
        overflow: hidden; contain: strict;
      }
      .laser-project-opening-progress::after {
        content: ""; display: block; width: 42%; height: 100%;
        background: linear-gradient(90deg, transparent, var(--app-accent), transparent);
        will-change: transform; animation: laser-loading-bar 1.05s linear infinite;
      }
      body.laser-project-opening-active::before,
      body.laser-project-opening-active::after,
      body.laser-project-opening-active .laser-loading-progress { display: none !important; }
      @media (prefers-reduced-motion: reduce) {
        .laser-project-opening-progress::after {
          animation: none; transform: translate3d(70%, 0, 0); will-change: auto;
        }
      }

      @keyframes laser-loading-bar {
        from {
          transform: translate3d(-100%, 0, 0);
        }
        to {
          transform: translate3d(240%, 0, 0);
        }
      }
      /* Live scientific criteria rerun matching while the previous grid stays
         visible. Use an actual Streamlit spinner, not a stale-DOM overlay:
         its context removes it on success, failure and interrupted reruns. */
      [class*="st-key-review-loading-"] [data-testid="stSpinner"] {
        position: fixed;
        top: 0.75rem;
        left: 50%;
        transform: translateX(-50%);
        z-index: 2147483646;
        width: max-content;
        max-width: calc(100vw - 2rem);
        padding: 0.75rem 1rem;
        border: 1px solid var(--app-border-strong);
        border-radius: 10px;
        background: var(--app-surface);
        color: var(--app-text);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.14);
        pointer-events: none;
      }
      /* The sequence monitor refreshes one small fragment every second. Keep
         its existing controls visible without showing the full-app loading
         modal for each lightweight log/progress refresh. */
__PAGE_LOADING_MONITOR_SUPPRESSION__
      body:has(.sequence-live-monitor-active) [data-testid="stStatusWidget"] {
        display: none !important;
      }
      .criteria-section-heading {
        margin: 0.85rem 0 0.6rem;
        padding: 0;
        border: 0;
        border-radius: 0;
        background: var(--app-section-header);
        color: var(--app-text);
        font-size: var(--app-type-control);
        font-weight: 600;
        letter-spacing: -0.005em;
      }
      [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background: var(--app-sidebar) !important;
        color: var(--app-text) !important;
        border-right: 1px solid var(--app-border) !important;
      }
      [data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
      }
      [data-testid="stSidebar"] * {
        color: var(--app-text) !important;
        font-family: var(--app-font) !important;
      }
      div, p, label, span, input, textarea, button {
        font-family: var(--app-font) !important;
      }
      [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
        font-weight: 400 !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
      }
      h1, h2, h3, h4 {
        margin-top: 0.25rem;
        color: var(--app-text) !important;
        font-family: var(--app-font) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        line-height: 1.3 !important;
      }
      h1 { font-size: 1.375rem !important; padding-bottom: 0.65rem !important; }
      h2 { font-size: 1rem !important; padding-bottom: 0.4rem !important; }
      h3 { font-size: 0.9375rem !important; padding-bottom: 0.35rem !important; }
      h4 { font-size: var(--app-type-control) !important; }
      strong, b { font-weight: 600; }
      code, pre, code *, pre *, [data-testid="stCode"] * {
        font-family: var(--app-font-mono) !important;
        font-size: var(--app-type-data) !important;
        font-variant-numeric: tabular-nums;
      }
      input[type="number"], [data-testid="stMetricValue"],
      [data-testid="stDataFrame"], [data-testid="stTable"] {
        font-variant-numeric: tabular-nums;
      }
      .matlab-title {
        border: 0;
        border-radius: 0;
        padding: 0.5rem 0;
        font-size: 0.9375rem;
        font-weight: 600;
        letter-spacing: -0.005em;
        background: transparent;
        color: var(--app-text);
        box-shadow: none;
      }
      .matlab-status {
        font-size: var(--app-type-small);
        color: var(--app-muted);
        margin-bottom: 0.35rem;
      }
      .run-criteria-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 0.45rem;
      }
      .run-criteria-group {
        min-width: 0;
        padding: 0.85rem 1rem;
        border: 0;
        border-radius: 10px;
        background: var(--app-surface);
      }
      .run-criteria-heading {
        margin-bottom: 0.4rem;
        color: var(--app-text);
        font-size: var(--app-type-control);
        font-weight: 600;
      }
      .run-criteria-row {
        display: grid;
        grid-template-columns: minmax(5.6rem, 0.8fr) minmax(0, 1.5fr);
        gap: 0.55rem;
        padding: 0.36rem 0;
        border-top: 1px solid var(--app-divider);
        font-size: var(--app-type-data);
        line-height: 1.5;
      }
      .run-criteria-label { color: var(--app-muted); }
      .run-criteria-value {
        color: var(--app-text);
        font-weight: 500;
        overflow-wrap: anywhere;
      }
      @media (max-width: 1100px) {
        .run-criteria-grid { grid-template-columns: 1fr; }
      }
      [data-testid="stVerticalBlockBorderWrapper"],
      [data-testid="stForm"],
      [data-testid="stVerticalBlock"][data-test-border="true"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        box-shadow: none;
      }
      div[data-testid="stMarkdownContainer"] p,
      div[data-testid="stCaptionContainer"],
      label {
        color: inherit;
        font-size: var(--app-type-control);
        letter-spacing: -0.003em;
      }
      div[data-testid="stMarkdownContainer"] p {
        font-size: var(--app-type-body);
        line-height: 1.5;
      }
      [data-testid="stWidgetLabel"] p,
      [data-testid="stCheckbox"] p, [data-testid="stRadio"] p,
      [data-testid="stToggle"] p {
        font-size: var(--app-type-control) !important;
      }
      body:has(.annotation-viewer-active) [data-testid="stMarkdownContainer"] p {
        font-size: var(--app-type-control);
      }
      div[data-testid="stCaptionContainer"] {
        color: var(--app-muted) !important;
      }
      /* Help belongs to its label. Captions remain readable supporting text,
         never detached question-mark popups. */
      [data-testid="stWidgetLabel"] {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 0.35rem;
      }
      [data-testid="stWidgetLabel"] > :is(label, span) {
        flex: 0 1 auto;
        min-width: 0;
      }
      [data-testid="stWidgetLabel"] > :has([data-testid="stTooltipIcon"]) {
        flex: 0 0 auto;
        width: auto;
      }
      [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] {
        flex: 0 0 auto;
        align-self: center;
      }
      div[data-testid="stCaptionContainer"] > p {
        color: var(--app-muted) !important;
        font-size: var(--app-type-small) !important;
        line-height: 1.4 !important;
      }
      .stButton button,
      .stDownloadButton button,
      [data-testid="stFormSubmitButton"] button {
        border-radius: 7px !important;
        border: 1px solid var(--app-border-strong) !important;
        background: var(--app-surface) !important;
        color: var(--app-text) !important;
        min-height: 2rem;
        padding: 0.3rem 0.75rem;
        font-size: var(--app-type-control) !important;
        font-weight: 500 !important;
        box-shadow: none;
        transition: background-color 80ms ease-out;
      }


      .stButton button[kind="primary"],
      .stDownloadButton button[kind="primary"],
      [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: var(--app-accent) !important;
        border-color: var(--app-accent) !important;
        color: #ffffff !important;
        box-shadow: none;
      }

      /* Streamlit renders button labels as Markdown paragraphs. They must inherit
         the button's white primary label, not the normal body-text styling. */
      .stButton button [data-testid="stMarkdownContainer"] p,
      .stDownloadButton button [data-testid="stMarkdownContainer"] p,
      [data-testid="stFormSubmitButton"] button [data-testid="stMarkdownContainer"] p {
        color: inherit !important;
        font-size: var(--app-type-control) !important;
        font-weight: 500 !important;
        line-height: 1.3 !important;
      }
      .stButton button:disabled, .stDownloadButton button:disabled,
      [data-testid="stFormSubmitButton"] button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }

      [class*="st-key-review-controls-collapse"],
      [class*="st-key-review-controls-reveal"] {
        position: sticky;
        top: 0.5rem;
        z-index: 20;
        width: 2rem;
      }
      [class*="st-key-review-controls-panel-hidden"],
      .st-key-standard-review-controls-panel-hidden {
        display: none !important;
      }
      .st-key-standard-review-controls-reveal button,
      .st-key-standard-review-controls-collapse button {
        width: 1.85rem !important;
        min-width: 1.85rem !important;
        min-height: 1.85rem !important;
        height: 1.85rem !important;
        padding: 0 !important;
        line-height: 1 !important;
      }
      [class*="st-key-page-header-standard-back-to-start-"] h1 {
        margin: 0 !important;
        padding-block: 0.1rem !important;
      }
      .st-key-standard-grid-stream-row h3 {
        margin: 0 !important;
        padding-block: 0.1rem !important;
      }
      .st-key-standard-grid-stream-row [data-testid="stSelectbox"] {
        margin: 0 !important;
      }
      .st-key-standard-grid-stream-row [data-testid="stWidgetLabel"] {
        display: none !important;
      }
      body:has(.st-key-standard-grid-stream-row) [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
        gap: 0.45rem !important;
      }
      [class*="st-key-review-controls-collapse"] .stButton button,
      [class*="st-key-review-controls-reveal"] .stButton button {
        width: 1.85rem !important;
        min-width: 1.85rem !important;
        min-height: 1.85rem !important;
        height: 1.85rem !important;
        padding: 0 !important;
        border-radius: 999px !important;
        font-size: 18px !important;
        line-height: 1 !important;
        box-shadow: none !important;
      }
      button:focus-visible,
      input:focus-visible,
      textarea:focus-visible,
      [role="button"]:focus-visible,
      [role="slider"]:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px var(--app-focus) !important;
      }
      [data-baseweb="input"],
      [data-baseweb="select"] > div,
      [data-baseweb="base-input"] {
        background: var(--app-control) !important;
        border-color: var(--app-border-strong) !important;
        border-radius: 7px !important;
        box-shadow: none;
        transition: border-color 140ms ease, box-shadow 140ms ease;
      }
      [data-baseweb="input"]:focus-within,
      [data-baseweb="select"] > div:focus-within,
      [data-baseweb="base-input"]:focus-within {
        border-color: var(--app-accent) !important;
        box-shadow: 0 0 0 3px var(--app-focus) !important;
      }
      input, textarea {
        color: var(--app-text) !important;
        caret-color: var(--app-accent) !important;
        font-size: var(--app-type-control) !important;
      }
      [data-baseweb="select"] {
        font-size: var(--app-type-control) !important;
      }
      input::placeholder, textarea::placeholder {
        color: var(--app-tertiary) !important;
      }
      [data-testid="stCheckbox"] input,
      [data-testid="stRadio"] input {
        accent-color: var(--app-accent) !important;
      }
      [data-testid="stSlider"] [role="slider"] {
        background: var(--app-surface) !important;
        border: 1px solid rgba(0, 122, 255, 0.32) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.14) !important;
      }
      [data-testid="stExpander"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        box-shadow: none;
        overflow: hidden;
      }
      [data-testid="stExpander"] details {
        border: 1px solid transparent !important;
        border-radius: 8px !important;
      }
      [data-testid="stExpander"] summary {
        background: transparent !important;
        border-radius: 8px !important;
      }
      [data-testid="stExpander"] summary:hover {
        background: transparent !important;
      }
      [data-testid="stExpander"] summary:active {
        background: var(--app-group) !important;
      }
      [data-testid="stMetric"] {
        background: var(--app-group) !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.75rem !important;
        box-shadow: none;
      }
      [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
      }
      [data-testid="stMetricLabel"] p {
        font-size: var(--app-type-control) !important;
        color: var(--app-muted);
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--app-border);
        border-radius: 8px;
        overflow: hidden;
        background: var(--app-surface);
        box-shadow: none;
      }
      [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.56) !important;
        border: 1px dashed var(--app-border-strong) !important;
        border-radius: 12px !important;
      }
      [data-testid="stAlert"] {
        border: 1px solid var(--app-border) !important;
        border-radius: 8px !important;
        box-shadow: none;
      }
      [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        color: var(--app-text) !important;
      }
      [data-baseweb="popover"] {
        background: var(--app-surface) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1) !important;
      }
      [data-baseweb="tab-list"] {
        padding: 2px !important;
        gap: 2px !important;
        background: rgba(118, 118, 128, 0.12) !important;
        border-radius: 10px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        justify-content: flex-start !important;
        scrollbar-width: thin;
      }
      [data-baseweb="tab"] {
        min-height: 1.9rem !important;
        flex: 0 0 auto !important;
        padding-left: 13px !important;
        padding-right: 13px !important;
        white-space: nowrap !important;
        border-radius: 8px !important;
        color: var(--app-muted) !important;
        font-size: var(--app-type-control) !important;
      }
      [data-baseweb="tab"] p {
        font-size: var(--app-type-control) !important;
        color: inherit !important;
      }
      [data-baseweb="tab"][aria-selected="true"] [data-testid="stMarkdownContainer"],
      [data-baseweb="tab"][aria-selected="true"] p {
        color: var(--app-selection-ink) !important;
      }
      [data-baseweb="tab"][aria-selected="true"] {
        background: var(--app-accent) !important;
        color: var(--app-selection-ink) !important;
        font-weight: 600 !important;
        box-shadow: none;
      }
      [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
      /* Settings use the macOS white-field / grey-group hierarchy. */
      [data-testid="stTextInputRootElement"],
      [data-testid="stNumberInputContainer"] {
        min-height: 2rem !important;
        height: 2rem !important;
        background: var(--app-control) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 6px !important;
      }
      [data-testid="stTextInputRootElement"] input,
      [data-testid="stNumberInputContainer"] input {
        min-height: 0 !important;
        height: 100% !important;
        padding-block: 0.25rem !important;
        padding-inline: 0.625rem !important;
      }
      [data-testid="stNumberInputContainer"] button {
        background: transparent !important;
        color: var(--app-muted) !important;
        min-height: 0 !important;
        border: 0 !important;
      }
      [data-baseweb="select"] > div { min-height: 2rem !important; }
      /* Labels identify a setting; values and option text remain regular. */
      [data-testid="stWidgetLabel"] p {
        color: var(--app-text) !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
      }
      input, textarea, [data-baseweb="select"],
      [data-testid="stCheckbox"] label p,
      [data-testid="stRadio"] label[data-baseweb="radio"] p,
      [data-testid="stToggle"] label p {
        font-weight: 400 !important;
      }
      [data-testid="stExpander"] summary p { font-weight: 500; }
      [class*="st-key-settings-group-"] {
        background: transparent !important;
        padding: 0 !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        gap: 0.75rem !important;
      }
      [class*="st-key-settings-group-"] [data-testid="stExpander"] {
        border-color: var(--app-divider) !important;
      }
      /* Related controls in columns share one continuous surface. Section
         headings and action-only rows remain on the workspace canvas. */
      __HORIZONTAL_SETTINGS_ROW__:not(:where(
        [data-testid="stSidebar"],
        .st-key-review-controls-panel, .st-key-review-controls-panel-hidden,
        [class*="st-key-settings-surface-"]
      ) *) {
        background: var(--app-group) !important;
        border-radius: 10px;
        padding: 0.625rem 0.75rem;
      }
      __HORIZONTAL_SETTINGS_ROW__ __HORIZONTAL_SETTINGS_ROW__ {
        background: transparent !important;
        padding: 0;
      }
      /* Each setting owns its grey input surface. Group and form wrappers
         stay on the canvas so their titles and action buttons remain outside. */
      [data-testid="stElementContainer"]:has(> :is(
        [data-testid="stTextInput"], [data-testid="stNumberInput"],
        [data-testid="stTextArea"], [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"], [data-testid="stRadio"],
        [data-testid="stCheckbox"], [data-testid="stToggle"],
        [data-testid="stSlider"], [data-testid="stSelectSlider"],
        [data-testid="stDateInput"], [data-testid="stTimeInput"],
        [data-testid="stColorPicker"], [data-testid="stFileUploader"],
        [data-testid="stSegmentedControl"], [data-testid="stPills"],
        [data-testid="stButtonGroup"]
      )):not(:where(
        [data-testid="stSidebar"],
        .st-key-review-controls-panel, .st-key-review-controls-panel-hidden,
        [class*="st-key-settings-surface-"],
        __HORIZONTAL_SETTINGS_ROW__
      ) *) {
        background: var(--app-group) !important;
        border-radius: 10px;
        width: 100% !important;
        min-width: 0;
        padding: 0.625rem 0.75rem;
      }
      /* Put setting titles in the first grid row on the canvas, with a
         rounded grey surface behind only the control in the second row.
         Inline checkbox/toggle labels remain inside their setting row. */
      [data-testid="stElementContainer"]:has(> :is(
        [data-testid="stTextInput"], [data-testid="stNumberInput"],
        [data-testid="stTextArea"], [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"], [data-testid="stRadio"],
        [data-testid="stSlider"], [data-testid="stSelectSlider"],
        [data-testid="stDateInput"], [data-testid="stTimeInput"],
        [data-testid="stFileUploader"], [data-testid="stSegmentedControl"],
        [data-testid="stPills"], [data-testid="stButtonGroup"]
      )):not(:where(
        [data-testid="stSidebar"],
        .st-key-review-controls-panel, .st-key-review-controls-panel-hidden,
        [class*="st-key-settings-surface-"],
        __HORIZONTAL_SETTINGS_ROW__
      ) *) {
        background: transparent !important;
        padding: 0;
      }
      __TITLED_SETTING__ {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 0;
      }
      __TITLED_SETTING__::before {
        content: "";
        grid-area: 2 / 1;
        background: var(--app-group);
        border-radius: 10px;
      }
      __TITLED_SETTING__ > [data-testid="stWidgetLabel"] {
        grid-area: 1 / 1;
        margin: 0 0 0.5rem;
        padding: 0 0.75rem;
        width: 100%;
      }
      __TITLED_SETTING__ > [data-testid="stWidgetLabel"] p {
        font-weight: 600 !important;
      }
      __TITLED_SETTING__ > :nth-child(2) {
        grid-area: 2 / 1;
        position: relative;
        margin: 0.625rem 0.75rem;
        width: calc(100% - 1.5rem);
        min-width: 0;
      }
      /* Opt-in multirow groups. These surfaces are used by settings pages;
         the lipid annotation viewer does not use or inherit them. */
      [class*="st-key-settings-surface-"] {
        background: var(--app-group) !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 0.625rem 0.75rem !important;
        gap: 0.75rem !important;
      }
      [class*="st-key-settings-surface-"] [data-testid="stHorizontalBlock"] {
        background: transparent !important;
        padding: 0;
      }
      [class*="st-key-settings-surface-"] [data-testid="stExpander"] {
        background: transparent !important;
        border-color: var(--app-divider) !important;
      }
      /* Annotation forms share the same content edge as the module controls. */
      [data-testid="stForm"][aria-label="matlab-engine-inputs"],
      [data-testid="stForm"][aria-label="standard-lipidomics-inputs"] {
        padding: 0 !important;
        border: 0 !important;
      }
      /* A form may contain several titled groups without one enclosing card. */
      [data-testid="stForm"]:has(.criteria-section-heading) {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
      }
      .st-key-ms-viewer-lipid-selector {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
      }
      .st-key-review-controls-panel, .st-key-review-controls-panel-hidden {
        background: var(--app-group) !important;
        border-color: transparent !important;
        border-radius: 10px !important;
      }
      /* Home modules read as grouped navigation rows, with quiet separators. */
      [class*="st-key-home-navigation-"] {
        background: var(--app-group) !important;
        border-color: transparent !important;
        border-radius: 10px !important;
        gap: 0 !important;
        padding: 0.3rem 0.6rem !important;
      }
      [class*="st-key-home-navigation-"] [data-testid="stVerticalBlock"] {
        gap: 0 !important;
      }
      [class*="st-key-home-navigation-"] .stButton button {
        min-height: 2.5rem;
        padding: 0.5rem 0.75rem;
        background: transparent !important;
        border: 0 !important;
        border-radius: 6px !important;
        justify-content: space-between;
        text-align: left;
      }
      [class*="st-key-home-navigation-"] .stButton button > div,
      [class*="st-key-home-navigation-"] .stButton button [data-testid="stMarkdownContainer"] {
        justify-content: flex-start !important;
        flex: 1;
        text-align: left;
      }
      [class*="st-key-home-navigation-"] .stButton button::after {
        content: "";
        width: 0.35rem;
        height: 0.35rem;
        flex: 0 0 auto;
        border-top: 1.5px solid var(--app-tertiary);
        border-right: 1.5px solid var(--app-tertiary);
        transform: rotate(45deg);
        margin-left: 0.5rem;
      }
      [class*="st-key-home-navigation-"] .stElementContainer + .stElementContainer {
        border-top: 1px solid var(--app-divider);
      }

      [class*="st-key-home-navigation-"] .stButton button [data-testid="stMarkdownContainer"] p {
        font-weight: 500 !important;
        text-align: left;
      }
      hr {
        border-color: var(--app-divider) !important;
      }
      .plot-status {
        font-size: var(--app-type-data);
        color: var(--app-text);
        background: var(--app-surface-material);
        border: 1px solid var(--app-border);
        border-radius: 8px;
        padding: 6px 10px;
        margin-top: 0.15rem;
        box-shadow: none;
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        backdrop-filter: blur(18px) saturate(180%);
      }
      [data-testid="stPlotlyChart"] {
        background: var(--app-surface) !important;
        border-radius: 8px;
        overflow: hidden;
      }
      [class*="st-key-review-controls-panel"] {
        background: var(--app-surface-soft) !important;
        border-right: 1px solid var(--app-divider);
      }
      ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
      }
      ::-webkit-scrollbar-track {
        background: transparent;
      }
      ::-webkit-scrollbar-thumb {
        background: rgba(60, 60, 67, 0.24);
        border: 3px solid transparent;
        border-radius: 999px;
        background-clip: padding-box;
      }
      ::-webkit-scrollbar-thumb:hover {
        background: rgba(60, 60, 67, 0.36);
        border: 3px solid transparent;
        background-clip: padding-box;
      }
      @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
          scroll-behavior: auto !important;
          transition-duration: 0.01ms !important;
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
        }
        /* Windows' global animation preference must produce a steady loading
           indicator, not a stepped animation. Retain the short-rerun delay. */
__PAGE_LOADING_REDUCED_MOTION__
      }
      /* Zero-height plot event bridges should not reserve two full row gaps. */
      .st-key-2d_interaction--grid-viewer,
      .st-key-3d_click--grid-viewer {
        margin-block: -0.875rem;
      }
      /* Keep the fixed diagnostic header close to its scrollable plot body. */
      [class*="st-key-"][class*="-diagnostics-shell"],
      [class*="st-key-"][class*="-diagnostics-shell"] [data-testid="stVerticalBlock"] {
        gap: 0.05rem !important;
      }
      /* Keep appearance available on every page without taking a header row. */
      .st-key-laser-theme-toolbar {
        position: fixed !important;
        right: max(1rem, env(safe-area-inset-right, 0px));
        bottom: max(1rem, env(safe-area-inset-bottom, 0px));
        width: max-content !important;
        max-width: calc(100vw - 2rem);
        min-height: 2.25rem;
        margin: 0 !important;
        z-index: 100;
        border-radius: 7px;
        background: var(--app-surface);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
      }
      body:has(.annotation-viewer-active) [data-testid="stMainBlockContainer"],
      [data-testid="stMainBlockContainer"] {
        padding-bottom: calc(4.5rem + env(safe-area-inset-bottom, 0px)) !important;
      }
      [data-testid="stPopoverButton"] {
        background: var(--app-surface) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border-strong) !important;
        border-radius: 7px !important;
        font-size: var(--app-type-control) !important;
      }
      /* Hover is visually neutral. A press gives immediate feedback, then
         fades back in 80 ms; keyboard focus keeps its own visible ring. */
      __ACTION_BUTTON__ {
        --app-button-fill: var(--app-surface);
        --app-button-border: var(--app-border-strong);
        --app-button-ink: var(--app-text);
        background: var(--app-button-fill) !important;
        border-color: var(--app-button-border) !important;
        color: var(--app-button-ink) !important;
        box-shadow: none !important;
        filter: none !important;
        transform: none !important;
        transition: background-color 80ms ease-out;
      }
      [data-testid="stFileUploader"] button,
      [data-testid="stNumberInputContainer"] button {
        --app-button-fill: var(--app-control);
      }
      [data-testid="stNumberInputContainer"] button {
        --app-button-ink: var(--app-muted);
      }
      __ACTION_BUTTON__[kind="primary"] {
        --app-button-fill: var(--app-accent);
        --app-button-border: var(--app-accent);
        --app-button-ink: var(--app-selection-ink);
      }
      [class*="st-key-home-navigation-"] .stButton button {
        --app-button-fill: transparent;
        --app-button-border: transparent;
        border-color: transparent !important;
      }
      __ACTION_BUTTON__:not(:disabled):hover {
        background: var(--app-button-fill) !important;
        border-color: var(--app-button-border) !important;
        color: var(--app-button-ink) !important;
        box-shadow: none !important;
        filter: none !important;
        transform: none !important;
      }
      __ACTION_BUTTON__:not(:disabled):active {
        background: var(--app-group) !important;
        transition: none;
      }
      __ACTION_BUTTON__[kind="primary"]:not(:disabled):active {
        background: var(--app-accent-pressed) !important;
      }
      [class*="st-key-home-navigation-"] .stButton button:not(:disabled):active {
        background: var(--app-control) !important;
      }
      __ACTION_BUTTON__:not(:disabled):focus-visible {
        box-shadow: 0 0 0 3px var(--app-focus) !important;
      }
      /* Streamlit uses a separate button kind for form submissions. Limit
         this correction to the opted-in calibration/settings forms. */
      [data-testid="stForm"]:has([class*="st-key-settings-surface-"])
      button[kind="primaryFormSubmit"] {
        --app-button-fill: var(--app-accent);
        --app-button-border: var(--app-accent);
        --app-button-ink: var(--app-selection-ink);
      }
      [data-testid="stForm"]:has([class*="st-key-settings-surface-"])
      button[kind="primaryFormSubmit"]:not(:disabled):active {
        background: var(--app-accent-pressed) !important;
      }
      [data-baseweb="tab"] { --app-tab-fill: transparent; }
      [data-baseweb="tab"][aria-selected="true"] {
        --app-tab-fill: var(--app-accent);
      }
      [data-baseweb="tab"]:hover {
        background: var(--app-tab-fill) !important;
        box-shadow: none !important;
      }
      [data-baseweb="tab"]:active {
        background: var(--app-accent-pressed) !important;
      }
      [data-testid="stPopoverBody"], [data-testid="stPopoverBody"] > div {
        background: var(--app-surface) !important;
        color: var(--app-text) !important;
      }
    </style>
""".replace("__TITLED_SETTING__", _TITLED_SETTING).replace("__ACTION_BUTTON__", _ACTION_BUTTON).replace("__HORIZONTAL_SETTINGS_ROW__", _HORIZONTAL_SETTINGS_ROW)


APP_THEME_CSS = _loading_styles(_APP_THEME_TEMPLATE, enabled=False)
VIEWER_THEME_CSS = _loading_styles(_APP_THEME_TEMPLATE, enabled=False)


_MIDNIGHT_THEME_TEMPLATE = r"""
<style>
  :root {
    color-scheme: dark;
    --app-canvas: #172b3a;
    --app-loading-veil: #172b3a;
    --app-surface: #213e50;
    --app-surface-material: rgba(33, 62, 80, 0.98);
    --app-surface-soft: #2b4c60;
    --app-control: #1a3344;
    --app-group: #213e50;
    --app-sidebar: #213e50;
    --app-selection-ink: #06131e;
    --app-border: rgba(125, 227, 244, 0.24);
    --app-border-strong: rgba(125, 227, 244, 0.46);
    --app-divider: rgba(125, 227, 244, 0.18);
    --app-text: #e9f6fa;
    --app-muted: #bdd3dd;
    --app-tertiary: #a8c3cf;
    --app-accent: #7de3f4;
    --app-accent-pressed: #9beaf7;
    --app-focus: rgba(125, 227, 244, 0.32);
  }
  /* A quiet HUD grid and arc-reactor glow; no moving or flashing decoration. */
  [data-testid="stMain"] {
    background:
      radial-gradient(ellipse at 90% 0%, rgba(28, 153, 191, 0.10), transparent 55%),
      linear-gradient(rgba(81, 217, 245, 0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(81, 217, 245, 0.025) 1px, transparent 1px),
      var(--app-canvas) !important;
    background-size: auto, 40px 40px, 40px 40px, auto !important;
  }
  h1 {
    color: var(--app-accent) !important;
    border-bottom: 1px solid var(--app-border);
    text-shadow: 0 0 24px rgba(81, 217, 245, 0.15);
  }
  body:not(:has(.annotation-viewer-active)) [data-testid="stMain"] {
    background: var(--app-canvas) !important;
    background-image: none !important;
  }
  body:not(:has(.annotation-viewer-active)) h1 {
    color: var(--app-text) !important;
    border-bottom: 0;
    text-shadow: none;
  }
  body:not(:has(.annotation-viewer-active)) input:is([type="text"], [type="number"]):disabled,
  body:not(:has(.annotation-viewer-active)) textarea:disabled {
    color: var(--app-muted) !important;
    -webkit-text-fill-color: var(--app-muted) !important;
    opacity: 1;
  }
  [data-testid="stSidebar"], [data-testid="stSidebarContent"],
  div[data-testid="stCaptionContainer"] > p,
  [data-testid="stFileUploaderDropzone"],
  [data-testid="stCode"], pre, code,
  [data-baseweb="menu"], [role="listbox"], [role="tooltip"],
  [data-testid="stTooltipContent"] {
    background: var(--app-surface) !important;
    color: var(--app-text) !important;
  }
  [data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"],
  [data-testid="stText"], [data-testid="stMetricValue"],
  [data-testid="stMetricLabel"], [data-testid="stCheckbox"],
  [data-testid="stRadio"], [data-testid="stToggle"],
  [data-testid="stExpander"] summary,
  [data-baseweb="select"], [data-baseweb="select"] *,
  button[aria-label^="Help for"],
  [data-baseweb="input"] svg, [data-baseweb="menu"] *,
  [role="listbox"] *, [role="option"], [role="option"] *,
  [data-testid="stPopoverButton"] * {
    color: var(--app-text) !important;
  }
  button[aria-label^="Help for"] svg {
    stroke: var(--app-muted) !important;
  }
  [data-baseweb="textarea"], textarea, input,
  [data-testid="stFileUploader"] button {
    background: var(--app-control) !important;
    color: var(--app-text) !important;
    border-color: var(--app-border-strong) !important;
  }
  [role="option"]:hover, [aria-selected="true"][role="option"] {
    background: var(--app-surface-soft) !important;
  }
  label[data-baseweb="radio"] > div:first-child {
    background: var(--app-control) !important;
    border: 1px solid var(--app-border-strong);
  }
  label[data-baseweb="radio"]:has(input:checked) > div:first-child {
    background: var(--app-accent) !important;
  }
  label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {
    background: #06131e !important;
  }
  .stButton button[kind="primary"],
  .stDownloadButton button[kind="primary"],
  [data-testid="stFormSubmitButton"] button[kind="primary"] {
    color: #06131e !important;
    box-shadow: none;
  }
  .stButton button [data-testid="stMarkdownContainer"],
  .stDownloadButton button [data-testid="stMarkdownContainer"],
  [data-testid="stFormSubmitButton"] button [data-testid="stMarkdownContainer"] {
    color: inherit !important;
  }


  a { color: var(--app-accent); }
  [data-testid="stAlert"] {
    background: var(--app-surface-soft) !important;
    color: var(--app-text) !important;
  }
  [data-testid="stProgress"] [role="progressbar"] > div {
    background-color: var(--app-accent) !important;
  }
__PAGE_LOADING_MIDNIGHT_VEIL__
__PAGE_LOADING_MIDNIGHT_MESSAGE__
  ::-webkit-scrollbar-thumb { background-color: rgba(100, 215, 240, 0.30); }
  ::-webkit-scrollbar-thumb:hover { background-color: rgba(100, 215, 240, 0.48); }
  /* Plotly figures get real dark layouts at their rendering boundary. */
  [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: var(--app-surface) !important;
    color-scheme: dark;
  }
  [data-testid="stElementToolbar"], [data-testid="stDataFrameColumnMenu"],
  [data-testid="stDataFrameColumnVisibilityMenu"],
  [data-testid="stDataFrameColumnFormattingMenu"],
  [data-testid="stDataFrameTooltipContent"] {
    background: var(--app-surface) !important;
    color: var(--app-text) !important;
  }
  /* Glide paints underneath a transparent scroll/event plane. An opaque
     scroller masks all cells even though their values remain accessible. */
  [data-testid="stDataFrame"] .dvn-scroller {
    background: transparent !important;
  }
  /* Adapt only neutral native numeric tables, not scientific plot colours.
     Keep native sorting, search, copy/download, and virtual scrolling intact. */
  body:has(.annotation-viewer-active) [data-testid="stDataFrame"] .stDataFrameGlideDataEditor {
    filter: invert(0.86) sepia(0.3) saturate(1.6) hue-rotate(165deg);
  }
  /* The search bar is inside the adapted numeric-table surface. Give it the
     same source palette as that canvas, so it remains dark after adaptation. */
  body:has(.annotation-viewer-active) [data-testid="stDataFrame"] .gdg-search-bar,
  body:has(.annotation-viewer-active) [data-testid="stDataFrame"] .gdg-search-bar input,
  body:has(.annotation-viewer-active) [data-testid="stDataFrame"] .gdg-search-bar button {
    background: #f5f5f7 !important;
    color: #1d1d1f !important;
  }
  [data-testid="stElementToolbar"] button,
  [data-testid="stElementToolbarButtonIcon"] {
    color: var(--app-muted) !important;
  }
  [data-testid="stPlotlyChart"] .modebar {
    background: var(--app-surface-material) !important;
  }
  [data-testid="stPlotlyChart"] .modebar-btn path {
    fill: var(--app-muted) !important;
  }
</style>
"""


MIDNIGHT_THEME_CSS = _loading_styles(_MIDNIGHT_THEME_TEMPLATE, enabled=False)
VIEWER_MIDNIGHT_THEME_CSS = _loading_styles(_MIDNIGHT_THEME_TEMPLATE, enabled=False)


def normalize_theme(theme: object) -> str:
    """Ignore obsolete preferences without touching workflow/session settings."""
    return theme if isinstance(theme, str) and theme in THEMES else THEMES[0]


def current_theme() -> str:
    return normalize_theme(st.session_state.get(THEME_STATE_KEY))


def _remember_theme() -> None:
    st.session_state[THEME_STATE_KEY] = normalize_theme(
        st.session_state.get(THEME_WIDGET_KEY)
    )


def apply_app_theme(*, viewer: bool = False) -> None:
    """Apply this session's appearance and its fixed, page-independent chooser."""
    selected = current_theme()
    st.markdown(VIEWER_THEME_CSS if viewer else APP_THEME_CSS, unsafe_allow_html=True)
    midnight = VIEWER_MIDNIGHT_THEME_CSS if viewer else MIDNIGHT_THEME_CSS
    st.markdown(
        midnight if selected == "Midnight" else "<style></style>",
        unsafe_allow_html=True,
    )
    # Separate durable preference from the widget key so page/widget cleanup
    # cannot reset appearance. Never mutate shared Streamlit configuration.
    with st.container(
        key="laser-theme-toolbar", horizontal=True, horizontal_alignment="right"
    ):
        st.session_state[THEME_WIDGET_KEY] = selected
        with st.popover("Theme", icon=":material/palette:"):
            st.radio(
                "Appearance",
                THEMES,
                key=THEME_WIDGET_KEY,
                on_change=_remember_theme,
            )
