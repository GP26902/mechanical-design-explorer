"""Streamlit-facing helpers shared by app.py and every chapter's standalone mode."""
from __future__ import annotations

import html

import streamlit as st

from common import units
from common.stepping import clamp, decimals, nice_step, snap_range
from common.validation import ValidationError

DISCLAIMER_TEXT = "For learning and screening, not certified design."
ILLUSTRATIVE_TEXT = "Example inputs are illustrative, not from real projects."
_KEY = "unit_system"
REPO_URL = "https://github.com/GP26902/mechanical-design-explorer"


def show_disclaimer() -> None:
    """The banner every chapter must display."""
    st.warning(f"**{DISCLAIMER_TEXT}** {ILLUSTRATIVE_TEXT}", icon="⚠️")


def unit_selector() -> str:
    """Sidebar SI/US toggle. State is shared via session_state, so the shell and chapters agree."""
    st.sidebar.radio("Units", units.SYSTEMS, key=_KEY, horizontal=True,
                     help="Calculations always run in SI internally; this only changes inputs and display.")
    return get_unit_system()


def get_unit_system() -> str:
    return st.session_state.get(_KEY, units.SI)


def render_safely(render) -> None:
    """Run a chapter's render(); show bad-input messages and unexpected errors without crashing."""
    try:
        render()
    except ValidationError as exc:
        st.error(f"Input problem: {exc}")
    except Exception as exc:  # last resort so one chapter never takes down the app
        st.error(f"This chapter hit an unexpected error: {exc}")


def _slider_changed(key: str) -> None:
    st.session_state[f"{key}_box"] = st.session_state[key]


def _box_changed(key: str, lo: float, hi: float) -> None:
    st.session_state[key] = clamp(st.session_state[f"{key}_box"], lo, hi)


def slider_box(label: str, lo: float, hi: float, default: float, key: str, step: float | None = None) -> float:
    """Slider for quick dragging plus a number box for an exact value; the two stay in sync.

    Works in the units shown to the user. The slider step is a round 1-2-5 number (see common/stepping.py)
    and the typed value may be anything inside [lo, hi]. The slider keeps `key`, the box uses `key_box`.
    """
    step = step or nice_step(hi - lo)
    dec = decimals(step)
    lo, hi = snap_range(lo, hi, step)
    box_key = f"{key}_box"
    if key not in st.session_state:
        st.session_state[key] = clamp(default, lo, hi)
    st.session_state[key] = clamp(st.session_state[key], lo, hi)
    st.session_state[box_key] = st.session_state[key]
    st.markdown(f"<div style='font-size:0.875rem'>{html.escape(label)}</div>",
                unsafe_allow_html=True)
    left, right = st.columns([3, 2], vertical_alignment="center")
    left.slider(label, float(lo), float(hi), step=float(step), format=f"%.{dec}f", key=key,
                label_visibility="collapsed", on_change=_slider_changed, args=(key,))
    right.number_input(label, float(lo), float(hi), step=float(step), format=f"%.{max(dec + 1, 2)}f",
                       key=box_key, label_visibility="collapsed", on_change=_box_changed, args=(key, lo, hi),
                       help="Type an exact value here.")
    return round(float(st.session_state[key]), dec + 4)


def _reset(prefix: str) -> None:
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefix)]:
        del st.session_state[k]


def chapter_sidebar(key_prefix: str = "", about: str = "") -> None:
    """Sidebar extras for the selected chapter: reset button and the about expander (both optional)."""
    if key_prefix:
        st.sidebar.button("Reset to defaults", key="reset_defaults", on_click=_reset, args=(key_prefix,),
                          help="Put every input of this chapter back to its starting value.")
    if about:
        with st.sidebar.expander("About this chapter"):
            st.markdown(about)


def sidebar_links() -> None:
    st.sidebar.markdown(f"[GitHub repository]({REPO_URL}) · [README]({REPO_URL}#readme)")


def run_standalone(title: str, description: str, render, key_prefix: str = "", about: str = "") -> None:
    """Used by each chapter's `if __name__ == "__main__"` block (streamlit run chapters/x.py)."""
    st.set_page_config(page_title=title, page_icon="🛠️", layout="wide")
    unit_selector()
    chapter_sidebar(key_prefix, about)
    sidebar_links()
    st.title(title)
    st.caption(description)
    show_disclaimer()
    render_safely(render)
