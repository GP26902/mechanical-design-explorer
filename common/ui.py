"""Streamlit-facing helpers shared by app.py and every chapter's standalone mode."""
from __future__ import annotations

import streamlit as st

from common import units
from common.validation import ValidationError

DISCLAIMER_TEXT = "For learning and screening, not certified design."
ILLUSTRATIVE_TEXT = "Example inputs are illustrative, not from real projects."
_KEY = "unit_system"


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


def run_standalone(title: str, description: str, render) -> None:
    """Used by each chapter's `if __name__ == "__main__"` block (streamlit run chapters/x.py)."""
    st.set_page_config(page_title=title, page_icon="🛠️", layout="wide")
    unit_selector()
    st.title(title)
    st.caption(description)
    show_disclaimer()
    render_safely(render)
