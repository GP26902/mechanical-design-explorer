"""Chapter template. Copy this file to chapters/<your_chapter>.py (no leading underscore).

Rules: import only from common/, never from other chapters. Keep calculations in pure functions
(no Streamlit) so pytest can test them. Raise ValidationError for impossible inputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `streamlit run chapters/x.py` find common/

import streamlit as st

from common import units
from common.theme import apply_theme
from common.ui import get_unit_system, run_standalone
from common.validation import require_positive

TITLE = "Template chapter"
DESCRIPTION = "One sentence on what this chapter shows and the method it uses."
KEY_PREFIX = "tpl_"   # optional: prefix of this chapter's widget keys; the sidebar Reset button clears them
ABOUT = ""            # optional: markdown (method, checks, limitations) shown in the sidebar expander


# ---- Calculations: pure functions, SI in / SI out, no Streamlit ----------------------------
def calculate(length_m: float) -> float:
    require_positive(length_m, "Length")
    return length_m * 2


# ---- UI ------------------------------------------------------------------------------------
def render() -> None:
    system = get_unit_system()
    value = st.number_input(f"Length ({units.label('length', system)})", value=10.0)
    result_si = calculate(units.to_si(value, "length", system))
    st.metric("Result", units.fmt(result_si, "length", system))


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render, KEY_PREFIX, ABOUT)
