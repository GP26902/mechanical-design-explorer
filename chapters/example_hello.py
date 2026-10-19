"""Example chapter: proves auto-discovery, standalone running, units and validation.
Delete it once real chapters exist; nothing else depends on it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `streamlit run chapters/x.py` find common/

import plotly.graph_objects as go
import streamlit as st

from common import units
from common.theme import ACCENT, apply_theme
from common.ui import get_unit_system, run_standalone
from common.validation import require_positive

TITLE = "Example: unit & validation demo"
DESCRIPTION = "Trivial chapter showing auto-discovery, SI/US units and input validation."


# ---- Calculations (pure) -------------------------------------------------------------------
def circle_area_m2(diameter_m: float) -> float:
    require_positive(diameter_m, "Diameter")
    return 3.141592653589793 * diameter_m**2 / 4


# ---- UI ------------------------------------------------------------------------------------
def render() -> None:
    system = get_unit_system()
    u = units.label("length", system)
    d = st.number_input(f"Diameter ({u})", value=25.4 if system == units.SI else 1.0,
                        help="Try 0 or a negative value to see the validation message.")
    area = circle_area_m2(units.to_si(d, "length", system))   # raises ValidationError if invalid
    st.metric("Circle area", f"{area * 1e6:.1f} mm²")
    fig = go.Figure(go.Bar(x=["Area (mm²)"], y=[area * 1e6], marker_color=ACCENT))
    st.plotly_chart(apply_theme(fig, "Area for the chosen diameter", 300), width="stretch")


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render)
