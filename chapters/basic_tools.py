"""Chapter 4: Basic tools & examples. Small calculators the other chapters rely on.

Step 1 tabs: units and validation demos, and thread and fastener sizes (ISO metric and UNC/UNF). Calculations are pure
functions (no Streamlit). Thread standards are tabulated in their own units, so thread dimensions are in mm (metric)
or inch (inch series), named accordingly, rather than SI metres.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `streamlit run chapters/x.py` find common/

import math
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import units
from common.theme import ACCENT, FAIL_COLOR, PASS_COLOR, apply_theme
from common.ui import get_unit_system, run_standalone
from common.validation import ValidationError, require_less_than, require_positive

TITLE = "4 · Basic tools & examples"
DESCRIPTION = ("Small calculators the other chapters rely on: unit and validation demos, and ISO metric and UNC/UNF "
               "thread and fastener sizes with clearance holes and the floating-fastener position tolerance.")
ORDER = 90          # listed after chapters 1 to 3 (they use the default, 50)
KEY_PREFIX = "bt_"
ABOUT = """**Method.** Each tab is a small closed-form calculator or a lookup of published standard values, shown with the formula it uses.

**Threads & fasteners.** ISO metric (60° thread) basic geometry from the pitch P: pitch diameter d₂ = d − (3√3/8)P, external minor diameter d₃ = d − (17√3/24)P, internal minor diameter D₁ = d − (5√3/8)P, and the tensile stress area A_s = (π/4)((d₂ + d₃)/2)² = (π/4)(d − 0.9382 P)² (the ISO 898-1 definition). Unified inch threads (UNC and UNF) use A_s = (π/4)(D − 0.9743/n)² with n threads per inch. The approximate tap drill is d − P. Clearance holes are the close and normal series of ASME B18.2.8, which follow ISO 273 (fine and medium). The floating-fastener position tolerance is T = H − F, the same formula as Chapter 3, with H the smallest clearance hole and F the nominal fastener diameter.

**Units & validation.** The circle-area demo and two small cases (a hollow tube and a simply supported beam) show how impossible inputs get a clear message instead of a crash.

**Checked in pytest:** the thread geometry formulas by hand; tensile stress areas against the published ISO 898-1 table (coarse and fine pitches) and the UN tables (UNC and UNF); the clearance-hole tables for consistency; T = H − F by hand; every input check.

**Limitations:** nominal dimensions only (no tolerance classes such as 6g/6H), no proof or tensile loads, and a limited list of sizes (M3 to M30, #4 to 1-1/4 in). Charts from other sources round some medium clearance holes differently (for example M12 and M16). Values are for learning and screening: confirm them against the standard before using them for real work.

For learning and screening, not certified design."""

SQRT3 = math.sqrt(3.0)
COLOR_AMBER = "#d97706"

# ---- reference data (published standard values; nominal) -------------------------------------------
METRIC_COARSE = {3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 16: 2.0, 20: 2.5, 24: 3.0, 30: 3.5}
METRIC_FINE = {3: [0.35], 4: [0.35, 0.5], 5: [0.35, 0.5, 0.75], 6: [0.5, 0.75], 8: [0.5, 0.75, 1.0],
               10: [0.5, 0.75, 1.0, 1.25], 12: [0.5, 0.75, 1.0, 1.25, 1.5], 16: [0.5, 0.75, 1.0, 1.25, 1.5],
               20: [1.0, 1.5, 2.0], 24: [1.0, 1.5, 2.0], 30: [1.0, 1.5, 2.0, 3.0]}
# clearance hole diameters, mm: (close / fine series, normal / medium series), ASME B18.2.8 = ISO 273
CLEARANCE_MM = {3: (3.2, 3.4), 4: (4.3, 4.5), 5: (5.3, 5.5), 6: (6.4, 6.6), 8: (8.4, 9.0), 10: (10.5, 11.0),
                12: (13.0, 13.5), 16: (17.0, 17.5), 20: (21.0, 22.0), 24: (25.0, 26.0), 30: (31.0, 33.0)}
# size -> (major diameter in, threads per inch)
UNC = {"#4": (0.112, 40), "#6": (0.138, 32), "#8": (0.164, 32), "#10": (0.190, 24), "1/4": (0.25, 20),
       "5/16": (0.3125, 18), "3/8": (0.375, 16), "7/16": (0.4375, 14), "1/2": (0.5, 13), "9/16": (0.5625, 12),
       "5/8": (0.625, 11), "3/4": (0.75, 10), "7/8": (0.875, 9), "1": (1.0, 8), "1-1/8": (1.125, 7), "1-1/4": (1.25, 7)}
UNF = {"#4": (0.112, 48), "#6": (0.138, 40), "#8": (0.164, 36), "#10": (0.190, 32), "1/4": (0.25, 28),
       "5/16": (0.3125, 24), "3/8": (0.375, 24), "7/16": (0.4375, 20), "1/2": (0.5, 20), "9/16": (0.5625, 18),
       "5/8": (0.625, 18), "3/4": (0.75, 16), "7/8": (0.875, 14), "1": (1.0, 12), "1-1/8": (1.125, 12), "1-1/4": (1.25, 12)}
INCH_SERIES = {"UNC (coarse)": UNC, "UNF (fine)": UNF}
MM_PER_IN = 25.4


# ---- Calculations: pure functions ------------------------------------------------------------------
def circle_area_m2(diameter_m: float) -> float:
    require_positive(diameter_m, "Diameter")
    return math.pi * diameter_m**2 / 4


def hollow_tube_area_mm2(od_mm: float, wall_mm: float) -> float:
    """Cross-section area of a round tube from its outer diameter and wall thickness (mm in, mm^2 out)."""
    require_positive(od_mm, "Outer diameter")
    require_positive(wall_mm, "Wall thickness")
    require_less_than(wall_mm, "Wall thickness", od_mm / 2, "half the outer diameter")
    id_mm = od_mm - 2 * wall_mm
    return math.pi / 4 * (od_mm**2 - id_mm**2)


def midspan_moment_nm(load_n: float, span_m: float) -> float:
    """Maximum bending moment M = PL/4 of a simply supported beam with a central point load."""
    require_positive(load_n, "Load")
    require_positive(span_m, "Span")
    return load_n * span_m / 4


@dataclass(frozen=True)
class MetricThread:
    d: float            # nominal (major) diameter, mm
    pitch: float        # mm
    d2: float           # pitch diameter, external thread
    d3: float           # minor diameter, external thread
    D1: float           # minor diameter, internal thread
    stress_area: float  # tensile stress area, mm^2
    tap_drill: float    # approximate, d - P


def metric_thread(d_mm: float, pitch_mm: float) -> MetricThread:
    """ISO 724 basic profile of a 60 degree thread plus the ISO 898-1 tensile stress area."""
    require_positive(d_mm, "Nominal diameter")
    require_positive(pitch_mm, "Pitch")
    require_less_than(pitch_mm, "Pitch", d_mm, "nominal diameter")
    d2 = d_mm - 3 * SQRT3 / 8 * pitch_mm
    d3 = d_mm - 17 * SQRT3 / 24 * pitch_mm
    if d3 <= 0:
        raise ValidationError(f"A pitch of {pitch_mm:g} mm is too large for a {d_mm:g} mm thread.")
    D1 = d_mm - 5 * SQRT3 / 8 * pitch_mm
    return MetricThread(d_mm, pitch_mm, d2, d3, D1, math.pi / 4 * ((d2 + d3) / 2) ** 2, d_mm - pitch_mm)


@dataclass(frozen=True)
class InchThread:
    D: float            # nominal (major) diameter, in
    tpi: float
    pitch: float        # in, 1 / tpi
    E: float            # pitch diameter, external thread, in
    Kn: float           # minor diameter, external thread, in
    stress_area: float  # tensile stress area, in^2
    tap_drill: float    # approximate, D - P, in


def inch_thread(D_in: float, tpi: float) -> InchThread:
    """Unified (UN) basic profile with the ASME B1.1 tensile stress area (pi/4)(D - 0.9743/n)^2."""
    require_positive(D_in, "Nominal diameter")
    require_positive(tpi, "Threads per inch")
    p = 1.0 / tpi
    require_less_than(p, "Pitch", D_in, "nominal diameter")
    kn = D_in - 17 * SQRT3 / 24 * p
    if kn <= 0:
        raise ValidationError(f"{tpi:g} threads per inch is too coarse for a {D_in:g} in thread.")
    return InchThread(D_in, tpi, p, D_in - 3 * SQRT3 / 8 * p, kn, math.pi / 4 * (D_in - 9 * SQRT3 / 16 * p) ** 2, D_in - p)


def floating_fastener_tolerance_mm(hole_mm: float, fastener_mm: float) -> float:
    """Position tolerance T = H - F for a floating fastener (hole at MMC H, fastener at MMC F), as in Chapter 3."""
    require_positive(hole_mm, "Hole size")
    require_positive(fastener_mm, "Fastener size")
    if hole_mm < fastener_mm:
        raise ValidationError(f"The hole ({hole_mm:g} mm) is smaller than the fastener ({fastener_mm:g} mm): "
                              "the fastener would not fit.")
    return hole_mm - fastener_mm


def coarse_series_rows() -> list:
    """One dict per metric coarse size, for the table and the chart."""
    rows = []
    for d, p in METRIC_COARSE.items():
        t, (close, normal) = metric_thread(d, p), CLEARANCE_MM[d]
        rows.append({"Size": f"M{d}", "d": d, "Pitch (mm)": p, "Pitch dia d₂ (mm)": round(t.d2, 3),
                     "Minor dia d₃ (mm)": round(t.d3, 3), "Stress area (mm²)": round(t.stress_area, 1),
                     "Tap drill ≈ d − P (mm)": round(t.tap_drill, 2), "Close hole (mm)": close, "Normal hole (mm)": normal})
    return rows


# ---- UI ------------------------------------------------------------------------------------------
def _entry(label: str, kind: str, default_si: float, key: str, system: str, step_si: float, fmt: str = "%.3f") -> float:
    """Number box in the chosen unit system; returns the SI value."""
    u = units.label(kind, system)
    v = st.number_input(f"{label} ({u})", value=float(units.from_si(default_si, kind, system)),
                        step=float(units.from_si(step_si, kind, system)), format=fmt, key=f"{key}_{system}")
    return units.to_si(v, kind, system)


def _show_error(exc: ValidationError) -> None:
    st.error(f"Input problem: {exc}")


def _render_validation(system: str) -> None:
    st.subheader("Unit and validation demo")
    st.caption("Chapters compute in SI and convert only at the edges. Try a zero or negative value to see the message.")
    d = _entry("Diameter", "length", 0.0254, "bt_dia", system, 0.001, "%.3f")
    try:
        area = circle_area_m2(d)
    except ValidationError as exc:
        _show_error(exc)
    else:
        st.metric("Circle area", f"{area * 1e6:,.1f} mm²")
        fig = go.Figure(go.Bar(x=["Area (mm²)"], y=[area * 1e6], marker_color=ACCENT))
        st.plotly_chart(apply_theme(fig, "Area for the chosen diameter", 280), width="stretch")

    st.subheader("Impossible inputs get a clear message")
    st.caption("Every chapter checks its inputs the same way and explains what is wrong instead of crashing. "
               "Example inputs are illustrative.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Round tube cross-section**")
        od = _entry("Outer diameter", "length", 0.040, "bt_tube_od", system, 0.001, "%.2f")
        wall = _entry("Wall thickness", "length", 0.003, "bt_tube_wall", system, 0.0005, "%.2f")
        st.caption("Try a wall thickness of half the outer diameter or more.")
        try:
            area = hollow_tube_area_mm2(od * 1e3, wall * 1e3)
        except ValidationError as exc:
            _show_error(exc)
        else:
            st.metric("Tube area", f"{area:,.1f} mm²", f"inner diameter {units.fmt(od - 2 * wall, 'length', system, 3)}", delta_color="off")
    with right:
        st.markdown("**Simply supported beam, central load**")
        load = _entry("Load", "force", 2000.0, "bt_beam_load", system, 100.0, "%.1f")
        span = _entry("Span", "length_long", 1.5, "bt_beam_span", system, 0.1, "%.2f")
        st.caption("Try a negative span or a zero load.")
        try:
            m = midspan_moment_nm(load, span)
        except ValidationError as exc:
            _show_error(exc)
        else:
            st.metric("Maximum moment M = PL/4", units.fmt(m, "torque", system, 3))


def _metric_tab() -> None:
    d = st.selectbox("Nominal diameter", list(METRIC_COARSE), format_func=lambda x: f"M{x}", index=list(METRIC_COARSE).index(8),
                     key="bt_md")
    pitches = [METRIC_COARSE[d]] + [p for p in METRIC_FINE[d] if p != METRIC_COARSE[d]]
    pitch = st.selectbox("Pitch", pitches, format_func=lambda p: f"{p:g} mm" + (" (coarse)" if p == METRIC_COARSE[d] else " (fine)"),
                         key=f"bt_mp_{d}")
    t = metric_thread(d, pitch)
    close, normal = CLEARANCE_MM[d]
    st.markdown(f"#### M{d} × {pitch:g}")
    a, b, c, e = st.columns(4)
    a.metric("Pitch diameter d₂", f"{t.d2:.3f} mm")
    b.metric("Minor diameter d₃", f"{t.d3:.3f} mm")
    c.metric("Tensile stress area", f"{t.stress_area:.1f} mm²")
    e.metric("Approx. tap drill", f"{t.tap_drill:.2f} mm")
    st.caption(f"Internal thread minor diameter D₁ = {t.D1:.3f} mm. Standard drills round the tap drill (for example M8 uses 6.8 mm).")
    st.markdown("**Clearance holes and the floating-fastener position tolerance (T = H − F, as in Chapter 3)**")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Close hole H", f"{close:g} mm")
    f2.metric("T with close hole", f"{floating_fastener_tolerance_mm(close, d):.2f} mm")
    f3.metric("Normal hole H", f"{normal:g} mm")
    f4.metric("T with normal hole", f"{floating_fastener_tolerance_mm(normal, d):.2f} mm")

    rows = coarse_series_rows()
    fig = go.Figure(go.Scatter(x=[r["d"] for r in rows], y=[r["Stress area (mm²)"] for r in rows], mode="lines+markers",
                               name="Coarse series", line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=[d], y=[t.stress_area], mode="markers", name=f"M{d} × {pitch:g}",
                             marker=dict(size=15, color=PASS_COLOR if pitch == METRIC_COARSE[d] else COLOR_AMBER,
                                         line=dict(width=2, color="#222"))))
    apply_theme(fig, "Tensile stress area by size (a finer pitch gives a larger stress area)", 340)
    fig.update_xaxes(title="Nominal diameter (mm)")
    fig.update_yaxes(title="Stress area (mm²)")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Full coarse series"):
        st.dataframe(pd.DataFrame(rows).drop(columns="d"), hide_index=True, width="stretch")


def _inch_tab() -> None:
    series = st.radio("Series", list(INCH_SERIES), horizontal=True, key="bt_iseries")
    table = INCH_SERIES[series]
    size = st.selectbox("Nominal size", list(table), index=list(table).index("1/4"), key=f"bt_isize_{series[:3]}")
    D, tpi = table[size]
    t = inch_thread(D, tpi)
    st.markdown(f"#### {size}-{tpi:g} {series.split()[0]}")
    a, b, c, e = st.columns(4)
    a.metric("Major diameter", f"{t.D:.4f} in", f"{t.D * MM_PER_IN:.2f} mm", delta_color="off")
    b.metric("Pitch diameter E", f"{t.E:.4f} in")
    c.metric("Tensile stress area", f"{t.stress_area:.4f} in²", f"{t.stress_area * MM_PER_IN**2:.1f} mm²", delta_color="off")
    e.metric("Approx. tap drill", f"{t.tap_drill:.4f} in", f"{t.tap_drill * MM_PER_IN:.2f} mm", delta_color="off")
    st.caption(f"Pitch P = 1/{tpi:g} = {t.pitch:.4f} in; external minor diameter {t.Kn:.4f} in. Standard tap drills round the approximate value.")
    fig = go.Figure()
    for name, tab, col in (("UNC (coarse)", UNC, ACCENT), ("UNF (fine)", UNF, FAIL_COLOR)):
        fig.add_trace(go.Scatter(x=[v[0] for v in tab.values()], y=[inch_thread(*v).stress_area for v in tab.values()], mode="lines+markers",
                                 name=name, line=dict(color=col, width=3)))
    fig.add_trace(go.Scatter(x=[D], y=[t.stress_area], mode="markers", name=f"{size}-{tpi:g}",
                             marker=dict(size=15, color=COLOR_AMBER, line=dict(width=2, color="#222"))))
    apply_theme(fig, "Tensile stress area by size (the fine series is a little larger)", 340)
    fig.update_xaxes(title="Major diameter (in)")
    fig.update_yaxes(title="Stress area (in²)")
    st.plotly_chart(fig, width="stretch")


def _render_threads() -> None:
    st.caption("Thread standards are tabulated in their own units, so this tab shows mm for metric threads and inches for "
               "inch threads regardless of the units setting. Values are nominal and illustrative of the standard tables.")
    kind = st.radio("Thread type", ["Metric (ISO)", "Inch (UNC / UNF)"], horizontal=True, key="bt_kind")
    _metric_tab() if kind.startswith("Metric") else _inch_tab()


def render() -> None:
    system = get_unit_system()
    tab_val, tab_thr = st.tabs(["Units & validation", "Threads & fasteners"])
    with tab_val:
        _render_validation(system)
    with tab_thr:
        _render_threads()


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render, KEY_PREFIX, ABOUT)
