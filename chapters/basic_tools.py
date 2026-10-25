"""Chapter 4: Basic tools & examples. Small calculators the other chapters rely on.

Tabs: units and validation demos; thread and fastener sizes (ISO metric and UNC/UNF); pipe sizes (ASME B36.10, schedule 40
and 80); wall thickness under internal pressure (thin-wall and Lame thick-wall); thickness sensitivity of a cantilever
plate; ISO 286 standard tolerance grades. Calculations are pure functions (no Streamlit). Thread and pipe standards are tabulated in their own units, so
those dimensions are in mm (or inch for the inch threads), named accordingly; the other functions are SI in, SI out.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `streamlit run chapters/x.py` find common/

import math
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from common import units
from common.theme import ACCENT, FAIL_COLOR, PASS_COLOR, apply_theme
from common.ui import get_unit_system, run_standalone, slider_box
from common.validation import ValidationError, require_less_than, require_non_negative, require_positive

TITLE = "4 · Basic tools & examples"
DESCRIPTION = ("Small calculators the other chapters rely on: unit and validation demos, ISO metric and UNC/UNF thread "
               "sizes with clearance holes, pipe sizes, wall thickness under pressure, plate thickness sensitivity and "
               "ISO 286 tolerance grades.")
ORDER = 90          # listed after chapters 1 to 3 (they use the default, 50)
KEY_PREFIX = "bt_"
ABOUT = """**Method.** Each tab is a small closed-form calculator or a lookup of published standard values, shown with the formula it uses.

**Threads & fasteners.** ISO metric (60° thread) basic geometry from the pitch P: pitch diameter d₂ = d − (3√3/8)P, external minor diameter d₃ = d − (17√3/24)P, internal minor diameter D₁ = d − (5√3/8)P, and the tensile stress area A_s = (π/4)((d₂ + d₃)/2)² = (π/4)(d − 0.9382 P)² (the ISO 898-1 definition). Unified inch threads (UNC and UNF) use A_s = (π/4)(D − 0.9743/n)² with n threads per inch. The approximate tap drill is d − P. Clearance holes are the close and normal series of ASME B18.2.8, which follow ISO 273 (fine and medium). The floating-fastener position tolerance is T = H − F, the same formula as Chapter 3, with H the smallest clearance hole and F the nominal fastener diameter.

**Pipe sizes.** Outside diameter and wall thickness of NPS 1/2 to 8 steel pipe, schedule 40 and 80 (ASME B36.10M, millimetre values). Inside diameter = OD − 2 × wall, flow area = (π/4) ID², and the mean velocity at a flow Q is v = Q/A. Use the inside diameter as the pipe diameter in Chapter 2.

**Wall thickness under pressure.** Thin-wall (membrane) stresses for internal pressure p, inside diameter D_i and wall t: hoop σ_h = pD_i/(2t), axial (closed ends) σ_a = pD_i/(4t). The thick-wall (Lamé) inner-surface hoop stress is σ_h = p(r_o² + r_i²)/(r_o² − r_i²), radial σ_r = −p, axial σ_a = p r_i²/(r_o² − r_i²). The required wall for an allowable stress S is t = pD_i/(2S) (thin) or t = r_i(√((S + p)/(S − p)) − 1) (Lamé), plus the corrosion allowance; stresses are computed on the wall left after the corrosion allowance. This is a stress screening of a plain cylinder, not a pressure-vessel or piping code design.

**Thickness sensitivity.** Cantilever plate of width b, length L and thickness t with a tip load P: deflection δ = 4PL³/(Ebt³), stiffness k = P/δ = Ebt³/(4L³), root stress σ = 6PL/(bt²). Deflection falls with the cube of the thickness and stress with its square. Beam theory assumes a long plate (L/t of about 10 or more) and small deflections; the tab warns otherwise.

**Tolerance grades.** ISO 286-1 standard tolerance grades IT5 to IT12 for nominal sizes up to 500 mm (the size range is "over a, up to and including b"). The tolerance width T is the table value; a hole H class is +T/0, a shaft h class is 0/−T and a js class is ±T/2. These are the values to enter in Chapter 3's stack-up table. The table follows T = k·i with the standard tolerance factor i = 0.45 ∛D + 0.001 D (µm, D the geometric mean of the size range) and k = 7, 10, 16, 25, 40, 64, 100, 160 for IT5 to IT12, rounded as the standard rounds them. Only the grades, not the fundamental deviations or fit classes (such as H7/g6), are included.

**Units & validation.** The circle-area demo and two small cases (a hollow tube and a simply supported beam) show how impossible inputs get a clear message instead of a crash.

**Checked in pytest:** the thread geometry formulas by hand; tensile stress areas against the published ISO 898-1 table (coarse and fine pitches) and the UN tables (UNC and UNF); the clearance-hole tables for consistency; T = H − F by hand; pipe inside diameters, flow area and velocity by hand; thin-wall and Lamé stresses and required walls by hand, including the round trip (a wall sized for S gives exactly S); cantilever deflection, stiffness and root stress by hand with the cube and square laws; the IT grade table against the published ISO 286-1 values, its size-range boundary rule and the T = k·i formula; every input check.

**Limitations:** nominal dimensions only (no tolerance classes such as 6g/6H or manufacturing tolerances on pipe wall), no proof or tensile loads, and a limited list of sizes (M3 to M30, #4 to 1-1/4 in, NPS 1/2 to 8, IT5 to IT12 up to 500 mm). Pipe dimensions are carbon-steel B36.10M values; stainless pipe (B36.19M) differs for some schedules and sizes. The wall-thickness tab ignores weld joint efficiency, temperature, external pressure, bends and the piping and pressure-vessel codes. The plate tab uses linear beam theory. Charts from other sources round some medium clearance holes differently (for example M12 and M16). Values are for learning and screening: confirm them against the standard before using them for real work.

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


# NPS -> (outside diameter mm, schedule 40 wall mm, schedule 80 wall mm), carbon steel, ASME B36.10M
PIPE_SIZES = {"1/2": (21.3, 2.77, 3.73), "3/4": (26.7, 2.87, 3.91), "1": (33.4, 3.38, 4.55), "1-1/4": (42.2, 3.56, 4.85),
              "1-1/2": (48.3, 3.68, 5.08), "2": (60.3, 3.91, 5.54), "2-1/2": (73.0, 5.16, 7.01), "3": (88.9, 5.49, 7.62),
              "3-1/2": (101.6, 5.74, 8.08), "4": (114.3, 6.02, 8.56), "5": (141.3, 6.55, 9.53), "6": (168.3, 7.11, 10.97),
              "8": (219.1, 8.18, 12.70)}
SCHEDULES = (40, 80)
MATERIAL_E_GPA = {"Steel (typical)": 200.0, "Aluminium (typical)": 69.0, "Titanium (typical)": 114.0}


@dataclass(frozen=True)
class PipeSize:
    nps: str
    schedule: int
    od_mm: float
    wall_mm: float
    id_mm: float
    flow_area_m2: float


def pipe_size(nps: str, schedule: int) -> PipeSize:
    if nps not in PIPE_SIZES:
        raise ValidationError(f"NPS {nps} is not in the table (NPS 1/2 to 8).")
    if schedule not in SCHEDULES:
        raise ValidationError("The schedule must be 40 or 80.")
    od, w40, w80 = PIPE_SIZES[nps]
    wall = w40 if schedule == 40 else w80
    id_mm = od - 2 * wall
    return PipeSize(nps, schedule, od, wall, id_mm, math.pi / 4 * (id_mm / 1000) ** 2)


def pipe_velocity_m_s(flow_m3s: float, id_mm: float) -> float:
    """Mean velocity v = Q / A in a full pipe of the given inside diameter."""
    require_positive(flow_m3s, "Flow")
    require_positive(id_mm, "Inside diameter")
    return flow_m3s / (math.pi / 4 * (id_mm / 1000) ** 2)


@dataclass(frozen=True)
class WallStress:
    hoop_thin: float        # Pa, p Di / (2 t)
    axial_thin: float       # Pa, p Di / (4 t)
    hoop_thick: float       # Pa, Lame, inner surface (the largest)
    radial_inner: float     # Pa, -p at the inner surface
    axial_thick: float      # Pa, Lame, closed ends
    d_over_t: float


def wall_stresses(p_pa: float, di_m: float, t_m: float) -> WallStress:
    """Stresses in a cylinder with internal gauge pressure p, inside diameter Di and wall t (closed ends for the axial stress)."""
    require_positive(p_pa, "Pressure")
    require_positive(di_m, "Inside diameter")
    require_positive(t_m, "Wall thickness")
    ri = di_m / 2
    ro = ri + t_m
    return WallStress(p_pa * ri / t_m, p_pa * ri / (2 * t_m), p_pa * (ro**2 + ri**2) / (ro**2 - ri**2), -p_pa,
                      p_pa * ri**2 / (ro**2 - ri**2), di_m / t_m)


def required_wall_thin_m(p_pa: float, di_m: float, allowable_pa: float, corrosion_m: float = 0.0) -> float:
    """t = p Di / (2 S) + corrosion allowance."""
    for v, n in ((p_pa, "Pressure"), (di_m, "Inside diameter"), (allowable_pa, "Allowable stress")):
        require_positive(v, n)
    require_non_negative(corrosion_m, "Corrosion allowance")
    return p_pa * di_m / (2 * allowable_pa) + corrosion_m


def required_wall_thick_m(p_pa: float, di_m: float, allowable_pa: float, corrosion_m: float = 0.0) -> float:
    """Lame: the inner hoop stress equals S when r_o = r_i sqrt((S + p) / (S - p)); plus the corrosion allowance."""
    for v, n in ((p_pa, "Pressure"), (di_m, "Inside diameter"), (allowable_pa, "Allowable stress")):
        require_positive(v, n)
    require_non_negative(corrosion_m, "Corrosion allowance")
    if allowable_pa <= p_pa:
        raise ValidationError("The allowable stress must be greater than the pressure: no wall thickness can carry it.")
    return di_m / 2 * (math.sqrt((allowable_pa + p_pa) / (allowable_pa - p_pa)) - 1) + corrosion_m


@dataclass(frozen=True)
class PlateResult:
    deflection_m: float
    stiffness_n_per_m: float
    root_stress_pa: float


def cantilever_plate(load_n: float, length_m: float, width_m: float, thickness_m: float, e_pa: float) -> PlateResult:
    """Tip-loaded cantilever plate (linear beam theory): delta = 4 P L^3 / (E b t^3), sigma = 6 P L / (b t^2)."""
    for v, n in ((load_n, "Load"), (length_m, "Length"), (width_m, "Width"), (thickness_m, "Thickness"), (e_pa, "Young's modulus")):
        require_positive(v, n)
    delta = 4 * load_n * length_m**3 / (e_pa * width_m * thickness_m**3)
    return PlateResult(delta, load_n / delta, 6 * load_n * length_m / (width_m * thickness_m**2))


# ISO 286-1 standard tolerance grades, micrometres. Size ranges are "over a, up to and including b" (mm).
IT_RANGES = [(0, 3), (3, 6), (6, 10), (10, 18), (18, 30), (30, 50), (50, 80), (80, 120), (120, 180), (180, 250),
             (250, 315), (315, 400), (400, 500)]
IT_GRADES = (5, 6, 7, 8, 9, 10, 11, 12)
IT_TABLE_UM = {
    5: [4, 5, 6, 8, 9, 11, 13, 15, 18, 20, 23, 25, 27],
    6: [6, 8, 9, 11, 13, 16, 19, 22, 25, 29, 32, 36, 40],
    7: [10, 12, 15, 18, 21, 25, 30, 35, 40, 46, 52, 57, 63],
    8: [14, 18, 22, 27, 33, 39, 46, 54, 63, 72, 81, 89, 97],
    9: [25, 30, 36, 43, 52, 62, 74, 87, 100, 115, 130, 140, 155],
    10: [40, 48, 58, 70, 84, 100, 120, 140, 160, 185, 210, 230, 250],
    11: [60, 75, 90, 110, 130, 160, 190, 220, 250, 290, 320, 360, 400],
    12: [100, 120, 150, 180, 210, 250, 300, 350, 400, 460, 520, 570, 630]}


def it_range_index(size_mm: float) -> int:
    """Index of the ISO 286 size range 'over a, up to and including b' that contains the size."""
    require_positive(size_mm, "Nominal size")
    if size_mm > IT_RANGES[-1][1]:
        raise ValidationError(f"{size_mm:g} mm is above the 500 mm limit of this table.")
    return next(i for i, (_, hi) in enumerate(IT_RANGES) if size_mm <= hi)


def it_tolerance_um(size_mm: float, grade: int) -> float:
    """ISO 286-1 standard tolerance (width of the tolerance zone) in micrometres."""
    if grade not in IT_TABLE_UM:
        raise ValidationError(f"IT{grade} is not in this table (IT5 to IT12).")
    return float(IT_TABLE_UM[grade][it_range_index(size_mm)])


def it_zone_entries_um(size_mm: float, grade: int) -> dict:
    """(upper, lower) deviations in micrometres for hole H, shaft h and symmetric js zones of the same grade."""
    t = it_tolerance_um(size_mm, grade)
    return {"Hole, H": (t, 0.0), "Shaft, h": (0.0, -t), "Symmetric, js": (t / 2, -t / 2)}


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


def _slider(label: str, kind: str, lo: float, hi: float, default: float, key: str, system: str) -> float:
    lo_u, hi_u, d_u = (units.from_si(v, kind, system) for v in (lo, hi, default))
    return units.to_si(slider_box(f"{label} ({units.label(kind, system)})", lo_u, hi_u, d_u, f"{key}_{system}"), kind, system)


def _area_text(area_m2: float, system: str) -> str:
    return f"{area_m2 * 1e6:,.0f} mm²" if system == units.SI else f"{area_m2 / 0.00064516:.2f} in²"


def _render_pipes(system: str) -> None:
    st.caption("Carbon-steel pipe dimensions from ASME B36.10M. Use the inside diameter as the pipe diameter in Chapter 2.")
    c1, c2 = st.columns(2)
    sch = c1.radio("Schedule", list(SCHEDULES), horizontal=True, key="bt_sch", format_func=lambda x: f"Sch {x}")
    nps = c2.selectbox("Nominal pipe size (NPS, in)", list(PIPE_SIZES), index=list(PIPE_SIZES).index("2"), key="bt_nps")
    ps = pipe_size(nps, sch)
    a, b, c, d = st.columns(4)
    a.metric("Outside diameter", units.fmt(ps.od_mm / 1e3, "length", system, 4))
    b.metric("Wall thickness", units.fmt(ps.wall_mm / 1e3, "length", system, 3))
    c.metric("Inside diameter", units.fmt(ps.id_mm / 1e3, "length", system, 4))
    d.metric("Flow area", _area_text(ps.flow_area_m2, system))
    q = _entry("Flow", "flow", 0.010, "bt_pipe_q", system, 0.001, "%.2f")
    try:
        v = pipe_velocity_m_s(q, ps.id_mm)
    except ValidationError as exc:
        _show_error(exc)
    else:
        st.metric("Mean velocity at that flow", units.fmt(v, "velocity", system, 3))
    names = list(PIPE_SIZES)
    fig = go.Figure()
    for sc, col in ((40, ACCENT), (80, FAIL_COLOR)):
        fig.add_trace(go.Bar(x=names, y=[float(units.from_si(pipe_size(n, sc).id_mm / 1e3, "length", system)) for n in names],
                             name=f"Sch {sc}", marker_color=col, opacity=0.85))
    fig.add_trace(go.Scatter(x=[nps], y=[float(units.from_si(ps.id_mm / 1e3, "length", system))], mode="markers", name="Selected",
                             marker=dict(size=14, color=COLOR_AMBER, line=dict(width=2, color="#222"))))
    apply_theme(fig, "Inside diameter by nominal pipe size", 340)
    fig.update_xaxes(title="NPS (in)")
    fig.update_yaxes(title=f"Inside diameter ({units.label('length', system)})")
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Full table"):
        rows = [{"NPS (in)": n, f"OD ({units.label('length', system)})": round(float(units.from_si(PIPE_SIZES[n][0] / 1e3, "length", system)), 3),
                 **{f"Sch {sc} ID ({units.label('length', system)})": round(float(units.from_si(pipe_size(n, sc).id_mm / 1e3, "length", system)), 3)
                    for sc in SCHEDULES}} for n in names]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_wall(system: str) -> None:
    st.caption("Internal pressure on a plain cylinder with closed ends. A stress screening, not a pressure-vessel or piping code design. "
               "Example inputs are illustrative.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        p = _entry("Internal gauge pressure", "pressure", 5.0e6, "bt_wall_p", system, 1e5, "%.1f")
        di = _slider("Inside diameter", "length", 0.010, 0.500, 0.100, "bt_wall_di", system)
        t = _slider("Wall thickness", "length", 0.0005, 0.050, 0.004, "bt_wall_t", system)
        s_allow = _entry("Allowable stress", "stress", 100e6, "bt_wall_s", system, 5e6, "%.1f")
        ca = _entry("Corrosion allowance", "length", 0.0, "bt_wall_ca", system, 0.0005, "%.2f")
    with right:
        try:
            require_positive(s_allow, "Allowable stress")
            require_non_negative(ca, "Corrosion allowance")
            if ca >= t:
                raise ValidationError("The corrosion allowance must be smaller than the wall thickness: nothing would be left to carry the pressure.")
            te = t - ca
            ws = wall_stresses(p, di, te)
            t_thin, t_thick = required_wall_thin_m(p, di, s_allow, ca), required_wall_thick_m(p, di, s_allow, ca)
        except ValidationError as exc:
            _show_error(exc)
            return
        util = ws.hoop_thick / s_allow
        a, b = st.columns(2)
        a.metric("Hoop stress, thin-wall", units.fmt(ws.hoop_thin, "stress", system, 3))
        b.metric("Hoop stress, thick-wall (Lamé)", units.fmt(ws.hoop_thick, "stress", system, 3))
        c, d = st.columns(2)
        c.metric("Axial stress (closed ends)", units.fmt(ws.axial_thick, "stress", system, 3))
        d.metric("Required wall, Lamé", units.fmt(t_thick, "length", system, 3))
        msg = (f"Hoop stress is {util * 100:.0f} % of the allowable stress. Required wall {units.fmt(t_thick, 'length', system, 3)} "
               f"including the corrosion allowance (thin-wall formula: {units.fmt(t_thin, 'length', system, 3)}).")
        (st.success if util <= 1 else st.error)(("Pass. " if util <= 1 else "Fail. ") + msg)
        if ws.d_over_t < 20:
            st.info(f"D/t = {ws.d_over_t:.1f} is below 20, so the thin-wall formula is not reliable here: the thick-wall (Lamé) value governs "
                    f"(thin-wall is {(1 - ws.hoop_thin / ws.hoop_thick) * 100:.0f} % low).")
        ts = [te * f for f in (0.25 + 2.75 * i / 80 for i in range(81))]
        ri = di / 2
        thin = [float(units.from_si(p * ri / x, "stress", system)) for x in ts]
        thick = [float(units.from_si(p * ((ri + x) ** 2 + ri**2) / ((ri + x) ** 2 - ri**2), "stress", system)) for x in ts]
        xs = [float(units.from_si(x + ca, "length", system)) for x in ts]
        fig = go.Figure([go.Scatter(x=xs, y=thin, mode="lines", name="Thin-wall hoop", line=dict(color=COLOR_AMBER, width=3, dash="dash")),
                         go.Scatter(x=xs, y=thick, mode="lines", name="Thick-wall (Lamé) hoop", line=dict(color=ACCENT, width=3))])
        fig.add_hline(y=float(units.from_si(s_allow, "stress", system)), line=dict(color=FAIL_COLOR, dash="dot"), annotation_text="allowable stress")
        fig.add_trace(go.Scatter(x=[float(units.from_si(t, "length", system))], y=[float(units.from_si(ws.hoop_thick, "stress", system))],
                                 mode="markers", name="This wall", marker=dict(size=14, color=PASS_COLOR if util <= 1 else FAIL_COLOR,
                                                                                 line=dict(width=2, color="#222"))))
        apply_theme(fig, "Hoop stress against wall thickness", 360)
        fig.update_xaxes(title=f"Wall thickness ({units.label('length', system)})")
        fig.update_yaxes(title=f"Hoop stress ({units.label('stress', system)})", rangemode="tozero")
        st.plotly_chart(fig, width="stretch")


def _render_plate(system: str) -> None:
    st.caption("Tip-loaded cantilever plate, linear beam theory. Deflection falls with the cube of the thickness and stress with its square. "
               "Example inputs are illustrative.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        mat = st.selectbox("Material (typical Young's modulus)", list(MATERIAL_E_GPA) + ["Custom"], key="bt_plate_mat")
        e_unit = "GPa" if system == units.SI else "Msi"
        e_per = 1e9 if system == units.SI else 6.894757293168e9
        e_default = (MATERIAL_E_GPA[mat] if mat != "Custom" else 100.0) * 1e9 / e_per
        e_val = st.number_input(f"Young's modulus E ({e_unit})", value=float(e_default), step=1.0 if system == units.SI else 0.5,
                                format="%.2f", key=f"bt_plate_E_{mat}_{system}", disabled=mat != "Custom")
        e_pa = e_val * e_per
        load = _entry("Tip load", "force", 100.0, "bt_plate_load", system, 10.0, "%.1f")
        length = _slider("Length", "length", 0.050, 0.500, 0.200, "bt_plate_L", system)
        width = _slider("Width", "length", 0.010, 0.200, 0.050, "bt_plate_b", system)
        thick = _slider("Thickness", "length", 0.001, 0.020, 0.004, "bt_plate_t", system)
    with right:
        try:
            r = cantilever_plate(load, length, width, thick, e_pa)
            r_up = cantilever_plate(load, length, width, thick * 1.1, e_pa)
        except ValidationError as exc:
            _show_error(exc)
            return
        a, b, c = st.columns(3)
        a.metric("Tip deflection", units.fmt(r.deflection_m, "length", system, 3))
        b.metric("Stiffness", f"{r.stiffness_n_per_m / 1e3:,.3g} N/mm" if system == units.SI else f"{r.stiffness_n_per_m / 175.12684:,.3g} lbf/in")
        c.metric("Root bending stress", units.fmt(r.root_stress_pa, "stress", system, 3))
        st.info(f"A 10 % thicker plate is {r_up.stiffness_n_per_m / r.stiffness_n_per_m:.3f}× as stiff ({(r_up.stiffness_n_per_m / r.stiffness_n_per_m - 1) * 100:.0f} % more) "
                f"and its root stress is {r_up.root_stress_pa / r.root_stress_pa:.3f}× ({(1 - r_up.root_stress_pa / r.root_stress_pa) * 100:.0f} % lower).")
        if length / thick < 10:
            st.warning(f"L/t = {length / thick:.1f} is below 10: this plate is short for its thickness, so beam theory ignores shear deflection and under-predicts the deflection.")
        if r.deflection_m > 0.1 * length:
            st.warning("The deflection is more than 10 % of the length: linear beam theory is not reliable for large deflections.")
        ts = [0.001 + 0.019 * i / 80 for i in range(81)]
        res = [cantilever_plate(load, length, width, x, e_pa) for x in ts]
        xs = [float(units.from_si(x, "length", system)) for x in ts]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Tip deflection", "Root bending stress"), horizontal_spacing=0.14)
        x0 = float(units.from_si(thick, "length", system))
        for col, name, ys, y0, color in ((1, "Tip deflection", [float(units.from_si(x.deflection_m, "length", system)) for x in res],
                                          float(units.from_si(r.deflection_m, "length", system)), ACCENT),
                                         (2, "Root stress", [float(units.from_si(x.root_stress_pa, "stress", system)) for x in res],
                                          float(units.from_si(r.root_stress_pa, "stress", system)), COLOR_AMBER)):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, line=dict(color=color, width=3), showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", name="This plate", showlegend=col == 1,
                                     marker=dict(size=13, color=PASS_COLOR, line=dict(width=2, color="#222"))), row=1, col=col)
        apply_theme(fig, "Deflection falls with t³, stress with t² (log scales)", 360)
        fig.update_xaxes(title=f"Thickness ({units.label('length', system)})")
        fig.update_yaxes(title=f"Tip deflection ({units.label('length', system)})", type="log", row=1, col=1)
        fig.update_yaxes(title=f"Stress ({units.label('stress', system)})", type="log", row=1, col=2)
        fig.update_layout(legend=dict(orientation="h", y=-0.28))
        st.plotly_chart(fig, width="stretch")


def _render_grades(system: str) -> None:
    st.caption("ISO 286-1 standard tolerance grades (the width of the tolerance zone). Enter these in Chapter 3's stack-up table. "
               "Example inputs are illustrative.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        size = _slider("Nominal size", "length", 0.001, 0.500, 0.025, "bt_it_size", system)
        grade = st.selectbox("Tolerance grade", list(IT_GRADES), index=IT_GRADES.index(7), format_func=lambda g: f"IT{g}", key="bt_it_grade")
        st.caption("Smaller grade numbers are tighter. IT5 to IT7 are typical for fits and bearings, IT11 and IT12 for loose general work.")
    with right:
        try:
            size_mm = size * 1e3                                    # the slider returns SI (metres); the ISO table is in mm
            t = it_tolerance_um(size_mm, grade)
            entries = it_zone_entries_um(size_mm, grade)
            lo, hi = IT_RANGES[it_range_index(size_mm)]
        except ValidationError as exc:
            _show_error(exc)
            return
        a, b = st.columns(2)
        a.metric(f"IT{grade} tolerance T", f"{t:g} µm" if system == units.SI else f"{t / 25.4:.2f} mil (0.001 in)")
        b.metric("Size range used", f"over {lo:g} to {hi:g} mm" if system == units.SI else f"over {lo / 25.4:.3f} to {hi / 25.4:.3f} in")
        if system == units.SI:
            fmt_dev = lambda v: f"{v:+g}".replace("+0", "0") if v else "0"
            unit = "µm"
        else:
            fmt_dev = lambda v: f"{v / 25.4:+.2f}" if v else "0"
            unit = "mil"
        rows = [{"Zone": k, f"Upper deviation ({unit})": fmt_dev(u), f"Lower deviation ({unit})": fmt_dev(l),
                 "Size limits": f"{units.fmt(size + u * 1e-6, 'length', system, 5)} / {units.fmt(size + l * 1e-6, 'length', system, 5)}"}
                for k, (u, l) in entries.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Table values follow T = k·i with i = 0.45 ∛D + 0.001 D (µm) and k = {dict(zip(IT_GRADES, (7, 10, 16, 25, 40, 64, 100, 160)))[grade]} for IT{grade}, "
                   "rounded as the standard rounds them.")
        fig = go.Figure()
        mids = [math.sqrt(max(a_, 1.0) * b_) for a_, b_ in IT_RANGES]
        for g in IT_GRADES:
            fig.add_trace(go.Scatter(x=mids, y=IT_TABLE_UM[g], mode="lines+markers", name=f"IT{g}",
                                     line=dict(width=4 if g == grade else 1.5, color=ACCENT if g == grade else None),
                                     marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=[size * 1e3], y=[t], mode="markers", name="This size", marker=dict(size=14, color=COLOR_AMBER,
                                                                                                         line=dict(width=2, color="#222"))))
        apply_theme(fig, "Standard tolerance against nominal size (log scales)", 360)
        fig.update_xaxes(title="Nominal size (mm)", type="log")
        fig.update_yaxes(title="Tolerance (µm)", type="log")
        st.plotly_chart(fig, width="stretch")


def _render_threads() -> None:
    st.caption("Thread standards are tabulated in their own units, so this tab shows mm for metric threads and inches for "
               "inch threads regardless of the units setting. Values are nominal and illustrative of the standard tables.")
    kind = st.radio("Thread type", ["Metric (ISO)", "Inch (UNC / UNF)"], horizontal=True, key="bt_kind")
    _metric_tab() if kind.startswith("Metric") else _inch_tab()


def render() -> None:
    system = get_unit_system()
    tabs = st.tabs(["Units & validation", "Threads & fasteners", "Pipe sizes", "Wall under pressure", "Thickness sensitivity",
                    "Tolerance grades"])
    for tab, fn in zip(tabs, (_render_validation, lambda _s: _render_threads(), _render_pipes, _render_wall, _render_plate,
                              _render_grades)):
        with tab:
            fn(system)


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render, KEY_PREFIX, ABOUT)
