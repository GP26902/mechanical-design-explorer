"""Chapter 2: Pump & pipe system.

Method (no CFD claims): one-dimensional system-curve analysis. Pipe friction from Darcy-Weisbach with
a laminar / Colebrook-White friction factor, fitting losses from loss coefficients K, a quadratic
centrifugal pump curve scaled with the affinity laws, and the operating point where pump head equals
system head. Bottleneck = section with the largest pressure drop. Cavitation screening compares NPSH
available with NPSH required. Energy: hydraulic power rho g Q H, an assumed efficiency curve through the best
efficiency point, motor and drive efficiencies, and a throttle-valve vs variable-speed-drive comparison at a
reduced target flow. One or two identical pumps in parallel or series (equivalent quadratic curve). The pump curve
can come from a design point or from a least-squares fit of entered points. Tank gauge pressures and site elevation
(standard atmosphere) enter the static head and NPSH. Water, or a custom fluid (density, viscosity, vapour pressure).
Example systems load illustrative inputs; the section table and the curves can be downloaded as CSV.
Pump data and loss coefficients are illustrative.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `streamlit run chapters/x.py` find common/

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

from common import units
from common.theme import ACCENT, FAIL_COLOR, PASS_COLOR, STRESS_COLORSCALE, apply_theme
from common.ui import get_unit_system, run_standalone, slider_box
from common.validation import (ValidationError, require_in_range, require_less_than, require_non_negative,
                               require_positive)

TITLE = "2 · Pump & pipe system"
KEY_PREFIX = "pp_"
ABOUT = """**Method.** System-curve analysis, not CFD: Darcy-Weisbach friction with a laminar or Colebrook-White friction factor, fitting losses from loss coefficients, and a quadratic pump curve scaled with the affinity laws. The operating point is where the pump curve meets the system curve. Cavitation risk compares NPSH available with NPSH required.

**Energy.** Hydraulic power is ρgQH. Pump efficiency follows an assumed curve (η = η_BEP (2x − x²), x = flow / best-efficiency flow, with the best-efficiency flow taken as the design flow) that keeps the same shape under the affinity laws. Shaft and electrical power follow from it and from the motor efficiency. A throttle-valve vs variable-speed-drive comparison at a lower flow shows the head the valve wastes and the power and yearly cost the drive saves.

**Examples and export.** The example dropdown loads a set of illustrative inputs (a baseline, an undersized pipe, a hot-water suction lift, two pumps in parallel, a pressurised glycol loop at altitude); edit any input afterwards. The section table and the curve data download as CSV, in the units you have chosen.

**Realism inputs.** Pipe roughness and the valve and fitting K can be entered directly. Tank gauge pressures add (p_discharge − p_suction)/ρg to the static head, and the site elevation lowers the atmospheric pressure (standard atmosphere), which lowers NPSH available. A custom fluid replaces water's density, viscosity and vapour pressure.

**Pumps and pump data.** One or two identical pumps can run in parallel (same head, flows add) or in series (same flow, heads add); both reduce to an equivalent quadratic curve. The pump curve can also be fitted to entered points by regressing head on Q², which gives the shutoff head and the coefficient a directly; the fit residual is shown.

**Checked in pytest:** laminar f = 64/Re; Colebrook against the Moody chart and Swamee-Jain; Darcy-Weisbach and minor-loss hand values; system head by hand; affinity laws; operating point; NPSH and cavitation thresholds; water vapour pressure; energy balance; hydraulic, shaft and electrical power and yearly cost by hand; the drive speed against the closed form and the operating-point solver; the throttle loss against an equivalent valve added to the layout; the pump-curve fit against a hand regression and numpy; parallel and series curves and operating points against closed forms; the standard atmosphere and tank-pressure static head by hand; a custom fluid with water's own properties against the water result; density scaling of pressure drop and power.

**Limitations:** one series path, one pipe material. The pump curve is not corrected for fluid viscosity, so results for viscous liquids are indicative only; a custom fluid's properties do not change with temperature. Loss coefficients are rounded textbook values. The efficiency curve shape and the constant motor and drive efficiencies are assumptions, not manufacturer data. Pump pairs are identical, run at the same speed and are always both running; interaction between pumps and check-valve losses are ignored. Water hammer, two-phase flow and parallel branches are not modelled.

For learning and screening, not certified design."""
DESCRIPTION = ("Darcy-Weisbach pipe and fitting losses against a centrifugal pump curve: operating point, "
               "bottleneck section, NPSH cavitation check, example systems, CSV export, pump power and energy cost with a throttle vs "
               "variable-speed comparison, parallel or series pumps, a pump curve fitted from points, tank pressures, "
               "altitude and custom fluids (system-curve analysis, not CFD).")

# =============================================================================================
# CALCULATIONS (pure functions, SI units in and out, no Streamlit)
# =============================================================================================
G = 9.81                    # m/s^2
P_ATM = 101_325.0           # Pa, sea level
NPSH_MARGIN = 1.3           # NPSHa / NPSHr below this is flagged "marginal"
RE_LAMINAR, RE_TURBULENT = 2300.0, 4000.0

CUSTOM_ROUGH, CUSTOM_K = "Custom roughness", "Custom K"

# ---- example systems (illustrative, SI) --------------------------------------------------------------
DEFAULT_EXAMPLE = "Cooling-water transfer (default)"
BASE_VALUES = dict(material="Commercial steel", d_s=0.125, l_s=3.0, z_s=1.0, d_d=0.100, l_d=60.0, z_d=12.0, bends=4,
                   valve="Gate valve (open)", n_valves=1, fitting="Tee, flow through run", n_fit=1, p_s=0.0, p_d=0.0,
                   elev=0.0, speed=100, fluid="Water", temp=20, n_pumps=1, arrangement="parallel", h0=40.0, qd=0.020,
                   hd=30.0, nr=3.5)
EXAMPLES = {
    DEFAULT_EXAMPLE: dict(note="Baseline transfer line. The pump runs well to the right of its best-efficiency flow, so look at "
                               "the efficiency warning and the throttle vs variable-speed saving.", values={}),
    "Undersized discharge pipe": dict(note="A 65 mm, 80 m discharge line with six bends: the discharge pipe takes most of the "
                                           "pressure drop and the flow falls to about half of the baseline.",
                                      values=dict(d_d=0.065, l_d=80.0, bends=6)),
    "Hot water with a suction lift": dict(note="75 °C water and a pump 3 m above the tank: NPSH available falls below NPSH "
                                               "required and the cavitation check turns red.",
                                          values=dict(z_s=-3.0, temp=75)),
    "Two pumps in parallel": dict(note="A larger, longer system. One pump runs at about 130 % of its best-efficiency flow; "
                                       "two in parallel put each close to it. Switch the pump count to compare.",
                                  values=dict(d_s=0.150, d_d=0.125, l_d=150.0, z_d=18.0, bends=6, n_pumps=2,
                                              arrangement="parallel")),
    "Pressurised glycol loop at altitude": dict(note="A 30 % glycol-water style custom fluid, 150 kPa on the discharge tank and "
                                                     "a 1500 m site: higher static head, lower atmospheric NPSH. The pump "
                                                     "curve is not corrected for viscosity.",
                                                values=dict(z_d=5.0, l_d=80.0, p_d=150e3, elev=1500.0, fluid="Custom fluid")),
}


def build_example(name: str) -> tuple:
    """(Layout, PumpData, water temperature C, custom Fluid or None) for an example system, at rated speed."""
    V = {**BASE_VALUES, **EXAMPLES[name]["values"]}
    lay = Layout(V["d_s"], V["l_s"], V["d_d"], V["l_d"], ROUGHNESS[V["material"]], V["z_s"], V["z_d"], V["bends"],
                 V["n_valves"], VALVE_K[V["valve"]], V["n_fit"], FITTING_K[V["fitting"]], V["p_s"], V["p_d"], V["elev"])
    pump = PumpData(V["h0"], V["qd"], V["hd"], V["nr"], 1750.0, 0.75, V["n_pumps"], V["arrangement"])
    fluid = custom_fluid(1040.0, 3.0e-3, 2.0e3) if V["fluid"] == "Custom fluid" else None
    return lay, pump, float(V["temp"]), fluid


def sections_csv(r: "Result", system: str) -> bytes:
    """The section table at the operating point, with units in the column names. Plain table, one row per section."""
    if r.q_op is None:
        raise ValidationError("There is no operating point, so there are no sections to export.")
    fu = lambda v, kind, nd: round(float(units.from_si(v, kind, system)), nd)
    df = pd.DataFrame({
        "Section": [s.name for s in r.sections],
        "Side": [s.side for s in r.sections],
        f"Diameter ({units.label('length', system)})": [fu(s.diameter, "length", 4) for s in r.sections],
        f"Velocity ({units.label('velocity', system)})": [fu(s.velocity, "velocity", 4) for s in r.sections],
        "Reynolds number": pd.array([None if s.Re is None else round(s.Re) for s in r.sections], dtype="Int64"),   # whole numbers, blank for fittings
        "Friction factor": [None if s.f is None else round(s.f, 5) for s in r.sections],
        f"Head loss ({units.label('length_long', system)})": [fu(s.head_loss, "length_long", 4) for s in r.sections],
        f"Pressure drop ({units.label('pressure', system)})": [fu(d, "pressure", 4) for d in r.dp],
        "Share of drop (%)": [round(x * 100, 2) for x in r.share],
        "Bottleneck": ["yes" if s.name == r.bottleneck else "" for s in r.sections]})
    return df.to_csv(index=False).encode("utf-8")


def curves_csv(r: "Result", system: str) -> bytes:
    """Pump, system and NPSH curves on the plotted flow grid (at the current pump speed), units in the column names."""
    hu = units.label("length_long", system)
    df = pd.DataFrame({
        f"Flow ({units.label('flow', system)})": np.round(units.from_si(r.q, "flow", system), 5),
        f"Pump head ({hu})": np.round(units.from_si(r.h_pump, "length_long", system), 4),
        f"System head ({hu})": np.round(units.from_si(r.h_sys, "length_long", system), 4),
        f"NPSH available ({hu})": np.round(units.from_si(r.npsha_curve, "length_long", system), 4),
        f"NPSH required ({hu})": np.round(units.from_si(r.npshr_curve, "length_long", system), 4)})
    return df.to_csv(index=False).encode("utf-8")

DEFAULT_POINTS_Q = (0.0, 0.010, 0.020, 0.030)       # m^3/s, illustrative pump points for the "fitted from points" mode
DEFAULT_POINTS_H = (40.0, 37.6, 30.2, 17.4)         # m, scattered slightly around H = 40 - 25000 Q^2

# Typical textbook loss coefficients, rounded. Illustrative: real values depend on size, type and source.
K_ENTRANCE, K_EXIT, K_ELBOW_90 = 0.5, 1.0, 0.9
VALVE_K = {"Gate valve (open)": 0.15, "Ball valve (open)": 0.05, "Swing check valve": 2.0, "Globe valve (open)": 10.0}
FITTING_K = {"Tee, flow through run": 0.9, "Tee, flow through branch": 2.0}
ROUGHNESS = {"Commercial steel": 0.045e-3, "Galvanised iron": 0.15e-3, "Cast iron": 0.26e-3,
             "Drawn tubing / copper": 0.0015e-3, "PVC / plastic": 0.0015e-3}      # absolute roughness, m


def _finite(v, name: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number.") from None
    if not math.isfinite(x):
        raise ValidationError(f"{name} must be a finite number.")
    return x


def _count(v, name: str) -> int:
    x = _finite(v, name)
    if x < 0 or x != int(x):
        raise ValidationError(f"{name} must be a whole number, zero or more.")
    return int(x)


# ---- water properties (handbook table values, rounded; linear in T for density, log-linear for mu and Pv)
_T = np.arange(0.0, 101.0, 10.0)
_RHO = np.array([999.8, 999.7, 998.2, 995.7, 992.2, 988.0, 983.2, 977.8, 971.8, 965.3, 958.4])
_MU = np.array([1.787e-3, 1.307e-3, 1.002e-3, 0.797e-3, 0.653e-3, 0.547e-3, 0.466e-3, 0.404e-3, 0.355e-3,
                0.315e-3, 0.282e-3])
_PV = np.array([0.611e3, 1.228e3, 2.339e3, 4.246e3, 7.384e3, 12.35e3, 19.94e3, 31.19e3, 47.39e3, 70.14e3, 101.33e3])


@dataclass(frozen=True)
class Fluid:
    temp_c: float
    rho: float      # kg/m^3
    mu: float       # Pa.s
    pv: float       # vapour pressure, Pa


def water(temp_c: float) -> Fluid:
    t = require_in_range(_finite(temp_c, "Water temperature"), "Water temperature", 0.0, 100.0)
    return Fluid(t, float(np.interp(t, _T, _RHO)), float(np.exp(np.interp(t, _T, np.log(_MU)))),
                 float(np.exp(np.interp(t, _T, np.log(_PV)))))


def custom_fluid(rho: float, mu: float, pv: float) -> Fluid:
    """A user-defined liquid (kg/m^3, Pa.s, Pa absolute). Properties are fixed: no temperature dependence."""
    require_in_range(_finite(rho, "Density"), "Density", 100.0, 3000.0)
    require_in_range(_finite(mu, "Viscosity"), "Viscosity", 1e-5, 10.0)
    require_in_range(_finite(pv, "Vapour pressure"), "Vapour pressure", 0.0, 5e6)
    return Fluid(float("nan"), float(rho), float(mu), float(pv))


# ---- hydraulics ------------------------------------------------------------------------------
def atm_pressure(elevation_m: float) -> float:
    """Standard atmosphere: P = 101325 (1 - 2.25577e-5 h)^5.25588 Pa, h = site elevation in m."""
    h = require_in_range(_finite(elevation_m, "Site elevation"), "Site elevation", -500.0, 6000.0)
    return P_ATM * (1.0 - 2.25577e-5 * h) ** 5.25588


def velocity(Q: float, D: float) -> float:
    return Q / (math.pi * D**2 / 4)


def reynolds(rho: float, V: float, D: float, mu: float) -> float:
    return rho * V * D / mu


def friction_factor(Re: float, rel_rough: float) -> float:
    """Darcy friction factor: 64/Re for laminar flow (Re < 2300), otherwise Colebrook-White solved by
    fixed-point iteration. 2300 <= Re <= 4000 is transitional: the value is a guide only."""
    require_positive(_finite(Re, "Reynolds number"), "Reynolds number")
    require_non_negative(_finite(rel_rough, "Relative roughness"), "Relative roughness")
    if Re < RE_LAMINAR:
        return 64.0 / Re
    x = 1.0 / math.sqrt(0.25 / math.log10(rel_rough / 3.7 + 5.74 / Re**0.9) ** 2)   # Swamee-Jain start
    for _ in range(100):
        x_new = -2.0 * math.log10(rel_rough / 3.7 + 2.51 * x / Re)
        if abs(x_new - x) < 1e-13:
            x = x_new
            break
        x = x_new
    return 1.0 / x**2


def darcy_head(f: float, L: float, D: float, V: float) -> float:
    """Friction head loss h = f (L/D) V^2 / 2g, in metres of fluid."""
    return f * (L / D) * V**2 / (2 * G)


def minor_head(K: float, V: float) -> float:
    """Fitting head loss h = K V^2 / 2g."""
    return K * V**2 / (2 * G)


def npsha(p_surface: float, pv: float, rho: float, z_suction: float, h_loss_suction: float) -> float:
    """NPSH available (m). p_surface: absolute pressure on the suction tank surface (atmospheric for an open tank).
    z_suction > 0: liquid level above the pump."""
    return (p_surface - pv) / (rho * G) + z_suction - h_loss_suction


def cavitation_status(npsh_a: float, npsh_r: float, margin: float = NPSH_MARGIN) -> str:
    if npsh_a < npsh_r:
        return "cavitation"
    return "marginal" if npsh_a < margin * npsh_r else "ok"


# ---- layout, pump ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    d_suction: float
    l_suction: float
    d_discharge: float
    l_discharge: float
    roughness: float            # absolute, m (same pipe material throughout)
    z_suction: float            # liquid level in the suction tank above the pump centreline (m), + = flooded
    z_discharge: float          # liquid level in the discharge tank above the pump centreline (m)
    n_bends: int = 0            # 90-degree elbows on the discharge side
    n_valves: int = 0
    k_valve: float = 0.0
    n_fittings: int = 0
    k_fitting: float = 0.0
    p_suction: float = 0.0      # gauge pressure on the suction tank surface, Pa (0 = open tank)
    p_discharge: float = 0.0    # gauge pressure on the discharge tank surface, Pa
    site_elevation: float = 0.0     # m above sea level (sets the atmospheric pressure)


MAX_REL_ROUGHNESS = 0.05        # upper end of the Colebrook-White / Moody chart range


def validate_layout(L: Layout) -> Layout:
    require_positive(_finite(L.d_suction, "Suction pipe diameter"), "Suction pipe diameter")
    require_positive(_finite(L.d_discharge, "Discharge pipe diameter"), "Discharge pipe diameter")
    require_positive(_finite(L.l_suction, "Suction pipe length"), "Suction pipe length")
    require_positive(_finite(L.l_discharge, "Discharge pipe length"), "Discharge pipe length")
    require_non_negative(_finite(L.roughness, "Pipe roughness"), "Pipe roughness")
    _finite(L.z_suction, "Suction tank level")
    _finite(L.z_discharge, "Discharge tank level")
    for v, name in ((L.n_bends, "Number of bends"), (L.n_valves, "Number of valves"),
                    (L.n_fittings, "Number of fittings")):
        _count(v, name)
    require_non_negative(_finite(L.k_valve, "Valve loss coefficient"), "Valve loss coefficient")
    require_non_negative(_finite(L.k_fitting, "Fitting loss coefficient"), "Fitting loss coefficient")
    for name, d in (("suction", L.d_suction), ("discharge", L.d_discharge)):
        if L.roughness / d > MAX_REL_ROUGHNESS:
            raise ValidationError(f"The pipe roughness is too large for the {name} pipe diameter: roughness/diameter = "
                                  f"{L.roughness / d:.3f} is above {MAX_REL_ROUGHNESS:g}, the limit of the friction-factor "
                                  "equation. Lower the roughness or use a larger pipe.")
    p_atm = atm_pressure(L.site_elevation)
    for p, name in ((L.p_suction, "Suction tank pressure"), (L.p_discharge, "Discharge tank pressure")):
        if _finite(p, name) <= -p_atm:
            raise ValidationError(f"{name} must be above a perfect vacuum: gauge pressure must be greater than "
                                  f"{-p_atm / 1000:.1f} kPa at this elevation.")
    return L


@dataclass(frozen=True)
class PumpData:
    """Quadratic curve H = H0 - a Q^2 at rated speed, through the shutoff head and the design point."""
    h_shutoff: float
    q_design: float
    h_design: float
    npshr_design: float         # NPSH required at the design flow, m
    rpm_rated: float
    eta_bep: float = 0.75       # pump efficiency at the best-efficiency (= design) flow, assumed
    n_pumps: int = 1            # identical pumps, all at the same speed
    arrangement: str = "parallel"   # how n_pumps > 1 are connected: "parallel" or "series"

    def a(self) -> float:
        """Coefficient of one pump: H = H0 - a Q^2."""
        return (self.h_shutoff - self.h_design) / self.q_design**2


ARRANGEMENTS = ("parallel", "series")
MAX_PUMPS = 4


def validate_pump(p: PumpData) -> PumpData:
    require_positive(_finite(p.h_design, "Pump design head"), "Pump design head")
    require_positive(_finite(p.q_design, "Pump design flow"), "Pump design flow")
    require_less_than(_finite(p.h_design, "Pump design head"), "Pump design head",
                      _finite(p.h_shutoff, "Pump shutoff head"), "shutoff head")
    require_positive(_finite(p.npshr_design, "NPSH required"), "NPSH required")
    require_positive(_finite(p.rpm_rated, "Rated speed"), "Rated speed")
    require_in_range(_finite(p.eta_bep, "Best pump efficiency"), "Best pump efficiency", 1e-6, 1.0)
    n = _finite(p.n_pumps, "Number of pumps")
    if n != int(n) or not 1 <= n <= MAX_PUMPS:
        raise ValidationError(f"Number of pumps must be a whole number from 1 to {MAX_PUMPS}.")
    if p.arrangement not in ARRANGEMENTS:
        raise ValidationError("Pump arrangement must be 'parallel' or 'series'.")
    return p


def speed_ratio(p: PumpData, rpm: float) -> float:
    return require_in_range(_finite(rpm, "Pump speed"), "Pump speed", 0.2 * p.rpm_rated, 1.5 * p.rpm_rated) / p.rpm_rated


def curve_coeffs(p: PumpData) -> tuple:
    """(H0, a) of the equivalent single quadratic curve H = H0 - a Q^2 in the total flow Q, at rated speed.
    n identical pumps in parallel: each carries Q/n at the same head, so H = H0 - a (Q/n)^2 -> (H0, a/n^2).
    In series: each carries Q and adds its head, so H = n (H0 - a Q^2) -> (n H0, n a)."""
    n = p.n_pumps
    if n == 1:
        return p.h_shutoff, p.a()
    if p.arrangement == "parallel":
        return p.h_shutoff, p.a() / n**2
    return n * p.h_shutoff, n * p.a()


def flow_per_pump(p: PumpData, Q):
    """Flow through each pump: Q / n in parallel, Q in series (and for one pump)."""
    Q = np.asarray(Q, dtype=float)
    return Q / p.n_pumps if p.arrangement == "parallel" else Q


def pump_head(p: PumpData, Q, r: float = 1.0):
    """Total head at speed ratio r. Affinity laws (Q ~ N, H ~ N^2) turn H0 - aQ^2 into r^2 H0 - aQ^2."""
    Q = np.asarray(Q, dtype=float)
    h0, a = curve_coeffs(p)
    return r**2 * h0 - a * Q**2


def pump_q_max(p: PumpData, r: float = 1.0) -> float:
    """Total flow at which the head falls to zero."""
    h0, a = curve_coeffs(p)
    return math.sqrt(r**2 * h0 / a)


def pump_npshr(p: PumpData, Q, r: float = 1.0):
    """NPSH required: rated-speed curve r0 + (rd - r0)(Q/Qd)^2 with r0 = 0.5 rd (assumed), scaled by affinity (~N^2)."""
    Q = flow_per_pump(p, Q)                                  # each pump sees its own flow
    r0 = 0.5 * p.npshr_design
    return r**2 * r0 + (p.npshr_design - r0) * (Q / p.q_design) ** 2


ETA_FLOOR = 0.05                # efficiency is never taken below 5 % (keeps power finite at very low or very high flow)
BEP_LOW, BEP_HIGH = 0.7, 1.2    # usual preferred operating range, fraction of best-efficiency flow (rule of thumb)


def pump_efficiency(p: PumpData, Q, r: float = 1.0):
    """Assumed efficiency curve eta = eta_bep (2x - x^2), x = Q / (r Q_design). Peaks at eta_bep when x = 1 and, because
    x uses the speed-scaled flow, keeps its shape under the affinity laws. Floored at ETA_FLOOR."""
    x = flow_per_pump(p, Q) / (r * p.q_design)             # per-pump flow, so a pair has the same eta curve per pump
    return np.maximum(p.eta_bep * (2.0 * x - x**2), ETA_FLOOR)


# ---- pump curve fitted from entered points -----------------------------------------------------
@dataclass(frozen=True)
class PumpFit:
    h_shutoff: float        # m, intercept
    a: float                # m per (m^3/s)^2
    rms: float              # m, root-mean-square residual
    max_resid: float        # m, largest absolute residual
    n_points: int

    def q_zero_head(self) -> float:
        return math.sqrt(self.h_shutoff / self.a)


def fit_pump_points(flows, heads) -> PumpFit:
    """Least-squares fit of H = H0 - a Q^2 to (flow, head) points at rated speed: ordinary linear regression of H on x = Q^2
    (slope = -a, intercept = H0). Flows in m^3/s, heads in m."""
    try:
        q = np.asarray(flows, dtype=float)
        h = np.asarray(heads, dtype=float)
    except (TypeError, ValueError):
        raise ValidationError("Pump points must be numbers.") from None
    if q.ndim != 1 or q.shape != h.shape:
        raise ValidationError("Each pump point needs one flow and one head.")
    if len(q) < 3:
        raise ValidationError("Enter at least 3 pump points to fit a curve.")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(h))):
        raise ValidationError("Pump points must be finite numbers; every row needs both a flow and a head.")
    if np.any(q < 0):
        raise ValidationError("Pump point flows cannot be negative.")
    if np.any(h <= 0):
        raise ValidationError("Pump point heads must be greater than zero.")
    if len(np.unique(q)) < 2:
        raise ValidationError("Pump points need at least 2 different flows.")
    x = q**2
    sxx = float(np.sum((x - x.mean()) ** 2))
    slope = float(np.sum((x - x.mean()) * (h - h.mean())) / sxx)
    a, h0 = -slope, float(h.mean() - slope * x.mean())
    if a <= 0:
        raise ValidationError("The points do not show head falling with flow, so no pump curve can be fitted. "
                              "Check the entered values.")
    res = h - (h0 - a * x)
    return PumpFit(h0, a, float(math.sqrt(np.mean(res**2))), float(np.max(np.abs(res))), len(q))


def pump_from_fit(fit: PumpFit, q_design: float, npshr_design: float, rpm_rated: float, eta_bep: float = 0.75,
                  n_pumps: int = 1, arrangement: str = "parallel") -> PumpData:
    """PumpData whose curve is the fitted one; the head at the design flow comes from the fit."""
    require_positive(_finite(q_design, "Pump design flow"), "Pump design flow")
    if q_design >= fit.q_zero_head():
        raise ValidationError(f"The design flow ({q_design * 1000:.1f} L/s) is beyond the flow where the fitted head reaches "
                              f"zero ({fit.q_zero_head() * 1000:.1f} L/s). Lower the design flow.")
    return PumpData(fit.h_shutoff, q_design, fit.h_shutoff - fit.a * q_design**2, npshr_design, rpm_rated, eta_bep,
                    n_pumps, arrangement)


# ---- sections, system curve, operating point -------------------------------------------------
@dataclass(frozen=True)
class Section:
    name: str
    side: str               # "suction" or "discharge"
    diameter: float
    velocity: float
    Re: Optional[float]     # pipes only
    f: Optional[float]
    head_loss: float        # m


def _pipe(name, side, Q, D, Lp, eps, fl: Fluid) -> Section:
    V = velocity(Q, D)
    Re = reynolds(fl.rho, V, D, fl.mu)
    f = friction_factor(Re, eps / D)
    return Section(name, side, D, V, Re, f, darcy_head(f, Lp, D, V))


def _minor(name, side, Q, D, K) -> Section:
    V = velocity(Q, D)
    return Section(name, side, D, V, None, None, minor_head(K, V))


def sections_at(Q: float, L: Layout, fl: Fluid) -> list:
    """All loss sections in flow order (suction side then discharge side) at flow Q (m^3/s)."""
    if Q <= 0:
        return []
    s = [_minor("Suction entrance", "suction", Q, L.d_suction, K_ENTRANCE),
         _pipe("Suction pipe", "suction", Q, L.d_suction, L.l_suction, L.roughness, fl),
         _pipe("Discharge pipe", "discharge", Q, L.d_discharge, L.l_discharge, L.roughness, fl)]
    for name, n, k in (("Bends", L.n_bends, K_ELBOW_90), ("Valves", L.n_valves, L.k_valve),
                       ("Fittings", L.n_fittings, L.k_fitting)):
        if n > 0:
            s.append(_minor(name, "discharge", Q, L.d_discharge, n * k))
    s.append(_minor("Discharge exit", "discharge", Q, L.d_discharge, K_EXIT))
    return s


def static_head(L: Layout, fl: Optional[Fluid] = None) -> float:
    """Elevation difference plus the tank pressure difference as a head: (z_d - z_s) + (p_d - p_s) / (rho g)."""
    dp = L.p_discharge - L.p_suction
    if dp == 0:
        return L.z_discharge - L.z_suction
    if fl is None:
        raise ValidationError("A fluid is needed to turn the tank pressure difference into a head.")
    return L.z_discharge - L.z_suction + dp / (fl.rho * G)


def system_head(Q: float, L: Layout, fl: Fluid) -> float:
    return static_head(L, fl) + sum(s.head_loss for s in sections_at(Q, L, fl))


def solve_operating_point(h_pump: Callable[[float], float], h_sys: Callable[[float], float],
                          q_max: float, iterations: int = 80) -> Optional[float]:
    """Flow where pump head equals system head, by bisection on [0, q_max]; None if there is no crossing."""
    lo, hi = 0.0, q_max
    f = lambda q: h_pump(q) - h_sys(q)
    if f(lo) <= 0 or f(hi) >= 0:
        return None
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ProfileNode:
    label: str
    head: float             # total head, m (elevation + pressure + velocity)
    velocity: float
    pressure: float         # gauge pressure at the pump-centreline elevation, Pa


@dataclass
class Result:
    fluid: Fluid
    ratio: float
    static_head: float
    q: np.ndarray               # curve grid, m^3/s
    h_pump: np.ndarray
    h_sys: np.ndarray
    npsha_curve: np.ndarray
    npshr_curve: np.ndarray
    q_op: Optional[float]
    h_op: Optional[float]
    sections: list              # at the operating point (empty when there is none)
    dp: list                    # pressure drop of each section, Pa
    share: list                 # fraction of the total friction + fitting drop
    bottleneck: Optional[str]
    npsh_a: Optional[float]
    npsh_r: Optional[float]
    status: Optional[str]
    transitional: list          # names of pipe sections in the transitional regime
    profile: list


def analyse(L: Layout, pump: PumpData, rpm: float, temp_c: float = 20.0, n_points: int = 121,
            fluid: Optional[Fluid] = None) -> Result:
    """fluid: optional custom liquid (see custom_fluid); when given, temp_c is ignored."""
    validate_layout(L)
    validate_pump(pump)
    r = speed_ratio(pump, rpm)
    fl = water(temp_c) if fluid is None else fluid
    p_surface = atm_pressure(L.site_elevation) + L.p_suction          # absolute pressure on the suction tank surface
    q_max = pump_q_max(pump, r)
    q = np.linspace(0.0, q_max, n_points)
    h_sys = np.array([system_head(x, L, fl) for x in q])
    h_pump = pump_head(pump, q, r)
    suction_loss = lambda x: sum(s.head_loss for s in sections_at(x, L, fl) if s.side == "suction")
    npsha_c = np.array([npsha(p_surface, fl.pv, fl.rho, L.z_suction, suction_loss(x)) for x in q])
    npshr_c = pump_npshr(pump, q, r)

    q_op = solve_operating_point(lambda x: float(pump_head(pump, x, r)), lambda x: system_head(x, L, fl), q_max)
    if q_op is None:
        return Result(fl, r, static_head(L, fl), q, h_pump, h_sys, npsha_c, npshr_c, None, None, [], [], [], None,
                      None, None, None, [], [])
    h_op = float(pump_head(pump, q_op, r))
    secs = sections_at(q_op, L, fl)
    dp = [fl.rho * G * s.head_loss for s in secs]
    total = sum(dp)
    share = [d / total for d in dp] if total > 0 else [0.0] * len(dp)
    bottleneck = secs[int(np.argmax(dp))].name
    na = npsha(p_surface, fl.pv, fl.rho, L.z_suction, suction_loss(q_op))
    nr = float(pump_npshr(pump, q_op, r))
    trans = [s.name for s in secs if s.Re is not None and RE_LAMINAR <= s.Re <= RE_TURBULENT]

    # energy profile along the path: tank -> suction sections -> pump -> discharge sections -> tank
    node = lambda label, head, v: ProfileNode(label, head, v, fl.rho * G * (head - v**2 / (2 * G)))
    v_s, v_d = velocity(q_op, L.d_suction), velocity(q_op, L.d_discharge)
    head = L.z_suction + L.p_suction / (fl.rho * G)        # total head at the suction tank surface
    prof = [node("Suction tank surface", head, 0.0)]
    for s in secs:
        if s.side == "suction":
            head -= s.head_loss
            prof.append(node(s.name, head, s.velocity))
    prof.append(node("Pump inlet", head, v_s))
    head += h_op
    prof.append(node("Pump outlet", head, v_d))
    for s in secs:
        if s.side == "discharge":
            head -= s.head_loss
            prof.append(node(s.name, head, 0.0 if s.name == "Discharge exit" else s.velocity))
    return Result(fl, r, static_head(L, fl), q, h_pump, h_sys, npsha_c, npshr_c, q_op, h_op, secs, dp, share,
                  bottleneck, na, nr, cavitation_status(na, nr), trans, prof)


# ---- power, energy cost, throttle vs variable speed -----------------------------------------
@dataclass(frozen=True)
class EnergyInputs:
    hours_per_year: float
    price_per_kwh: float
    eta_motor: float
    eta_vfd: float = 0.97


def validate_energy(e: EnergyInputs) -> EnergyInputs:
    require_in_range(_finite(e.hours_per_year, "Running hours per year"), "Running hours per year", 1.0, 8760.0)
    require_non_negative(_finite(e.price_per_kwh, "Electricity price"), "Electricity price")
    require_in_range(_finite(e.eta_motor, "Motor efficiency"), "Motor efficiency", 1e-6, 1.0)
    require_in_range(_finite(e.eta_vfd, "Drive efficiency"), "Drive efficiency", 1e-6, 1.0)
    return e


@dataclass(frozen=True)
class PowerResult:
    eta: float          # pump efficiency
    hydraulic: float    # W, rho g Q H
    shaft: float        # W, hydraulic / eta
    electrical: float   # W, shaft / (motor efficiency x drive efficiency)


def operating_power(p: PumpData, Q: float, H: float, r: float, rho: float, eta_motor: float,
                    eta_drive: float = 1.0) -> PowerResult:
    eta = float(pump_efficiency(p, Q, r))
    hyd = rho * G * Q * H
    shaft = hyd / eta
    return PowerResult(eta, hyd, shaft, shaft / (eta_motor * eta_drive))


def annual_energy(power_w: float, hours_per_year: float, price_per_kwh: float) -> tuple:
    """(kWh per year, cost per year) for a constant electrical power."""
    kwh = power_w / 1000.0 * hours_per_year
    return kwh, kwh * price_per_kwh


@dataclass(frozen=True)
class ControlResult:
    q_target: float             # m^3/s
    h_system: float             # m, head the system needs at q_target
    h_pump_throttle: float      # m, head the pump delivers at q_target at the current speed
    h_valve: float              # m, head the throttle valve has to burn
    ratio_vfd: float            # speed ratio (to rated) that delivers q_target with no valve
    rpm_vfd: float
    throttle: PowerResult
    vfd: PowerResult
    saving_w: float             # electrical, throttle minus VFD
    saving_frac: float
    kwh_throttle: float
    kwh_vfd: float
    cost_throttle: float
    cost_vfd: float


def compare_control(r: Result, L: Layout, pump: PumpData, frac: float, energy: EnergyInputs) -> ControlResult:
    """Deliver frac x (operating flow) two ways: (1) throttle valve, pump stays at the current speed; (2) variable-speed drive,
    no valve. VFD speed from H_sys(Q) = ratio^2 H0 - a Q^2, i.e. ratio = sqrt((H_sys + a Q^2) / H0)."""
    validate_energy(energy)
    if r.q_op is None:
        raise ValidationError("There is no operating point to compare against.")
    require_in_range(_finite(frac, "Target flow fraction"), "Target flow fraction", 0.05, 1.0)
    q = frac * r.q_op
    h_sys = system_head(q, L, r.fluid)
    h_thr = float(pump_head(pump, q, r.ratio))
    h0_eq, a_eq = curve_coeffs(pump)
    ratio_v = math.sqrt((h_sys + a_eq * q**2) / h0_eq)
    thr = operating_power(pump, q, h_thr, r.ratio, r.fluid.rho, energy.eta_motor)
    vfd = operating_power(pump, q, h_sys, ratio_v, r.fluid.rho, energy.eta_motor, energy.eta_vfd)
    kt, ct = annual_energy(thr.electrical, energy.hours_per_year, energy.price_per_kwh)
    kv, cv = annual_energy(vfd.electrical, energy.hours_per_year, energy.price_per_kwh)
    return ControlResult(q, h_sys, h_thr, max(h_thr - h_sys, 0.0), ratio_v, ratio_v * pump.rpm_rated, thr, vfd,
                         thr.electrical - vfd.electrical, (thr.electrical - vfd.electrical) / thr.electrical,
                         kt, kv, ct, cv)


# =============================================================================================
# FIGURES
# =============================================================================================
GREY, AMBER = "#6b7280", "#f59e0b"
STATUS_COLOR = {"ok": PASS_COLOR, "marginal": AMBER, "cavitation": FAIL_COLOR}
STATUS_TEXT = {"ok": "No cavitation expected", "marginal": "Cavitation margin is thin",
               "cavitation": "Cavitation risk"}
# Schematic geometry (not to scale): suction tank -> suction pipe -> pump -> discharge route -> discharge tank
SUCTION = [(1.2, 0.0), (4.45, 0.0)]
PUMP_C, PUMP_R = (5.0, 0.0), 0.55
DISCHARGE = [(5.55, 0.0), (8.0, 0.0), (8.0, 3.0), (11.0, 3.0)]


def _route_point(pts, t):
    seg = [math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])]
    d = t * sum(seg)
    for (a, b), ln in zip(zip(pts[:-1], pts[1:]), seg):
        if d <= ln or (a, b) == (pts[-2], pts[-1]):
            u = min(d / ln, 1.0)
            return a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])
        d -= ln


def _colour(v, vmin, vmax):
    t = 0.5 if vmax <= vmin else min(max((v - vmin) / (vmax - vmin), 0.0), 1.0)
    return sample_colorscale(STRESS_COLORSCALE, [t])[0]


def layout_figure(r: Result, lay: Layout, color_by: str, system: str, pump_label: str = "PUMP",
                  pump_note: str = "") -> go.Figure:
    """Schematic (not to scale) coloured by gauge pressure at pump-centreline elevation, or by velocity."""
    prof = {n.label: n for n in r.profile}
    sec = {s.name: s for s in r.sections}
    pres = color_by == "pressure"
    conv = (lambda p: units.from_si(p, "pressure", system)) if pres else (lambda v: units.from_si(v, "velocity", system))
    unit = units.label("pressure" if pres else "velocity", system)
    P = lambda label: conv(prof[label].pressure)                      # pressure nodes (user units)
    Vv = lambda name: conv(sec[name].velocity)                        # section velocities (user units)

    # --- values along each run
    groups = [(n, c) for n, c in (("Bends", lay.n_bends), ("Valves", lay.n_valves), ("Fittings", lay.n_fittings)) if c > 0]
    total_items = sum(c for _, c in groups)
    fr = list(np.linspace(0.12, 0.88, total_items)) if total_items > 1 else [0.5]
    items, k = [], 0                                                  # (group, position along discharge route)
    for g, c in groups:
        for _ in range(c):
            items.append((g, fr[k]))
            k += 1
    prev = {"Bends": "Discharge pipe", "Valves": "Bends" if lay.n_bends else "Discharge pipe",
            "Fittings": "Valves" if lay.n_valves else ("Bends" if lay.n_bends else "Discharge pipe")}
    drop = {g: (P(prev[g]) - P(g)) / c for g, c in groups}            # pressure drop per item (pressure mode)
    p_out, p_pipe_end = P("Pump outlet"), P("Discharge pipe")
    d_val = lambda t: (p_out - (p_out - p_pipe_end) * t - sum(drop[g] for g, pos in items if pos <= t)) if pres \
        else Vv("Discharge pipe")
    s_val = lambda t: (P("Suction entrance") + (P("Suction pipe") - P("Suction entrance")) * t) if pres \
        else Vv("Suction pipe")

    vals = [d_val(t) for t in np.linspace(0, 1, 30)] + [s_val(t) for t in np.linspace(0, 1, 10)]
    vals += [P("Suction tank surface")] if pres else [0.0]
    vmin, vmax = (min(vals), max(vals)) if pres else (0.0, max(vals))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0

    dmax = max(lay.d_suction, lay.d_discharge)
    width = lambda D: 5 + 15 * D / dmax
    fig = go.Figure()
    # bottleneck highlight underneath
    bn = r.bottleneck
    if bn in ("Suction pipe", "Discharge pipe"):
        pts, D = (SUCTION, lay.d_suction) if bn == "Suction pipe" else (DISCHARGE, lay.d_discharge)
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", hoverinfo="skip",
                                 line=dict(color="rgba(198,40,40,0.35)", width=width(D) + 14), showlegend=False))
    # pipes: many short coloured segments
    for runs, valf, D, n in ((SUCTION, s_val, lay.d_suction, 10), (DISCHARGE, d_val, lay.d_discharge, 60)):
        ts = np.linspace(0, 1, n + 1)
        for a, b in zip(ts[:-1], ts[1:]):
            (x0, y0), (x1, y1) = _route_point(runs, a), _route_point(runs, b)
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=_colour(valf((a + b) / 2), vmin, vmax), width=width(D), shape="linear")))
    # fitting markers
    symbol = {"Bends": "diamond", "Valves": "square", "Fittings": "triangle-up"}
    for g, pos in items:
        x, y = _route_point(DISCHARGE, pos)
        ring = g == bn
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", showlegend=False,
                                 marker=dict(symbol=symbol[g], size=16, color=_colour(d_val(pos), vmin, vmax),
                                             line=dict(width=3 if ring else 1.5, color=FAIL_COLOR if ring else "#222")),
                                 hovertemplate=f"{g[:-1] if g != 'Fittings' else 'Fitting'}: head loss {sec[g].head_loss / (dict(groups)[g]):.2f} m<extra></extra>"))
    for name, (x, y), sym in (("Suction entrance", SUCTION[0], "triangle-right"), ("Discharge exit", DISCHARGE[-1], "triangle-right")):
        v = (P("Suction entrance") if name == "Suction entrance" else P("Discharge exit")) if pres else Vv(name)
        ring = name == bn
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", showlegend=False,
                                 marker=dict(symbol=sym, size=16, color=_colour(v, vmin, vmax),
                                             line=dict(width=3 if ring else 1.5, color=FAIL_COLOR if ring else "#222")),
                                 hovertemplate=f"{name}: head loss {sec[name].head_loss:.2f} m<extra></extra>"))
    # tanks, pump
    fig.add_shape(type="rect", x0=0, x1=1.2, y0=-1.3, y1=1.3, fillcolor="rgba(31,119,180,0.18)", line=dict(color="#555", width=2))
    fig.add_shape(type="rect", x0=11.0, x1=12.5, y0=1.8, y1=4.2, fillcolor="rgba(31,119,180,0.18)", line=dict(color="#555", width=2))
    col = STATUS_COLOR[r.status]
    fig.add_shape(type="circle", x0=PUMP_C[0] - PUMP_R, x1=PUMP_C[0] + PUMP_R, y0=-PUMP_R, y1=PUMP_R,
                  fillcolor="white", line=dict(color=col, width=4))
    u_h = units.label("length_long", system)
    fig.add_annotation(x=PUMP_C[0], y=0, text=f"<b>{pump_label}</b><br>{units.from_si(r.h_op, 'length_long', system):.1f} {u_h}",
                       showarrow=False, font=dict(size=11))
    fig.add_annotation(x=0.6, y=-1.3, text="Suction tank", showarrow=False, yshift=-12)
    fig.add_annotation(x=11.75, y=1.8, text="Discharge tank", showarrow=False, yshift=-12)
    if pump_note:
        fig.add_annotation(x=PUMP_C[0], y=PUMP_R, text=pump_note, showarrow=False, yshift=14, font=dict(size=12))
    fig.add_annotation(x=PUMP_C[0], y=-PUMP_R, text=STATUS_TEXT[r.status], showarrow=False, yshift=-14,
                       font=dict(color=col, size=12))
    share = r.share[int(np.argmax(r.dp))]
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top", showarrow=False,
                       text=f"<b>Bottleneck: {bn}</b> ({share * 100:.0f} % of the friction + fitting pressure drop)",
                       font=dict(color=FAIL_COLOR, size=13))
    # colour bar
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(colorscale=STRESS_COLORSCALE, cmin=vmin, cmax=vmax, color=[vmin], showscale=True,
                                         colorbar=dict(title=f"{'Pressure' if pres else 'Velocity'} ({unit})", len=0.7))))
    apply_theme(fig, f"Pipe layout coloured by {'gauge pressure' if pres else 'velocity'} (schematic, not to scale)", 430)
    fig.update_xaxes(visible=False, range=[-0.4, 12.9])
    fig.update_yaxes(visible=False, range=[-2.0, 5.0], scaleanchor="x", scaleratio=1)
    return fig


def curves_figure(r: Result, pump: PumpData, system: str, points=None) -> go.Figure:
    qu, hu = units.label("flow", system), units.label("length_long", system)
    Lq, Lh = (lambda v: units.from_si(v, "flow", system)), (lambda v: units.from_si(v, "length_long", system))
    fig = go.Figure()
    if abs(r.ratio - 1.0) > 1e-9:
        qr = np.linspace(0, pump_q_max(pump, 1.0), 100)
        fig.add_trace(go.Scatter(x=Lq(qr), y=Lh(pump_head(pump, qr, 1.0)), mode="lines", name="Pump at rated speed",
                                 line=dict(color=GREY, dash="dot", width=2)))
    if pump.n_pumps > 1:
        one = replace(pump, n_pumps=1)
        q1 = np.linspace(0, pump_q_max(one, r.ratio), 100)
        fig.add_trace(go.Scatter(x=Lq(q1), y=Lh(pump_head(one, q1, r.ratio)), mode="lines", name="One pump (this speed)",
                                 line=dict(color=GREY, dash="dot", width=2)))
    name = "Pump curve (this speed)" if pump.n_pumps == 1 else f"{pump.n_pumps} pumps in {pump.arrangement} (this speed)"
    fig.add_trace(go.Scatter(x=Lq(r.q), y=Lh(r.h_pump), mode="lines", name=name, line=dict(color=ACCENT, width=3)))
    if points is not None:
        fig.add_trace(go.Scatter(x=Lq(points[0]), y=Lh(points[1]), mode="markers", name="Entered points (rated speed)",
                                 marker=dict(size=10, color=AMBER, symbol="diamond", line=dict(width=1.5, color="#222"))))
    fig.add_trace(go.Scatter(x=Lq(r.q), y=Lh(r.h_sys), mode="lines", name="System curve",
                             line=dict(color="#d97706", width=3)))
    fig.add_hline(y=Lh(r.static_head), line=dict(color=GREY, dash="dash", width=1), annotation_text="static head",
                  annotation_position="top left")
    if r.q_op is not None:
        fig.add_trace(go.Scatter(x=[Lq(r.q_op)], y=[Lh(r.h_op)], mode="markers+text", name="Operating point",
                                 marker=dict(size=14, color=STATUS_COLOR[r.status], line=dict(width=2, color="#222")),
                                 text=[f"  {Lq(r.q_op):.1f} {qu}, {Lh(r.h_op):.1f} {hu}"], textposition="top right",
                                 hovertemplate="%{x:.2f}, %{y:.2f}<extra></extra>"))
    else:
        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=FAIL_COLOR, size=14),
                           text="<b>Curves do not cross: no operating point</b>")
    apply_theme(fig, "Pump curve vs system curve", 400)
    fig.update_xaxes(title=f"Flow ({qu})", rangemode="tozero")
    fig.update_yaxes(title=f"Head ({hu})", range=[0, 1.25 * float(Lh(r.h_pump.max()))])   # keep the steep system curve from squashing the pump curve
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return fig


def npsh_figure(r: Result, system: str) -> go.Figure:
    qu, hu = units.label("flow", system), units.label("length_long", system)
    Lq, Lh = (lambda v: units.from_si(v, "flow", system)), (lambda v: units.from_si(v, "length_long", system))
    fig = go.Figure([
        go.Scatter(x=Lq(r.q), y=Lh(r.npsha_curve), mode="lines", name="NPSH available", line=dict(color=ACCENT, width=3)),
        go.Scatter(x=Lq(r.q), y=Lh(r.npshr_curve), mode="lines", name="NPSH required", line=dict(color=FAIL_COLOR, width=3)),
        go.Scatter(x=Lq(r.q), y=Lh(NPSH_MARGIN * r.npshr_curve), mode="lines", name=f"{NPSH_MARGIN:g} × NPSH required",
                   line=dict(color=AMBER, width=2, dash="dot"))])
    if r.q_op is not None:
        fig.add_vline(x=Lq(r.q_op), line=dict(color=GREY, dash="dash", width=1))
        fig.add_trace(go.Scatter(x=[Lq(r.q_op)], y=[Lh(r.npsh_a)], mode="markers", name="Operating point",
                                 marker=dict(size=13, color=STATUS_COLOR[r.status], line=dict(width=2, color="#222"))))
    apply_theme(fig, "NPSH available vs required (cavitation screening)", 340)
    fig.update_xaxes(title=f"Flow ({qu})", rangemode="tozero")
    top = max(float(r.npsha_curve[0]), NPSH_MARGIN * float(r.npshr_curve.max())) * 1.15
    low = min(0.0, r.npsh_a if r.npsh_a is not None else 0.0) - 0.1 * top        # a steep fall of NPSHa must not flatten the rest
    fig.update_yaxes(title=f"NPSH ({hu})", range=[float(Lh(low)), float(Lh(top))])
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return fig


def efficiency_figure(r: Result, pump: PumpData, system: str) -> go.Figure:
    qu = units.label("flow", system)
    Lq = lambda v: units.from_si(v, "flow", system)
    share = pump.n_pumps if pump.arrangement == "parallel" else 1     # total flow per unit of per-pump flow
    q_bep = r.ratio * pump.q_design * share                        # best-efficiency flow, total
    x_op = r.q_op / q_bep
    inside = BEP_LOW <= x_op <= BEP_HIGH
    fig = go.Figure()
    fig.add_vrect(x0=float(Lq(BEP_LOW * q_bep)), x1=float(Lq(BEP_HIGH * q_bep)), fillcolor="rgba(46,125,50,0.12)",
                  line_width=0, annotation_text="usual preferred range (70–120 % of BEP flow)",
                  annotation_position="top left", annotation_font_size=11)
    xq = r.q / q_bep
    raw = pump.eta_bep * (2 * xq - xq**2)                               # hide the 5 % floor so the curve ends where it falls to it
    fig.add_trace(go.Scatter(x=Lq(r.q), y=np.where(raw > ETA_FLOOR, 100 * pump_efficiency(pump, r.q, r.ratio), np.nan), mode="lines",
                             name="Pump efficiency (assumed shape)", line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=[Lq(r.q_op)], y=[100 * float(pump_efficiency(pump, r.q_op, r.ratio))], mode="markers",
                             name="Operating point", marker=dict(size=14, color=PASS_COLOR if inside else AMBER,
                                                                 line=dict(width=2, color="#222"))))
    apply_theme(fig, "Pump efficiency at this speed (assumed curve, not manufacturer data)", 320)
    fig.update_xaxes(title=f"Flow ({qu})", rangemode="tozero")
    fig.update_yaxes(title="Efficiency (%)", range=[0, 100 * pump.eta_bep * 1.2])
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return fig


def control_figure(c: ControlResult, r: Result, pump: PumpData, system: str) -> go.Figure:
    qu, hu = units.label("flow", system), units.label("length_long", system)
    Lq, Lh = (lambda v: units.from_si(v, "flow", system)), (lambda v: units.from_si(v, "length_long", system))
    qv = np.linspace(0.0, pump_q_max(pump, c.ratio_vfd), 100)
    q = float(Lq(c.q_target))
    fig = go.Figure([
        go.Scatter(x=Lq(r.q), y=Lh(r.h_sys), mode="lines", name="System curve", line=dict(color="#d97706", width=3)),
        go.Scatter(x=Lq(r.q), y=Lh(r.h_pump), mode="lines", name="Pump, current speed", line=dict(color=ACCENT, width=3)),
        go.Scatter(x=Lq(qv), y=Lh(pump_head(pump, qv, c.ratio_vfd)), mode="lines", name="Pump, variable speed",
                   line=dict(color=PASS_COLOR, width=3, dash="dash"))])
    fig.add_shape(type="line", x0=q, x1=q, y0=float(Lh(c.h_system)), y1=float(Lh(c.h_pump_throttle)),
                  line=dict(color=FAIL_COLOR, width=4))
    fig.add_annotation(x=q, y=float(Lh((c.h_system + c.h_pump_throttle) / 2)), xanchor="left", showarrow=False, xshift=8,
                       text=f"<b>valve burns {float(Lh(c.h_valve)):.1f} {hu}</b>", font=dict(color=FAIL_COLOR, size=12),
                       bgcolor="rgba(128,128,128,0.18)")
    fig.add_trace(go.Scatter(x=[q], y=[float(Lh(c.h_pump_throttle))], mode="markers", name="Throttled point",
                             marker=dict(size=13, color=AMBER, line=dict(width=2, color="#222"))))
    fig.add_trace(go.Scatter(x=[q], y=[float(Lh(c.h_system))], mode="markers", name="Variable-speed point",
                             marker=dict(size=13, color=PASS_COLOR, line=dict(width=2, color="#222"))))
    apply_theme(fig, f"Same flow ({q:.1f} {qu}): throttle valve vs variable speed", 400)
    fig.update_xaxes(title=f"Flow ({qu})", rangemode="tozero")
    fig.update_yaxes(title=f"Head ({hu})", range=[0, 1.25 * float(Lh(r.h_pump.max()))])
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return fig


# =============================================================================================
# UI
# =============================================================================================
def _slider(label, kind, lo, hi, default, key, system):
    lo_u, hi_u, d_u = (units.from_si(v, kind, system) for v in (lo, hi, default))
    v = slider_box(f"{label} ({units.label(kind, system)})", lo_u, hi_u, d_u, f"{key}_{system}")
    return units.to_si(v, kind, system)


def _metric(col, label: str, value: str, note: str = "") -> None:
    """Metric with a plain caption underneath (st.metric's delta would draw a misleading up-arrow)."""
    col.metric(label, value)
    if note:
        col.caption(note)


def _num(label, kind, default, key, system, step_si, fmt="%.2f"):
    v = st.number_input(f"{label} ({units.label(kind, system)})", value=float(units.from_si(default, kind, system)),
                        step=float(units.from_si(step_si, kind, system)), format=fmt, key=f"{key}_{system}")
    return units.to_si(v, kind, system)


def render() -> None:
    system = get_unit_system()
    st.caption("Illustrative example values, not from a real project. Pump data and loss coefficients are generic "
               "textbook-style numbers. Water or a custom fluid. System-curve analysis, not CFD.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        example = st.selectbox("Example system (loads the inputs below)", list(EXAMPLES), key="pp_example")
        V = {**BASE_VALUES, **EXAMPLES[example]["values"]}
        ek = (lambda base: base) if example == DEFAULT_EXAMPLE else (lambda base: f"{base}_e{list(EXAMPLES).index(example)}")
        st.caption(f"Illustrative example, not from a real project. {EXAMPLES[example]['note']} Edit any input afterwards.")
        st.subheader("System")
        mats = list(ROUGHNESS) + [CUSTOM_ROUGH]
        mat = st.selectbox("Pipe material (roughness)", mats, index=mats.index(V["material"]), key=ek("pp_material"))
        if mat == CUSTOM_ROUGH:
            eps = _num("Pipe roughness", "length", 0.045e-3, ek("pp_eps"), system, 0.005e-3, "%.4f")
        d_s = _slider("Suction pipe diameter", "length", 0.040, 0.250, V["d_s"], ek("pp_ds"), system)
        l_s = _slider("Suction pipe length", "length_long", 0.5, 20.0, V["l_s"], ek("pp_ls"), system)
        z_s = _slider("Suction tank level above pump (− = lift)", "length_long", -6.0, 6.0, V["z_s"], ek("pp_zs"), system)
        d_d = _slider("Discharge pipe diameter", "length", 0.040, 0.250, V["d_d"], ek("pp_dd"), system)
        l_d = _slider("Discharge pipe length", "length_long", 5.0, 300.0, V["l_d"], ek("pp_ld"), system)
        z_d = _slider("Discharge tank level above pump", "length_long", 0.0, 40.0, V["z_d"], ek("pp_zd"), system)
        n_b = st.slider("90° bends on the discharge side", 0, 12, V["bends"], key=ek("pp_bends"))
        vt, vn = st.columns([2, 1])
        valves = list(VALVE_K) + [CUSTOM_K]
        valve = vt.selectbox("Valve type", valves, index=valves.index(V["valve"]), key=ek("pp_valve"))
        n_v = vn.slider("Valves", 0, 4, V["n_valves"], key=ek("pp_nvalve"))
        if valve == CUSTOM_K:
            kv = st.number_input("Valve loss coefficient K", 0.0, 1000.0, 2.0, 0.1, key=ek("pp_kvalve"), format="%.2f")
        ft, fn = st.columns([2, 1])
        fittings = list(FITTING_K) + [CUSTOM_K]
        fitting = ft.selectbox("Fitting type", fittings, index=fittings.index(V["fitting"]), key=ek("pp_fitting"))
        n_f = fn.slider("Fittings", 0, 6, V["n_fit"], key=ek("pp_nfit"))
        if fitting == CUSTOM_K:
            kf = st.number_input("Fitting loss coefficient K", 0.0, 1000.0, 1.0, 0.1, key=ek("pp_kfit"), format="%.2f")
        with st.expander("Tanks and site (illustrative)", expanded=example != DEFAULT_EXAMPLE and bool(V["p_d"] or V["p_s"] or V["elev"])):
            p_s = _num("Suction tank gauge pressure (0 = open tank)", "pressure", V["p_s"], ek("pp_ps"), system, 10e3, "%.1f")
            p_d = _num("Discharge tank gauge pressure (0 = open tank)", "pressure", V["p_d"], ek("pp_pd"), system, 10e3, "%.1f")
            elev = _num("Site elevation above sea level", "length_long", V["elev"], ek("pp_elev"), system, 100.0, "%.0f")
        st.subheader("Pump and fluid")
        pct = st.slider("Pump speed (% of rated speed)", 50, 120, V["speed"], key=ek("pp_speed"))
        fluid_mode = st.radio("Fluid", ["Water", "Custom fluid"], index=["Water", "Custom fluid"].index(V["fluid"]),
                              horizontal=True, key=ek("pp_fluid"))
        temp = st.slider("Water temperature (°C)", 0, 100, V["temp"], key=ek("pp_temp"), disabled=fluid_mode != "Water")
        if fluid_mode != "Water":
            st.caption("The water temperature is ignored for a custom fluid. Defaults are illustrative, roughly a "
                       "30 % glycol-water mix at 20 °C; the properties stay fixed.")
            rho_c = _num("Density", "density", 1040.0, ek("pp_rho"), system, 10.0, "%.1f")
            mu_c = st.number_input("Viscosity (mPa·s = cP)", 0.01, 10000.0, 3.0, 0.1, key=ek("pp_mu"), format="%.2f")
            pv_c = _num("Vapour pressure (absolute)", "pressure", 2e3, ek("pp_pv"), system, 500.0, "%.2f")
        pc1, pc2 = st.columns(2)
        n_p = pc1.selectbox("Identical pumps", [1, 2], index=[1, 2].index(V["n_pumps"]), key=ek("pp_npumps"))
        arr = pc2.selectbox("Connected in", list(ARRANGEMENTS), index=list(ARRANGEMENTS).index(V["arrangement"]),
                            key=ek("pp_arrangement"), disabled=n_p == 1)
        with st.expander("Pump data (illustrative, per pump)"):
            curve_mode = st.radio("Pump curve from", ["Design point", "Fitted from points"], horizontal=True, key=ek("pp_curve_mode"))
            if curve_mode == "Design point":
                h0 = _num("Shutoff head at rated speed", "length_long", V["h0"], ek("pp_h0"), system, 0.5, "%.1f")
            qd = _num("Design flow", "flow", V["qd"], ek("pp_qd"), system, 0.001 if system == units.SI else 6.3e-5, "%.1f")
            if curve_mode == "Design point":
                hd = _num("Head at design flow", "length_long", V["hd"], ek("pp_hd"), system, 0.5, "%.1f")
            else:
                qu_, hu_ = units.label("flow", system), units.label("length_long", system)
                default_pts = pd.DataFrame({f"Flow ({qu_})": [round(units.from_si(v, "flow", system), 1) for v in DEFAULT_POINTS_Q],
                                            f"Head ({hu_})": [round(units.from_si(v, "length_long", system), 1) for v in DEFAULT_POINTS_H]})
                st.caption("Pump points at rated speed (illustrative). Add or remove rows; at least 3.")
                pts_df = st.data_editor(default_pts, num_rows="dynamic", hide_index=True, key=f"{ek('pp_pts')}_{system}", width="stretch")
            nr = _num("NPSH required at design flow", "length_long", V["nr"], ek("pp_nr"), system, 0.1, "%.1f")
            rpm_rated = st.number_input("Rated speed (rpm)", value=1750.0, step=50.0, key=ek("pp_rpm_rated"))
            eta_bep = st.number_input("Best efficiency, at design flow (%)", 1.0, 100.0, 75.0, 1.0, key=ek("pp_eta_bep"),
                                      format="%.0f", help="Assumed efficiency curve peaks here; not manufacturer data.")
        color_by = st.radio("Colour the layout by", ["pressure", "velocity"], format_func=str.capitalize,
                            horizontal=True, key=ek("pp_color"))
        st.caption(f"Pump speed: {pct / 100 * rpm_rated:,.0f} rpm")
        st.subheader("Energy and control")
        target = st.slider("Flow actually needed (% of the operating flow)", 30, 100, 70, key=ek("pp_target"),
                           help="Compares a throttle valve with a variable-speed drive at this lower flow.")
        with st.expander("Energy cost inputs (illustrative)"):
            hours = st.number_input("Running hours per year", 1.0, 8760.0, 4000.0, 100.0, key=ek("pp_hours"), format="%.0f")
            price = st.number_input("Electricity price ($/kWh)", 0.0, 5.0, 0.10, 0.01, key=ek("pp_price"), format="%.3f")
            eta_m = st.number_input("Motor efficiency (%)", 1.0, 100.0, 92.0, 1.0, key=ek("pp_eta_motor"), format="%.0f")
            eta_v = st.number_input("Drive (VFD) efficiency (%)", 1.0, 100.0, 97.0, 1.0, key=ek("pp_eta_vfd"), format="%.0f")

    lay = Layout(d_s, l_s, d_d, l_d, eps if mat == CUSTOM_ROUGH else ROUGHNESS[mat], z_s, z_d, n_b, n_v,
                 kv if valve == CUSTOM_K else VALVE_K[valve], n_f, kf if fitting == CUSTOM_K else FITTING_K[fitting],
                 p_s, p_d, elev)
    fluid = custom_fluid(rho_c, mu_c * 1e-3, pv_c) if fluid_mode != "Water" else None
    fit, pts_si = None, None
    if curve_mode == "Design point":
        pump = PumpData(h0, qd, hd, nr, rpm_rated, eta_bep / 100, n_p, arr)
    else:
        pts = pts_df.dropna(how="all")
        if pts.isna().any().any():
            raise ValidationError("Every pump point needs both a flow and a head.")
        pts_si = (units.to_si(pts.iloc[:, 0].to_numpy(dtype=float), "flow", system),
                  units.to_si(pts.iloc[:, 1].to_numpy(dtype=float), "length_long", system))
        fit = fit_pump_points(*pts_si)
        pump = pump_from_fit(fit, qd, nr, rpm_rated, eta_bep / 100, n_p, arr)
    r = analyse(lay, pump, pct / 100 * rpm_rated, float(temp), fluid=fluid)

    with right:
        fl_text = (f"water at {r.fluid.temp_c:.0f} °C" if fluid is None else
                   f"custom fluid (density {units.fmt(r.fluid.rho, 'density', system, 4)}, viscosity {r.fluid.mu * 1e3:.2f} mPa·s, "
                   f"vapour pressure {units.fmt(r.fluid.pv, 'pressure', system, 3)} abs)")
        st.caption(f"Fluid: {fl_text}. Atmospheric pressure at the site: {units.fmt(atm_pressure(lay.site_elevation), 'pressure', system, 4)}.")
        if fit is not None:
            hu = units.label("length_long", system)
            st.info(f"Fitted pump curve (per pump, rated speed): shutoff head {units.fmt(fit.h_shutoff, 'length_long', system, 3)}, "
                    f"head at the design flow {units.fmt(pump.h_design, 'length_long', system, 3)}, zero head at "
                    f"{units.fmt(fit.q_zero_head(), 'flow', system, 3)}. Residual: RMS {units.from_si(fit.rms, 'length_long', system):.2f} {hu}, "
                    f"largest {units.from_si(fit.max_resid, 'length_long', system):.2f} {hu} over {fit.n_points} points.")
        if r.q_op is None:
            st.error("No operating point: the pump cannot overcome the static head plus losses at this speed "
                     "(the curves do not cross). Raise the pump speed or shutoff head, or lower the static head.")
            st.plotly_chart(curves_figure(r, pump, system, pts_si), width="stretch")
        else:
            msg = f"{STATUS_TEXT[r.status]}: NPSH available {units.fmt(r.npsh_a, 'length_long', system)} vs required " \
                  f"{units.fmt(r.npsh_r, 'length_long', system)} at the operating flow."
            {"ok": st.success, "marginal": st.warning, "cavitation": st.error}[r.status](msg)
            if r.transitional:
                st.warning("Transitional flow (Re 2300–4000) in: " + ", ".join(r.transitional) +
                           ". The friction factor there is only a guide.")
            top = int(np.argmax(r.dp))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Operating flow", units.fmt(r.q_op, "flow", system, 3))
            m2.metric("Pump head", units.fmt(r.h_op, "length_long", system, 3))
            m3.metric("Largest pressure drop", r.bottleneck, f"{units.fmt(r.dp[top], 'pressure', system, 3)}", delta_color="off")
            m4.metric("NPSH margin", f"{r.npsh_a / r.npsh_r:.1f} × required")
            note = "" if pump.n_pumps == 1 else f"{pump.n_pumps} identical pumps in {pump.arrangement}"
            st.plotly_chart(layout_figure(r, lay, color_by, system, "PUMP", note), width="stretch")
            st.plotly_chart(curves_figure(r, pump, system, pts_si), width="stretch")
            st.plotly_chart(npsh_figure(r, system), width="stretch")
            pu, vu, hu = (units.label("pressure", system), units.label("velocity", system), units.label("length_long", system))
            st.dataframe(pd.DataFrame({
                "Section": [s.name + ("  ◀ bottleneck" if s.name == r.bottleneck else "") for s in r.sections],
                f"Velocity ({vu})": [round(units.from_si(s.velocity, "velocity", system), 2) for s in r.sections],
                "Reynolds": [None if s.Re is None else round(s.Re) for s in r.sections],
                "Friction f": [None if s.f is None else round(s.f, 4) for s in r.sections],
                f"Head loss ({hu})": [round(units.from_si(s.head_loss, "length_long", system), 3) for s in r.sections],
                f"Pressure drop ({pu})": [round(units.from_si(d, "pressure", system), 2) for d in r.dp],
                "Share of drop": [round(x * 100, 1) for x in r.share]}),
                hide_index=True, width="stretch",
                column_config={"Share of drop": st.column_config.ProgressColumn("Share of drop (%)", min_value=0, max_value=100, format="%.1f")})
            d1, d2 = st.columns(2)
            d1.download_button("Download sections (CSV)", sections_csv(r, system), "pump_pipe_sections.csv", "text/csv",
                               key="pp_dl_sections")
            d2.download_button("Download curves (CSV)", curves_csv(r, system), "pump_pipe_curves.csv", "text/csv",
                               key="pp_dl_curves")

            st.subheader("Energy and cost at the operating point")
            en = validate_energy(EnergyInputs(hours, price, eta_m / 100, eta_v / 100))
            pw = operating_power(pump, r.q_op, r.h_op, r.ratio, r.fluid.rho, en.eta_motor)
            kwh, cost = annual_energy(pw.electrical, en.hours_per_year, en.price_per_kwh)
            x_bep = float(flow_per_pump(pump, r.q_op)) / (r.ratio * pump.q_design)
            e1, e2, e3, e4 = st.columns(4)
            _metric(e1, "Pump efficiency", f"{pw.eta * 100:.0f} %", f"at {x_bep * 100:.0f} % of best-efficiency flow")
            _metric(e2, "Shaft power", units.fmt(pw.shaft, "power", system, 3))
            _metric(e3, "Electrical input", units.fmt(pw.electrical, "power", system, 3))
            _metric(e4, "Energy cost per year", f"${cost:,.0f}", f"{kwh:,.0f} kWh per year")
            if not BEP_LOW <= x_bep <= BEP_HIGH:
                st.warning(f"Operating at {x_bep * 100:.0f} % of the best-efficiency flow, outside the usual 70–120 % range "
                           "(rule of thumb): efficiency drops there and wear typically rises.")
            st.plotly_chart(efficiency_figure(r, pump, system), width="stretch")

            st.subheader("Throttle valve vs variable-speed drive")
            c = compare_control(r, lay, pump, target / 100, en)
            st.caption(f"To deliver {units.fmt(c.q_target, 'flow', system, 3)} ({target} % of the operating flow), a throttle "
                       f"valve has to burn {units.fmt(c.h_valve, 'length_long', system, 3)} of head. A variable-speed drive "
                       f"runs the pump at {c.rpm_vfd:,.0f} rpm ({c.ratio_vfd * 100:.0f} % of rated) instead.")
            s1, s2, s3, s4 = st.columns(4)
            _metric(s1, "Throttle valve, electrical", units.fmt(c.throttle.electrical, "power", system, 3), f"${c.cost_throttle:,.0f} per year")
            _metric(s2, "Variable speed, electrical", units.fmt(c.vfd.electrical, "power", system, 3), f"${c.cost_vfd:,.0f} per year")
            _metric(s3, "Power saved", units.fmt(c.saving_w, "power", system, 3), f"{c.saving_frac * 100:.0f} % of the throttle power")
            _metric(s4, "Saved per year", f"${c.cost_throttle - c.cost_vfd:,.0f}", f"{c.kwh_throttle - c.kwh_vfd:,.0f} kWh per year")
            st.plotly_chart(control_figure(c, r, pump, system), width="stretch")
        with st.expander("Method and limits"):
            st.markdown(
                "- Series system: suction tank → entrance → suction pipe → pump → discharge pipe with bends, valves and fittings "
                "→ exit into the discharge tank. One pipe material throughout.\n"
                "- Pipe friction: Darcy-Weisbach, h = f (L/D) V²/2g. f = 64/Re for Re < 2300, otherwise Colebrook-White (iterated). "
                "Re 2300–4000 is flagged as transitional. Fittings: h = K V²/2g with rounded, typical textbook K values "
                "(real values depend on size, type and source).\n"
                "- Pump: quadratic curve H = H₀ − aQ² through the shutoff and design points, scaled with the affinity laws "
                "(Q ∝ N, H ∝ N²), or fitted to entered points by regressing head on Q² (slope −a, intercept H₀; the residual is shown). "
                "The NPSH-required shape (half the design value at zero flow, rising with Q²) is an assumption.\n"
                "- Two identical pumps: parallel gives H = H₀ − a(Q/2)² (flows add at equal head), series gives H = 2(H₀ − aQ²) "
                "(heads add at equal flow). NPSH required and efficiency use the flow through each pump. Both pumps run at the same "
                "speed and are always both running; interaction between them and check-valve losses are ignored.\n"
                "- Static head = (z_discharge − z_suction) + (p_discharge − p_suction)/ρg with tank gauge pressures; atmospheric pressure "
                "at the site elevation comes from the standard atmosphere, 101325 (1 − 2.25577×10⁻⁵ h)^5.25588 Pa. Roughness divided by "
                "diameter must stay at or below 0.05 (the friction-factor equation's range). A custom fluid has fixed properties and "
                "the pump curve is not corrected for its viscosity.\n"
                "- Operating point: pump head = system head (static head + losses at that flow), found by bisection.\n"
                "- Bottleneck: the section with the largest pressure drop at the operating point.\n"
                "- Cavitation screening: NPSH available = (P_surface − P_vapour)/ρg + suction level − suction losses, with P_surface the absolute "
                "pressure on the suction tank (site atmosphere plus tank gauge pressure). Flagged \"marginal\" below 1.3 × NPSH required.\n"
                "- Power: hydraulic ρgQH; shaft = hydraulic / η; electrical = shaft / (motor × drive efficiency). Pump efficiency "
                "η = η_BEP (2x − x²), x = Q / (speed ratio × design flow), is an assumed shape (floored at 5 %), not manufacturer data. "
                "Motor and drive efficiencies are held constant. Yearly cost = kWh × price at constant running conditions.\n"
                "- Throttle vs variable speed: both deliver the same lower flow. Throttle: pump stays at the current speed and the "
                "valve burns the surplus head. Variable speed: speed ratio = √((H_sys + aQ²)/H₀), no valve. At 100 % of the flow the "
                "drive saves nothing and its own losses make it slightly worse.\n"
                "- Pressure colours are gauge pressure referred to the pump-centreline elevation; pipe elevation changes, "
                "water hammer, two-phase flow and parallel branches are not modelled.")


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render, KEY_PREFIX, ABOUT)
