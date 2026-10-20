"""Chapter 2: Pump & pipe system.

Method (no CFD claims): one-dimensional system-curve analysis. Pipe friction from Darcy-Weisbach with
a laminar / Colebrook-White friction factor, fitting losses from loss coefficients K, a quadratic
centrifugal pump curve scaled with the affinity laws, and the operating point where pump head equals
system head. Bottleneck = section with the largest pressure drop. Cavitation screening compares NPSH
available with NPSH required. Water only. Pump data and loss coefficients are illustrative.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `streamlit run chapters/x.py` find common/

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

from common import units
from common.theme import ACCENT, FAIL_COLOR, PASS_COLOR, STRESS_COLORSCALE, apply_theme
from common.ui import get_unit_system, run_standalone
from common.validation import (ValidationError, require_in_range, require_less_than, require_non_negative,
                               require_positive)

TITLE = "2 · Pump & pipe system"
DESCRIPTION = ("Darcy-Weisbach pipe and fitting losses against a centrifugal pump curve: operating point, "
               "bottleneck section and NPSH cavitation check (system-curve analysis, not CFD).")

# =============================================================================================
# CALCULATIONS (pure functions, SI units in and out, no Streamlit)
# =============================================================================================
G = 9.81                    # m/s^2
P_ATM = 101_325.0           # Pa, sea level
NPSH_MARGIN = 1.3           # NPSHa / NPSHr below this is flagged "marginal"
RE_LAMINAR, RE_TURBULENT = 2300.0, 4000.0

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


# ---- hydraulics ------------------------------------------------------------------------------
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


def npsha(p_atm: float, pv: float, rho: float, z_suction: float, h_loss_suction: float) -> float:
    """NPSH available (m) for a suction tank open to the atmosphere. z_suction > 0: liquid level above the pump."""
    return (p_atm - pv) / (rho * G) + z_suction - h_loss_suction


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
    return L


@dataclass(frozen=True)
class PumpData:
    """Quadratic curve H = H0 - a Q^2 at rated speed, through the shutoff head and the design point."""
    h_shutoff: float
    q_design: float
    h_design: float
    npshr_design: float         # NPSH required at the design flow, m
    rpm_rated: float

    def a(self) -> float:
        return (self.h_shutoff - self.h_design) / self.q_design**2


def validate_pump(p: PumpData) -> PumpData:
    require_positive(_finite(p.h_design, "Pump design head"), "Pump design head")
    require_positive(_finite(p.q_design, "Pump design flow"), "Pump design flow")
    require_less_than(_finite(p.h_design, "Pump design head"), "Pump design head",
                      _finite(p.h_shutoff, "Pump shutoff head"), "shutoff head")
    require_positive(_finite(p.npshr_design, "NPSH required"), "NPSH required")
    require_positive(_finite(p.rpm_rated, "Rated speed"), "Rated speed")
    return p


def speed_ratio(p: PumpData, rpm: float) -> float:
    return require_in_range(_finite(rpm, "Pump speed"), "Pump speed", 0.2 * p.rpm_rated, 1.5 * p.rpm_rated) / p.rpm_rated


def pump_head(p: PumpData, Q, r: float = 1.0):
    """Head at speed ratio r. Affinity laws (Q ~ N, H ~ N^2) turn H0 - aQ^2 into r^2 H0 - aQ^2."""
    Q = np.asarray(Q, dtype=float)
    return r**2 * p.h_shutoff - p.a() * Q**2


def pump_q_max(p: PumpData, r: float = 1.0) -> float:
    """Flow at which the pump head falls to zero."""
    return math.sqrt(r**2 * p.h_shutoff / p.a())


def pump_npshr(p: PumpData, Q, r: float = 1.0):
    """NPSH required: rated-speed curve r0 + (rd - r0)(Q/Qd)^2 with r0 = 0.5 rd (assumed), scaled by affinity (~N^2)."""
    Q = np.asarray(Q, dtype=float)
    r0 = 0.5 * p.npshr_design
    return r**2 * r0 + (p.npshr_design - r0) * (Q / p.q_design) ** 2


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


def static_head(L: Layout) -> float:
    return L.z_discharge - L.z_suction


def system_head(Q: float, L: Layout, fl: Fluid) -> float:
    return static_head(L) + sum(s.head_loss for s in sections_at(Q, L, fl))


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


def analyse(L: Layout, pump: PumpData, rpm: float, temp_c: float = 20.0, n_points: int = 121) -> Result:
    validate_layout(L)
    validate_pump(pump)
    r = speed_ratio(pump, rpm)
    fl = water(temp_c)
    q_max = pump_q_max(pump, r)
    q = np.linspace(0.0, q_max, n_points)
    h_sys = np.array([system_head(x, L, fl) for x in q])
    h_pump = pump_head(pump, q, r)
    suction_loss = lambda x: sum(s.head_loss for s in sections_at(x, L, fl) if s.side == "suction")
    npsha_c = np.array([npsha(P_ATM, fl.pv, fl.rho, L.z_suction, suction_loss(x)) for x in q])
    npshr_c = pump_npshr(pump, q, r)

    q_op = solve_operating_point(lambda x: float(pump_head(pump, x, r)), lambda x: system_head(x, L, fl), q_max)
    if q_op is None:
        return Result(fl, r, static_head(L), q, h_pump, h_sys, npsha_c, npshr_c, None, None, [], [], [], None,
                      None, None, None, [], [])
    h_op = float(pump_head(pump, q_op, r))
    secs = sections_at(q_op, L, fl)
    dp = [fl.rho * G * s.head_loss for s in secs]
    total = sum(dp)
    share = [d / total for d in dp] if total > 0 else [0.0] * len(dp)
    bottleneck = secs[int(np.argmax(dp))].name
    na = npsha(P_ATM, fl.pv, fl.rho, L.z_suction, suction_loss(q_op))
    nr = float(pump_npshr(pump, q_op, r))
    trans = [s.name for s in secs if s.Re is not None and RE_LAMINAR <= s.Re <= RE_TURBULENT]

    # energy profile along the path: tank -> suction sections -> pump -> discharge sections -> tank
    node = lambda label, head, v: ProfileNode(label, head, v, fl.rho * G * (head - v**2 / (2 * G)))
    v_s, v_d = velocity(q_op, L.d_suction), velocity(q_op, L.d_discharge)
    head = L.z_suction
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
    return Result(fl, r, static_head(L), q, h_pump, h_sys, npsha_c, npshr_c, q_op, h_op, secs, dp, share,
                  bottleneck, na, nr, cavitation_status(na, nr), trans, prof)


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


def layout_figure(r: Result, lay: Layout, color_by: str, system: str) -> go.Figure:
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
    fig.add_annotation(x=PUMP_C[0], y=0, text=f"<b>PUMP</b><br>{units.from_si(r.h_op, 'length_long', system):.1f} {u_h}",
                       showarrow=False, font=dict(size=11))
    fig.add_annotation(x=0.6, y=-1.3, text="Suction tank", showarrow=False, yshift=-12)
    fig.add_annotation(x=11.75, y=1.8, text="Discharge tank", showarrow=False, yshift=-12)
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


def curves_figure(r: Result, pump: PumpData, system: str) -> go.Figure:
    qu, hu = units.label("flow", system), units.label("length_long", system)
    Lq, Lh = (lambda v: units.from_si(v, "flow", system)), (lambda v: units.from_si(v, "length_long", system))
    fig = go.Figure()
    if abs(r.ratio - 1.0) > 1e-9:
        qr = np.linspace(0, pump_q_max(pump, 1.0), 100)
        fig.add_trace(go.Scatter(x=Lq(qr), y=Lh(pump_head(pump, qr, 1.0)), mode="lines", name="Pump at rated speed",
                                 line=dict(color=GREY, dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=Lq(r.q), y=Lh(r.h_pump), mode="lines", name="Pump curve (this speed)",
                             line=dict(color=ACCENT, width=3)))
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


# =============================================================================================
# UI
# =============================================================================================
def _slider(label, kind, lo, hi, default, key, system, fmt=None):
    lo_u, hi_u, d_u = (units.from_si(v, kind, system) for v in (lo, hi, default))
    v = st.slider(f"{label} ({units.label(kind, system)})", float(lo_u), float(hi_u), float(d_u),
                  step=float((hi_u - lo_u) / 200), format=fmt, key=f"{key}_{system}")
    return units.to_si(v, kind, system)


def _num(label, kind, default, key, system, step_si, fmt="%.2f"):
    v = st.number_input(f"{label} ({units.label(kind, system)})", value=float(units.from_si(default, kind, system)),
                        step=float(units.from_si(step_si, kind, system)), format=fmt, key=f"{key}_{system}")
    return units.to_si(v, kind, system)


def render() -> None:
    system = get_unit_system()
    st.caption("Illustrative example values, not from a real project. Pump data and loss coefficients are generic "
               "textbook-style numbers. Water only. System-curve analysis, not CFD.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("System")
        mat = st.selectbox("Pipe material (roughness)", list(ROUGHNESS), key="pp_material")
        d_s = _slider("Suction pipe diameter", "length", 0.040, 0.250, 0.125, "pp_ds", system, "%.1f" if system == units.SI else "%.2f")
        l_s = _slider("Suction pipe length", "length_long", 0.5, 20.0, 3.0, "pp_ls", system, "%.1f")
        z_s = _slider("Suction tank level above pump (− = lift)", "length_long", -6.0, 6.0, 1.0, "pp_zs", system, "%.1f")
        d_d = _slider("Discharge pipe diameter", "length", 0.040, 0.250, 0.100, "pp_dd", system, "%.1f" if system == units.SI else "%.2f")
        l_d = _slider("Discharge pipe length", "length_long", 5.0, 300.0, 60.0, "pp_ld", system, "%.0f")
        z_d = _slider("Discharge tank level above pump", "length_long", 0.0, 40.0, 12.0, "pp_zd", system, "%.1f")
        n_b = st.slider("90° bends on the discharge side", 0, 12, 4, key="pp_bends")
        vt, vn = st.columns([2, 1])
        valve = vt.selectbox("Valve type", list(VALVE_K), key="pp_valve")
        n_v = vn.slider("Valves", 0, 4, 1, key="pp_nvalve")
        ft, fn = st.columns([2, 1])
        fitting = ft.selectbox("Fitting type", list(FITTING_K), key="pp_fitting")
        n_f = fn.slider("Fittings", 0, 6, 1, key="pp_nfit")
        st.subheader("Pump and fluid")
        pct = st.slider("Pump speed (% of rated speed)", 50, 120, 100, key="pp_speed")
        temp = st.slider("Water temperature (°C)", 0, 100, 20, key="pp_temp")
        with st.expander("Pump data (illustrative)"):
            h0 = _num("Shutoff head at rated speed", "length_long", 40.0, "pp_h0", system, 0.5, "%.1f")
            qd = _num("Design flow", "flow", 0.020, "pp_qd", system, 0.001 if system == units.SI else 6.3e-5, "%.1f")
            hd = _num("Head at design flow", "length_long", 30.0, "pp_hd", system, 0.5, "%.1f")
            nr = _num("NPSH required at design flow", "length_long", 3.5, "pp_nr", system, 0.1, "%.1f")
            rpm_rated = st.number_input("Rated speed (rpm)", value=1750.0, step=50.0, key="pp_rpm_rated")
        color_by = st.radio("Colour the layout by", ["pressure", "velocity"], format_func=str.capitalize,
                            horizontal=True, key="pp_color")
        st.caption(f"Pump speed: {pct / 100 * rpm_rated:,.0f} rpm")

    lay = Layout(d_s, l_s, d_d, l_d, ROUGHNESS[mat], z_s, z_d, n_b, n_v, VALVE_K[valve], n_f, FITTING_K[fitting])
    pump = PumpData(h0, qd, hd, nr, rpm_rated)
    r = analyse(lay, pump, pct / 100 * rpm_rated, float(temp))

    with right:
        if r.q_op is None:
            st.error("No operating point: the pump cannot overcome the static head plus losses at this speed "
                     "(the curves do not cross). Raise the pump speed or shutoff head, or lower the static head.")
            st.plotly_chart(curves_figure(r, pump, system), width="stretch")
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
            st.plotly_chart(layout_figure(r, lay, color_by, system), width="stretch")
            st.plotly_chart(curves_figure(r, pump, system), width="stretch")
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
        with st.expander("Method and limits"):
            st.markdown(
                "- Series system: suction tank → entrance → suction pipe → pump → discharge pipe with bends, valves and fittings "
                "→ exit into the discharge tank. One pipe material throughout.\n"
                "- Pipe friction: Darcy-Weisbach, h = f (L/D) V²/2g. f = 64/Re for Re < 2300, otherwise Colebrook-White (iterated). "
                "Re 2300–4000 is flagged as transitional. Fittings: h = K V²/2g with rounded, typical textbook K values "
                "(real values depend on size, type and source).\n"
                "- Pump: quadratic curve H = H₀ − aQ² through the shutoff and design points, scaled with the affinity laws "
                "(Q ∝ N, H ∝ N²). The NPSH-required shape (half the design value at zero flow, rising with Q²) is an assumption.\n"
                "- Operating point: pump head = system head (static head + losses at that flow), found by bisection.\n"
                "- Bottleneck: the section with the largest pressure drop at the operating point.\n"
                "- Cavitation screening: NPSH available = (P_atm − P_vapour)/ρg + suction level − suction losses, at sea-level "
                "pressure. Flagged \"marginal\" below 1.3 × NPSH required.\n"
                "- Pressure colours are gauge pressure referred to the pump-centreline elevation; pipe elevation changes, "
                "pump efficiency and power, water hammer, two-phase flow and parallel branches are not modelled.")


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render)
