"""Chapter 1: Beam & shaft stress.

Methods (no FEA/CFD claims): closed-form Euler-Bernoulli beam equations for reactions, shear,
moment, stress and deflection, plus a small 1D Euler-Bernoulli beam finite-element model that
draws the deformed shape and is cross-checked against the closed form in the tests.
Shaft mode adds torsion (tau = T*c/J) and von Mises combination.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `streamlit run chapters/x.py` find common/

from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import units
from common.theme import ACCENT, STRESS_COLORSCALE, apply_theme
from common.ui import get_unit_system, run_standalone, slider_box
from common.validation import (ValidationError, require_in_range, require_less_than,
                               require_non_negative, require_positive)

TITLE = "1 · Beam & shaft stress"
KEY_PREFIX = "bs_"
ABOUT = """**Method.** Closed-form Euler-Bernoulli beam equations for reactions, shear, moment and deflection, cross-checked against a small 1D beam finite-element model (a cross-check, not a general FEA tool). Shaft mode adds torsion (τ = Tc/J) and a von Mises equivalent stress. The optional fatigue check uses the Goodman relation.

**Checked in pytest:** reactions, shear and moment (PL/4); deflection against PL³/48EI and PL³/3EI; finite-element model against the closed form; section properties; torsion; von Mises; endurance limit and Goodman; combined bending and torsion by hand.

**Limitations:** linear elastic, one point load, self-weight ignored. Transverse shear, stress concentrations, buckling and contact stresses are not included.

For learning and screening, not certified design."""
DESCRIPTION = ("Closed-form beam equations plus a small 1D finite-element model: stress map, "
               "deformed shape, shear and moment diagrams, safety factor and optional Goodman fatigue check.")

# =============================================================================================
# CALCULATIONS (pure functions, SI units in and out, no Streamlit)
# =============================================================================================
SS = "simply_supported"
CANT = "cantilever"
SUPPORT_LABELS = {SS: "Simply supported", CANT: "Cantilever (fixed at left end)"}
SHAPE_LABELS = {"rectangle": "Rectangle (b × h)", "solid_round": "Solid round (d)",
                "hollow_round": "Hollow round / tube (Do, Di)"}


@dataclass(frozen=True)
class Material:
    name: str
    E: float    # Pa
    Sy: float   # Pa
    Su: float   # Pa


# Typical handbook values, rounded. Illustrative: always use certified data for real work.
MATERIALS = {m.name: m for m in [
    Material("Steel AISI 1018 (cold drawn)", 205e9, 370e6, 440e6),
    Material("Steel AISI 1045 (hot rolled)", 205e9, 310e6, 565e6),
    Material("Stainless 304 (annealed)", 193e9, 215e6, 505e6),
    Material("Aluminium 6061-T6", 68.9e9, 276e6, 310e6),
    Material("Aluminium 7075-T6", 71.7e9, 503e6, 572e6),
    Material("Titanium Ti-6Al-4V (annealed)", 114e9, 880e6, 950e6),
]}


@dataclass(frozen=True)
class SectionProps:
    shape: str
    A: float            # area, m^2
    I: float            # second moment of area about the bending axis, m^4
    J: Optional[float]  # polar moment, m^4 (round sections only)
    c: float            # distance to extreme fibre, m
    depth: float        # overall depth (diameter), m
    d_inner: float      # bore diameter, m (0 if solid)


def section_properties(shape: str, **dims: float) -> SectionProps:
    """Rectangle: b, h. Solid round: d. Hollow round: d_o, d_i."""
    if shape == "rectangle":
        b = require_positive(dims.get("b"), "Width b")
        h = require_positive(dims.get("h"), "Depth h")
        return SectionProps(shape, b * h, b * h**3 / 12, None, h / 2, h, 0.0)
    if shape == "solid_round":
        d = require_positive(dims.get("d"), "Diameter d")
        return SectionProps(shape, np.pi * d**2 / 4, np.pi * d**4 / 64, np.pi * d**4 / 32, d / 2, d, 0.0)
    if shape == "hollow_round":
        do = require_positive(dims.get("d_o"), "Outer diameter")
        di = require_positive(dims.get("d_i"), "Inner diameter")
        require_less_than(di, "Inner diameter", do, "outer diameter")
        return SectionProps(shape, np.pi * (do**2 - di**2) / 4, np.pi * (do**4 - di**4) / 64,
                            np.pi * (do**4 - di**4) / 32, do / 2, do, di)
    raise ValidationError(f"Unknown cross-section '{shape}'.")


def _check_load_case(support, L, P, a):
    if support not in (SS, CANT):
        raise ValidationError(f"Unknown support type '{support}'.")
    require_positive(L, "Span")
    require_non_negative(P, "Load")
    require_in_range(a, "Load position", 0.0, L)


def reactions(support: str, L: float, P: float, a: float) -> dict:
    """Downward point load P at distance a from the left (fixed) end."""
    _check_load_case(support, L, P, a)
    if support == SS:
        return {"R_left": P * (L - a) / L, "R_right": P * a / L}
    return {"R_fixed": P, "M_fixed": P * a}   # M_fixed is the (hogging) wall moment magnitude


def shear(support, L, P, a, x):
    x = np.asarray(x, dtype=float)
    if support == SS:
        r1 = P * (L - a) / L
        return np.where(x < a, r1, r1 - P)
    return np.where(x < a, P, 0.0)


def moment(support, L, P, a, x):
    """Bending moment, sagging positive (cantilever therefore negative / hogging)."""
    x = np.asarray(x, dtype=float)
    if support == SS:
        r1 = P * (L - a) / L
        return np.where(x <= a, r1 * x, r1 * x - P * (x - a))
    return np.where(x <= a, -P * (a - x), 0.0)


def max_moment(support, L, P, a) -> float:
    """Magnitude of the largest bending moment: P*a*b/L (simply supported) or P*a (cantilever)."""
    return P * a * (L - a) / L if support == SS else P * a


def deflection_closed_form(support, L, P, a, E, I, x):
    """Deflection (downward positive) from the standard closed-form elastic-curve equations."""
    x = np.asarray(x, dtype=float)
    if support == SS:
        b = L - a
        left = P * b * x * (L**2 - b**2 - x**2) / (6 * E * I * L)
        right = P * a * (L - x) * (2 * L * x - x**2 - a**2) / (6 * E * I * L)
        return np.where(x <= a, left, right)
    left = P * x**2 * (3 * a - x) / (6 * E * I)
    right = P * a**2 * (3 * x - a) / (6 * E * I)
    return np.where(x <= a, left, right)


def max_deflection(support, L, P, a, E, I) -> float:
    if support == SS:
        t = min(a, L - a)     # distance from the load to the nearer support
        return P * t * (L**2 - t**2) ** 1.5 / (9 * np.sqrt(3) * E * I * L)
    return P * a**2 * (3 * L - a) / (6 * E * I)    # tip deflection


def fe_deflection(support, L, P, a, E, I, n_el: int = 80):
    """Small 1D Euler-Bernoulli beam finite-element model (2 DOF per node: deflection, slope).
    A node is placed exactly at the load. Returns (node x, downward deflection)."""
    xs = np.linspace(0.0, L, n_el + 1)
    tol = L / (4 * n_el)
    keep = (np.abs(xs - a) > tol) | (xs == 0.0) | (xs == L)
    nodes = np.unique(np.concatenate([xs[keep], [a]]))
    n = len(nodes)
    K = np.zeros((2 * n, 2 * n))
    for e in range(n - 1):
        l = nodes[e + 1] - nodes[e]
        k = E * I / l**3 * np.array([[12, 6 * l, -12, 6 * l],
                                     [6 * l, 4 * l**2, -6 * l, 2 * l**2],
                                     [-12, -6 * l, 12, -6 * l],
                                     [6 * l, 2 * l**2, -6 * l, 4 * l**2]])
        K[2 * e:2 * e + 4, 2 * e:2 * e + 4] += k
    F = np.zeros(2 * n)
    F[2 * int(np.argmin(np.abs(nodes - a)))] = -P          # downward load
    fixed = [0, 2 * (n - 1)] if support == SS else [0, 1]
    free = [i for i in range(2 * n) if i not in fixed]
    u = np.zeros(2 * n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return nodes, -u[0::2]


def bending_stress(M, c, I):
    return np.abs(M) * c / I


def torsion_shear(T, c, J):
    return T * c / J


def von_mises(sigma, tau):
    """Plane stress with one normal and one shear component: sqrt(sigma^2 + 3 tau^2)."""
    return np.sqrt(np.asarray(sigma, dtype=float) ** 2 + 3 * np.asarray(tau, dtype=float) ** 2)


def endurance_limit(Su: float, k: float = 1.0) -> float:
    """Simplified endurance limit Se = k * 0.5*Su, with 0.5*Su capped at 700 MPa (steel rule of thumb)."""
    return k * min(0.5 * Su, 700e6)


def goodman_factor(sigma_a: float, sigma_m: float, Se: float, Su: float) -> float:
    """Goodman safety factor n = 1 / (sigma_a/Se + sigma_m/Su)."""
    d = sigma_a / Se + sigma_m / Su
    return float("inf") if d <= 0 else 1.0 / d


@dataclass
class Result:
    x: np.ndarray
    V: np.ndarray
    M: np.ndarray
    x_fe: np.ndarray
    w_fe: np.ndarray
    vm_outer: np.ndarray        # von Mises at the outer fibre along x
    section: SectionProps
    T: float
    reactions: dict
    sigma_max: float            # bending, at the critical section
    tau_max: float              # torsion
    vm_max: float
    w_max: float                # closed form
    w_max_fe: float
    sf: float
    sf_fatigue: Optional[float]
    n_req: float
    passed: bool


def analyse(support, L, P, a, shape, dims, material: Material, T=0.0, n_req=1.5,
            fatigue_R: Optional[float] = None, fatigue_k: float = 0.6) -> Result:
    _check_load_case(support, L, P, a)
    require_non_negative(T, "Torque")
    require_positive(n_req, "Required safety factor")
    for name, v in (("E", material.E), ("Yield strength", material.Sy), ("Ultimate strength", material.Su)):
        require_positive(v, name)
    sec = section_properties(shape, **dims)
    if T > 0 and sec.J is None:
        raise ValidationError("Torque needs a round section (solid or hollow). Rectangles are not modelled for torsion.")

    x = np.unique(np.concatenate([np.linspace(0.0, L, 401), [a, max(a - 1e-9 * L, 0.0)]]))
    V, M = shear(support, L, P, a, x), moment(support, L, P, a, x)
    tau = torsion_shear(T, sec.c, sec.J) if sec.J else 0.0
    vm_x = von_mises(bending_stress(M, sec.c, sec.I), tau)

    sigma_max = max_moment(support, L, P, a) * sec.c / sec.I
    vm_max = float(von_mises(sigma_max, tau))
    sf = float("inf") if vm_max == 0 else material.Sy / vm_max

    sf_f = None
    if fatigue_R is not None:
        require_in_range(fatigue_R, "Stress ratio R", -1.0, 1.0)
        require_in_range(fatigue_k, "Fatigue reduction factor", 0.05, 1.0)
        s_a, s_m = vm_max * (1 - fatigue_R) / 2, vm_max * (1 + fatigue_R) / 2
        sf_f = goodman_factor(s_a, s_m, endurance_limit(material.Su, fatigue_k), material.Su)

    x_fe, w_fe = fe_deflection(support, L, P, a, material.E, sec.I)
    passed = sf >= n_req and (sf_f is None or sf_f >= n_req)
    return Result(x, V, M, x_fe, w_fe, vm_x, sec, T, reactions(support, L, P, a), sigma_max,
                  float(tau), vm_max, max_deflection(support, L, P, a, material.E, sec.I),
                  float(w_fe.max()), sf, sf_f, n_req, passed)


# =============================================================================================
# FIGURES
# =============================================================================================
def auto_exaggeration(L: float, w_max: float) -> float:
    return 1.0 if w_max <= 0 else float(np.clip(0.08 * L / w_max, 1.0, 500.0))


def stress_map_figure(res: Result, material: Material, support, L, P, a, scale, system) -> go.Figure:
    """Section view through the axis, drawn on the deformed shape (vertical shear mapping of the
    undeformed field, so the pattern is exact). Colour = von Mises / yield: blue low, red at yield."""
    sec = res.section
    f = lambda v: units.from_si(v, "length_long", system)
    xg = np.linspace(0, L, 320)
    wg = np.interp(xg, res.x_fe, res.w_fe) * scale
    Mg = np.abs(moment(support, L, P, a, xg))
    half = sec.depth / 2
    y_hi, y_lo = half, -wg.max() - half
    span_y = y_hi - y_lo
    Yg = np.linspace(y_lo - 0.15 * span_y, y_hi + 0.30 * span_y, 400)
    absy = np.abs(Yg[:, None] + wg[None, :])               # material coordinate of each pixel
    sigma = Mg[None, :] * absy / sec.I
    tau = res.T * absy / sec.J if sec.J else 0.0
    z = von_mises(sigma, tau) / material.Sy
    mask = absy > half
    if sec.d_inner > 0:
        mask |= absy < sec.d_inner / 2
    z = np.where(mask, np.nan, z)

    fig = go.Figure(go.Heatmap(x=f(xg), y=f(Yg), z=z, zmin=0, zmax=1, colorscale=STRESS_COLORSCALE,
                               hoverinfo="skip", colorbar=dict(title="σ<sub>vm</sub> / yield", len=0.9)))
    for sign in (1, -1):
        fig.add_trace(go.Scatter(x=f(xg), y=f(-wg + sign * half), mode="lines", showlegend=False,
                                 line=dict(color="#333", width=1.5), hoverinfo="skip"))
    base = -half - 0.03 * (Yg[-1] - Yg[0])
    if support == SS:
        fig.add_trace(go.Scatter(x=[f(0), f(L)], y=[f(base)] * 2, mode="markers", showlegend=False,
                                 marker=dict(symbol="triangle-up", size=16, color="#444"), hoverinfo="skip"))
    else:
        fig.add_shape(type="rect", x0=f(-0.03 * L), x1=f(0), y0=f(-2 * half), y1=f(2 * half),
                      fillcolor="#555", line=dict(width=0))
    if P > 0:
        top = -float(np.interp(a, xg, wg)) + half
        fig.add_annotation(x=f(a), y=f(top), ax=0, ay=-45, showarrow=True, arrowhead=2, arrowwidth=3,
                           arrowcolor="#111", text=f"P = {units.fmt(P, 'force', system)}", font=dict(size=13))
    xr = 1.1 * L
    h = int(np.clip(850 * (Yg[-1] - Yg[0]) / xr + 110, 260, 460))
    apply_theme(fig, f"Stress map on deformed shape (deflection exaggerated ×{scale:.0f})", h)
    fig.update_xaxes(range=[f(-0.05 * L), f(1.05 * L)], showgrid=False,
                     title=f"Position along span ({units.label('length_long', system)})")
    fig.update_yaxes(scaleanchor="x", showticklabels=False, showgrid=False)
    return fig


def diagram_figure(res: Result, system) -> go.Figure:
    fx = units.from_si(res.x, "length_long", system)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Shear force V", "Bending moment M (sagging positive)"))
    for row, y, kind in ((1, res.V, "force"), (2, res.M, "torque")):
        fig.add_trace(go.Scatter(x=fx, y=units.from_si(y, kind, system), mode="lines", fill="tozeroy",
                                 line=dict(color=ACCENT, width=2), showlegend=False), row=row, col=1)
        fig.update_yaxes(title_text=units.label(kind, system), row=row, col=1)
    fig.update_xaxes(title_text=f"Position ({units.label('length_long', system)})", row=2, col=1)
    apply_theme(fig, None, 480)
    fig.update_layout(margin=dict(l=50, r=20, t=40, b=45))
    return fig


# =============================================================================================
# UI
# =============================================================================================
def _slider(label, kind, lo, hi, default, key, system):
    lo_u, hi_u, d_u = (units.from_si(v, kind, system) for v in (lo, hi, default))
    v = slider_box(f"{label} ({units.label(kind, system)})", lo_u, hi_u, d_u, f"{key}_{system}")
    return units.to_si(v, kind, system)


def _size(label, default_si, key, system):
    d = units.from_si(default_si, "length", system)
    v = st.number_input(f"{label} ({units.label('length', system)})", value=float(d), step=float(d / 10),
                        format="%.3f", key=f"{key}_{system}")
    return units.to_si(v, "length", system)


def render() -> None:
    system = get_unit_system()
    left, right = st.columns([1, 2], gap="large")
    with left:
        mode = st.radio("Mode", ["Beam", "Shaft (bending + torque)"], key="bs_mode", horizontal=True)
        shaft = mode.startswith("Shaft")
        support = st.selectbox("Supports", [SS, CANT], format_func=SUPPORT_LABELS.get, key="bs_support")
        L = _slider("Span", "length_long", 0.2, 5.0, 1.0, "bs_span", system)
        P = _slider("Load", "force", 0.0, 20000.0, 5000.0, "bs_load", system)
        pct = st.slider("Load position (% of span from left end)", 0, 100, 50, key="bs_pos")
        a = L * pct / 100
        shapes = ["solid_round", "hollow_round"] if shaft else list(SHAPE_LABELS)
        shape = st.selectbox("Cross-section", shapes, format_func=SHAPE_LABELS.get,
                             key="bs_shape_shaft" if shaft else "bs_shape_beam")
        if shape == "rectangle":
            dims = {"b": _size("Width b", 0.030, "bs_b", system), "h": _size("Depth h", 0.040, "bs_h", system)}
        elif shape == "solid_round":
            dims = {"d": _size("Diameter d", 0.040, "bs_d", system)}
        else:
            dims = {"d_o": _size("Outer diameter", 0.050, "bs_do", system),
                    "d_i": _size("Inner diameter", 0.040, "bs_di", system)}
        T = _slider("Torque", "torque", 0.0, 1000.0, 150.0, "bs_T", system) if shaft else 0.0
        mat_name = st.selectbox("Material (typical values, illustrative)", list(MATERIALS), key="bs_material")
        material = MATERIALS[mat_name]
        st.caption(f"E = {units.fmt(material.E, 'stress', system, 4)}, yield = {units.fmt(material.Sy, 'stress', system)}, "
                   f"ultimate = {units.fmt(material.Su, 'stress', system)}")
        n_req = st.slider("Required safety factor", 1.0, 4.0, 1.5, 0.1, key="bs_nreq")
        fatigue = st.checkbox("Fatigue check (Goodman)", key="bs_fatigue")
        R = k = None
        if fatigue:
            R = st.slider("Load ratio R = P_min / P_max (−1 = fully reversed)", -1.0, 1.0, 0.0, 0.1, key="bs_R")
            k = st.slider("Fatigue reduction factor k (surface, size, reliability)", 0.2, 1.0, 0.6, 0.05, key="bs_k")

    res = analyse(support, L, P, a, shape, dims, material, T, n_req, R, k if k else 0.6)

    with right:
        if L / res.section.depth < 5:
            st.warning("Span is under 5× the section depth: beam theory is unreliable for such short, deep members.")
        if res.passed:
            st.success(f"PASS: safety factor {res.sf:.2f} ≥ required {n_req:.1f}" if np.isfinite(res.sf)
                       else "PASS: no load applied")
        else:
            st.error(f"FAIL: safety factor below the required {n_req:.1f}"
                     + (" (yielding predicted)" if res.sf < 1 else ""))
        c1, c2 = st.columns(2)
        c1.metric("Max von Mises stress", units.fmt(res.vm_max, "stress", system))
        c2.metric("Safety factor (yield)", "∞" if not np.isfinite(res.sf) else f"{res.sf:.2f}")
        c3, c4 = st.columns(2)
        c3.metric("Max deflection", units.fmt(res.w_max, "length", system))
        c4.metric("Fatigue safety factor (Goodman)" if fatigue else "Torsion shear stress",
                  ("∞" if not np.isfinite(res.sf_fatigue) else f"{res.sf_fatigue:.2f}") if fatigue
                  else units.fmt(res.tau_max, "stress", system))
        scale_auto = auto_exaggeration(L, res.w_max)
        auto = st.toggle("Auto deflection exaggeration", value=True, key="bs_auto")
        scale = scale_auto if auto else float(st.slider("Exaggeration ×", 1, 500, 50, key="bs_scale"))
        st.plotly_chart(stress_map_figure(res, material, support, L, P, a, scale, system), width="stretch")
        st.plotly_chart(diagram_figure(res, system), width="stretch")
        diff = abs(res.w_max_fe - res.w_max) / res.w_max * 100 if res.w_max > 0 else 0.0
        st.caption(f"Deformed shape: 1D beam finite-element model. Max deflection: closed form "
                   f"(FE differs by {diff:.4f} %). Reactions: "
                   + ", ".join(f"{k_} = {units.fmt(v, 'force' if k_.startswith('R') else 'torque', system)}"
                               for k_, v in res.reactions.items()))
        with st.expander("Method and limits"):
            st.markdown(
                "- Linear-elastic Euler-Bernoulli beam; one point load; self-weight ignored.\n"
                "- Stress at the outer fibre of the critical section: σ = M·c/I; torsion τ = T·c/J; "
                "von Mises σ<sub>vm</sub> = √(σ² + 3τ²). Safety factor = yield / σ<sub>vm</sub>.\n"
                "- Transverse shear, stress concentrations, buckling and contact stresses are **not** included.\n"
                "- Goodman: n = 1/(σ<sub>a</sub>/S<sub>e</sub> + σ<sub>m</sub>/S<sub>u</sub>), with simplified "
                "S<sub>e</sub> = k·0.5·S<sub>u</sub> (cap 700 MPa); the same load ratio R is applied to bending and torque.\n"
                "- The 1D finite-element model is not a general FEA tool.", unsafe_allow_html=True)


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render, KEY_PREFIX, ABOUT)
