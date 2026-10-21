"""Chapter 1: Beam & shaft stress.

Methods (no FEA/CFD claims): a small 1D Euler-Bernoulli beam finite-element model gives the
reactions and deflection for every support case (simply supported, cantilever, fixed-fixed, propped
cantilever, overhanging) under a point load and/or a full-length uniform load; shear and moment follow
from equilibrium. Closed-form beam equations are kept as independent references and checked against it in the tests.
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
ABOUT = """**Method.** A small 1D Euler-Bernoulli beam finite-element model (a cross-check-grade tool, not general FEA) gives reactions and deflection for five support cases under a point load and/or a full-length uniform load. Shear and moment follow from equilibrium. For simply supported and cantilever beams the maximum deflection is shown from closed-form equations and the finite-element result is the cross-check. Shaft mode adds torsion (τ = Tc/J) and a von Mises equivalent stress. The optional fatigue check uses the Goodman relation.

**Checked in pytest:** reactions, shear and moment (PL/4, wL²/8); deflection against PL³/48EI, PL³/3EI, 5wL⁴/384EI and wL⁴/8EI; fixed-fixed end moments (Pab²/L², wL²/12) and centre deflections; propped cantilever reactions and wall moment; overhanging beam reactions, support moment and tip deflection; force and moment equilibrium for every support and load case; section properties (rectangle, round, tube, rectangular tube, I-beam); torsion; von Mises; custom material and deflection limit checks; endurance limit and Goodman.

**Limitations:** linear elastic, one point load and/or a full-length uniform load, self-weight ignored. Transverse shear, web crippling, local buckling of thin walls, stress concentrations and contact stresses are not included. Rectangular tubes and I-beams are bending only (no torsion).

For learning and screening, not certified design."""
DESCRIPTION = ("Five support cases under point and uniform loads using a small 1D finite-element model checked against "
               "textbook equations: stress map, deformed shape, shear and moment diagrams, safety factor and optional Goodman fatigue check.")

# =============================================================================================
# CALCULATIONS (pure functions, SI units in and out, no Streamlit)
# =============================================================================================
SS = "simply_supported"
CANT = "cantilever"
FIXED = "fixed_fixed"
PROPPED = "propped_cantilever"
OVER = "overhanging"
SUPPORTS = (SS, CANT, FIXED, PROPPED, OVER)
SUPPORT_LABELS = {SS: "Simply supported", CANT: "Cantilever (fixed at left end)",
                  FIXED: "Fixed at both ends",
                  PROPPED: "Propped cantilever (fixed left, pinned right)",
                  OVER: "Overhanging (overhang at right)"}
LOAD_TYPES = ["Point load", "Uniform load (UDL)", "Point + uniform"]
SHAPE_LABELS = {"rectangle": "Rectangle (b × h)", "solid_round": "Solid round (d)",
                "hollow_round": "Hollow round / tube (Do, Di)",
                "hollow_rect": "Hollow rectangular tube (b, h, t)",
                "i_beam": "I-beam (bf, tf, h, tw)"}
CUSTOM_MATERIAL = "Custom (enter E, yield, ultimate)"
DEFL_LIMITS = {"Off": None, "L/180": 180, "L/250": 250, "L/360": 360, "L/500": 500}


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
    """Rectangle: b, h. Solid round: d. Hollow round: d_o, d_i. Hollow rectangular tube: b, h, t (wall).
    I-beam (bending about the strong axis): bf, tf, h, tw."""
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
    if shape == "hollow_rect":
        b = require_positive(dims.get("b"), "Width b")
        h = require_positive(dims.get("h"), "Depth h")
        t = require_positive(dims.get("t"), "Wall thickness t")
        require_less_than(2 * t, "Twice the wall thickness", min(b, h), "the smaller outer dimension")
        inner = (b - 2 * t) * (h - 2 * t)
        return SectionProps(shape, b * h - inner, (b * h**3 - (b - 2 * t) * (h - 2 * t)**3) / 12, None, h / 2, h,
                            h - 2 * t)
    if shape == "i_beam":
        bf = require_positive(dims.get("bf"), "Flange width")
        tf = require_positive(dims.get("tf"), "Flange thickness")
        h = require_positive(dims.get("h"), "Total depth")
        tw = require_positive(dims.get("tw"), "Web thickness")
        require_less_than(2 * tf, "Twice the flange thickness", h, "the total depth")
        require_less_than(tw, "Web thickness", bf, "flange width")
        hw = h - 2 * tf
        return SectionProps(shape, bf * h - (bf - tw) * hw, (bf * h**3 - (bf - tw) * hw**3) / 12, None, h / 2, h, 0.0)
    raise ValidationError(f"Unknown cross-section '{shape}'.")


def _check_load_case(support, L, P, a, w=0.0, overhang=0.0):
    if support not in SUPPORTS:
        raise ValidationError(f"Unknown support type '{support}'.")
    require_positive(L, "Span")
    require_non_negative(P, "Load")
    require_non_negative(w, "Distributed load")
    require_in_range(a, "Load position", 0.0, L)
    if support == OVER:
        require_positive(overhang, "Overhang")
        require_less_than(overhang, "Overhang", L, "beam length")


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


def deflection_udl_closed_form(support, L, w, E, I, x):
    """Deflection (downward positive) under a full-length uniform load w (N/m): textbook equations."""
    x = np.asarray(x, dtype=float)
    if support == SS:
        return w * x * (L**3 - 2 * L * x**2 + x**3) / (24 * E * I)
    if support == CANT:
        return w * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * E * I)
    raise ValidationError("Closed-form UDL deflection is only provided for simply supported and cantilever beams.")


def closed_form_max_deflection(support, L, P, a, w, E, I) -> Optional[float]:
    """Largest deflection from closed-form equations (superposition of point load and UDL), or None
    for supports where the chapter uses the finite-element result instead."""
    if support not in (SS, CANT):
        return None
    if w == 0:
        return float(max_deflection(support, L, P, a, E, I))
    if P == 0:
        return float(5 * w * L**4 / (384 * E * I) if support == SS else w * L**4 / (8 * E * I))
    x = np.linspace(0.0, L, 4001)
    return float((deflection_closed_form(support, L, P, a, E, I, x)
                  + deflection_udl_closed_form(support, L, w, E, I, x)).max())


def support_layout(support: str, L: float, overhang: float = 0.0) -> list:
    """Support positions and kinds, left to right: (x, 'pin' or 'fixed'). The first is always at x = 0."""
    if support == SS:
        return [(0.0, "pin"), (L, "pin")]
    if support == CANT:
        return [(0.0, "fixed")]
    if support == FIXED:
        return [(0.0, "fixed"), (L, "fixed")]
    if support == PROPPED:
        return [(0.0, "fixed"), (L, "pin")]
    if support == OVER:
        return [(0.0, "pin"), (L - overhang, "pin")]
    raise ValidationError(f"Unknown support type '{support}'.")


@dataclass
class BeamSolution:
    nodes: np.ndarray
    w: np.ndarray        # downward deflection at the nodes
    sup_x: list          # support positions
    sup_kind: list       # 'pin' or 'fixed'
    R: np.ndarray        # upward reaction force at each support
    Mc: np.ndarray       # counter-clockwise reaction moment at each support (0 for simple supports)


def fe_solve(support, L, P, a, w, E, I, overhang: float = 0.0, n_el: int = 80) -> BeamSolution:
    """Small 1D Euler-Bernoulli beam finite-element model (2 DOF per node: deflection, slope).
    Handles any support layout above, a point load P at a and a full-length uniform load w (N/m, downward).
    Nodes are placed exactly at the load and at every support. Returns deflection and reactions."""
    _check_load_case(support, L, P, a, w, overhang)
    sup = support_layout(support, L, overhang)
    mand = np.array(sorted({0.0, L, a, *[x for x, _ in sup]}))
    xs = np.linspace(0.0, L, n_el + 1)
    tol = L / (4 * n_el)
    keep = np.all(np.abs(xs[:, None] - mand[None, :]) > tol, axis=1)
    nodes = np.unique(np.concatenate([xs[keep], mand]))
    nodes = nodes[np.insert(np.diff(nodes) > 1e-9 * L, 0, True)]
    n = len(nodes)
    K = np.zeros((2 * n, 2 * n))
    F = np.zeros(2 * n)
    for e in range(n - 1):
        l = nodes[e + 1] - nodes[e]
        k = 1.0 / l**3 * np.array([[12, 6 * l, -12, 6 * l],                                  # stiffness with EI = 1
                                   [6 * l, 4 * l**2, -6 * l, 2 * l**2],
                                   [-12, -6 * l, 12, -6 * l],
                                   [6 * l, 2 * l**2, -6 * l, 4 * l**2]])
        K[2 * e:2 * e + 4, 2 * e:2 * e + 4] += k
        F[2 * e:2 * e + 4] += np.array([-w * l / 2, -w * l**2 / 12, -w * l / 2, w * l**2 / 12])   # consistent UDL
    F[2 * int(np.argmin(np.abs(nodes - a)))] -= P                                                # downward point load
    fixed, idx = [], []
    for x_s, kind in sup:
        i = int(np.argmin(np.abs(nodes - x_s)))
        idx.append(i)
        fixed += [2 * i] + ([2 * i + 1] if kind == "fixed" else [])
    free = [i for i in range(2 * n) if i not in fixed]
    u = np.zeros(2 * n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    r = K @ u - F          # reactions at constrained DOFs (independent of E and I, which only scale the deflection)
    u = u / (E * I)        # K was built with EI = 1
    R = np.array([r[2 * i] for i in idx])
    Mc = np.array([r[2 * i + 1] if kind == "fixed" else 0.0 for i, (_, kind) in zip(idx, sup)])
    return BeamSolution(nodes, -u[0::2], [x for x, _ in sup], [k for _, k in sup], R, Mc)


def fe_deflection(support, L, P, a, E, I, n_el: int = 80):
    """Point-load finite-element deflection. Returns (node x, downward deflection)."""
    sol = fe_solve(support, L, P, a, 0.0, E, I, 0.0, n_el)
    return sol.nodes, sol.w


def internal_forces(sol: BeamSolution, L, P, a, w, x):
    """Shear V (up positive on the left face) and bending moment M (sagging positive) from equilibrium,
    using the support reactions. Exact for point and uniform loads."""
    x = np.asarray(x, dtype=float)
    V, M = np.zeros_like(x), np.zeros_like(x)
    for x_s, Rj in zip(sol.sup_x, sol.R):
        if x_s < L * (1 - 1e-12):          # a reaction exactly at the right end is not part of V inside the beam
            V = V + np.where(x >= x_s, Rj, 0.0)
        M = M + Rj * np.maximum(x - x_s, 0.0)
    if sol.sup_kind[0] == "fixed":
        M = M - sol.Mc[0]                  # wall moment at the left end (counter-clockwise reaction = hogging)
    V = V - P * (x >= a) - w * x
    M = M - P * np.maximum(x - a, 0.0) - w * x**2 / 2
    return V, M


def reaction_summary(support, sol: BeamSolution) -> dict:
    """Reactions for display: forces are upward-positive, moments are magnitudes."""
    R, Mc = sol.R, np.abs(sol.Mc)
    if support == SS:
        return {"R_left": float(R[0]), "R_right": float(R[1])}
    if support == CANT:
        return {"R_fixed": float(R[0]), "M_fixed": float(Mc[0])}
    if support == FIXED:
        return {"R_left": float(R[0]), "R_right": float(R[1]), "M_left": float(Mc[0]), "M_right": float(Mc[1])}
    if support == PROPPED:
        return {"R_fixed": float(R[0]), "M_fixed": float(Mc[0]), "R_pinned": float(R[1])}
    return {"R_left": float(R[0]), "R_right": float(R[1])}


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
    w_max: float                # closed form where available, otherwise the finite-element value
    w_max_fe: float
    sf: float
    sf_fatigue: Optional[float]
    n_req: float
    passed: bool
    w_closed: Optional[float] = None    # closed-form maximum deflection (None: finite element only)
    strength_ok: bool = True            # safety factor (and fatigue factor if checked) meets the requirement
    defl_ratio: Optional[float] = None  # deflection limit as span / ratio (None: not checked)
    defl_items: tuple = ()              # ((label, actual, allowed), ...) in metres
    defl_ok: bool = True


def deflection_limit_items(support, L, overhang, sol: BeamSolution, w_max, ratio) -> tuple:
    """Serviceability check items (label, actual, allowed), in metres. Between supports the allowed deflection
    is span / ratio; the overhang tip is checked against 2 x overhang / ratio; a cantilever uses its length
    (the conservative choice; some codes use twice the length)."""
    require_positive(ratio, "Deflection limit ratio")
    if support == OVER:
        span = L - overhang
        in_span = sol.nodes <= span + 1e-9 * L
        return (("Span deflection", float(np.max(np.abs(sol.w[in_span]))), span / ratio),
                ("Overhang tip deflection", float(abs(sol.w[-1])), 2 * overhang / ratio))
    return (("Tip deflection" if support == CANT else "Max deflection", float(w_max), L / ratio),)


def analyse(support, L, P, a, shape, dims, material: Material, T=0.0, n_req=1.5,
            fatigue_R: Optional[float] = None, fatigue_k: float = 0.6, *, w=0.0, overhang=0.0,
            defl_ratio: Optional[float] = None) -> Result:
    """P: point load (N) at a (m from the left end); w: full-length uniform load (N/m);
    overhang: length of the overhang (m) for the overhanging beam."""
    _check_load_case(support, L, P, a, w, overhang)
    require_non_negative(T, "Torque")
    require_positive(n_req, "Required safety factor")
    for name, v in (("E", material.E), ("Yield strength", material.Sy), ("Ultimate strength", material.Su)):
        require_positive(v, name)
    if material.Su < material.Sy:
        raise ValidationError("Ultimate strength must be at least the yield strength.")
    sec = section_properties(shape, **dims)
    if T > 0 and sec.J is None:
        raise ValidationError("Torque needs a round section (solid or hollow). Rectangles are not modelled for torsion.")

    sol = fe_solve(support, L, P, a, w, material.E, sec.I, overhang)
    extra = [a, max(a - 1e-9 * L, 0.0)]
    for x_s in sol.sup_x:
        extra += [x_s, max(x_s - 1e-9 * L, 0.0)]
    x = np.unique(np.concatenate([np.linspace(0.0, L, 401), extra]))
    V, M = internal_forces(sol, L, P, a, w, x)
    cross = np.nonzero(V[:-1] * V[1:] < 0)[0]            # zero-shear points: where a uniform load puts the moment peak
    if len(cross):
        x0 = x[cross] + (x[cross + 1] - x[cross]) * V[cross] / (V[cross] - V[cross + 1])
        x = np.unique(np.concatenate([x, x0]))
        V, M = internal_forces(sol, L, P, a, w, x)
    tau = torsion_shear(T, sec.c, sec.J) if sec.J else 0.0
    vm_x = von_mises(bending_stress(M, sec.c, sec.I), tau)

    sigma_max = float(np.max(np.abs(M))) * sec.c / sec.I
    vm_max = float(von_mises(sigma_max, tau))
    sf = float("inf") if vm_max == 0 else material.Sy / vm_max

    sf_f = None
    if fatigue_R is not None:
        require_in_range(fatigue_R, "Stress ratio R", -1.0, 1.0)
        require_in_range(fatigue_k, "Fatigue reduction factor", 0.05, 1.0)
        s_a, s_m = vm_max * (1 - fatigue_R) / 2, vm_max * (1 + fatigue_R) / 2
        sf_f = goodman_factor(s_a, s_m, endurance_limit(material.Su, fatigue_k), material.Su)

    w_fe_max = float(np.max(np.abs(sol.w)))
    w_cf = closed_form_max_deflection(support, L, P, a, w, material.E, sec.I)
    strength_ok = sf >= n_req and (sf_f is None or sf_f >= n_req)
    w_show = w_cf if w_cf is not None else w_fe_max
    items = deflection_limit_items(support, L, overhang, sol, w_show, defl_ratio) if defl_ratio else ()
    defl_ok = all(actual <= allowed for _, actual, allowed in items)
    return Result(x, V, M, sol.nodes, sol.w, vm_x, sec, T, reaction_summary(support, sol), sigma_max,
                  float(tau), vm_max, w_show, w_fe_max, sf, sf_f, n_req, strength_ok and defl_ok, w_cf,
                  strength_ok, defl_ratio, items, defl_ok)


# =============================================================================================
# FIGURES
# =============================================================================================
def auto_exaggeration(L: float, w_max: float) -> float:
    return 1.0 if w_max <= 0 else float(np.clip(0.08 * L / w_max, 1.0, 500.0))


def stress_map_figure(res: Result, material: Material, support, L, P, a, scale, system,
                      w=0.0, overhang=0.0) -> go.Figure:
    """Section view through the axis, drawn on the deformed shape (vertical shear mapping of the
    undeformed field, so the pattern is exact). Colour = von Mises / yield: blue low, red at yield."""
    sec = res.section
    f = lambda v: units.from_si(v, "length_long", system)
    xg = np.linspace(0, L, 320)
    wg = np.interp(xg, res.x_fe, res.w_fe) * scale
    Mg = np.abs(np.interp(xg, res.x, res.M))
    half = sec.depth / 2
    y_hi, y_lo = float(np.max(-wg)) + half, float(np.min(-wg)) - half
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
    sup = support_layout(support, L, overhang)
    pins = [x for x, kind in sup if kind == "pin"]
    if pins:
        fig.add_trace(go.Scatter(x=[f(x) for x in pins], y=[f(base)] * len(pins), mode="markers", showlegend=False,
                                 marker=dict(symbol="triangle-up", size=16, color="#444"), hoverinfo="skip"))
    for x, kind in sup:
        if kind == "fixed":
            x0, x1 = (-0.03 * L, 0.0) if x == 0 else (L, 1.03 * L)
            fig.add_shape(type="rect", x0=f(x0), x1=f(x1), y0=f(-2 * half), y1=f(2 * half),
                          fillcolor="#555", line=dict(width=0))
    top_at = lambda xv: -float(np.interp(xv, xg, wg)) + half
    if w > 0:
        for xv in np.linspace(0.0, L, 11):
            fig.add_annotation(x=f(xv), y=f(top_at(xv)), ax=0, ay=-30, showarrow=True, arrowhead=2,
                               arrowwidth=1.5, arrowcolor="#111", text="")
        right_side = P > 0 and a < 0.35 * L
        fig.add_annotation(x=f(L if right_side else 0.0), y=f(top_at(L if right_side else 0.0)), showarrow=False,
                           yshift=42, xanchor="right" if right_side else "left",
                           text=f"w = {units.fmt(w, 'line_load', system)}", font=dict(size=13))
    if P > 0:
        fig.add_annotation(x=f(a), y=f(top_at(a)), ax=0, ay=-45, showarrow=True, arrowhead=2, arrowwidth=3,
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


def _stress_input(label, default_si, key, system):
    d = units.from_si(default_si, "stress", system)
    v = st.number_input(f"{label} ({units.label('stress', system)})", value=float(d), step=float(d / 20),
                        format="%.1f", key=f"{key}_{system}")
    return units.to_si(v, "stress", system)


def render() -> None:
    system = get_unit_system()
    left, right = st.columns([1, 2], gap="large")
    with left:
        mode = st.radio("Mode", ["Beam", "Shaft (bending + torque)"], key="bs_mode", horizontal=True)
        shaft = mode.startswith("Shaft")
        support = st.selectbox("Supports", list(SUPPORTS), format_func=SUPPORT_LABELS.get, key="bs_support")
        L = _slider("Beam length" if support == OVER else "Span", "length_long", 0.2, 5.0, 1.0, "bs_span", system)
        overhang = L * st.slider("Overhang (% of beam length)", 5, 40, 20, key="bs_overhang") / 100 if support == OVER else 0.0
        load_type = st.radio("Load type", LOAD_TYPES, key="bs_loadtype", horizontal=True)
        has_point, has_udl = load_type != LOAD_TYPES[1], load_type != LOAD_TYPES[0]
        P = _slider("Point load", "force", 0.0, 20000.0, 5000.0, "bs_load", system) if has_point else 0.0
        a = L * st.slider("Point load position (% of beam length from left end)", 0, 100, 50, key="bs_pos") / 100 \
            if has_point else L / 2
        w = _slider("Uniform load (full length)", "line_load", 0.0, 20000.0, 5000.0, "bs_udl", system) if has_udl else 0.0
        shapes = ["solid_round", "hollow_round"] if shaft else list(SHAPE_LABELS)
        shape = st.selectbox("Cross-section", shapes, format_func=SHAPE_LABELS.get,
                             key="bs_shape_shaft" if shaft else "bs_shape_beam")
        if shape == "rectangle":
            dims = {"b": _size("Width b", 0.030, "bs_b", system), "h": _size("Depth h", 0.040, "bs_h", system)}
        elif shape == "solid_round":
            dims = {"d": _size("Diameter d", 0.040, "bs_d", system)}
        elif shape == "hollow_round":
            dims = {"d_o": _size("Outer diameter", 0.050, "bs_do", system),
                    "d_i": _size("Inner diameter", 0.040, "bs_di", system)}
        elif shape == "hollow_rect":
            dims = {"b": _size("Width b", 0.040, "bs_rb", system), "h": _size("Depth h", 0.060, "bs_rh", system),
                    "t": _size("Wall thickness t", 0.004, "bs_rt", system)}
        else:
            dims = {"bf": _size("Flange width", 0.100, "bs_ibf", system),
                    "tf": _size("Flange thickness", 0.008, "bs_itf", system),
                    "h": _size("Total depth h", 0.200, "bs_ih", system),
                    "tw": _size("Web thickness", 0.006, "bs_itw", system)}
        T = _slider("Torque", "torque", 0.0, 1000.0, 150.0, "bs_T", system) if shaft else 0.0
        mat_name = st.selectbox("Material (typical values, illustrative)", [*MATERIALS, CUSTOM_MATERIAL],
                                key="bs_material")
        if mat_name == CUSTOM_MATERIAL:
            base = MATERIALS["Steel AISI 1018 (cold drawn)"]
            material = Material("Custom", _stress_input("Elastic modulus E", base.E, "bs_cE", system),
                                _stress_input("Yield strength", base.Sy, "bs_cSy", system),
                                _stress_input("Ultimate strength", base.Su, "bs_cSu", system))
            st.caption("Custom values are whatever you enter; check them against your material certificate.")
        else:
            material = MATERIALS[mat_name]
            st.caption(f"E = {units.fmt(material.E, 'stress', system, 4)}, yield = {units.fmt(material.Sy, 'stress', system)}, "
                       f"ultimate = {units.fmt(material.Su, 'stress', system)}")
        n_req = st.slider("Required safety factor", 1.0, 4.0, 1.5, 0.1, key="bs_nreq")
        defl_label = st.selectbox("Deflection limit (span ÷ ratio)", list(DEFL_LIMITS), key="bs_defl")
        fatigue = st.checkbox("Fatigue check (Goodman)", key="bs_fatigue")
        R = k = None
        if fatigue:
            R = st.slider("Load ratio R = P_min / P_max (−1 = fully reversed)", -1.0, 1.0, 0.0, 0.1, key="bs_R")
            k = st.slider("Fatigue reduction factor k (surface, size, reliability)", 0.2, 1.0, 0.6, 0.05, key="bs_k")

    res = analyse(support, L, P, a, shape, dims, material, T, n_req, R, k if k else 0.6, w=w, overhang=overhang,
                  defl_ratio=DEFL_LIMITS[defl_label])

    with right:
        if L / res.section.depth < 5:
            st.warning("Span is under 5× the section depth: beam theory is unreliable for such short, deep members.")
        gov = max(res.defl_items, key=lambda it: it[1] / it[2]) if res.defl_items else None
        fmt_len = lambda v: units.fmt(v, "length", system)
        if res.passed:
            msg = (f"PASS: safety factor {res.sf:.2f} ≥ required {n_req:.1f}" if np.isfinite(res.sf)
                   else "PASS: no load applied")
            if gov:
                msg += f"; {gov[0].lower()} {fmt_len(gov[1])} ≤ allowed {fmt_len(gov[2])} ({defl_label})"
            st.success(msg)
        else:
            parts = []
            if not res.strength_ok:
                parts.append(f"FAIL: safety factor below the required {n_req:.1f}"
                             + (" (yielding predicted)" if res.sf < 1 else ""))
            for label, actual, allowed in res.defl_items:
                if actual > allowed:
                    parts.append(f"FAIL: {label.lower()} {fmt_len(actual)} exceeds the allowed {fmt_len(allowed)} ({defl_label})")
            st.error(" · ".join(parts))
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
        st.plotly_chart(stress_map_figure(res, material, support, L, P, a, scale, system, w, overhang), width="stretch")
        st.plotly_chart(diagram_figure(res, system), width="stretch")
        reac = ", ".join(f"{k_} = {units.fmt(v, 'force' if k_.startswith('R') else 'torque', system)}"
                         for k_, v in res.reactions.items())
        if res.w_closed is not None:
            diff = abs(res.w_max_fe - res.w_max) / res.w_max * 100 if res.w_max > 0 else 0.0
            st.caption(f"Deformed shape: 1D beam finite-element model. Max deflection: closed form "
                       f"(FE differs by {diff:.4f} %). Reactions: {reac}")
        else:
            st.caption(f"Deformed shape and max deflection: 1D beam finite-element model (checked against textbook "
                       f"cases in the tests). Reactions: {reac}")
        with st.expander("Method and limits"):
            st.markdown(
                "- Linear-elastic Euler-Bernoulli beam; one point load and/or a full-length uniform load; self-weight ignored.\n"
                "- Stress at the outer fibre of the critical section: σ = M·c/I; torsion τ = T·c/J; "
                "von Mises σ<sub>vm</sub> = √(σ² + 3τ²). Safety factor = yield / σ<sub>vm</sub>.\n"
                "- Optional deflection limit: span ÷ ratio between supports, 2 × overhang ÷ ratio at an overhang tip, and the "
                "length for a cantilever (conservative).\n"
                "- Transverse shear, web crippling, local buckling of thin walls, stress concentrations and contact stresses are "
                "**not** included.\n"
                "- Goodman: n = 1/(σ<sub>a</sub>/S<sub>e</sub> + σ<sub>m</sub>/S<sub>u</sub>), with simplified "
                "S<sub>e</sub> = k·0.5·S<sub>u</sub> (cap 700 MPa); the same load ratio R is applied to bending and torque.\n"
                "- The 1D finite-element model is not a general FEA tool.", unsafe_allow_html=True)


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render, KEY_PREFIX, ABOUT)
