"""Step 2 checks: fixed-fixed, propped cantilever and overhanging beams, and uniform loads,
against textbook closed-form values and equilibrium."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from chapters import beam_shaft as bs
from common import units
from common.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]
E, B, H = 205e9, 0.030, 0.040
I = B * H**3 / 12
MAT = bs.MATERIALS["Steel AISI 1018 (cold drawn)"]
MAT_E = bs.Material("test steel", E, 370e6, 440e6)
DIMS = {"b": B, "h": H}


def run(support, L, P, a, w, overhang=0.0):
    return bs.analyse(support, L, P, a, "rectangle", DIMS, MAT_E, w=w, overhang=overhang)


def M_at(res, x):
    return float(np.interp(x, res.x, res.M))


# ---- simply supported and cantilever with a uniform load -------------------------------------
def test_ss_udl_reactions_moment_deflection():
    L, w = 2.0, 3000.0
    res = run(bs.SS, L, 0.0, L / 2, w)
    assert res.reactions["R_left"] == pytest.approx(w * L / 2, rel=1e-9)
    assert res.reactions["R_right"] == pytest.approx(w * L / 2, rel=1e-9)
    assert np.max(np.abs(res.M)) == pytest.approx(w * L**2 / 8, rel=1e-9)
    assert res.w_closed == pytest.approx(5 * w * L**4 / (384 * E * I), rel=1e-12)
    assert res.w_max_fe == pytest.approx(5 * w * L**4 / (384 * E * I), rel=1e-4)


def test_cantilever_udl():
    L, w = 1.5, 2500.0
    res = run(bs.CANT, L, 0.0, 0.0, w)
    assert res.reactions["R_fixed"] == pytest.approx(w * L, rel=1e-9)
    assert res.reactions["M_fixed"] == pytest.approx(w * L**2 / 2, rel=1e-9)
    assert np.max(np.abs(res.M)) == pytest.approx(w * L**2 / 2, rel=1e-9)
    assert res.w_closed == pytest.approx(w * L**4 / (8 * E * I), rel=1e-12)
    assert res.w_max_fe == pytest.approx(w * L**4 / (8 * E * I), rel=1e-4)


def test_point_plus_udl_is_the_sum_for_simply_supported():
    L, P, a, w = 2.0, 4000.0, 0.7, 1500.0
    res = run(bs.SS, L, P, a, w)
    assert res.reactions["R_left"] == pytest.approx(P * (L - a) / L + w * L / 2, rel=1e-9)
    assert res.w_max_fe == pytest.approx(res.w_closed, rel=2e-4)         # FE against closed-form superposition
    assert abs(M_at(res, L)) < 1e-6 * np.max(np.abs(res.M))              # free of moment at the pinned end


# ---- fixed at both ends ----------------------------------------------------------------------
def test_fixed_fixed_point_load_off_centre():
    L, P, a = 2.0, 6000.0, 0.6
    b = L - a
    res = run(bs.FIXED, L, P, a, 0.0)
    assert res.reactions["M_left"] == pytest.approx(P * a * b**2 / L**2, rel=1e-9)
    assert res.reactions["M_right"] == pytest.approx(P * a**2 * b / L**2, rel=1e-9)
    assert res.reactions["R_left"] == pytest.approx(P * b**2 * (3 * a + b) / L**3, rel=1e-9)
    assert res.reactions["R_right"] == pytest.approx(P * a**2 * (a + 3 * b) / L**3, rel=1e-9)


def test_fixed_fixed_centre_load_and_udl():
    L, P, w = 2.0, 5000.0, 2000.0
    res = run(bs.FIXED, L, P, L / 2, 0.0)
    assert res.reactions["M_left"] == pytest.approx(P * L / 8, rel=1e-9)
    assert res.w_max_fe == pytest.approx(P * L**3 / (192 * E * I), rel=1e-4)
    res = run(bs.FIXED, L, 0.0, 0.0, w)
    assert res.reactions["M_left"] == pytest.approx(w * L**2 / 12, rel=1e-9)
    assert M_at(res, L / 2) == pytest.approx(w * L**2 / 24, rel=1e-9)     # sagging at midspan
    assert res.w_max_fe == pytest.approx(w * L**4 / (384 * E * I), rel=1e-4)
    assert res.w_closed is None                                           # finite-element result is used here


# ---- propped cantilever (fixed left, simple support right) ------------------------------------
def test_propped_cantilever_udl():
    L, w = 2.0, 3000.0
    res = run(bs.PROPPED, L, 0.0, 0.0, w)
    assert res.reactions["R_pinned"] == pytest.approx(3 * w * L / 8, rel=1e-9)
    assert res.reactions["R_fixed"] == pytest.approx(5 * w * L / 8, rel=1e-9)
    assert res.reactions["M_fixed"] == pytest.approx(w * L**2 / 8, rel=1e-9)
    sol = bs.fe_solve(bs.PROPPED, L, 0.0, 0.0, w, E, I)
    x = sol.nodes
    textbook = w * x**2 * (3 * L**2 - 5 * L * x + 2 * x**2) / (48 * E * I)   # elastic curve, fixed at x = 0
    assert np.allclose(sol.w, textbook, rtol=1e-6, atol=1e-12 * textbook.max())
    assert res.w_max_fe == pytest.approx(w * L**4 / (185 * E * I), rel=5e-3)  # textbook wL^4/185EI


def test_propped_cantilever_centre_point_load():
    L, P = 2.0, 8000.0
    res = run(bs.PROPPED, L, P, L / 2, 0.0)
    assert res.reactions["R_pinned"] == pytest.approx(5 * P / 16, rel=1e-9)
    assert res.reactions["R_fixed"] == pytest.approx(11 * P / 16, rel=1e-9)
    assert res.reactions["M_fixed"] == pytest.approx(3 * P * L / 16, rel=1e-9)


# ---- overhanging beam -------------------------------------------------------------------------
def test_overhang_tip_load():
    Ls, c, P = 0.9, 0.3, 3000.0
    L = Ls + c
    res = run(bs.OVER, L, P, L, 0.0, overhang=c)
    assert res.reactions["R_left"] == pytest.approx(-P * c / Ls, rel=1e-9)         # pushed down
    assert res.reactions["R_right"] == pytest.approx(P * (1 + c / Ls), rel=1e-9)
    assert np.max(np.abs(res.M)) == pytest.approx(P * c, rel=1e-9)                 # at the support
    tip = P * c**2 * (Ls + c) / (3 * E * I)
    assert res.w_max_fe == pytest.approx(tip, rel=1e-4)                            # textbook tip deflection
    sol = bs.fe_solve(bs.OVER, L, P, L, 0.0, E, I, c)
    assert sol.w[-1] == pytest.approx(tip, rel=1e-4)
    assert sol.w[len(sol.w) // 4] < 0                                              # span lifts between the supports


def test_overhang_udl():
    Ls, c, w = 0.9, 0.3, 2000.0
    L = Ls + c
    res = run(bs.OVER, L, 0.0, 0.0, w, overhang=c)
    R2 = w * L**2 / (2 * Ls)
    assert res.reactions["R_right"] == pytest.approx(R2, rel=1e-9)
    assert res.reactions["R_left"] == pytest.approx(w * L - R2, rel=1e-9)
    assert M_at(res, Ls) == pytest.approx(-w * c**2 / 2, rel=1e-9)                 # hogging over the support


# ---- equilibrium for every support and load combination -----------------------------------------
@pytest.mark.parametrize("support", bs.SUPPORTS)
@pytest.mark.parametrize("P,w", [(5000.0, 0.0), (0.0, 3000.0), (4000.0, 2500.0)])
def test_equilibrium(support, P, w):
    L, a, c = 1.8, 0.55, 0.4
    sol = bs.fe_solve(support, L, P, a, w, E, I, c if support == bs.OVER else 0.0)
    assert sol.R.sum() == pytest.approx(P + w * L, rel=1e-9)                       # vertical forces
    moment_about_left = float(np.dot(sol.R, sol.sup_x) + sol.Mc.sum())            # counter-clockwise positive
    assert moment_about_left == pytest.approx(P * a + w * L**2 / 2, rel=1e-9)
    res = run(support, L, P, a, w, c if support == bs.OVER else 0.0)
    if support != bs.FIXED:
        assert abs(M_at(res, L)) < 1e-6 * np.max(np.abs(res.M))                    # no moment at a free or pinned end
    assert np.all(np.isfinite(res.M)) and np.all(np.isfinite(res.w_fe))


# ---- validation ---------------------------------------------------------------------------------
def test_validation_messages():
    with pytest.raises(ValidationError, match="Distributed load"):
        run(bs.SS, 1.0, 0.0, 0.5, -1.0)
    with pytest.raises(ValidationError, match="Overhang"):
        run(bs.OVER, 1.0, 1000.0, 0.5, 0.0, overhang=0.0)
    with pytest.raises(ValidationError, match="Overhang"):
        run(bs.OVER, 1.0, 1000.0, 0.5, 0.0, overhang=1.0)
    with pytest.raises(ValidationError, match="Unknown support"):
        run("magic", 1.0, 1000.0, 0.5, 0.0)
    assert run(bs.SS, 1.0, 0.0, 0.5, 0.0).sf == float("inf")                       # no load at all is fine


def test_line_load_units():
    assert units.to_si(1, "line_load", "US") == pytest.approx(units.to_si(1, "force", "US") / 0.3048, rel=1e-9)
    assert units.from_si(units.to_si(123.4, "line_load", "US"), "line_load", "US") == pytest.approx(123.4)
    assert units.label("line_load", "SI") == "N/m"


# ---- figures and the app --------------------------------------------------------------------------
@pytest.mark.parametrize("support", bs.SUPPORTS)
@pytest.mark.parametrize("P,w", [(5000.0, 0.0), (0.0, 3000.0), (4000.0, 2500.0)])
def test_figures_build(support, P, w):
    L, a, c = 1.2, 0.4, 0.3
    ov = c if support == bs.OVER else 0.0
    res = run(support, L, P, a, w, ov)
    fig = bs.stress_map_figure(res, MAT_E, support, L, P, a, 20.0, units.SI, w, ov)
    assert len(fig.data) >= 3
    assert bs.diagram_figure(res, units.SI) is not None


def _app():
    return AppTest.from_file(str(ROOT / "chapters" / "beam_shaft.py"), default_timeout=60).run()


@pytest.mark.parametrize("support", bs.SUPPORTS)
@pytest.mark.parametrize("load_type", bs.LOAD_TYPES)
def test_app_every_support_and_load_type(support, load_type):
    at = _app()
    at.selectbox(key="bs_support").set_value(support).run()
    at.radio(key="bs_loadtype").set_value(load_type).run()
    assert not at.exception
    assert not any("Input problem" in e.value or "unexpected" in e.value for e in at.error), [e.value for e in at.error]
    assert len(at.metric) == 4
    keys = {s.key for s in at.slider}
    assert ("bs_overhang" in keys) == (support == bs.OVER)
    assert ("bs_load_SI" in keys) == (load_type != bs.LOAD_TYPES[1])
    assert ("bs_udl_SI" in keys) == (load_type != bs.LOAD_TYPES[0])


def test_app_us_units_with_udl():
    at = _app()
    at.radio(key="bs_loadtype").set_value(bs.LOAD_TYPES[2]).run()
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception
    assert not any("Input problem" in e.value or "unexpected" in e.value for e in at.error)
    assert any("lbf/ft" in s.label for s in at.slider)
