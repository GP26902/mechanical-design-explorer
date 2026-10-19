"""Checks against textbook closed-form / hand calculations."""
import math

import numpy as np
import pytest

from chapters import beam_shaft as bs
from common.validation import ValidationError

E, I = 200e9, 8e-6
L, P = 2.0, 1000.0


# ---- reactions, shear, moment ---------------------------------------------------------------
def test_ss_centre_load_reactions_and_moment():
    r = bs.reactions(bs.SS, L, P, L / 2)
    assert r["R_left"] == pytest.approx(500) and r["R_right"] == pytest.approx(500)
    assert bs.max_moment(bs.SS, L, P, L / 2) == pytest.approx(P * L / 4)   # 500 N·m
    x = np.linspace(0, L, 401)
    assert np.max(bs.moment(bs.SS, L, P, L / 2, x)) == pytest.approx(500)
    assert bs.shear(bs.SS, L, P, L / 2, 0.5) == pytest.approx(500)
    assert bs.shear(bs.SS, L, P, L / 2, 1.5) == pytest.approx(-500)


def test_ss_off_centre_reactions_and_moment():
    a, b = 0.5, 1.5
    r = bs.reactions(bs.SS, L, P, a)
    assert r["R_left"] == pytest.approx(P * b / L) and r["R_right"] == pytest.approx(P * a / L)
    assert bs.max_moment(bs.SS, L, P, a) == pytest.approx(P * a * b / L)   # 375 N·m
    assert bs.moment(bs.SS, L, P, a, a) == pytest.approx(375)


def test_cantilever_tip_load():
    r = bs.reactions(bs.CANT, L, P, L)
    assert r["R_fixed"] == pytest.approx(P) and r["M_fixed"] == pytest.approx(P * L)  # 2000 N·m
    assert bs.moment(bs.CANT, L, P, L, 0.0) == pytest.approx(-2000)   # hogging at the wall
    assert bs.moment(bs.CANT, L, P, L, L) == pytest.approx(0)


# ---- deflection: closed form ----------------------------------------------------------------
def test_ss_centre_deflection_PL3_over_48EI():
    assert bs.max_deflection(bs.SS, L, P, L / 2, E, I) == pytest.approx(P * L**3 / (48 * E * I))
    assert bs.max_deflection(bs.SS, L, P, L / 2, E, I) == pytest.approx(1.04167e-4, rel=1e-4)  # hand value


def test_cantilever_tip_deflection_PL3_over_3EI():
    assert bs.max_deflection(bs.CANT, L, P, L, E, I) == pytest.approx(P * L**3 / (3 * E * I))


def test_cantilever_mid_load_tip_deflection():
    a = 1.0   # tip deflection = P a^2 (3L - a) / 6EI
    assert bs.max_deflection(bs.CANT, L, P, a, E, I) == pytest.approx(P * a**2 * (3 * L - a) / (6 * E * I))


@pytest.mark.parametrize("a", [0.3, 0.5, 1.0, 1.4, 1.9])
def test_ss_max_formula_matches_curve_and_is_symmetric(a):
    x = np.linspace(0, L, 20001)
    y = bs.deflection_closed_form(bs.SS, L, P, a, E, I, x)
    assert y.max() == pytest.approx(bs.max_deflection(bs.SS, L, P, a, E, I), rel=1e-6)
    # under the load: P a^2 b^2 / (3 E I L)
    b = L - a
    assert bs.deflection_closed_form(bs.SS, L, P, a, E, I, a) == pytest.approx(P * a**2 * b**2 / (3 * E * I * L))
    assert bs.max_deflection(bs.SS, L, P, a, E, I) == pytest.approx(bs.max_deflection(bs.SS, L, P, L - a, E, I))


# ---- finite-element model vs closed form ----------------------------------------------------
@pytest.mark.parametrize("support,a", [(bs.SS, 1.0), (bs.SS, 0.5), (bs.SS, 1.7), (bs.CANT, 2.0), (bs.CANT, 1.0)])
def test_fe_matches_closed_form(support, a):
    x, w = bs.fe_deflection(support, L, P, a, E, I)
    cf = bs.deflection_closed_form(support, L, P, a, E, I, x)
    assert np.max(np.abs(w - cf)) / cf.max() < 1e-6
    assert w.max() == pytest.approx(bs.max_deflection(support, L, P, a, E, I), rel=1e-4)


def test_fe_load_at_support_gives_zero_deflection():
    x, w = bs.fe_deflection(bs.SS, L, P, 0.0, E, I)
    assert np.max(np.abs(w)) < 1e-12


# ---- section properties ---------------------------------------------------------------------
def test_section_properties_hand_values():
    r = bs.section_properties("rectangle", b=0.03, h=0.04)
    assert r.A == pytest.approx(1.2e-3) and r.I == pytest.approx(1.6e-7) and r.c == pytest.approx(0.02) and r.J is None
    c = bs.section_properties("solid_round", d=0.04)
    assert c.I == pytest.approx(1.2566e-7, rel=1e-4) and c.J == pytest.approx(2.5133e-7, rel=1e-4)
    t = bs.section_properties("hollow_round", d_o=0.05, d_i=0.04)
    assert t.I == pytest.approx(1.8113e-7, rel=1e-4) and t.J == pytest.approx(2 * t.I)


# ---- torsion, von Mises, Goodman ------------------------------------------------------------
def test_shaft_torsion_hand_example():
    # 25 mm solid shaft, 200 N·m: tau = 16 T / (pi d^3) = 65.19 MPa
    s = bs.section_properties("solid_round", d=0.025)
    assert bs.torsion_shear(200, s.c, s.J) == pytest.approx(65.19e6, rel=1e-3)
    assert bs.torsion_shear(200, s.c, s.J) == pytest.approx(16 * 200 / (math.pi * 0.025**3))


def test_von_mises_hand_values():
    assert bs.von_mises(100e6, 50e6) == pytest.approx(math.sqrt(17500) * 1e6)    # 132.29 MPa
    assert bs.von_mises(0, 40e6) == pytest.approx(math.sqrt(3) * 40e6)           # pure torsion
    assert bs.von_mises(80e6, 0) == pytest.approx(80e6)                          # pure bending


def test_goodman_and_endurance_limit():
    assert bs.goodman_factor(100, 100, 250, 500) == pytest.approx(1 / (0.4 + 0.2))   # 1.667
    assert bs.goodman_factor(100, 0, 250, 500) == pytest.approx(2.5)                 # fully reversed: Se/sigma_a
    assert bs.endurance_limit(440e6) == pytest.approx(220e6)
    assert bs.endurance_limit(1600e6) == pytest.approx(700e6)                        # capped
    assert bs.endurance_limit(440e6, 0.6) == pytest.approx(132e6)


# ---- end to end -----------------------------------------------------------------------------
STEEL = bs.MATERIALS["Steel AISI 1018 (cold drawn)"]
RECT = {"b": 0.03, "h": 0.04}


def test_analyse_beam_hand_calc():
    # SS, 1 m, 5 kN at centre: M = 1250 N·m; sigma = M c / I = 1250*0.02/1.6e-7 = 156.25 MPa; SF = 370/156.25
    r = bs.analyse(bs.SS, 1.0, 5000, 0.5, "rectangle", RECT, STEEL, n_req=1.5)
    assert r.sigma_max == pytest.approx(156.25e6) and r.vm_max == pytest.approx(156.25e6)
    assert r.sf == pytest.approx(370 / 156.25) and r.passed
    assert not bs.analyse(bs.SS, 1.0, 5000, 0.5, "rectangle", RECT, STEEL, n_req=3.0).passed
    assert r.w_max_fe == pytest.approx(r.w_max, rel=1e-4)


def test_analyse_shaft_combined_hand_calc():
    # 40 mm solid shaft, SS 1 m, 5 kN centre, T = 150 N·m
    r = bs.analyse(bs.SS, 1.0, 5000, 0.5, "solid_round", {"d": 0.04}, STEEL, T=150)
    I = math.pi * 0.04**4 / 64
    sigma, tau = 1250 * 0.02 / I, 16 * 150 / (math.pi * 0.04**3)
    assert r.vm_max == pytest.approx(math.sqrt(sigma**2 + 3 * tau**2))


def test_material_changes_safety_factor_and_scale_not_pattern():
    weak, strong = bs.MATERIALS["Aluminium 6061-T6"], bs.MATERIALS["Titanium Ti-6Al-4V (annealed)"]
    a = bs.analyse(bs.SS, 1.0, 5000, 0.5, "rectangle", RECT, weak)
    b = bs.analyse(bs.SS, 1.0, 5000, 0.5, "rectangle", RECT, strong)
    assert np.allclose(a.vm_outer, b.vm_outer)                       # same stress pattern and magnitude
    assert a.sf == pytest.approx(276e6 / a.vm_max) and b.sf == pytest.approx(880e6 / b.vm_max)
    assert b.sf > a.sf
    assert np.allclose(a.vm_outer / weak.Sy * weak.Sy, b.vm_outer / strong.Sy * strong.Sy)


def test_fatigue_goodman_integration():
    r = bs.analyse(bs.SS, 1.0, 5000, 0.5, "rectangle", RECT, STEEL, fatigue_R=-1.0, fatigue_k=0.6)
    assert r.sf_fatigue == pytest.approx(0.6 * 220e6 / 156.25e6)      # fully reversed: Se / sigma
    r0 = bs.analyse(bs.SS, 1.0, 5000, 0.5, "rectangle", RECT, STEEL, fatigue_R=0.0, fatigue_k=0.6)
    assert r0.sf_fatigue == pytest.approx(1 / (78.125e6 / 132e6 + 78.125e6 / 440e6))


def test_no_load_is_infinite_safety_factor():
    r = bs.analyse(bs.SS, 1.0, 0.0, 0.5, "rectangle", RECT, STEEL)
    assert math.isinf(r.sf) and r.passed


# ---- validation -----------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(L=0.0), dict(L=-1.0), dict(P=-5.0), dict(a=-0.1), dict(a=1.5), dict(n_req=0.0), dict(L=math.nan),
])
def test_bad_load_case_raises(kwargs):
    args = dict(support=bs.SS, L=1.0, P=1000.0, a=0.5, shape="rectangle", dims=RECT, material=STEEL)
    args.update(kwargs)
    with pytest.raises(ValidationError):
        bs.analyse(**args)


@pytest.mark.parametrize("shape,dims", [
    ("solid_round", {"d": 0.0}), ("solid_round", {"d": -0.01}), ("rectangle", {"b": 0.03, "h": 0.0}),
    ("rectangle", {"b": -1, "h": 0.04}), ("hollow_round", {"d_o": 0.04, "d_i": 0.04}),
    ("hollow_round", {"d_o": 0.04, "d_i": 0.05}), ("hollow_round", {"d_o": 0.04, "d_i": 0.0}), ("triangle", {}),
])
def test_bad_section_raises(shape, dims):
    with pytest.raises(ValidationError):
        bs.analyse(bs.SS, 1.0, 1000.0, 0.5, shape, dims, STEEL)


def test_torque_on_rectangle_raises_and_bad_support_raises():
    with pytest.raises(ValidationError):
        bs.analyse(bs.SS, 1.0, 1000.0, 0.5, "rectangle", RECT, STEEL, T=10)
    with pytest.raises(ValidationError):
        bs.analyse("fixed-fixed", 1.0, 1000.0, 0.5, "rectangle", RECT, STEEL)


# ---- figures build --------------------------------------------------------------------------
@pytest.mark.parametrize("support", [bs.SS, bs.CANT])
@pytest.mark.parametrize("shape,dims,T", [("rectangle", RECT, 0), ("solid_round", {"d": 0.04}, 100),
                                          ("hollow_round", {"d_o": 0.05, "d_i": 0.04}, 100)])
def test_figures_build(support, shape, dims, T):
    r = bs.analyse(support, 1.0, 5000, 0.5, shape, dims, STEEL, T=T)
    sc = bs.auto_exaggeration(1.0, r.w_max)
    assert len(bs.stress_map_figure(r, STEEL, support, 1.0, 5000, 0.5, sc, "SI").data) >= 1
    assert len(bs.diagram_figure(r, "US").data) == 2
