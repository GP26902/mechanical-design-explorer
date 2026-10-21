"""Step 3 checks: hollow rectangular tube and I-beam sections, custom material, deflection limit."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from chapters import beam_shaft as bs
from common.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]
E, SY, SU = 205e9, 370e6, 440e6
MAT = bs.Material("test steel", E, SY, SU)


# ---- section properties against an independent build-up (parallel-axis theorem) -------------------
def test_hollow_rect_properties():
    b, h, t = 0.040, 0.060, 0.004
    sec = bs.section_properties("hollow_rect", b=b, h=h, t=t)
    flange = b * t**3 / 12 + b * t * ((h - t) / 2) ** 2
    web = t * (h - 2 * t) ** 3 / 12
    assert sec.I == pytest.approx(2 * flange + 2 * web, rel=1e-12)
    assert sec.A == pytest.approx(2 * b * t + 2 * t * (h - 2 * t), rel=1e-12)
    assert sec.c == pytest.approx(h / 2) and sec.depth == h and sec.d_inner == pytest.approx(h - 2 * t)
    assert sec.J is None


def test_i_beam_properties():
    bf, tf, h, tw = 0.100, 0.008, 0.200, 0.006
    sec = bs.section_properties("i_beam", bf=bf, tf=tf, h=h, tw=tw)
    flange = bf * tf**3 / 12 + bf * tf * ((h - tf) / 2) ** 2
    web = tw * (h - 2 * tf) ** 3 / 12
    assert sec.I == pytest.approx(2 * flange + web, rel=1e-12)
    assert sec.A == pytest.approx(2 * bf * tf + tw * (h - 2 * tf), rel=1e-12)
    assert sec.c == pytest.approx(h / 2) and sec.d_inner == 0.0 and sec.J is None


def test_thin_wall_limit_recovers_solid_rectangle_formula():
    """A tube with t -> b/2 and h/2 is a nearly solid rectangle."""
    sec = bs.section_properties("hollow_rect", b=0.04, h=0.06, t=0.0199999)
    assert sec.I == pytest.approx(0.04 * 0.06**3 / 12, rel=1e-3)


@pytest.mark.parametrize("shape,dims,match", [
    ("hollow_rect", dict(b=0.04, h=0.06, t=0.02), "wall thickness"),
    ("hollow_rect", dict(b=0.04, h=0.06, t=0.0), "Wall thickness"),
    ("hollow_rect", dict(b=0.0, h=0.06, t=0.004), "Width"),
    ("i_beam", dict(bf=0.1, tf=0.1, h=0.2, tw=0.006), "flange thickness"),
    ("i_beam", dict(bf=0.1, tf=0.008, h=0.2, tw=0.1), "Web thickness"),
    ("i_beam", dict(bf=0.1, tf=0.008, h=0.2, tw=-1), "Web thickness"),
])
def test_section_validation(shape, dims, match):
    with pytest.raises(ValidationError, match=match):
        bs.section_properties(shape, **dims)


def test_torsion_rejected_for_new_sections():
    with pytest.raises(ValidationError, match="round"):
        bs.analyse(bs.SS, 1.0, 1000.0, 0.5, "i_beam", dict(bf=0.1, tf=0.008, h=0.2, tw=0.006), MAT, T=50.0)


# ---- stress and deflection on the new sections ----------------------------------------------------
@pytest.mark.parametrize("shape,dims", [
    ("hollow_rect", dict(b=0.040, h=0.060, t=0.004)),
    ("i_beam", dict(bf=0.100, tf=0.008, h=0.200, tw=0.006)),
])
def test_stress_and_deflection_match_closed_form(shape, dims):
    L, P = 2.0, 6000.0
    res = bs.analyse(bs.SS, L, P, L / 2, shape, dims, MAT)
    sec = bs.section_properties(shape, **dims)
    assert res.sigma_max == pytest.approx(P * L / 4 * sec.c / sec.I, rel=1e-9)        # sigma = M c / I
    assert res.w_max == pytest.approx(P * L**3 / (48 * E * sec.I), rel=1e-9)          # PL^3 / 48EI
    assert res.w_max_fe == pytest.approx(res.w_max, rel=1e-4)


def test_stress_map_shows_the_core_of_the_tube_but_not_of_the_i_beam():
    for shape, dims, core_empty in [("hollow_rect", dict(b=0.04, h=0.06, t=0.004), True),
                                    ("i_beam", dict(bf=0.1, tf=0.008, h=0.2, tw=0.006), False)]:
        res = bs.analyse(bs.SS, 1.0, 5000.0, 0.5, shape, dims, MAT)
        fig = bs.stress_map_figure(res, MAT, bs.SS, 1.0, 5000.0, 0.5, 50.0, "SI")
        z, y = np.asarray(fig.data[0].z, dtype=float), np.asarray(fig.data[0].y, dtype=float)
        assert np.isnan(z[np.argmin(np.abs(y)), 0]) == core_empty         # axis at the support, where deflection is 0
        assert np.isfinite(z).any()


# ---- custom material --------------------------------------------------------------------------------
def test_custom_material_scales_safety_factor_and_deflection():
    dims = dict(b=0.03, h=0.04)
    base = bs.analyse(bs.SS, 1.0, 3000.0, 0.5, "rectangle", dims, MAT)
    stronger = bs.analyse(bs.SS, 1.0, 3000.0, 0.5, "rectangle", dims, bs.Material("c", E, 2 * SY, SU * 2))
    softer = bs.analyse(bs.SS, 1.0, 3000.0, 0.5, "rectangle", dims, bs.Material("c", E / 3, SY, SU))
    assert stronger.sf == pytest.approx(2 * base.sf, rel=1e-12) and stronger.w_max == pytest.approx(base.w_max)
    assert softer.w_max == pytest.approx(3 * base.w_max, rel=1e-12) and softer.sf == pytest.approx(base.sf)


@pytest.mark.parametrize("mat,match", [
    (bs.Material("c", 0.0, SY, SU), "E"),
    (bs.Material("c", E, -1.0, SU), "Yield"),
    (bs.Material("c", E, SY, 0.0), "Ultimate"),
    (bs.Material("c", E, 500e6, 400e6), "at least the yield"),
])
def test_custom_material_validation(mat, match):
    with pytest.raises(ValidationError, match=match):
        bs.analyse(bs.SS, 1.0, 3000.0, 0.5, "rectangle", dict(b=0.03, h=0.04), mat)


# ---- deflection limit -------------------------------------------------------------------------------
DIMS = dict(b=0.03, h=0.04)
I_R = 0.03 * 0.04**3 / 12


def test_limit_arithmetic_and_pass_fail_flip():
    L, P = 1.0, 2000.0
    w = P * L**3 / (48 * E * I_R)                      # closed-form midspan deflection
    for ratio, ok in [(0.5 * L / w, True), (2.0 * L / w, False)]:   # allowed = L/ratio = 2w (passes) or w/2 (fails)
        res = bs.analyse(bs.SS, L, P, L / 2, "rectangle", DIMS, MAT, defl_ratio=ratio)
        (label, actual, allowed), = res.defl_items
        assert actual == pytest.approx(w, rel=1e-9) and allowed == pytest.approx(L / ratio)
        assert res.defl_ok is ok and res.strength_ok and res.passed is ok


def test_limit_off_changes_nothing():
    res = bs.analyse(bs.SS, 1.0, 2000.0, 0.5, "rectangle", DIMS, MAT)
    assert res.defl_items == () and res.defl_ok and res.defl_ratio is None and res.passed == res.strength_ok


def test_limit_can_fail_while_strength_passes_and_vice_versa():
    soft = bs.Material("soft", E / 20, SY, SU)                       # low stiffness, same strength
    res = bs.analyse(bs.SS, 1.0, 1000.0, 0.5, "rectangle", DIMS, soft, defl_ratio=360)
    assert res.strength_ok and not res.defl_ok and not res.passed
    res = bs.analyse(bs.SS, 1.0, 20000.0, 0.5, "rectangle", DIMS, MAT, defl_ratio=5)
    assert not res.strength_ok and res.defl_ok and not res.passed


def test_cantilever_allowed_is_length_over_ratio():
    res = bs.analyse(bs.CANT, 1.2, 500.0, 1.2, "rectangle", DIMS, MAT, defl_ratio=250)
    (label, actual, allowed), = res.defl_items
    assert label == "Tip deflection" and allowed == pytest.approx(1.2 / 250)
    assert actual == pytest.approx(500 * 1.2**3 / (3 * E * I_R), rel=1e-9)


def test_overhang_checks_span_and_tip_separately():
    Ls, c, P = 0.9, 0.3, 3000.0
    res = bs.analyse(bs.OVER, Ls + c, P, Ls + c, "rectangle", DIMS, MAT, overhang=c, defl_ratio=360)
    (l1, span_actual, span_allowed), (l2, tip_actual, tip_allowed) = res.defl_items
    assert (l1, l2) == ("Span deflection", "Overhang tip deflection")
    assert span_allowed == pytest.approx(Ls / 360) and tip_allowed == pytest.approx(2 * c / 360)
    assert tip_actual == pytest.approx(P * c**2 * (Ls + c) / (3 * E * I_R), rel=1e-4)    # textbook tip deflection
    # the tip load puts an end moment M0 = P c on the simply supported span; its largest lift is M0 Ls^2 / (9 sqrt(3) EI)
    assert span_actual == pytest.approx(P * c * Ls**2 / (9 * np.sqrt(3) * E * I_R), rel=1e-3)


def test_limit_ratio_must_be_positive():
    with pytest.raises(ValidationError, match="Deflection limit"):
        bs.analyse(bs.SS, 1.0, 1000.0, 0.5, "rectangle", DIMS, MAT, defl_ratio=-5)


# ---- the app --------------------------------------------------------------------------------------------
def _app():
    return AppTest.from_file(str(ROOT / "chapters" / "beam_shaft.py"), default_timeout=60).run()


def _clean(at):
    assert not at.exception
    assert not any("Input problem" in e.value or "unexpected" in e.value for e in at.error), [e.value for e in at.error]


@pytest.mark.parametrize("shape,keys", [("hollow_rect", {"bs_rb_SI", "bs_rh_SI", "bs_rt_SI"}),
                                         ("i_beam", {"bs_ibf_SI", "bs_itf_SI", "bs_ih_SI", "bs_itw_SI"})])
def test_app_new_sections(shape, keys):
    at = _app()
    at.selectbox(key="bs_shape_beam").set_value(shape).run()
    _clean(at)
    assert keys <= {n.key for n in at.number_input}
    at.sidebar.radio(key="unit_system").set_value("US").run()
    _clean(at)


def test_app_shaft_mode_has_no_new_sections():
    at = _app()
    at.radio(key="bs_mode").set_value("Shaft (bending + torque)").run()
    assert set(at.selectbox(key="bs_shape_shaft").options) == {bs.SHAPE_LABELS["solid_round"], bs.SHAPE_LABELS["hollow_round"]}


def test_app_custom_material_inputs_and_validation_message():
    at = _app()
    at.selectbox(key="bs_material").set_value(bs.CUSTOM_MATERIAL).run()
    _clean(at)
    assert {"bs_cE_SI", "bs_cSy_SI", "bs_cSu_SI"} <= {n.key for n in at.number_input}
    at.number_input(key="bs_cSu_SI").set_value(100.0).run()           # ultimate below yield (MPa)
    assert not at.exception and any("at least the yield" in e.value for e in at.error)
    at.number_input(key="bs_cSu_SI").set_value(600.0).run()
    _clean(at)


def test_app_deflection_limit_messages():
    at = _app()
    at.selectbox(key="bs_shape_beam").set_value("i_beam").run()
    at.selectbox(key="bs_defl").set_value("L/500").run()
    _clean(at)
    assert any("deflection" in m.value and "allowed" in m.value for m in at.success)
    at.selectbox(key="bs_material").set_value(bs.CUSTOM_MATERIAL).run()
    at.number_input(key="bs_cE_SI").set_value(2000.0).run()           # very soft material (MPa)
    assert any("exceeds the allowed" in e.value for e in at.error)
