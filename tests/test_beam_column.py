"""Step 4 checks: column buckling (Euler / Johnson) and the shaft stress concentration factor Kt."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from chapters import beam_shaft as bs
from common import units
from common.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]
E, SY, SU = 205e9, 370e6, 440e6
MAT = bs.Material("test steel", E, SY, SU)
RECT = dict(b=0.020, h=0.030)                 # buckles about the 20 mm direction
I_MIN = 0.030 * 0.020**3 / 12                 # 20 000 mm^4


# ---- section axes ---------------------------------------------------------------------------------------
def test_rectangle_buckles_about_the_thin_direction():
    sec = bs.column_section("rectangle", **RECT)
    assert sec.I_min == pytest.approx(I_MIN, rel=1e-12) and sec.I_x == pytest.approx(0.020 * 0.030**3 / 12)
    assert sec.r == pytest.approx(0.020 / np.sqrt(12), rel=1e-12)              # r = b / sqrt(12)


def test_round_sections_are_the_same_about_every_axis():
    sec = bs.column_section("solid_round", d=0.020)
    assert sec.I_min == pytest.approx(np.pi * 0.020**4 / 64, rel=1e-12) and sec.r == pytest.approx(0.020 / 4)
    tube = bs.column_section("hollow_round", d_o=0.050, d_i=0.040)
    assert tube.I_min == pytest.approx(np.pi * (0.050**4 - 0.040**4) / 64, rel=1e-12)
    assert tube.r == pytest.approx(np.sqrt(0.050**2 + 0.040**2) / 4, rel=1e-12)   # r = sqrt(Do^2 + Di^2) / 4


def test_i_beam_and_box_tube_use_the_weak_axis():
    bf, tf, h, tw = 0.100, 0.008, 0.200, 0.006
    sec = bs.column_section("i_beam", bf=bf, tf=tf, h=h, tw=tw)
    assert sec.I_y == pytest.approx(2 * tf * bf**3 / 12 + (h - 2 * tf) * tw**3 / 12, rel=1e-12)
    assert sec.I_min == sec.I_y < sec.I_x
    b, hh, t = 0.040, 0.060, 0.004                                              # box tube: weak axis is across b
    box = bs.column_section("hollow_rect", b=b, h=hh, t=t)
    outer, inner = hh * b**3 / 12, (hh - 2 * t) * (b - 2 * t) ** 3 / 12
    assert box.I_y == pytest.approx(outer - inner, rel=1e-12) and box.I_min == box.I_y


# ---- Euler --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("end,K", [("Pinned-pinned", 1.0), ("Fixed-free (flagpole)", 2.0),
                                   ("Fixed-pinned", 0.7), ("Fixed-fixed", 0.5)])
def test_euler_load_for_each_end_condition(end, K):
    L = 3.0                                                                      # long enough to be in the Euler range
    res = bs.analyse_column(end, L, 1000.0, "rectangle", RECT, MAT)
    expected = np.pi**2 * E * I_MIN / (K * L) ** 2                              # textbook pi^2 E I / (K L)^2
    assert res.regime == "Euler" and res.K == K
    assert res.P_cr == pytest.approx(expected, rel=1e-12)
    assert res.sf == pytest.approx(expected / 1000.0, rel=1e-12)
    assert res.slenderness == pytest.approx(K * L / (0.020 / np.sqrt(12)), rel=1e-12)


def test_hand_calculation_pinned_steel_bar():
    """20 x 30 mm steel bar, 1.5 m, pinned: lambda = 259.8 > 104.6, Pcr = pi^2 E I / L^2 = 18.0 kN."""
    res = bs.analyse_column("Pinned-pinned", 1.5, 10000.0, "rectangle", RECT, MAT)
    assert res.slenderness == pytest.approx(259.8, abs=0.1)
    assert res.P_cr == pytest.approx(np.pi**2 * 205e9 * 2.0e-8 / 1.5**2, rel=1e-12)
    assert res.P_cr / 1000 == pytest.approx(17.98, abs=0.01)


# ---- Johnson -----------------------------------------------------------------------------------------------
def test_critical_slenderness_and_continuity_with_euler():
    lam_c = bs.critical_slenderness(E, SY)
    assert lam_c == pytest.approx(np.sqrt(2 * np.pi**2 * E / SY), rel=1e-12) and lam_c == pytest.approx(104.58, abs=0.01)
    euler, johnson = bs.column_curve(E, SY, [lam_c])
    assert euler[0] == pytest.approx(johnson[0], rel=1e-12) and johnson[0] == pytest.approx(SY / 2, rel=1e-12)
    # the slopes match too, so there is no kink in the strength chart
    d = 1e-4
    e1, j1 = bs.column_curve(E, SY, [lam_c * (1 + d)])
    e0, j0 = bs.column_curve(E, SY, [lam_c * (1 - d)])
    assert (e1[0] - e0[0]) == pytest.approx(j1[0] - j0[0], rel=1e-3)


def test_short_column_uses_johnson_and_hand_value():
    """20 x 30 mm, 0.5 m pinned: lambda = 86.6 < 104.6, Pcr = A Sy (1 - Sy lam^2 / (4 pi^2 E))."""
    res = bs.analyse_column("Pinned-pinned", 0.5, 10000.0, "rectangle", RECT, MAT)
    lam = 0.5 / (0.020 / np.sqrt(12))
    A = 0.020 * 0.030
    expected = A * SY * (1 - SY * lam**2 / (4 * np.pi**2 * E))
    assert res.regime == "Johnson" and res.slenderness == pytest.approx(lam, rel=1e-12)
    assert res.P_cr == pytest.approx(expected, rel=1e-12) and res.P_cr < res.P_euler      # Euler would overpredict
    assert res.P_cr < A * SY                                                               # below the squash load
    assert res.P_cr / 1000 == pytest.approx(expected / 1000)


def test_regime_switches_at_the_critical_slenderness():
    r = 0.020 / np.sqrt(12)
    lam_c = bs.critical_slenderness(E, SY)
    below = bs.analyse_column("Pinned-pinned", (lam_c - 0.01) * r, 1000.0, "rectangle", RECT, MAT)
    above = bs.analyse_column("Pinned-pinned", (lam_c + 0.01) * r, 1000.0, "rectangle", RECT, MAT)
    assert below.regime == "Johnson" and above.regime == "Euler"
    assert below.P_cr == pytest.approx(above.P_cr, rel=1e-3)                   # no jump across the boundary


def test_critical_load_never_exceeds_squash_load_and_is_monotonic():
    loads = [bs.analyse_column("Pinned-pinned", L, 1000.0, "rectangle", RECT, MAT).P_cr for L in np.linspace(0.05, 4, 60)]
    assert max(loads) <= 0.020 * 0.030 * SY + 1e-6
    assert all(a >= b for a, b in zip(loads, loads[1:]))


def test_safety_factor_and_pass_fail():
    res = bs.analyse_column("Pinned-pinned", 1.5, 1000.0, "rectangle", RECT, MAT, n_req=1.5)
    assert res.sf == pytest.approx(res.P_cr / 1000.0) and res.passed
    res = bs.analyse_column("Pinned-pinned", 1.5, 20000.0, "rectangle", RECT, MAT, n_req=1.5)
    assert res.sf < 1 and not res.passed
    assert bs.analyse_column("Pinned-pinned", 1.5, 0.0, "rectangle", RECT, MAT).sf == float("inf")
    assert res.sigma_applied == pytest.approx(20000.0 / (0.020 * 0.030))


@pytest.mark.parametrize("kwargs,match", [
    (dict(L=0.0), "Length"), (dict(L=-1.0), "Length"), (dict(P=-5.0), "Axial load"), (dict(end="magic"), "Unknown end"),
    (dict(n_req=0.0), "Required safety factor"), (dict(material=bs.Material("c", 0.0, SY, SU)), "E"),
    (dict(material=bs.Material("c", E, 500e6, 400e6)), "at least the yield"),
    (dict(dims=dict(b=0.0, h=0.03)), "Width"),
])
def test_column_validation(kwargs, match):
    args = dict(end="Pinned-pinned", L=1.0, P=1000.0, shape="rectangle", dims=RECT, material=MAT, n_req=1.5)
    args.update(kwargs)
    with pytest.raises(ValidationError, match=match):
        bs.analyse_column(args["end"], args["L"], args["P"], args["shape"], args["dims"], args["material"], args["n_req"])


# ---- buckling mode shapes ----------------------------------------------------------------------------------
def test_mode_shape_boundary_conditions():
    xi = np.linspace(0, 1, 100001)
    d = 1e-6
    f = lambda end, x: bs.buckled_shape(end, np.array(x))
    for end in bs.COLUMN_ENDS:
        y = bs.buckled_shape(end, xi)
        assert np.max(np.abs(y)) == pytest.approx(1.0, abs=1e-5)      # scaled to a peak of 1 (amplitude is arbitrary)
    # base: zero deflection everywhere; zero slope except for the pinned base
    for end in ("Fixed-free (flagpole)", "Fixed-pinned", "Fixed-fixed"):
        assert f(end, [0.0])[0] == pytest.approx(0, abs=1e-12) and f(end, [d])[0] == pytest.approx(0, abs=1e-9)
    assert f("Pinned-pinned", [0.0])[0] == pytest.approx(0, abs=1e-12) and f("Pinned-pinned", [1.0])[0] == pytest.approx(0, abs=1e-12)
    # pinned top
    for end in ("Fixed-pinned", "Pinned-pinned"):
        assert f(end, [1.0])[0] == pytest.approx(0, abs=1e-9)
    # fixed top (fixed-fixed): zero slope and deflection at both ends; flagpole: free top at maximum
    assert f("Fixed-fixed", [1.0])[0] == pytest.approx(0, abs=1e-12) and f("Fixed-fixed", [1 - d])[0] == pytest.approx(0, abs=1e-9)
    assert f("Fixed-free (flagpole)", [1.0])[0] == pytest.approx(1.0)


def test_fixed_pinned_mode_satisfies_the_buckling_equation():
    """y'' + k^2 y = const for the fixed-pinned mode, and the bending moment vanishes at the pinned end."""
    k = bs.KL_FIXED_PINNED
    assert np.tan(k) == pytest.approx(k, rel=1e-12)                              # tan(kL) = kL
    x = np.linspace(0, 1, 20001)
    y = bs._mode("Fixed-pinned", x)
    ypp = np.gradient(np.gradient(y, x), x)
    assert abs(ypp[-2]) < 1e-3 * np.max(np.abs(ypp))                              # M ~ y'' = 0 at the pinned top
    # effective length factor from the root: K = pi / kL = 0.699, which the chapter rounds to 0.7
    assert np.pi / k == pytest.approx(0.699, abs=1e-3)


# ---- Kt in the Goodman check --------------------------------------------------------------------------------
SHAFT = dict(d=0.040)


def _shaft(kt, R=-1.0, T=150.0, P=1000.0):
    return bs.analyse(bs.SS, 1.0, P, 0.5, "solid_round", SHAFT, MAT, T, 1.5, fatigue_R=R, fatigue_k=0.6, kt=kt)


def test_kt_one_is_the_old_result():
    assert _shaft(1.0).sf_fatigue == pytest.approx(
        bs.analyse(bs.SS, 1.0, 1000.0, 0.5, "solid_round", SHAFT, MAT, 150.0, 1.5, fatigue_R=-1.0, fatigue_k=0.6).sf_fatigue)


@pytest.mark.parametrize("R", [-1.0, 0.0, 0.5])
def test_goodman_with_kt_matches_hand_formula(R):
    kt = 2.2
    res = _shaft(kt, R=R)
    s_a, s_m = res.vm_max * (1 - R) / 2, res.vm_max * (1 + R) / 2
    Se = 0.6 * min(0.5 * SU, 700e6)
    assert res.sf_fatigue == pytest.approx(1 / (kt * s_a / Se + s_m / SU), rel=1e-12)


def test_kt_only_touches_the_alternating_part_and_not_the_static_factor():
    base, notched = _shaft(1.0, R=0.0), _shaft(2.0, R=0.0)
    assert notched.sf == base.sf and notched.vm_max == base.vm_max                    # static check unchanged
    assert notched.sf_fatigue < base.sf_fatigue
    steady = _shaft(3.0, R=1.0)                                                       # R = 1: no alternating stress at all
    assert steady.sf_fatigue == pytest.approx(_shaft(1.0, R=1.0).sf_fatigue, rel=1e-12)
    assert _shaft(2.0, R=-1.0).sf_fatigue == pytest.approx(_shaft(1.0, R=-1.0).sf_fatigue / 2, rel=1e-12)  # fully reversed: n ~ 1/Kt


def test_kt_validation():
    for bad in (0.5, 11.0):
        with pytest.raises(ValidationError, match="Kt"):
            _shaft(bad)


# ---- figures and the app -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("end", list(bs.COLUMN_ENDS))
@pytest.mark.parametrize("L", [0.4, 2.5])
def test_column_figures_build(end, L):
    res = bs.analyse_column(end, L, 30000.0, "rectangle", RECT, MAT)
    assert len(bs.column_figure(res, units.SI).data) >= 2
    fig = bs.strength_chart_figure(res, MAT, units.SI)
    assert len(fig.data) >= 5
    assert bs.strength_chart_figure(res, MAT, units.US) is not None


def _app():
    return AppTest.from_file(str(ROOT / "chapters" / "beam_shaft.py"), default_timeout=60).run()


def _clean(at):
    assert not at.exception
    assert not any("Input problem" in e.value or "unexpected" in e.value for e in at.error), [e.value for e in at.error]


@pytest.mark.parametrize("end", list(bs.COLUMN_ENDS))
@pytest.mark.parametrize("shape", list(bs.SHAPE_LABELS))
def test_app_column_mode(end, shape):
    at = _app()
    at.radio(key="bs_mode").set_value(bs.MODES[2]).run()
    at.selectbox(key="bs_cend").set_value(end).run()
    at.selectbox(key="bs_shape_col").set_value(shape).run()
    _clean(at)
    assert len(at.metric) == 4 and any("buckling" in m.value for m in [*at.success, *at.error])


def test_app_column_validation_and_units():
    at = _app()
    at.radio(key="bs_mode").set_value(bs.MODES[2]).run()
    at.number_input(key="bs_clen_SI_box").set_value(0.1).run()
    _clean(at)
    at.sidebar.radio(key="unit_system").set_value("US").run()
    _clean(at)
    assert any("lbf" in s.label for s in at.slider)


def test_app_kt_slider_only_in_shaft_mode_with_fatigue():
    at = _app()
    assert "bs_kt" not in {s.key for s in at.slider}
    at.radio(key="bs_mode").set_value(bs.MODES[1]).run()
    at.checkbox(key="bs_fatigue").set_value(True).run()
    _clean(at)
    assert "bs_kt" in {s.key for s in at.slider}
    at.slider(key="bs_kt").set_value(2.5).run()
    _clean(at)
    at.radio(key="bs_mode").set_value(bs.MODES[0]).run()
    assert "bs_kt" not in {s.key for s in at.slider}


def test_app_modes_still_load_and_switching_back_works():
    at = _app()
    for mode in (bs.MODES[2], bs.MODES[1], bs.MODES[0]):
        at.radio(key="bs_mode").set_value(mode).run()
        _clean(at)
