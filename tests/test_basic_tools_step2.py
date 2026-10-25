"""Chapter 4, step 2: pipe sizes, wall thickness under pressure, plate thickness sensitivity (hand values in comments)."""
import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from chapters import basic_tools as bt
from common.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]


# ---- pipe sizes ----------------------------------------------------------------------------------
def test_pipe_size_hand_values_nps2_sch40():
    # NPS 2: OD 60.3, Sch 40 wall 3.91 -> ID = 60.3 - 7.82 = 52.48 mm; area = pi/4 * 0.05248^2 = 2.1631e-3 m^2
    ps = bt.pipe_size("2", 40)
    assert (ps.od_mm, ps.wall_mm) == (60.3, 3.91) and ps.id_mm == pytest.approx(52.48)
    assert ps.flow_area_m2 == pytest.approx(2.1631e-3, rel=1e-4)
    assert bt.pipe_size("2", 80).id_mm == pytest.approx(60.3 - 2 * 5.54)                                 # Sch 80 wall 5.54
    assert bt.pipe_size("6", 80).wall_mm == 10.97 and bt.pipe_size("8", 40).od_mm == 219.1


def test_pipe_velocity_hand_value():
    # 10 L/s through an NPS 2 Sch 40 pipe: v = Q / A = 0.010 / 2.1631e-3 = 4.623 m/s
    assert bt.pipe_velocity_m_s(0.010, 52.48) == pytest.approx(4.623, abs=2e-3)
    assert bt.pipe_velocity_m_s(0.020, 52.48) == pytest.approx(2 * bt.pipe_velocity_m_s(0.010, 52.48))


def test_pipe_table_is_consistent():
    sizes = list(bt.PIPE_SIZES)
    ods = [bt.PIPE_SIZES[n][0] for n in sizes]
    assert ods == sorted(ods) and len(set(ods)) == len(ods)                                              # OD grows with NPS
    for n in sizes:
        od, w40, w80 = bt.PIPE_SIZES[n]
        assert 0 < w40 < w80 < od / 2                                                                    # heavier schedule, thicker wall, positive bore
        assert bt.pipe_size(n, 80).id_mm < bt.pipe_size(n, 40).id_mm


@pytest.mark.parametrize("call", [lambda: bt.pipe_size("7", 40), lambda: bt.pipe_size("2", 160), lambda: bt.pipe_velocity_m_s(0, 52.48),
                                  lambda: bt.pipe_velocity_m_s(0.01, -1)])
def test_bad_pipe_inputs_raise(call):
    with pytest.raises(ValidationError):
        call()


# ---- wall thickness under pressure -----------------------------------------------------------------
def test_wall_stresses_by_hand_and_force_balance():
    # p = 10 MPa, Di = 100 mm, t = 10 mm -> ri = 50, ro = 60 mm
    w = bt.wall_stresses(10e6, 0.100, 0.010)
    assert w.hoop_thin == pytest.approx(50e6) and w.axial_thin == pytest.approx(25e6) and w.d_over_t == pytest.approx(10.0)
    assert w.hoop_thick == pytest.approx(10e6 * (3600 + 2500) / (3600 - 2500))                       # p (ro^2 + ri^2)/(ro^2 - ri^2) = 55.4545 MPa
    assert w.hoop_thick == pytest.approx(55.4545e6, rel=1e-6)
    assert w.axial_thick == pytest.approx(10e6 * 2500 / 1100)                                          # p ri^2/(ro^2 - ri^2) = 22.7273 MPa
    assert w.axial_thick == pytest.approx(22.7273e6, rel=2e-6) and w.radial_inner == -10e6
    ri, ro = 0.05, 0.06
    assert w.axial_thick * (ro**2 - ri**2) == pytest.approx(10e6 * ri**2)                               # axial force balance: sigma_a (ro^2 - ri^2) = p ri^2
    A, B = w.axial_thick, 10e6 * ri**2 * ro**2 / (ro**2 - ri**2)                                       # Lame: sigma_h(r) = A + B / r^2
    assert A + B / ri**2 == pytest.approx(w.hoop_thick)
    assert A * (ro - ri) + B * (1 / ri - 1 / ro) == pytest.approx(10e6 * ri)                           # hoop force per length balances p ri


def test_thin_and_thick_wall_agree_for_a_thin_wall():
    w = bt.wall_stresses(1e6, 1.0, 0.002)                                                              # D/t = 500
    assert w.hoop_thin == pytest.approx(w.hoop_thick, rel=2e-3)
    assert bt.wall_stresses(10e6, 0.1, 0.010).hoop_thin < bt.wall_stresses(10e6, 0.1, 0.010).hoop_thick       # thin-wall is low for a thick wall


def test_required_wall_by_hand_and_round_trip():
    # S = 100 MPa: thin t = p Di / (2 S) = 10e6 * 0.1 / 200e6 = 5.0 mm; Lame ro = 50 sqrt(110/90) = 55.277 -> t = 5.277 mm
    assert bt.required_wall_thin_m(10e6, 0.1, 100e6) == pytest.approx(0.005)
    t = bt.required_wall_thick_m(10e6, 0.1, 100e6)
    assert t == pytest.approx(0.0052771, abs=1e-7) and t > 0.005
    assert bt.wall_stresses(10e6, 0.1, t).hoop_thick == pytest.approx(100e6, rel=1e-12)             # a wall sized for S gives exactly S
    assert bt.required_wall_thick_m(10e6, 0.1, 100e6, 0.0015) == pytest.approx(t + 0.0015)           # corrosion allowance is added
    assert bt.required_wall_thin_m(10e6, 0.1, 100e6, 0.0015) == pytest.approx(0.0065)


@pytest.mark.parametrize("call", [
    lambda: bt.wall_stresses(0, 0.1, 0.01), lambda: bt.wall_stresses(1e6, 0, 0.01), lambda: bt.wall_stresses(1e6, 0.1, -0.01),
    lambda: bt.required_wall_thick_m(100e6, 0.1, 100e6), lambda: bt.required_wall_thick_m(120e6, 0.1, 100e6),
    lambda: bt.required_wall_thin_m(1e6, 0.1, 0), lambda: bt.required_wall_thin_m(1e6, 0.1, 100e6, -0.001)])
def test_bad_pressure_inputs_raise(call):
    with pytest.raises(ValidationError):
        call()


def test_allowable_stress_not_above_pressure_message():
    with pytest.raises(ValidationError, match="allowable stress must be greater than the pressure"):
        bt.required_wall_thick_m(100e6, 0.1, 100e6)


# ---- thickness sensitivity ----------------------------------------------------------------------------
def test_cantilever_plate_hand_values_and_textbook_cross_check():
    # P = 100 N, L = 200 mm, b = 50 mm, t = 4 mm, E = 200 GPa: delta = 4 P L^3 / (E b t^3) = 4*100*0.008 / (200e9 * 0.05 * 6.4e-11) = 5.0 mm
    r = bt.cantilever_plate(100, 0.2, 0.05, 0.004, 200e9)
    assert r.deflection_m == pytest.approx(0.005) and r.stiffness_n_per_m == pytest.approx(20000.0)
    assert r.root_stress_pa == pytest.approx(150e6)                                                   # 6 P L / (b t^2) = 120 / 8e-7
    I = 0.05 * 0.004**3 / 12                                                                          # textbook: delta = P L^3 / (3 E I), sigma = M c / I
    assert r.deflection_m == pytest.approx(100 * 0.2**3 / (3 * 200e9 * I))
    assert r.root_stress_pa == pytest.approx(100 * 0.2 * 0.002 / I)


def test_cube_and_square_laws_in_thickness():
    a, b = bt.cantilever_plate(100, 0.2, 0.05, 0.004, 200e9), bt.cantilever_plate(100, 0.2, 0.05, 0.008, 200e9)
    assert b.deflection_m == pytest.approx(a.deflection_m / 8) and b.stiffness_n_per_m == pytest.approx(8 * a.stiffness_n_per_m)
    assert b.root_stress_pa == pytest.approx(a.root_stress_pa / 4)
    up = bt.cantilever_plate(100, 0.2, 0.05, 0.0044, 200e9)                                           # 10 % thicker: 1.1^3 = 1.331 stiffer, 1/1.1^2 = 0.826 stress
    assert up.stiffness_n_per_m / a.stiffness_n_per_m == pytest.approx(1.331) and up.root_stress_pa / a.root_stress_pa == pytest.approx(1 / 1.21)
    al = bt.cantilever_plate(100, 0.2, 0.05, 0.004, 69e9)
    assert al.deflection_m == pytest.approx(0.005 * 200 / 69) and al.root_stress_pa == pytest.approx(a.root_stress_pa)   # stress does not depend on E


@pytest.mark.parametrize("args", [(0, .2, .05, .004, 200e9), (100, 0, .05, .004, 200e9), (100, .2, -1, .004, 200e9),
                                  (100, .2, .05, 0, 200e9), (100, .2, .05, .004, 0), (100, .2, .05, .004, float("nan"))])
def test_bad_plate_inputs_raise(args):
    with pytest.raises(ValidationError):
        bt.cantilever_plate(*args)


# ---- the app ---------------------------------------------------------------------------------------------
def _run():
    return AppTest.from_file(str(ROOT / "chapters" / "basic_tools.py"), default_timeout=30).run()


def _val(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_five_tabs_and_pipe_tab_values():
    at = _run()
    assert not at.exception and len(at.tabs) == 6
    assert _val(at, "Inside diameter") == "52.48 mm" and _val(at, "Outside diameter") == "60.3 mm" and _val(at, "Wall thickness") == "3.91 mm"
    assert _val(at, "Mean velocity at that flow") == "4.62 m/s"                                      # 10 L/s default
    at.radio(key="bt_sch").set_value(80).run()
    assert not at.exception and _val(at, "Inside diameter") == "49.22 mm"
    at.number_input(key="bt_pipe_q_SI").set_value(0.0).run()
    assert not at.exception and any("Flow" in e.value for e in at.error)


def test_wall_tab_pass_fail_and_messages():
    at = _run()
    # 5 MPa, Di 100 mm, t 4 mm: thin = p ri / t = 62.5 MPa; Lame = 5e6 (0.054^2 + 0.05^2)/(0.054^2 - 0.05^2) = 65.1 MPa; required wall 2.57 mm
    assert _val(at, "Hoop stress, thin-wall") == "62.5 MPa" and _val(at, "Hoop stress, thick-wall (Lamé)") == "65.1 MPa"
    assert _val(at, "Required wall, Lamé") == "2.57 mm"
    assert any(s.value.startswith("Pass.") for s in at.success)
    at.number_input(key="bt_wall_p_SI").set_value(20000.0).run()                                     # 20 MPa: 260 % of the allowable stress
    assert not at.exception and any(e.value.startswith("Fail.") for e in at.error)
    assert not any("thin-wall formula is not reliable" in i.value for i in at.info)                  # D/t = 25 here, thin-wall is fine
    at.slider(key="bt_wall_t_SI").set_value(10.0).run()                                              # 10 mm wall: D/t = 10, 111 % of allowable
    assert not at.exception and any(e.value.startswith("Fail.") and "111 %" in e.value for e in at.error)
    assert any("thin-wall formula is not reliable" in i.value for i in at.info)
    at.number_input(key="bt_wall_ca_SI").set_value(12.0).run()                                       # corrosion allowance larger than the 10 mm wall
    assert any("corrosion allowance must be smaller" in e.value for e in at.error) and not at.exception


def test_plate_tab_values_warnings_and_units():
    at = _run()
    assert _val(at, "Tip deflection") == "5 mm" and _val(at, "Root bending stress") == "150 MPa"
    at.selectbox(key="bt_plate_mat").set_value("Aluminium (typical)").run()
    assert not at.exception and _val(at, "Tip deflection") == "14.5 mm"
    at.slider(key="bt_plate_t_SI").set_value(2.0).run()                                              # 8x the deflection: 116 mm, over 10 % of the length
    assert not at.exception and any("more than 10 %" in w.value for w in at.warning)
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception and "in" in _val(at, "Tip deflection") and "ksi" in _val(at, "Root bending stress")
