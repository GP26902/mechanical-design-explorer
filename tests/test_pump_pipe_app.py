from pathlib import Path
import pytest
from chapters import pump_pipe as pp_mod

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _run():
    return AppTest.from_file(str(ROOT / "chapters" / "pump_pipe.py"), default_timeout=60).run()


def _labels(at):
    return [m.label for m in at.metric]


def test_chapter_loads_standalone_with_disclaimer_and_results():
    at = _run()
    assert not at.exception
    assert any("not certified design" in w.value for w in at.warning)
    assert _labels(at)[:4] == ["Operating flow", "Pump head", "Largest pressure drop", "NPSH margin"]
    assert at.metric[2].value == "Discharge pipe"                          # default bottleneck
    assert any("No cavitation expected" in s.value for s in at.success)
    assert "L/s" in at.metric[0].value and "m" in at.metric[1].value


def test_raising_pump_speed_raises_flow():
    at = _run()
    q0 = float(at.metric[0].value.split()[0])
    at.slider(key="pp_speed").set_value(110).run()
    assert not at.exception and float(at.metric[0].value.split()[0]) > q0


def test_cavitation_is_flagged_for_suction_lift_with_hot_water():
    at = _run()
    at.slider(key="pp_zs_SI").set_value(-5.0).run()
    at.slider(key="pp_temp").set_value(70).run()
    assert not at.exception
    assert any("Cavitation risk" in e.value for e in at.error)


def test_weak_pump_shows_no_operating_point_message_not_a_crash():
    at = _run()
    at.number_input(key="pp_h0_SI").set_value(12.0).run()                   # shutoff 12 m below the 30 m design head
    assert not at.exception
    assert any("Input problem" in e.value for e in at.error)               # rejected with a message
    at = _run()
    at.number_input(key="pp_h0_SI").set_value(31.0).run()                   # valid pump...
    at.slider(key="pp_zd_SI").set_value(40.0).run()                         # ...but static head 39 m > 31 m shutoff
    assert not at.exception
    assert any("No operating point" in e.value for e in at.error)


def test_bad_pump_inputs_show_validation_message():
    at = _run()
    at.number_input(key="pp_qd_SI").set_value(0.0).run()
    assert not at.exception
    assert any("Input problem" in e.value and "Pump design flow" in e.value for e in at.error)


def test_us_units_and_velocity_colouring_do_not_crash():
    at = _run()
    at.radio(key="pp_color").set_value("velocity").run()
    assert not at.exception
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception
    assert "gpm" in at.metric[0].value
    assert not any("Input problem" in e.value or "unexpected" in e.value for e in at.error)


def test_energy_and_control_section_shows_and_responds():
    at = _run()
    assert not at.exception
    assert _labels(at)[4:] == ["Pump efficiency", "Shaft power", "Electrical input", "Energy cost per year",
                                "Throttle valve, electrical", "Variable speed, electrical", "Power saved", "Saved per year"]
    assert "kW" in at.metric[5].value and at.metric[7].value.startswith("$")
    saved_70 = float(at.metric[11].value.lstrip("$").replace(",", ""))
    at.slider(key="pp_target").set_value(40).run()
    assert not at.exception and float(at.metric[11].value.lstrip("$").replace(",", "")) > saved_70   # lower flow, bigger saving


def test_energy_section_in_us_units_and_hidden_without_operating_point():
    at = _run()
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception and "hp" in at.metric[5].value
    at = _run()
    at.number_input(key="pp_h0_SI").set_value(31.0).run()
    at.slider(key="pp_zd_SI").set_value(40.0).run()
    assert not at.exception and "Pump efficiency" not in _labels(at)


def test_two_pumps_raise_the_flow_and_label_the_arrangement():
    at = _run()
    q1 = float(at.metric[0].value.split()[0])
    at.selectbox(key="pp_npumps").set_value(2).run()
    assert not at.exception and float(at.metric[0].value.split()[0]) > q1                 # parallel by default
    q_par = float(at.metric[0].value.split()[0])
    at.selectbox(key="pp_arrangement").set_value("series").run()
    assert not at.exception and float(at.metric[0].value.split()[0]) > q1 and not any("Input problem" in e.value for e in at.error)
    assert q_par != float(at.metric[0].value.split()[0])


def test_fitted_from_points_mode_loads_shows_the_fit_and_works_in_us_units():
    at = _run()
    at.radio(key="pp_curve_mode").set_value("Fitted from points").run()
    assert not at.exception and not any("Input problem" in e.value for e in at.error)
    assert any("Fitted pump curve" in i.value and "Residual" in i.value for i in at.info)
    assert _labels(at)[:4] == ["Operating flow", "Pump head", "Largest pressure drop", "NPSH margin"]
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception and any("Fitted pump curve" in i.value for i in at.info)
    at.number_input(key="pp_qd_US").set_value(1500.0).run()                                  # design flow beyond zero-head flow
    assert not at.exception and any("beyond the flow where the fitted head reaches zero" in e.value for e in at.error)


def _q(at):
    return float(at.metric[0].value.split()[0])


def test_custom_roughness_and_k_inputs_change_the_flow():
    at = _run()
    q0 = _q(at)
    at.selectbox(key="pp_material").set_value("Custom roughness").run()
    assert not at.exception and _q(at) == pytest.approx(q0, rel=1e-3)                  # default 0.045 mm = commercial steel
    at.number_input(key="pp_eps_SI").set_value(0.5).run()
    assert not at.exception and _q(at) < q0
    at.number_input(key="pp_eps_SI").set_value(6.0).run()                              # 6 mm in a 100 mm pipe: above the 0.05 limit
    assert any("roughness" in e.value for e in at.error)
    at = _run()
    at.selectbox(key="pp_valve").set_value("Custom K").run()
    at.number_input(key="pp_kvalve").set_value(30.0).run()
    assert not at.exception and _q(at) < q0


def test_tank_pressure_and_altitude_inputs():
    at = _run()
    q0 = _q(at)
    at.number_input(key="pp_pd_SI").set_value(100.0).run()                             # 100 kPa on the discharge tank
    assert not at.exception and _q(at) < q0
    at = _run()
    at.number_input(key="pp_ps_SI").set_value(-200.0).run()                            # below a perfect vacuum
    assert any("vacuum" in e.value for e in at.error)
    at = _run()
    at.number_input(key="pp_elev_SI").set_value(3000.0).run()
    assert not at.exception and any("Atmospheric pressure at the site: 70" in c.value for c in at.caption)


def test_custom_fluid_disables_temperature_and_flags_bad_values():
    at = _run()
    assert not at.slider(key="pp_temp").disabled
    at.radio(key="pp_fluid").set_value("Custom fluid").run()
    assert not at.exception and at.slider(key="pp_temp").disabled
    assert any("ignored for a custom fluid" in c.value for c in at.caption)
    assert any("custom fluid (density" in c.value for c in at.caption)
    at.number_input(key="pp_rho_SI").set_value(0.0).run()
    assert not at.exception and any("Density" in e.value for e in at.error)
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception


@pytest.mark.parametrize("name", list(pp_mod.EXAMPLES))
def test_every_example_loads_in_the_app_with_the_same_flow_as_the_calculation(name):
    at = _run()
    if name != pp_mod.DEFAULT_EXAMPLE:
        at.selectbox(key="pp_example").set_value(name).run()
    assert not at.exception and not any("Input problem" in e.value for e in at.error)   # the cavitation box is also an error element
    lay, pump, temp, fluid = pp_mod.build_example(name)
    q = pp_mod.analyse(lay, pump, 1750.0, temp, fluid=fluid).q_op * 1000
    assert _q(at) == pytest.approx(q, rel=1e-2)                                 # 3 significant digits shown
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception and not any("Input problem" in e.value for e in at.error)   # the cavitation box is also an error element


def test_example_inputs_stay_editable_and_downloads_are_offered():
    at = _run()
    assert len(at.get("download_button")) == 2
    at.selectbox(key="pp_example").set_value("Undersized discharge pipe").run()
    q_small = _q(at)
    at.slider(key="pp_dd_e1_SI").set_value(100.0).run()                         # widen the pipe back to 100 mm
    assert not at.exception and _q(at) > 1.5 * q_small
    at.selectbox(key="pp_example").set_value(pp_mod.DEFAULT_EXAMPLE).run()
    assert not at.exception and at.slider(key="pp_dd_SI").value == 100.0
