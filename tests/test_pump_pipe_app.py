from pathlib import Path

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
