from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _run():
    return AppTest.from_file(str(ROOT / "chapters" / "beam_shaft.py"), default_timeout=60).run()


def test_chapter_loads_standalone_with_disclaimer_and_pass():
    at = _run()
    assert not at.exception and not at.error
    assert any("not certified design" in w.value for w in at.warning)
    assert any("PASS" in s.value for s in at.success)
    assert len(at.metric) == 4


def test_zero_diameter_shows_message_not_crash():
    at = _run()
    at.radio(key="bs_mode").set_value("Shaft (bending + torque)").run()
    at.number_input(key="bs_d_SI").set_value(0.0).run()
    assert not at.exception
    assert any("greater than zero" in e.value for e in at.error)


def test_shaft_mode_unit_toggle_and_fatigue():
    at = _run()
    at.radio(key="bs_mode").set_value("Shaft (bending + torque)").run()
    at.checkbox(key="bs_fatigue").check().run()
    assert not at.exception and any("Fatigue" in m.label for m in at.metric)
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception
    assert not any("Input problem" in e.value or "unexpected" in e.value for e in at.error)
    assert at.metric[0].value == "29 ksi"      # 198.9 MPa shown in US units


def test_overload_fails():
    at = _run()
    at.slider(key="bs_load_SI").set_value(20000.0).run()
    assert not at.exception
    assert any("FAIL" in e.value for e in at.error)


def test_shell_lists_both_chapters():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.sidebar.selectbox[0].options) == 2
    at.sidebar.selectbox[0].select_index(1) if "Beam" not in at.title[0].value else None
    at.run()
    assert not at.exception
