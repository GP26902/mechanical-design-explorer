from pathlib import Path

from streamlit.testing.v1 import AppTest

from common.discovery import discover_chapters

ROOT = Path(__file__).resolve().parents[1]


def _app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def test_chapters_declare_optional_prefix_and_about():
    chapters = {c.name: c for c in discover_chapters(ROOT / "chapters")}
    for name, prefix in [("beam_shaft", "bs_"), ("tolerance_stack", "ts_"), ("pump_pipe", "pp_")]:
        assert chapters[name].key_prefix == prefix and "Limitations" in chapters[name].about
    assert chapters["example_hello"].key_prefix == "" and chapters["example_hello"].about == ""


def test_typing_exact_value_sets_slider_to_exact_value():
    at = _app()
    at.number_input(key="bs_span_SI_box").set_value(1.37).run()
    assert not at.exception
    assert at.slider(key="bs_span_SI").value == 1.37
    assert at.metric  # still computing


def test_moving_slider_updates_box():
    at = _app()
    at.slider(key="bs_load_SI").set_value(12000.0).run()
    assert at.number_input(key="bs_load_SI_box").value == 12000.0


def test_reset_button_restores_defaults():
    at = _app()
    at.slider(key="bs_load_SI").set_value(12000.0).run()
    at.sidebar.button(key="reset_defaults").click().run()
    assert not at.exception
    assert at.slider(key="bs_load_SI").value == 5000.0
    assert at.number_input(key="bs_load_SI_box").value == 5000.0


def test_sidebar_has_about_and_links():
    at = _app()
    assert any("About this chapter" in e.label for e in at.sidebar.expander)
    assert any("GitHub repository" in m.value for m in at.sidebar.markdown)


def test_standalone_chapters_show_reset_and_about():
    for name in ("beam_shaft", "tolerance_stack", "pump_pipe"):
        at = AppTest.from_file(str(ROOT / "chapters" / f"{name}.py"), default_timeout=60).run()
        assert not at.exception, name
        assert at.sidebar.button(key="reset_defaults")
