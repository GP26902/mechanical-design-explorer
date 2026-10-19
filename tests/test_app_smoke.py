from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_shell_loads_with_chapter_and_disclaimer():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("not certified design" in w.value for w in at.warning)
    assert at.sidebar.selectbox[0].options  # chapter dropdown populated


def test_invalid_input_shows_message_not_crash():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    at.number_input[0].set_value(0).run()
    assert not at.exception
    assert any("greater than zero" in e.value for e in at.error)


def test_example_runs_standalone_with_disclaimer():
    at = AppTest.from_file(str(ROOT / "chapters" / "example_hello.py"), default_timeout=30).run()
    assert not at.exception
    assert any("not certified design" in w.value for w in at.warning)
    assert len(at.metric) == 1


def test_template_runs_standalone():
    at = AppTest.from_file(str(ROOT / "chapters" / "_template.py"), default_timeout=30).run()
    assert not at.exception
