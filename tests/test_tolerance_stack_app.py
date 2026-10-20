from pathlib import Path

from streamlit.testing.v1 import AppTest

from chapters import tolerance_stack as ts
from common import units

ROOT = Path(__file__).resolve().parents[1]


def _run():
    return AppTest.from_file(str(ROOT / "chapters" / "tolerance_stack.py"), default_timeout=60).run()


def test_chapter_loads_standalone_with_disclaimer_and_both_tabs():
    at = _run()
    assert not at.exception
    assert any("not certified design" in w.value for w in at.warning)
    assert len(at.tabs) == 2 and len(at.metric) == 8
    assert at.metric[0].value == units.fmt(5e-3, "length", "SI", 4)
    assert any("RSS is inside" in w.value for w in at.warning)          # default stack: worst-case 4.75-5.25 vs spec 4.8-5.2
    assert any("PASS" in s.value for s in at.success)                   # default GD&T hole passes with bonus


def test_bad_spec_limits_show_message_and_gdt_tab_still_works():
    at = _run()
    at.number_input(key="ts_lsl_SI").set_value(6.0).run()               # LSL above USL
    assert not at.exception
    assert any("Input problem" in e.value and "Lower spec limit" in e.value for e in at.error)
    assert any("Bonus" in m.label for m in at.metric)                   # other tab unaffected


def test_gdt_min_above_max_shows_message_and_stack_tab_still_works():
    at = _run()
    at.number_input(key="ts_g_smin_hole_SI").set_value(11.0).run()
    assert not at.exception
    assert any("Input problem" in e.value and "Minimum size" in e.value for e in at.error)
    assert any("Cpk" in m.label for m in at.metric)


def test_gdt_large_offset_fails_and_rfs_toggle_removes_bonus():
    at = _run()
    at.slider(key="ts_g_dx_hole_SI").set_value(0.5).run()
    assert not at.exception and any("FAIL" in e.value for e in at.error)
    at = _run()
    at.radio(key="ts_g_mod").set_value(ts.RFS).run()                    # 0.31 offset diameter > 0.20 with no bonus
    assert not at.exception and any("FAIL" in e.value for e in at.error)
    assert at.metric[4].value == units.fmt(0.0, "length", "SI", 4)      # bonus metric


def test_pin_feature_and_us_units_do_not_crash():
    at = _run()
    at.radio(key="ts_g_feature").set_value(ts.PIN).run()
    assert not at.exception and any("PASS" in s.value for s in at.success)
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception
    assert not any("Input problem" in e.value or "unexpected" in e.value for e in at.error)
    assert at.metric[0].value == units.fmt(5e-3, "length", "US", 4)
