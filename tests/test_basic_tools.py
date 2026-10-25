"""Chapter 4 (basic tools & examples): thread and fastener sizes, validation demos (hand values in comments)."""
import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from chapters import basic_tools as bt
from common.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SQRT3 = math.sqrt(3)


# ---- ISO metric thread geometry ----------------------------------------------------------------
def test_metric_thread_geometry_by_hand_m10x1_5():
    # d2 = 10 - 0.649519 * 1.5 = 9.0257; d3 = 10 - 1.226869 * 1.5 = 8.1597; D1 = 10 - 1.082532 * 1.5 = 8.3762 (ISO 724 tables: 9.026, 8.160, 8.376)
    t = bt.metric_thread(10, 1.5)
    assert t.d2 == pytest.approx(9.0257, abs=1e-4) and t.d3 == pytest.approx(8.1597, abs=1e-4) and t.D1 == pytest.approx(8.3762, abs=1e-4)
    # A_s = pi/4 * ((d2 + d3) / 2)^2 = pi/4 * 8.59270^2 = 57.99 mm^2 (ISO 898-1 lists 58)
    assert t.stress_area == pytest.approx(math.pi / 4 * ((t.d2 + t.d3) / 2) ** 2)
    assert t.stress_area == pytest.approx(math.pi / 4 * (10 - 0.9382 * 1.5) ** 2, rel=2e-5)         # the (d - 0.9382 P) form
    assert t.tap_drill == 8.5


@pytest.mark.parametrize("d,p,published", [(3, .5, 5.03), (4, .7, 8.78), (5, .8, 14.2), (6, 1, 20.1), (8, 1.25, 36.6), (10, 1.5, 58),
                                           (12, 1.75, 84.3), (16, 2, 157), (20, 2.5, 245), (24, 3, 353), (30, 3.5, 561),
                                           (8, 1, 39.2), (10, 1.25, 61.2), (10, 1, 64.5), (12, 1.5, 88.1), (12, 1.25, 92.1),
                                           (16, 1.5, 167), (20, 1.5, 272), (24, 2, 384), (30, 2, 621)])
def test_metric_stress_area_matches_the_published_iso_898_1_table(d, p, published):
    assert bt.metric_thread(d, p).stress_area == pytest.approx(published, rel=3e-3)                  # table values are rounded to 3 digits


def test_finer_pitch_gives_a_larger_stress_area():
    assert bt.metric_thread(12, 1.5).stress_area > bt.metric_thread(12, 1.75).stress_area
    assert bt.metric_thread(10, 1.0).stress_area > bt.metric_thread(10, 1.25).stress_area > bt.metric_thread(10, 1.5).stress_area


def test_every_listed_metric_size_and_pitch_is_valid_and_consistent():
    for d, coarse in bt.METRIC_COARSE.items():
        assert coarse not in bt.METRIC_FINE[d] and all(p < coarse for p in bt.METRIC_FINE[d])        # fine pitches are finer than coarse
        for p in [coarse, *bt.METRIC_FINE[d]]:
            t = bt.metric_thread(d, p)
            assert t.d > t.d2 > t.d3 > 0 and t.D1 > t.d3 and 0 < t.stress_area < math.pi / 4 * d**2


# ---- inch threads ------------------------------------------------------------------------------
@pytest.mark.parametrize("series,size,published", [("UNF", "#10", 0.0200), ("UNF", "1/4", 0.0364), ("UNF", "5/16", 0.0580),
                                                    ("UNF", "3/8", 0.0878), ("UNF", "7/16", 0.1187), ("UNF", "1/2", 0.1599),
                                                    ("UNF", "9/16", 0.2030), ("UNF", "5/8", 0.2560), ("UNF", "3/4", 0.3730),
                                                    ("UNC", "1/4", 0.0318), ("UNC", "3/8", 0.0775), ("UNC", "1/2", 0.1419), ("UNC", "3/4", 0.3340)])
def test_inch_stress_area_matches_published_tables(series, size, published):
    D, n = bt.INCH_SERIES[f"{series} ({'coarse' if series == 'UNC' else 'fine'})"][size]
    assert bt.inch_thread(D, n).stress_area == pytest.approx(published, rel=3e-3, abs=3e-4)


def test_inch_thread_by_hand_quarter_20():
    # 1/4-20: P = 0.05 in; E = 0.25 - 0.649519 * 0.05 = 0.21752; A_s = pi/4 (0.25 - 0.9743 / 20)^2 = pi/4 (0.201285)^2 = 0.031820 in^2
    t = bt.inch_thread(0.25, 20)
    assert t.pitch == pytest.approx(0.05) and t.E == pytest.approx(0.21752, abs=1e-5)
    assert t.stress_area == pytest.approx(0.031820, abs=2e-6) and t.tap_drill == pytest.approx(0.20)


def test_unf_has_more_threads_and_a_larger_area_than_unc_for_the_same_size():
    for size, (D, n_c) in bt.UNC.items():
        n_f = bt.UNF[size][1]
        assert n_f > n_c or (size in ("1-1/8", "1-1/4") and n_f > n_c)
        assert bt.inch_thread(D, n_f).stress_area > bt.inch_thread(D, n_c).stress_area


# ---- clearance holes and floating-fastener tolerance ----------------------------------------------
def test_clearance_tables_are_consistent():
    assert set(bt.CLEARANCE_MM) == set(bt.METRIC_COARSE)
    sizes = sorted(bt.CLEARANCE_MM)
    for d in sizes:
        close, normal = bt.CLEARANCE_MM[d]
        assert d < close < normal < d * 1.4
    assert [bt.CLEARANCE_MM[d][1] for d in sizes] == sorted(bt.CLEARANCE_MM[d][1] for d in sizes)


def test_floating_fastener_tolerance_hand_values():
    # T = H - F: M6 normal 6.6 - 6 = 0.6 mm, M8 normal 9 - 8 = 1.0 mm, M10 close 10.5 - 10 = 0.5 mm, M12 normal 13.5 - 12 = 1.5 mm
    assert bt.floating_fastener_tolerance_mm(6.6, 6) == pytest.approx(0.6)
    assert bt.floating_fastener_tolerance_mm(9, 8) == pytest.approx(1.0)
    assert bt.floating_fastener_tolerance_mm(*[bt.CLEARANCE_MM[10][0], 10]) == pytest.approx(0.5)
    assert bt.floating_fastener_tolerance_mm(bt.CLEARANCE_MM[12][1], 12) == pytest.approx(1.5)
    with pytest.raises(ValidationError, match="would not fit"):
        bt.floating_fastener_tolerance_mm(5.9, 6)


# ---- validation demos --------------------------------------------------------------------------
def test_circle_tube_and_beam_hand_values():
    assert bt.circle_area_m2(0.0254) * 1e6 == pytest.approx(506.707, abs=1e-3)                  # pi/4 * 25.4^2 mm^2
    # tube OD 40, wall 3 -> ID 34: pi/4 (1600 - 1156) = 348.71 mm^2
    assert bt.hollow_tube_area_mm2(40, 3) == pytest.approx(math.pi / 4 * 444)
    assert bt.midspan_moment_nm(2000, 1.5) == pytest.approx(750.0)                               # PL/4


@pytest.mark.parametrize("call", [
    lambda: bt.circle_area_m2(0), lambda: bt.circle_area_m2(-1), lambda: bt.circle_area_m2(float("nan")),
    lambda: bt.hollow_tube_area_mm2(40, 20), lambda: bt.hollow_tube_area_mm2(40, 25), lambda: bt.hollow_tube_area_mm2(40, 0),
    lambda: bt.hollow_tube_area_mm2(0, 3), lambda: bt.midspan_moment_nm(0, 1.5), lambda: bt.midspan_moment_nm(2000, -1),
    lambda: bt.metric_thread(0, 1), lambda: bt.metric_thread(10, 0), lambda: bt.metric_thread(10, 12), lambda: bt.metric_thread(10, 9),
    lambda: bt.inch_thread(0.25, 0), lambda: bt.inch_thread(-0.25, 20), lambda: bt.inch_thread(0.25, 4)])
def test_impossible_inputs_raise_a_clear_message(call):
    with pytest.raises(ValidationError):
        call()


def test_tube_message_names_the_problem():
    with pytest.raises(ValidationError, match="Wall thickness .* must be smaller than half the outer diameter"):
        bt.hollow_tube_area_mm2(40, 25)


# ---- the app --------------------------------------------------------------------------------------
def _run():
    return AppTest.from_file(str(ROOT / "chapters" / "basic_tools.py"), default_timeout=30).run()


def test_chapter_loads_with_its_tabs_and_default_thread():
    at = _run()
    assert not at.exception and len(at.tabs) == 5
    labels = [m.label for m in at.metric]
    assert "Tensile stress area" in labels
    assert at.metric[labels.index("Tensile stress area")].value == "36.6 mm²"                    # M8 x 1.25 default


def test_picking_size_pitch_and_inch_series():
    at = _run()
    at.selectbox(key="bt_md").set_value(12).run()
    assert not at.exception and next(m for m in at.metric if m.label == "Tensile stress area").value == "84.3 mm²"
    at.selectbox(key="bt_mp_12").set_value(1.5).run()
    assert not at.exception and next(m for m in at.metric if m.label == "Tensile stress area").value == "88.1 mm²"
    at.radio(key="bt_kind").set_value("Inch (UNC / UNF)").run()
    assert not at.exception and next(m for m in at.metric if m.label == "Tensile stress area").value.startswith("0.0318")      # 1/4-20 UNC
    at.radio(key="bt_iseries").set_value("UNF (fine)").run()
    assert not at.exception and next(m for m in at.metric if m.label == "Tensile stress area").value.startswith("0.0364")      # 1/4-28 UNF


def test_validation_demos_show_messages_inline_and_in_us_units():
    at = _run()
    at.number_input(key="bt_tube_wall_SI").set_value(25.0).run()
    assert not at.exception and any("half the outer diameter" in e.value for e in at.error)
    at.number_input(key="bt_beam_span_SI").set_value(-1.0).run()
    assert any("Span" in e.value and "greater than zero" in e.value for e in at.error)
    at.number_input(key="bt_dia_SI").set_value(0.0).run()
    assert any("Diameter" in e.value for e in at.error) and not at.exception
    at = _run()
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception and not at.error
