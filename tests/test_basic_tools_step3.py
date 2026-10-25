"""Chapter 4, step 3: ISO 286 standard tolerance grades (hand values from the published ISO 286-1 table)."""
import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from chapters import basic_tools as bt
from common.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size,grade,expected", [
    (2, 7, 10), (4, 7, 12), (8, 7, 15), (15, 7, 18), (25, 7, 21), (40, 7, 25), (65, 7, 30), (100, 7, 35),   # IT7 column
    (150, 7, 40), (200, 7, 46), (300, 7, 52), (350, 7, 57), (450, 7, 63),
    (15, 6, 11), (25, 6, 13), (40, 6, 16), (65, 6, 19),                                                         # IT6, as quoted in published tables
    (15, 8, 27), (25, 8, 33), (40, 8, 39), (65, 8, 46),                                                         # IT8
    (15, 5, 8), (25, 9, 52), (15, 10, 70), (15, 11, 110), (15, 12, 180), (450, 12, 630)])
def test_it_tolerance_matches_the_published_iso_286_1_table(size, grade, expected):
    assert bt.it_tolerance_um(size, grade) == expected


def test_size_range_boundaries_are_over_a_up_to_and_including_b():
    # "over 3 up to and including 6": 3 mm is in the first range (IT7 10 um), anything just above 3 is in the next (12 um)
    assert bt.it_tolerance_um(3, 7) == 10 and bt.it_tolerance_um(3.0001, 7) == 12 and bt.it_tolerance_um(6, 7) == 12
    assert bt.it_tolerance_um(6.0001, 7) == 15 and bt.it_tolerance_um(18, 7) == 18 and bt.it_tolerance_um(18.5, 7) == 21
    assert bt.it_tolerance_um(500, 7) == 63 and bt.it_tolerance_um(0.5, 7) == 10


def test_table_grows_with_size_and_with_grade():
    for g in bt.IT_GRADES:
        col = bt.IT_TABLE_UM[g]
        assert col == sorted(col) and len(set(col)) == len(col) == len(bt.IT_RANGES)       # tolerance grows with nominal size
    for i in range(len(bt.IT_RANGES)):
        row = [bt.IT_TABLE_UM[g][i] for g in bt.IT_GRADES]
        assert row == sorted(row) and len(set(row)) == len(row)                          # and with the grade number


def test_every_grade_five_steps_up_is_about_ten_times_larger():
    # ISO 286: from IT6 on, the tolerance is multiplied by 10 every fifth grade (values are rounded, so allow 8 %)
    for g in (6, 7):
        for i in range(len(bt.IT_RANGES)):
            assert bt.IT_TABLE_UM[g + 5][i] == pytest.approx(10 * bt.IT_TABLE_UM[g][i], rel=0.08)


def test_table_follows_the_standard_tolerance_factor_formula():
    # T = k i with i = 0.45 D^(1/3) + 0.001 D (um), D the geometric mean of the size range, k = 7, 10, 16, 25, 40, 64, 100, 160.
    # The standard rounds its values (and treats the first range specially), so check the ranges over 3 mm within 10 %.
    k = dict(zip(bt.IT_GRADES, (7, 10, 16, 25, 40, 64, 100, 160)))
    for idx, (lo, hi) in enumerate(bt.IT_RANGES):
        if lo < 3:
            continue
        i = 0.45 * math.sqrt(lo * hi) ** (1 / 3) + 0.001 * math.sqrt(lo * hi)
        for g in bt.IT_GRADES:
            assert bt.IT_TABLE_UM[g][idx] == pytest.approx(k[g] * i, rel=0.10, abs=1.0)
    # worked example for 25 mm IT7: D = sqrt(18 * 30) = 23.24, i = 0.45 * 2.854 + 0.0232 = 1.308, 16 i = 20.9 -> table 21
    assert 16 * (0.45 * math.sqrt(18 * 30) ** (1 / 3) + 0.001 * math.sqrt(18 * 30)) == pytest.approx(20.93, abs=0.02)


def test_zone_entries_h_h_and_js():
    # published example: 4 H7 = 4.000 to 4.012 mm, 4 h7 = 3.988 to 4.000 mm, 4 js7 = 4 +/- 0.006 mm (IT7 for 3-6 mm is 12 um)
    e = bt.it_zone_entries_um(4, 7)
    assert e["Hole, H"] == (12.0, 0.0) and e["Shaft, h"] == (0.0, -12.0) and e["Symmetric, js"] == (6.0, -6.0)


@pytest.mark.parametrize("call", [lambda: bt.it_tolerance_um(0, 7), lambda: bt.it_tolerance_um(-5, 7), lambda: bt.it_tolerance_um(500.1, 7),
                                  lambda: bt.it_tolerance_um(25, 4), lambda: bt.it_tolerance_um(25, 13), lambda: bt.it_tolerance_um(float("nan"), 7),
                                  lambda: bt.it_zone_entries_um(25, 1)])
def test_bad_inputs_raise(call):
    with pytest.raises(ValidationError):
        call()


def test_size_limit_message():
    with pytest.raises(ValidationError, match="500 mm limit"):
        bt.it_tolerance_um(600, 7)


def _run():
    return AppTest.from_file(str(ROOT / "chapters" / "basic_tools.py"), default_timeout=30).run()


def _val(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_grades_tab_values_and_units():
    at = _run()
    assert not at.exception and len(at.tabs) == 6
    assert _val(at, "IT7 tolerance T") == "21 µm" and _val(at, "Size range used") == "over 18 to 30 mm"            # 25 mm default
    at.selectbox(key="bt_it_grade").set_value(6).run()
    assert not at.exception and _val(at, "IT6 tolerance T") == "13 µm"
    at.slider(key="bt_it_size_SI").set_value(3.0).run()
    assert not at.exception and _val(at, "IT6 tolerance T") == "6 µm" and _val(at, "Size range used") == "over 0 to 3 mm"
    at.sidebar.radio(key="unit_system").set_value("US").run()
    assert not at.exception and "mil" in _val(at, "IT6 tolerance T") and " in" in _val(at, "Size range used")
