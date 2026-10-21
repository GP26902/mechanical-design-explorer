import pytest

from common.stepping import clamp, decimals, nice_step, snap_range


@pytest.mark.parametrize("span,expected", [(4.8, 0.01), (20000, 50), (210, 0.5), (1.2, 0.002), (1000, 2), (12, 0.02)])
def test_nice_step_is_1_2_5(span, expected):
    assert nice_step(span) == pytest.approx(expected)


def test_nice_step_rejects_bad_span():
    with pytest.raises(ValueError):
        nice_step(0)


@pytest.mark.parametrize("step,d", [(0.02, 2), (0.5, 1), (100, 0), (0.002, 3), (0.0001, 4)])
def test_decimals(step, d):
    assert decimals(step) == d


def test_snap_range_puts_limits_on_the_grid():
    lo, hi = snap_range(1.5748, 9.8425, 0.02)          # 40 mm to 250 mm shown in inches
    assert (lo, hi) == (1.58, 9.84)
    assert snap_range(0.2, 5.0, 0.01) == (0.2, 5.0)


def test_round_target_is_on_the_slider_grid():
    """The reported bug: 10 mm must be reachable, not 9.96 or 10.02."""
    lo, hi = 1.0, 60.0
    step = nice_step(hi - lo)
    assert (10.0 - lo) / step == pytest.approx(round((10.0 - lo) / step))
    old_step = (hi - lo) / 200                         # the previous step size
    assert (10.0 - lo) / old_step != pytest.approx(round((10.0 - lo) / old_step))


def test_clamp():
    assert clamp(12, 0, 10) == 10 and clamp(-1, 0, 10) == 0 and clamp(5, 0, 10) == 5
