"""Step B checks: pump curve fitted from points, and two identical pumps in parallel or series (hand values in comments)."""
import math
from dataclasses import replace

import numpy as np
import pytest

from chapters import pump_pipe as pp
from common.validation import ValidationError
from tests.test_pump_pipe import BASE, PUMP

PAR = replace(PUMP, n_pumps=2, arrangement="parallel")
SER = replace(PUMP, n_pumps=2, arrangement="series")
EN = pp.EnergyInputs(4000.0, 0.10, 0.92, 0.97)


# ---- fit from points -------------------------------------------------------------------------
def test_fit_recovers_exact_curve_from_noise_free_points():
    q = np.array([0.0, 0.008, 0.016, 0.024, 0.032])
    f = pp.fit_pump_points(q, 40.0 - 25000.0 * q**2)
    assert f.h_shutoff == pytest.approx(40.0, abs=1e-9) and f.a == pytest.approx(25000.0, rel=1e-9)
    assert f.rms == pytest.approx(0.0, abs=1e-9) and f.max_resid == pytest.approx(0.0, abs=1e-9)
    assert f.q_zero_head() == pytest.approx(math.sqrt(40.0 / 25000.0))


def test_fit_matches_hand_regression_and_numpy():
    # x = Q^2 = 0, 1e-4, 4e-4, 9e-4 (mean 3.5e-4); H = 40.2, 37.4, 30.4, 17.2 (mean 31.3)
    # Sxx = 4.9e-7, Sxy = -0.01244 -> slope = -25387.755, a = 25387.755, H0 = 31.3 + 25387.755 * 3.5e-4 = 40.18571
    # residuals 0.0143, -0.2469, 0.3694, -0.1367 -> RMS = sqrt(0.216328 / 4) = 0.23255, largest 0.3694
    q, h = [0.0, 0.01, 0.02, 0.03], [40.2, 37.4, 30.4, 17.2]
    f = pp.fit_pump_points(q, h)
    assert f.a == pytest.approx(25387.755, rel=1e-6) and f.h_shutoff == pytest.approx(40.18571, rel=1e-6)
    assert f.rms == pytest.approx(0.23255, rel=1e-4) and f.max_resid == pytest.approx(0.36939, rel=1e-4)
    slope, icpt = np.polyfit(np.array(q) ** 2, h, 1)                   # independent library fit
    assert f.a == pytest.approx(-slope) and f.h_shutoff == pytest.approx(icpt) and f.n_points == 4


def test_fitted_pump_uses_the_fit_and_keeps_affinity_scaling():
    f = pp.fit_pump_points([0.0, 0.01, 0.02, 0.03], [40.2, 37.4, 30.4, 17.2])
    p = pp.pump_from_fit(f, 0.020, 3.5, 1750.0)
    assert p.h_shutoff == pytest.approx(f.h_shutoff) and p.a() == pytest.approx(f.a)
    assert p.h_design == pytest.approx(40.18571 - 25387.755 * 0.02**2, rel=1e-6)          # 30.0305 m
    assert float(pp.pump_head(p, 0.01, 0.8)) == pytest.approx(0.64 * 40.18571 - 25387.755 * 1e-4, rel=1e-6)
    assert pp.analyse(BASE, p, 1750.0, 20.0).q_op is not None


@pytest.mark.parametrize("q,h", [
    ([0.0, 0.01], [40.0, 35.0]),                                   # fewer than 3 points
    ([0.01, 0.01, 0.01], [30.0, 31.0, 29.0]),                      # one distinct flow
    ([0.0, -0.01, 0.02], [40.0, 38.0, 30.0]),                      # negative flow
    ([0.0, 0.01, 0.02], [40.0, 0.0, 30.0]),                        # zero head
    ([0.0, 0.01, 0.02], [40.0, float("nan"), 30.0]),               # missing value
    ([0.0, 0.01, 0.02], [40.0, 30.0]),                             # unequal lengths
    ([0.0, 0.01, 0.02], [20.0, 30.0, 40.0]),                       # head rises with flow: a < 0
    ([0.0, 0.01, 0.02], [30.0, 30.0, 30.0]),                       # flat: a = 0
])
def test_bad_fit_points_raise(q, h):
    with pytest.raises(ValidationError):
        pp.fit_pump_points(q, h)


def test_design_flow_beyond_zero_head_raises():
    f = pp.fit_pump_points([0.0, 0.01, 0.02, 0.03], [40.2, 37.4, 30.4, 17.2])      # zero head near 0.0398 m^3/s
    with pytest.raises(ValidationError):
        pp.pump_from_fit(f, 0.045, 3.5, 1750.0)
    with pytest.raises(ValidationError):
        pp.pump_from_fit(f, 0.0, 3.5, 1750.0)


# ---- parallel and series ---------------------------------------------------------------------
def test_parallel_and_series_curves_hand_values():
    # single pump: H = 40 - 25000 Q^2. Parallel at Q = 0.030 -> each pump 0.015 -> 40 - 25000 * 2.25e-4 = 34.375 m
    assert float(pp.pump_head(PAR, 0.030)) == pytest.approx(34.375)
    assert float(pp.pump_head(PAR, 0.030)) == pytest.approx(float(pp.pump_head(PUMP, 0.015)))
    # series at Q = 0.020 -> each pump gives 30 m -> 60 m
    assert float(pp.pump_head(SER, 0.020)) == pytest.approx(60.0) == pytest.approx(2 * float(pp.pump_head(PUMP, 0.020)))
    assert float(pp.pump_head(PAR, 0.0)) == pytest.approx(40.0) and float(pp.pump_head(SER, 0.0)) == pytest.approx(80.0)
    assert pp.pump_q_max(PAR) == pytest.approx(2 * pp.pump_q_max(PUMP))              # zero-head flow doubles in parallel
    assert pp.pump_q_max(SER) == pytest.approx(math.sqrt(80.0 / 50000.0))            # series: sqrt(2 H0 / 2a), same flow as one pump
    assert pp.pump_q_max(SER) == pytest.approx(pp.pump_q_max(PUMP))
    assert float(pp.pump_head(PAR, 0.030, 0.8)) == pytest.approx(0.64 * 40.0 - 25000 * 0.015**2)       # affinity scaling kept


@pytest.mark.parametrize("pump,k_eq", [(PAR, 25000 / 4), (SER, 2 * 25000)])
def test_operating_point_against_closed_form(pump, k_eq):
    # pump H0e - ae Q^2 = Hs + k Q^2  ->  Q = sqrt((H0e - Hs) / (ae + k)); Hs = 10 m, k = 12000
    h0e = 40.0 if pump is PAR else 80.0
    q = pp.solve_operating_point(lambda x: float(pp.pump_head(pump, x)), lambda x: 10.0 + 12000.0 * x**2, pp.pump_q_max(pump))
    assert q == pytest.approx(math.sqrt((h0e - 10.0) / (k_eq + 12000.0)), rel=1e-9)


def test_two_pumps_deliver_more_than_one_but_not_double_in_a_real_system():
    one = pp.analyse(BASE, PUMP, 1750.0, 20.0)
    par, ser = pp.analyse(BASE, PAR, 1750.0, 20.0), pp.analyse(BASE, SER, 1750.0, 20.0)
    assert one.q_op < par.q_op < 2 * one.q_op and one.q_op < ser.q_op < 2 * one.q_op
    assert par.h_op > one.h_op and ser.h_op > one.h_op
    for r in (par, ser):
        assert r.h_op == pytest.approx(float(pp.system_head(r.q_op, BASE, r.fluid)), abs=1e-6)       # energy balance still closes
        assert r.static_head + sum(s.head_loss for s in r.sections) == pytest.approx(r.h_op, abs=1e-6)


def test_npsh_and_efficiency_use_the_flow_through_each_pump():
    assert float(pp.pump_npshr(PAR, 0.040)) == pytest.approx(float(pp.pump_npshr(PUMP, 0.020)))     # 3.5 m at the design flow
    assert float(pp.pump_npshr(SER, 0.020)) == pytest.approx(float(pp.pump_npshr(PUMP, 0.020)))
    assert float(pp.pump_efficiency(PAR, 0.040)) == pytest.approx(0.75)
    assert float(pp.pump_efficiency(SER, 0.020)) == pytest.approx(0.75)
    assert float(pp.pump_efficiency(PAR, 0.020)) == pytest.approx(0.75 * 0.75)                      # each pump at half its design flow


def test_total_power_counts_both_pumps():
    one = pp.operating_power(PUMP, 0.020, 30.0, 1.0, 998.2, 0.92)
    par = pp.operating_power(PAR, 0.040, 30.0, 1.0, 998.2, 0.92)      # same head, double flow, each pump at its design point
    ser = pp.operating_power(SER, 0.020, 60.0, 1.0, 998.2, 0.92)      # same flow, double head, each pump at its design point
    assert par.shaft == pytest.approx(2 * one.shaft) and ser.shaft == pytest.approx(2 * one.shaft)
    assert par.electrical == pytest.approx(2 * one.electrical)


@pytest.mark.parametrize("pump", [PAR, SER])
def test_throttle_vs_drive_with_two_pumps(pump):
    r = pp.analyse(BASE, pump, 1750.0, 20.0)
    c = pp.compare_control(r, BASE, pump, 0.6, EN)
    assert float(pp.pump_head(pump, c.q_target, c.ratio_vfd)) == pytest.approx(c.h_system, abs=1e-9)
    assert pp.analyse(BASE, pump, c.ratio_vfd * pump.rpm_rated, 20.0).q_op == pytest.approx(c.q_target, rel=1e-6)
    assert c.h_valve > 0 and c.saving_frac > 0


@pytest.mark.parametrize("change", [dict(n_pumps=0), dict(n_pumps=5), dict(n_pumps=1.5), dict(arrangement="diagonal")])
def test_bad_pump_count_or_arrangement_raises(change):
    with pytest.raises(ValidationError):
        pp.analyse(BASE, replace(PUMP, **change), 1750.0, 20.0)
