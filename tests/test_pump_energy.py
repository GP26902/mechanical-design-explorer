"""Step A checks: pump efficiency, power, yearly energy cost and throttle vs variable speed (hand values in comments)."""
import math
from dataclasses import replace

import pytest

from chapters import pump_pipe as pp
from common import units
from common.validation import ValidationError
from tests.test_pump_pipe import BASE, PUMP

EN = pp.EnergyInputs(hours_per_year=4000.0, price_per_kwh=0.10, eta_motor=0.92, eta_vfd=0.97)


def test_efficiency_curve_hand_values_and_affinity_invariance():
    assert float(pp.pump_efficiency(PUMP, 0.020)) == pytest.approx(0.75)                 # peak at the design flow
    assert float(pp.pump_efficiency(PUMP, 0.010)) == pytest.approx(0.75 * (1 - 0.25))     # x = 0.5: 0.75 (2*0.5 - 0.25) = 0.5625
    assert float(pp.pump_efficiency(PUMP, 0.030)) == pytest.approx(0.75 * (3 - 2.25))     # x = 1.5: 0.5625
    assert float(pp.pump_efficiency(PUMP, 0.015, 0.8)) == pytest.approx(float(pp.pump_efficiency(PUMP, 0.015 / 0.8)))
    assert float(pp.pump_efficiency(PUMP, 0.0)) == pp.ETA_FLOOR                           # floor, never zero
    assert float(pp.pump_efficiency(PUMP, 0.080)) == pp.ETA_FLOOR                         # x = 4: negative, floored


def test_power_chain_and_yearly_cost_hand_values():
    # rho g Q H = 998.2 * 9.81 * 0.020 * 30 = 5875.4 W; shaft = /0.75 = 7833.9 W; electrical = /0.92 = 8515.1 W
    p = pp.operating_power(PUMP, 0.020, 30.0, 1.0, 998.2, 0.92)
    assert p.hydraulic == pytest.approx(5875.405, rel=1e-6)
    assert p.shaft == pytest.approx(7833.874, rel=1e-6)
    assert p.electrical == pytest.approx(8515.080, rel=1e-6)
    kwh, cost = pp.annual_energy(p.electrical, 4000.0, 0.10)                              # 8.515 kW x 4000 h = 34,060 kWh; x $0.10
    assert kwh == pytest.approx(34060.32, rel=1e-6) and cost == pytest.approx(3406.032, rel=1e-6)


def test_drive_losses_divide_electrical_power():
    a = pp.operating_power(PUMP, 0.020, 30.0, 1.0, 998.2, 0.92)
    b = pp.operating_power(PUMP, 0.020, 30.0, 1.0, 998.2, 0.92, eta_drive=0.97)
    assert b.electrical == pytest.approx(a.electrical / 0.97)


@pytest.fixture(scope="module")
def r():
    return pp.analyse(BASE, PUMP, 1750.0, 20.0)


def test_vfd_speed_matches_closed_form_and_the_operating_point_solver(r):
    c = pp.compare_control(r, BASE, PUMP, 0.6, EN)
    assert c.q_target == pytest.approx(0.6 * r.q_op)
    assert pp.pump_head(PUMP, c.q_target, c.ratio_vfd) == pytest.approx(c.h_system, abs=1e-9)   # pump at that speed gives the system head
    r2 = pp.analyse(BASE, PUMP, c.ratio_vfd * PUMP.rpm_rated, 20.0)                              # independent: bisection finds the same flow
    assert r2.q_op == pytest.approx(c.q_target, rel=1e-6)
    assert c.rpm_vfd == pytest.approx(c.ratio_vfd * 1750.0)


def test_throttle_loss_matches_an_equivalent_valve_added_to_the_layout(r):
    c = pp.compare_control(r, BASE, PUMP, 0.6, EN)
    assert c.h_valve == pytest.approx(c.h_pump_throttle - c.h_system) and c.h_valve > 0
    v = pp.velocity(c.q_target, BASE.d_discharge)
    k_extra = 2 * pp.G * c.h_valve / v**2                                                        # h = K V^2 / 2g solved for K
    throttled = replace(BASE, k_valve=BASE.k_valve + k_extra)                                    # one valve: K adds directly
    assert pp.analyse(throttled, PUMP, 1750.0, 20.0).q_op == pytest.approx(c.q_target, rel=1e-6)


def test_saving_grows_as_flow_falls_and_drive_costs_a_little_at_full_flow(r):
    s = [pp.compare_control(r, BASE, PUMP, f, EN).saving_frac for f in (0.4, 0.6, 0.8)]
    assert s[0] > s[1] > s[2] > 0
    full = pp.compare_control(r, BASE, PUMP, 1.0, EN)
    assert full.h_valve == pytest.approx(0.0, abs=1e-6) and full.ratio_vfd == pytest.approx(1.0, abs=1e-6)
    assert full.vfd.electrical == pytest.approx(full.throttle.electrical / EN.eta_vfd, rel=1e-6)   # only the drive loss differs
    assert full.saving_w < 0


def test_cost_fields_are_consistent(r):
    c = pp.compare_control(r, BASE, PUMP, 0.5, EN)
    assert c.cost_throttle == pytest.approx(c.throttle.electrical / 1000 * 4000 * 0.10)
    assert c.cost_vfd == pytest.approx(c.vfd.electrical / 1000 * 4000 * 0.10)
    assert c.saving_frac == pytest.approx(c.saving_w / c.throttle.electrical)


@pytest.mark.parametrize("change", [dict(hours_per_year=0.0), dict(hours_per_year=9000.0), dict(price_per_kwh=-0.1),
                                    dict(eta_motor=0.0), dict(eta_motor=1.2), dict(eta_vfd=0.0), dict(eta_vfd=float("nan"))])
def test_bad_energy_inputs_raise(r, change):
    with pytest.raises(ValidationError):
        pp.compare_control(r, BASE, PUMP, 0.7, replace(EN, **change))


@pytest.mark.parametrize("frac", [0.0, 0.01, 1.2, float("nan")])
def test_bad_target_fraction_raises(r, frac):
    with pytest.raises(ValidationError):
        pp.compare_control(r, BASE, PUMP, frac, EN)


@pytest.mark.parametrize("eta", [0.0, 1.1, -0.2])
def test_bad_pump_efficiency_raises(eta):
    with pytest.raises(ValidationError):
        pp.analyse(BASE, replace(PUMP, eta_bep=eta), 1750.0, 20.0)


def test_no_operating_point_gives_clear_message():
    weak = pp.analyse(replace(BASE, z_discharge=45.0), PUMP, 1750.0, 20.0)
    assert weak.q_op is None
    with pytest.raises(ValidationError):
        pp.compare_control(weak, BASE, PUMP, 0.7, EN)


def test_power_units():
    assert units.to_si(1, "power", "US") == pytest.approx(745.6999, rel=1e-6)    # 1 hp
    assert units.fmt(8515.08, "power", "SI") == "8.52 kW"
