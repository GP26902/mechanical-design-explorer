"""Step C checks: custom roughness / K, tank pressures, altitude and custom fluids (hand values in comments)."""
import math
from dataclasses import replace

import pytest

from chapters import pump_pipe as pp
from common import units
from common.validation import ValidationError
from tests.test_pump_pipe import BASE, PUMP

W20 = pp.water(20.0)                                  # rho 998.2, mu 1.002e-3, pv 2.339 kPa
RHO_G = W20.rho * pp.G                                # 9792.342 Pa per m of water


# ---- atmosphere ------------------------------------------------------------------------------
@pytest.mark.parametrize("h,expected", [(0.0, 101325.0), (1000.0, 89874.6), (2000.0, 79495.2)])
def test_standard_atmosphere_hand_values(h, expected):
    # P = 101325 (1 - 2.25577e-5 h)^5.25588; at 1000 m: 0.9774423^5.25588 = 0.88699 -> 89.87 kPa (ISA table 89.88 kPa)
    assert pp.atm_pressure(h) == pytest.approx(expected, rel=2e-4)


@pytest.mark.parametrize("h", [-600.0, 7000.0, float("nan")])
def test_bad_elevation_raises(h):
    with pytest.raises(ValidationError):
        pp.atm_pressure(h)


def test_altitude_lowers_npsh_available_by_the_pressure_difference_over_rho_g():
    # (101325 - 89874.6) / (998.2 * 9.81) = 1.1693 m; the hydraulics are unchanged so the whole shift shows in NPSHa
    sea = pp.analyse(BASE, PUMP, 1750.0, 20.0)
    high = pp.analyse(replace(BASE, site_elevation=1000.0), PUMP, 1750.0, 20.0)
    assert high.q_op == pytest.approx(sea.q_op, rel=1e-12)
    assert sea.npsh_a - high.npsh_a == pytest.approx((101325.0 - pp.atm_pressure(1000.0)) / RHO_G, rel=1e-9)
    assert sea.npsh_a - high.npsh_a == pytest.approx(1.1693, abs=2e-3)


# ---- tank pressures ----------------------------------------------------------------------------
def test_static_head_with_tank_pressure_hand_value():
    # 100 kPa on the discharge tank, water at 20 C: 100000 / (998.2 * 9.81) = 10.2121 m; plus 12 - 1 = 11 m of elevation
    lay = replace(BASE, p_discharge=100e3)
    assert pp.static_head(lay, W20) == pytest.approx(11.0 + 10.2121, abs=1e-3)
    assert pp.static_head(replace(BASE, p_suction=100e3), W20) == pytest.approx(11.0 - 10.2121, abs=1e-3)
    assert pp.static_head(BASE) == 11.0                                          # no fluid needed for open tanks
    with pytest.raises(ValidationError):
        pp.static_head(lay)


def test_pressurised_discharge_tank_lowers_flow_and_the_energy_balance_still_closes():
    open_ = pp.analyse(BASE, PUMP, 1750.0, 20.0)
    lay = replace(BASE, p_discharge=100e3)
    r = pp.analyse(lay, PUMP, 1750.0, 20.0)
    assert r.q_op < open_.q_op and r.h_op > open_.h_op
    assert r.static_head == pytest.approx(11.0 + 100e3 / RHO_G)
    assert r.static_head + sum(s.head_loss for s in r.sections) == pytest.approx(r.h_op, abs=1e-6)
    assert r.profile[-1].head == pytest.approx(lay.z_discharge + 100e3 / RHO_G, abs=1e-6)       # ends at the tank surface head


def test_suction_tank_pressure_raises_npsh_by_p_over_rho_g():
    # +50 kPa on the suction surface: 50000 / 9792.342 = 5.1060 m more NPSH available at the same flow and losses
    a = pp.npsha(pp.P_ATM, W20.pv, W20.rho, 1.0, 0.5)
    b = pp.npsha(pp.P_ATM + 50e3, W20.pv, W20.rho, 1.0, 0.5)
    assert b - a == pytest.approx(5.1060, abs=1e-3)
    r = pp.analyse(replace(BASE, p_suction=50e3), PUMP, 1750.0, 20.0)
    assert r.profile[0].head == pytest.approx(BASE.z_suction + 5.1060, abs=1e-3)
    assert r.profile[0].pressure == pytest.approx(RHO_G * BASE.z_suction + 50e3)                 # gauge pressure at pump elevation


@pytest.mark.parametrize("change", [dict(p_suction=-101325.0), dict(p_discharge=-120e3), dict(p_suction=float("nan")),
                                    dict(site_elevation=3000.0, p_suction=-70e3)])         # 3000 m: P_atm = 70.1 kPa, so -70 is fine...
def test_pressure_below_vacuum_raises(change):
    if change.get("site_elevation") == 3000.0:
        change = dict(site_elevation=3000.0, p_suction=-71e3)                              # ... but -71 kPa is below a vacuum
    with pytest.raises(ValidationError):
        pp.analyse(replace(BASE, **change), PUMP, 1750.0, 20.0)


# ---- custom fluid ------------------------------------------------------------------------------
def test_custom_fluid_with_water_properties_reproduces_water():
    w = pp.water(40.0)
    ref = pp.analyse(BASE, PUMP, 1750.0, 40.0)
    cf = pp.analyse(BASE, PUMP, 1750.0, 99.0, fluid=pp.custom_fluid(w.rho, w.mu, w.pv))       # temp_c ignored
    assert cf.q_op == pytest.approx(ref.q_op, rel=1e-12) and cf.h_op == pytest.approx(ref.h_op, rel=1e-12)
    assert cf.npsh_a == pytest.approx(ref.npsh_a, rel=1e-12) and cf.dp == pytest.approx(ref.dp)


def test_density_scales_pressure_drop_and_power_when_reynolds_is_unchanged():
    # doubling rho and mu keeps Re, f and every head loss; pressure drops (rho g h) and hydraulic power (rho g Q H) double
    ref = pp.analyse(BASE, PUMP, 1750.0, 20.0)
    f2 = pp.custom_fluid(2 * W20.rho, 2 * W20.mu, W20.pv)
    r2 = pp.analyse(BASE, PUMP, 1750.0, 20.0, fluid=f2)
    assert r2.q_op == pytest.approx(ref.q_op, rel=1e-9) and r2.h_op == pytest.approx(ref.h_op, rel=1e-9)
    assert [b / a for a, b in zip(ref.dp, r2.dp)] == pytest.approx([2.0] * len(ref.dp))
    p1 = pp.operating_power(PUMP, 0.020, 30.0, 1.0, W20.rho, 0.92)
    p2 = pp.operating_power(PUMP, 0.020, 30.0, 1.0, f2.rho, 0.92)
    assert p2.hydraulic == pytest.approx(2 * p1.hydraulic) and p2.electrical == pytest.approx(2 * p1.electrical)
    assert r2.npsh_a < ref.npsh_a                                                                # (p - pv)/(rho g) halves


def test_viscous_fluid_lowers_the_flow_and_raises_friction():
    ref = pp.analyse(BASE, PUMP, 1750.0, 20.0)
    thick = pp.analyse(BASE, PUMP, 1750.0, 20.0, fluid=pp.custom_fluid(W20.rho, 50e-3, W20.pv))
    assert thick.q_op < ref.q_op
    assert thick.sections[2].f > ref.sections[2].f                                               # discharge pipe friction factor


@pytest.mark.parametrize("rho,mu,pv", [(0.0, 1e-3, 2e3), (1000.0, 0.0, 2e3), (1000.0, -1e-3, 2e3), (1000.0, 1e-3, -1.0),
                                       (float("nan"), 1e-3, 2e3), (50.0, 1e-3, 2e3), (1000.0, 100.0, 2e3)])
def test_bad_custom_fluid_raises(rho, mu, pv):
    with pytest.raises(ValidationError):
        pp.custom_fluid(rho, mu, pv)


def test_density_units():
    assert units.to_si(1, "density", "US") == pytest.approx(16.0185, rel=1e-5)               # 1 lb/ft^3
    assert units.fmt(1000.0, "density", "SI") == "1,000 kg/m³"


# ---- custom roughness and K ----------------------------------------------------------------------
def test_rougher_pipe_and_larger_k_lower_the_flow():
    ref = pp.analyse(BASE, PUMP, 1750.0, 20.0)
    assert pp.analyse(replace(BASE, roughness=3e-3), PUMP, 1750.0, 20.0).q_op < ref.q_op
    assert pp.analyse(replace(BASE, k_valve=20.0), PUMP, 1750.0, 20.0).q_op < ref.q_op


def test_relative_roughness_above_the_equation_range_raises():
    # 0.08 m discharge pipe: 5 mm / 80 mm = 0.0625 > 0.05
    with pytest.raises(ValidationError, match="roughness"):
        pp.analyse(replace(BASE, roughness=5e-3), PUMP, 1750.0, 20.0)
    assert pp.analyse(replace(BASE, roughness=3.9e-3), PUMP, 1750.0, 20.0).q_op is not None     # 3.9/80 = 0.049: allowed
