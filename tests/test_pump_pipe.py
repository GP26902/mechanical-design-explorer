"""Checks against textbook / hand calculations (worked values in the comments)."""
import math
from dataclasses import replace

import numpy as np
import pytest

from chapters import pump_pipe as pp
from common.validation import ValidationError

BASE = pp.Layout(d_suction=0.100, l_suction=3.0, d_discharge=0.080, l_discharge=60.0, roughness=0.045e-3,
                 z_suction=1.0, z_discharge=12.0, n_bends=4, n_valves=1, k_valve=pp.VALVE_K["Gate valve (open)"],
                 n_fittings=1, k_fitting=pp.FITTING_K["Tee, flow through run"])
PUMP = pp.PumpData(h_shutoff=40.0, q_design=0.020, h_design=30.0, npshr_design=3.5, rpm_rated=1750.0)


# ---- water properties ------------------------------------------------------------------------
def test_water_table_values():
    w = pp.water(20.0)
    assert w.rho == pytest.approx(998.2) and w.mu == pytest.approx(1.002e-3) and w.pv == pytest.approx(2339.0)
    assert pp.water(100.0).pv == pytest.approx(101_330.0, rel=1e-3)           # boils at 1 atm
    assert pp.water(25.0).pv == pytest.approx(3169.0, rel=0.02)               # steam-table value 3.169 kPa


@pytest.mark.parametrize("t", [-1.0, 101.0, float("nan"), None])
def test_bad_temperature_raises(t):
    with pytest.raises(ValidationError):
        pp.water(t)


# ---- Reynolds, friction factor ---------------------------------------------------------------
def test_reynolds_and_laminar_friction():
    Re = pp.reynolds(998.0, 0.01, 0.05, 1e-3)                                  # 998*0.01*0.05/1e-3
    assert Re == pytest.approx(499.0)
    assert pp.friction_factor(Re, 0.001) == pytest.approx(64 / 499)           # 0.12826


def test_colebrook_against_moody_and_self_consistency():
    f = pp.friction_factor(1e5, 0.001)
    assert f == pytest.approx(0.0222, rel=0.01)                                # Moody chart reading
    assert 1 / math.sqrt(f) == pytest.approx(-2 * math.log10(0.001 / 3.7 + 2.51 / (1e5 * math.sqrt(f))), abs=1e-9)
    assert pp.friction_factor(1e5, 0.0) == pytest.approx(0.0180, rel=0.01)    # smooth pipe, Re = 1e5
    sj = 0.25 / math.log10(0.001 / 3.7 + 5.74 / 1e5**0.9) ** 2                 # Swamee-Jain explicit fit
    assert f == pytest.approx(sj, rel=0.015)


@pytest.mark.parametrize("Re,rr", [(0.0, 0.001), (-5.0, 0.001), (1e5, -0.1), (float("nan"), 0.001)])
def test_bad_friction_inputs_raise(Re, rr):
    with pytest.raises(ValidationError):
        pp.friction_factor(Re, rr)


# ---- head losses -----------------------------------------------------------------------------
def test_darcy_head_and_pressure_drop_hand_value():
    h = pp.darcy_head(0.02, 100.0, 0.1, 2.0)                                   # 0.02*1000*4/19.62
    assert h == pytest.approx(4.0775, abs=1e-4)
    assert 1000.0 * pp.G * h == pytest.approx(40_000.0)                        # 40 kPa for rho = 1000


def test_minor_head_and_velocity():
    assert pp.minor_head(0.9, 3.0) == pytest.approx(0.9 * 9 / 19.62)           # 0.4128 m
    assert pp.velocity(0.01, 0.1) == pytest.approx(0.01 / (math.pi * 0.0025))  # 1.2732 m/s


def test_system_head_matches_independent_hand_calculation():
    lay = pp.Layout(0.1, 2.0, 0.1, 50.0, 0.045e-3, 0.0, 5.0, n_bends=2)
    Q, fl = 0.01, pp.water(20.0)
    V = Q / (math.pi * 0.1**2 / 4)
    Re = fl.rho * V * 0.1 / fl.mu
    f = 0.25 / math.log10(4.5e-4 / 3.7 + 5.74 / Re**0.9) ** 2                  # independent explicit fit
    expected = 5.0 + (0.5 + f * 2 / 0.1 + f * 50 / 0.1 + 2 * 0.9 + 1.0) * V**2 / (2 * pp.G)
    assert pp.system_head(Q, lay, fl) == pytest.approx(expected, rel=0.015)
    assert pp.system_head(0.0, lay, fl) == pytest.approx(5.0)                  # no flow: static head only


# ---- pump curve and affinity laws ------------------------------------------------------------
def test_pump_curve_passes_through_points_and_affinity():
    assert pp.pump_head(PUMP, 0.0) == pytest.approx(40.0)
    assert pp.pump_head(PUMP, 0.020) == pytest.approx(30.0)
    assert pp.pump_head(PUMP, 0.0, 0.8) == pytest.approx(0.64 * 40.0)          # H ~ N^2
    assert pp.pump_q_max(PUMP, 0.8) == pytest.approx(0.8 * pp.pump_q_max(PUMP, 1.0))   # Q ~ N
    # a point (Q, H) at rated speed maps to (0.8 Q, 0.64 H) at 80 % speed
    q = 0.012
    assert pp.pump_head(PUMP, 0.8 * q, 0.8) == pytest.approx(0.64 * float(pp.pump_head(PUMP, q)))


def test_npshr_curve():
    assert pp.pump_npshr(PUMP, 0.020) == pytest.approx(3.5)                    # design point
    assert pp.pump_npshr(PUMP, 0.0) == pytest.approx(1.75)                     # 0.5 * design (assumed)
    assert pp.pump_npshr(PUMP, 0.0, 0.8) == pytest.approx(0.64 * 1.75)


# ---- operating point -------------------------------------------------------------------------
def test_operating_point_analytic():
    # pump 30 - 2000 Q^2 against system 10 + 3000 Q^2  ->  Q = sqrt(20/5000) = 0.063246, H = 22 m
    q = pp.solve_operating_point(lambda x: 30 - 2000 * x**2, lambda x: 10 + 3000 * x**2, 0.12)
    assert q == pytest.approx(math.sqrt(20 / 5000), rel=1e-9)
    assert 30 - 2000 * q**2 == pytest.approx(22.0)


def test_no_intersection_returns_none():
    assert pp.solve_operating_point(lambda x: 5 - 100 * x**2, lambda x: 10 + x**2, 0.2) is None


# ---- NPSH and cavitation flag ----------------------------------------------------------------
def test_npsha_hand_value():
    w = pp.water(20.0)                                                         # (101325 - 2339)/(998.2*9.81) = 10.109 m
    assert pp.npsha(pp.P_ATM, w.pv, w.rho, 2.0, 0.5) == pytest.approx(11.61, abs=0.01)


def test_cavitation_status_thresholds():
    assert pp.cavitation_status(3.0, 4.0) == "cavitation"
    assert pp.cavitation_status(5.0, 4.0) == "marginal"                       # below 1.3 * 4 = 5.2
    assert pp.cavitation_status(5.2, 4.0) == "ok"
    assert pp.cavitation_status(6.0, 4.0) == "ok"


# ---- full analysis ---------------------------------------------------------------------------
def test_analysis_balances_energy_and_ranks_bottleneck():
    r = pp.analyse(BASE, PUMP, 1750.0, 20.0)
    assert r.q_op is not None and r.h_op == pytest.approx(float(pp.system_head(r.q_op, BASE, r.fluid)), abs=1e-6)
    assert r.static_head + sum(s.head_loss for s in r.sections) == pytest.approx(r.h_op, abs=1e-6)
    assert sum(r.share) == pytest.approx(1.0)
    assert r.bottleneck == r.sections[int(np.argmax(r.dp))].name == "Discharge pipe"
    assert r.profile[-1].head == pytest.approx(BASE.z_discharge, abs=1e-6)     # energy closes at the discharge tank level
    assert r.profile[0].head == pytest.approx(BASE.z_suction)
    assert r.profile[-2].head > r.profile[-1].head                              # exit loss is the last drop
    assert r.profile[[n.label for n in r.profile].index("Pump outlet")].head - \
        r.profile[[n.label for n in r.profile].index("Pump inlet")].head == pytest.approx(r.h_op)


def test_trends_speed_temperature_and_pipe_size():
    base = pp.analyse(BASE, PUMP, 1750.0, 20.0)
    assert pp.analyse(BASE, PUMP, 2100.0, 20.0).q_op > base.q_op > pp.analyse(BASE, PUMP, 1400.0, 20.0).q_op
    assert pp.analyse(BASE, PUMP, 1750.0, 80.0).npsh_a < base.npsh_a            # hot water: higher vapour pressure
    assert pp.analyse(replace(BASE, d_discharge=0.06), PUMP, 1750.0, 20.0).q_op < base.q_op
    assert pp.analyse(replace(BASE, n_valves=1, k_valve=pp.VALVE_K["Globe valve (open)"]), PUMP, 1750.0).q_op < base.q_op


def test_cavitation_flagged_for_suction_lift_with_hot_water():
    r = pp.analyse(replace(BASE, z_suction=-6.0), PUMP, 1750.0, 70.0)           # NPSHa < 7.3 - 6 = 1.3 m < NPSHr
    assert r.status == "cavitation" and r.npsh_a < r.npsh_r
    assert pp.analyse(BASE, PUMP, 1750.0, 20.0).status == "ok"


def test_suction_bottleneck_when_suction_pipe_is_small():
    r = pp.analyse(replace(BASE, d_suction=0.03, l_suction=15.0, d_discharge=0.15), PUMP, 1750.0, 20.0)
    assert r.bottleneck == "Suction pipe"


def test_pump_too_weak_gives_no_operating_point_not_a_crash():
    weak = pp.PumpData(10.0, 0.02, 5.0, 3.5, 1750.0)                           # shutoff 10 m < static head 11 m
    r = pp.analyse(BASE, weak, 1750.0, 20.0)
    assert r.q_op is None and r.sections == [] and r.status is None
    assert len(r.q) == len(r.h_pump) == len(r.h_sys)                            # curves still available to plot


def test_transitional_regime_is_flagged():
    lay = pp.Layout(0.05, 1.0, 0.05, 1.0, 0.045e-3, 0.0, 10.5)
    tiny = pp.PumpData(11.0, 1.2e-4, 10.5, 1.0, 1750.0)                        # operating flow ~1.2e-4 m3/s, Re ~ 3000
    r = pp.analyse(lay, tiny, 1750.0, 20.0)
    assert r.q_op is not None and 2300 <= r.sections[1].Re <= 4000
    assert "Suction pipe" in r.transitional


# ---- validation ------------------------------------------------------------------------------
@pytest.mark.parametrize("change", [
    dict(d_suction=0.0), dict(d_discharge=-0.05), dict(l_suction=0.0), dict(l_discharge=-3.0),
    dict(roughness=-1e-5), dict(n_bends=-1), dict(n_valves=1.5), dict(n_fittings=float("nan")),
    dict(k_valve=-1.0), dict(z_suction=float("nan")), dict(d_discharge=None),
])
def test_bad_layout_raises(change):
    with pytest.raises(ValidationError):
        pp.analyse(replace(BASE, **change), PUMP, 1750.0, 20.0)


@pytest.mark.parametrize("change", [
    dict(h_design=0.0), dict(q_design=-0.01), dict(h_shutoff=30.0), dict(h_shutoff=20.0),
    dict(npshr_design=0.0), dict(rpm_rated=0.0), dict(h_shutoff=float("nan")),
])
def test_bad_pump_raises(change):
    with pytest.raises(ValidationError):
        pp.analyse(BASE, replace(PUMP, **change), 1750.0, 20.0)


@pytest.mark.parametrize("rpm", [0.0, -100.0, 100.0, 5000.0, float("nan")])
def test_bad_speed_raises(rpm):
    with pytest.raises(ValidationError):
        pp.analyse(BASE, PUMP, rpm, 20.0)
