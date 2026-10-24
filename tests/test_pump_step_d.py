"""Step D checks: example systems and CSV export (hand values in comments)."""
import io
import math
from dataclasses import replace

import pandas as pd
import pytest

from chapters import pump_pipe as pp
from common.validation import ValidationError

NAMES = list(pp.EXAMPLES)
RATED = 1750.0


def run(name):
    lay, pump, temp, fluid = pp.build_example(name)
    return lay, pump, pp.analyse(lay, pump, RATED, temp, fluid=fluid)


def test_first_example_is_the_original_default_system():
    lay, pump, _ = run(pp.DEFAULT_EXAMPLE)
    assert lay == pp.Layout(0.125, 3.0, 0.100, 60.0, 0.045e-3, 1.0, 12.0, 4, 1, 0.15, 1, 0.9, 0.0, 0.0, 0.0)
    assert pump == pp.PumpData(40.0, 0.020, 30.0, 3.5, 1750.0, 0.75, 1, "parallel")
    assert pp.DEFAULT_EXAMPLE == NAMES[0] and set(pp.EXAMPLES[NAMES[0]]["values"]) == set()


@pytest.mark.parametrize("name", NAMES)
def test_every_example_is_valid_and_inside_the_slider_ranges(name):
    lay, pump, r = run(name)
    assert r.q_op is not None and r.status in {"ok", "marginal", "cavitation"}
    assert 0.040 <= lay.d_suction <= 0.250 and 0.040 <= lay.d_discharge <= 0.250
    assert 0.5 <= lay.l_suction <= 20.0 and 5.0 <= lay.l_discharge <= 300.0
    assert -6.0 <= lay.z_suction <= 6.0 and 0.0 <= lay.z_discharge <= 40.0 and 0 <= lay.n_bends <= 12
    assert r.static_head + sum(s.head_loss for s in r.sections) == pytest.approx(r.h_op, abs=1e-6)       # energy balance closes


def test_undersized_pipe_example_is_a_clear_bottleneck_with_low_flow():
    _, _, base = run(NAMES[0])
    _, _, small = run("Undersized discharge pipe")
    assert small.bottleneck == "Discharge pipe" and max(small.share) > 0.70
    assert small.q_op < 0.6 * base.q_op


def test_hot_suction_lift_example_flags_cavitation():
    _, _, base = run(NAMES[0])
    _, _, hot = run("Hot water with a suction lift")
    assert base.status == "ok" and hot.status == "cavitation" and hot.npsh_a < hot.npsh_r
    # (P_atm - Pv)/(rho g) at 75 C, less the 3 m lift and the suction losses, is what is left of NPSH available
    w = pp.water(75.0)
    assert hot.npsh_a == pytest.approx((pp.P_ATM - w.pv) / (w.rho * pp.G) - 3.0 - hot.sections[0].head_loss - hot.sections[1].head_loss, abs=1e-6)


def test_parallel_example_puts_each_pump_near_its_best_efficiency_flow():
    lay, pump, two = run("Two pumps in parallel")
    one = pp.analyse(lay, replace(pump, n_pumps=1), RATED, 20.0)
    assert two.q_op > 1.3 * one.q_op and two.q_op < 2 * one.q_op
    assert 0.7 <= (two.q_op / 2) / pump.q_design <= 1.2 < one.q_op / pump.q_design        # in the preferred range vs outside it


def test_glycol_example_uses_pressure_altitude_and_the_custom_fluid():
    lay, pump, g = run("Pressurised glycol loop at altitude")
    assert lay.p_discharge == 150e3 and lay.site_elevation == 1500.0 and g.fluid.rho == 1040.0
    assert g.static_head == pytest.approx(lay.z_discharge - lay.z_suction + 150e3 / (1040.0 * pp.G))
    sea = pp.analyse(replace(lay, site_elevation=0.0), pump, RATED, 20.0, fluid=pp.custom_fluid(1040.0, 3e-3, 2e3))
    assert sea.npsh_a - g.npsh_a == pytest.approx((pp.P_ATM - pp.atm_pressure(1500.0)) / (1040.0 * pp.G), rel=1e-9)


# ---- CSV --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def default_run():
    return run(NAMES[0])[2]


def _read(b):
    return pd.read_csv(io.BytesIO(b))


def test_sections_csv_si_matches_the_results_and_hand_values(default_run):
    r = default_run
    df = _read(pp.sections_csv(r, "SI"))
    assert list(df.columns) == ["Section", "Side", "Diameter (mm)", "Velocity (m/s)", "Reynolds number", "Friction factor",
                                "Head loss (m)", "Pressure drop (kPa)", "Share of drop (%)", "Bottleneck"]
    assert len(df) == len(r.sections) == 7 and list(df["Section"]) == [s.name for s in r.sections]
    assert list(df["Bottleneck"].fillna("")).count("yes") == 1 and df.loc[df["Bottleneck"] == "yes", "Section"].item() == r.bottleneck
    suction = df[df["Section"] == "Suction pipe"].iloc[0]
    assert suction["Velocity (m/s)"] == pytest.approx(4 * r.q_op / (math.pi * 0.125**2), abs=1e-3)    # v = 4Q / (pi D^2)
    assert suction["Diameter (mm)"] == pytest.approx(125.0)
    assert df["Head loss (m)"].sum() == pytest.approx(r.h_op - r.static_head, abs=1e-3)               # losses = pump head - static head
    assert df["Share of drop (%)"].sum() == pytest.approx(100.0, abs=0.1)
    assert suction["Pressure drop (kPa)"] == pytest.approx(r.fluid.rho * pp.G * suction["Head loss (m)"] / 1000, rel=1e-3)


def test_sections_csv_in_us_units_converts_the_columns(default_run):
    r = default_run
    si, us = _read(pp.sections_csv(r, "SI")), _read(pp.sections_csv(r, "US"))
    assert list(us.columns)[2:4] == ["Diameter (in)", "Velocity (ft/s)"] and "Pressure drop (psi)" in us.columns and "Head loss (ft)" in us.columns
    assert us["Diameter (in)"].iloc[1] == pytest.approx(125.0 / 25.4, abs=1e-3)
    assert us["Velocity (ft/s)"].iloc[1] == pytest.approx(si["Velocity (m/s)"].iloc[1] / 0.3048, abs=1e-3)
    assert us["Pressure drop (psi)"].iloc[1] == pytest.approx(si["Pressure drop (kPa)"].iloc[1] / 6.894757, rel=1e-3)
    assert us["Head loss (ft)"].iloc[1] == pytest.approx(si["Head loss (m)"].iloc[1] / 0.3048, rel=1e-3)
    assert list(us["Reynolds number"].fillna(-1)) == list(si["Reynolds number"].fillna(-1))        # unitless columns unchanged


def test_curves_csv_matches_the_plotted_curves(default_run):
    r = default_run
    df = _read(pp.curves_csv(r, "SI"))
    assert list(df.columns) == ["Flow (L/s)", "Pump head (m)", "System head (m)", "NPSH available (m)", "NPSH required (m)"]
    assert len(df) == len(r.q) == 121 and df["Flow (L/s)"].iloc[0] == 0.0
    assert df["Pump head (m)"].iloc[0] == pytest.approx(40.0) and df["System head (m)"].iloc[0] == pytest.approx(11.0)    # shutoff, static head
    assert df["NPSH required (m)"].iloc[0] == pytest.approx(1.75)                                                   # half of 3.5 m at zero flow
    i = (df["Pump head (m)"] - df["System head (m)"]).abs().idxmin()                                                # curves cross at the operating point
    assert df["Flow (L/s)"].iloc[i] == pytest.approx(r.q_op * 1000, abs=0.5)
    us = _read(pp.curves_csv(r, "US"))
    assert list(us.columns)[0] == "Flow (gpm)" and us["Pump head (ft)"].iloc[0] == pytest.approx(40.0 / 0.3048, rel=1e-3)


def test_csv_with_no_operating_point():
    lay, pump, _, _ = pp.build_example(NAMES[0])
    r = pp.analyse(replace(lay, z_discharge=45.0), pump, RATED, 20.0)
    assert r.q_op is None
    with pytest.raises(ValidationError):
        pp.sections_csv(r, "SI")
    assert len(_read(pp.curves_csv(r, "SI"))) == len(r.q)                                   # curves still export
