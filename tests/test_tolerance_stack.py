"""Checks against hand calculations (values worked out on paper, see comments)."""
import math

import numpy as np
import pytest

from chapters import tolerance_stack as ts
from common.validation import ValidationError

# Illustrative stack: housing 100.0 ±0.10 (+), three parts 40, 30, 25 ±0.05 (−). Nominal gap = 5.0.
STACK = [ts.Dim("Housing", 100.0, 0.10, 0.10, +1), ts.Dim("Part A", 40.0, 0.05, 0.05, -1),
         ts.Dim("Part B", 30.0, 0.05, 0.05, -1), ts.Dim("Part C", 25.0, 0.05, 0.05, -1)]


# ---- worst case ------------------------------------------------------------------------------
def test_nominal_and_worst_case():
    assert ts.nominal_gap(STACK) == pytest.approx(5.0)
    lo, hi = ts.worst_case(STACK)                       # ±(0.10 + 3*0.05) = ±0.25
    assert (lo, hi) == (pytest.approx(4.75), pytest.approx(5.25))


def test_worst_case_asymmetric_negative_direction():
    d = [ts.Dim("X", 10.0, 0.3, 0.1, -1)]               # contributes -(10.3) .. -(9.9)
    lo, hi = ts.worst_case(d)
    assert (lo, hi) == (pytest.approx(-10.3), pytest.approx(-9.9))


# ---- RSS -------------------------------------------------------------------------------------
def test_rss_hand_values():
    r = ts.rss(STACK, k=3)                              # sqrt(0.1² + 3·0.05²) = sqrt(0.0175) = 0.132288
    assert r.mean == pytest.approx(5.0)
    assert r.hi - r.mean == pytest.approx(math.sqrt(0.0175))
    assert r.sigma == pytest.approx(math.sqrt(0.0175) / 3)          # 0.044096


def test_rss_limits_do_not_depend_on_k_but_sigma_does():
    r4 = ts.rss(STACK, k=4)
    assert r4.hi - r4.mean == pytest.approx(math.sqrt(0.0175))
    assert r4.sigma == pytest.approx(math.sqrt(0.0175) / 4)         # 0.033072


def test_rss_asymmetric_shifts_mean():
    r = ts.rss([ts.Dim("X", 10.0, 0.3, 0.1, +1)], k=3)  # band centre 10.1, sigma = 0.4/6
    assert r.mean == pytest.approx(10.1) and r.sigma == pytest.approx(0.4 / 6)
    r_neg = ts.rss([ts.Dim("X", 10.0, 0.3, 0.1, -1)], k=3)
    assert r_neg.mean == pytest.approx(-10.1)


# ---- Cpk / Cp --------------------------------------------------------------------------------
def test_cpk_centred_equals_cp():
    r = ts.rss(STACK, 3)
    assert ts.cpk(r.mean, r.sigma, 4.8, 5.2) == pytest.approx(0.2 / math.sqrt(0.0175))   # 1.5119
    assert ts.cp(r.sigma, 4.8, 5.2) == pytest.approx(1.5119, abs=1e-4)


def test_cpk_off_centre_uses_nearer_limit():
    r = ts.rss([ts.Dim("X", 10.0, 0.3, 0.1, +1)], 3)    # mean 10.1, sigma 0.066667
    assert ts.cpk(r.mean, r.sigma, 9.9, 10.5) == pytest.approx(1.0)   # (10.1 - 9.9) / 0.2
    assert ts.cpk(r.mean, r.sigma, 9.8, 10.5) == pytest.approx(1.5)   # (10.1 - 9.8) / 0.2


def test_cpk_zero_sigma():
    assert ts.cpk(5.0, 0.0, 4.8, 5.2) == float("inf")
    assert ts.cpk(6.0, 0.0, 4.8, 5.2) == 0.0
    zero = [ts.Dim("A", 10.0, 0.0, 0.0, +1), ts.Dim("B", 4.0, 0.0, 0.0, -1)]
    res = ts.analyse_stack(zero, 5.9, 6.1, n=2000)      # must not crash
    assert res.cpk_rss == float("inf") and res.frac_out_mc == 0.0


# ---- contribution ranking --------------------------------------------------------------------
def test_contribution_shares():
    c = ts.contributions(STACK)
    assert c.loc[0, "name"] == "Housing"
    assert c.loc[0, "variance_share"] == pytest.approx(0.04 / 0.07)             # 57.14 %
    assert c["variance_share"].iloc[1:].tolist() == pytest.approx([0.01 / 0.07] * 3)   # 14.29 % each
    assert c["variance_share"].sum() == pytest.approx(1.0)
    assert c.loc[0, "worst_case_share"] == pytest.approx(0.2 / 0.5)             # 40 %
    assert c["worst_case_share"].sum() == pytest.approx(1.0)


# ---- Monte Carlo (statistical checks, fixed seed) --------------------------------------------
def test_monte_carlo_matches_rss():
    n = 200_000
    s = ts.monte_carlo(STACK, 3, n, seed=7)
    sigma = math.sqrt(0.0175) / 3
    assert abs(s.mean() - 5.0) < 4 * sigma / math.sqrt(n)
    assert s.std(ddof=1) == pytest.approx(sigma, rel=0.01)


def test_monte_carlo_out_of_spec_fraction_matches_normal_tail():
    n = 200_000
    s = ts.monte_carlo(STACK, 3, n, seed=11)
    sigma = math.sqrt(0.0175) / 3
    expected = math.erfc((0.1 / sigma) / math.sqrt(2))   # two-sided tail beyond ±0.1 (2.268 sigma) = 0.0233
    assert ts.fraction_out_of_spec(s, 4.9, 5.1) == pytest.approx(expected, abs=0.002)
    assert ts.cpk(s.mean(), s.std(ddof=1), 4.9, 5.1) == pytest.approx(0.1 / (3 * sigma), rel=0.02)   # 0.756


def test_monte_carlo_is_reproducible():
    assert np.array_equal(ts.monte_carlo(STACK, 3, 5000, seed=3), ts.monte_carlo(STACK, 3, 5000, seed=3))


def test_analyse_stack_flags():
    res = ts.analyse_stack(STACK, 4.8, 5.2, k=3, n=50_000)
    assert res.out_wc is True and res.out_rss is False       # wc ±0.25 exceeds ±0.2 spec; RSS ±0.132 does not
    assert res.cpk_rss == pytest.approx(1.5119, abs=1e-4)
    assert res.cpk_mc == pytest.approx(res.cpk_rss, rel=0.05)


# ---- stack validation ------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    [],
    [ts.Dim("A", 0.0, 0.1, 0.1)],
    [ts.Dim("A", -5.0, 0.1, 0.1)],
    [ts.Dim("A", 10.0, -0.1, 0.1)],
    [ts.Dim("A", 10.0, 0.1, -0.1)],
    [ts.Dim("A", float("nan"), 0.1, 0.1)],
    [ts.Dim("A", 10.0, float("nan"), 0.1)],
    [ts.Dim("A", 10.0, 0.1, 0.1, 0)],
    [ts.Dim("A", None, 0.1, 0.1)],
])
def test_bad_rows_raise(bad):
    with pytest.raises(ValidationError):
        ts.analyse_stack(bad, 4.8, 5.2)


@pytest.mark.parametrize("lsl,usl", [(5.2, 4.8), (5.0, 5.0), (float("nan"), 5.0), (4.8, None)])
def test_bad_spec_limits_raise(lsl, usl):
    with pytest.raises(ValidationError):
        ts.analyse_stack(STACK, lsl, usl)


@pytest.mark.parametrize("k,n", [(1.0, 5000), (7.0, 5000), (3.0, 10), (3.0, 5_000_000), (3.0, 5000.5)])
def test_bad_k_or_samples_raise(k, n):
    with pytest.raises(ValidationError):
        ts.analyse_stack(STACK, 4.8, 5.2, k=k, n=n)


# ---- GD&T position ---------------------------------------------------------------------------
def test_hole_mmc_bonus_hand_values():
    # Hole Ø10.00–10.20, position Ø0.20 at MMC, actual Ø10.15: bonus 0.15, zone 0.35, VC 9.80.
    g = ts.gdt_position(ts.HOLE, 10.0, 10.2, 10.15, 0.2, 0.12, 0.10)
    assert g.mmc_size == 10.0
    assert g.bonus == pytest.approx(0.15) and g.zone_dia == pytest.approx(0.35)
    assert g.virtual_condition == pytest.approx(9.8)
    assert g.measured_dia == pytest.approx(2 * math.hypot(0.12, 0.10))      # 0.3124
    assert g.passed is True
    assert g.passed_rfs is False                                           # 0.3124 > 0.20 without bonus


def test_hole_at_mmc_has_no_bonus_and_at_lmc_full_bonus():
    assert ts.gdt_position(ts.HOLE, 10.0, 10.2, 10.0, 0.2, 0, 0).bonus == pytest.approx(0.0)
    assert ts.gdt_position(ts.HOLE, 10.0, 10.2, 10.2, 0.2, 0, 0).zone_dia == pytest.approx(0.4)


def test_pin_mmc_bonus_hand_values():
    # Pin Ø9.90–10.00 (MMC = 10.00), position Ø0.10, actual Ø9.92: bonus 0.08, zone 0.18, VC 10.10.
    g = ts.gdt_position(ts.PIN, 9.9, 10.0, 9.92, 0.1, 0.05, 0.05)
    assert g.mmc_size == 10.0
    assert g.bonus == pytest.approx(0.08) and g.zone_dia == pytest.approx(0.18)
    assert g.virtual_condition == pytest.approx(10.1)
    assert g.measured_dia == pytest.approx(0.1 * math.sqrt(2))              # 0.1414 <= 0.18
    assert g.passed is True and g.passed_rfs is False


def test_rfs_gives_no_bonus():
    g = ts.gdt_position(ts.HOLE, 10.0, 10.2, 10.15, 0.2, 0.0, 0.05, modifier=ts.RFS)
    assert g.bonus == 0.0 and g.zone_dia == pytest.approx(0.2)
    assert g.passed is True                                                # measured 0.10 <= 0.20


def test_exactly_on_boundary_passes():
    g = ts.gdt_position(ts.HOLE, 10.0, 10.2, 10.0, 0.2, 0.1, 0.0)         # measured 0.2 = zone 0.2
    assert g.passed is True


def test_size_out_of_limits_fails_even_if_position_is_perfect():
    g = ts.gdt_position(ts.HOLE, 10.0, 10.2, 10.3, 0.2, 0.0, 0.0)
    assert g.size_ok is False and g.passed is False
    assert g.bonus == pytest.approx(0.2)                                   # bonus capped at the size range


def test_bonus_curve():
    sizes, zone = ts.bonus_curve(ts.HOLE, 10.0, 10.2, 0.2)
    assert zone[0] == pytest.approx(0.2) and zone[-1] == pytest.approx(0.4)
    sizes, zone = ts.bonus_curve(ts.PIN, 9.9, 10.0, 0.1)
    assert zone[0] == pytest.approx(0.2) and zone[-1] == pytest.approx(0.1)   # largest pin = MMC = no bonus


@pytest.mark.parametrize("args", [
    (ts.HOLE, 10.2, 10.0, 10.1, 0.2, 0, 0),             # min > max
    (ts.HOLE, 10.0, 10.0, 10.0, 0.2, 0, 0),             # min == max
    (ts.HOLE, 0.0, 10.2, 10.1, 0.2, 0, 0),              # zero size
    (ts.HOLE, 10.0, 10.2, -1.0, 0.2, 0, 0),             # negative actual
    (ts.HOLE, 10.0, 10.2, 10.1, -0.2, 0, 0),            # negative position tolerance
    (ts.HOLE, 10.0, 10.2, 10.1, 0.2, float("nan"), 0),  # NaN offset
    ("slot", 10.0, 10.2, 10.1, 0.2, 0, 0),              # unknown feature
])
def test_bad_gdt_inputs_raise(args):
    with pytest.raises(ValidationError):
        ts.gdt_position(*args)


def test_bad_modifier_raises():
    with pytest.raises(ValidationError):
        ts.gdt_position(ts.HOLE, 10.0, 10.2, 10.1, 0.2, 0, 0, modifier="LMC")


# ---- table rows -> dims ----------------------------------------------------------------------
def test_rows_to_dims_converts_units_and_skips_blank_rows():
    rows = [("A", "+", 4.0, 0.002, 0.004), (None, None, float("nan"), float("nan"), float("nan")), ("B", "−", 1.0, 0.001, 0.001)]
    dims = ts.rows_to_dims(rows, 0.0254)                                   # inches -> metres
    assert len(dims) == 2
    assert dims[0].nominal == pytest.approx(4.0 * 0.0254) and dims[0].tol_minus == pytest.approx(0.004 * 0.0254)
    assert dims[0].direction == 1 and dims[1].direction == -1


@pytest.mark.parametrize("row", [
    ("A", None, 4.0, 0.1, 0.1),                  # direction not chosen
    ("A", "*", 4.0, 0.1, 0.1),                   # unknown direction
    ("A", "+", float("nan"), 0.1, 0.1),          # blank nominal
    ("A", "+", 4.0, None, 0.1),                  # blank tolerance
    ("A", "+", "abc", 0.1, 0.1),                 # text in a number cell
])
def test_rows_to_dims_bad_rows_raise(row):
    with pytest.raises(ValidationError):
        ts.rows_to_dims([row])


def test_all_blank_table_raises_clear_error():
    with pytest.raises(ValidationError):
        ts.validate_dims(ts.rows_to_dims([(None, None, None, None, None)]))
