"""Chapter 3 GD&T upgrade: LMC modifier and fastener formula. Expected values are hand calculations."""
import numpy as np
import pytest

from chapters import tolerance_stack as ts
from common.validation import ValidationError

# Hole 10.00 - 10.20 (MMC 10.00, LMC 10.20), tol 0.2.  Pin 9.90 - 10.00 (MMC 10.00, LMC 9.90), tol 0.1.
H = (ts.HOLE, 10.0, 10.2)
P = (ts.PIN, 9.9, 10.0)


def g(feature_sizes, actual, tol, dx=0.0, dy=0.0, mod=ts.LMC):
    return ts.gdt_position(*feature_sizes, actual, tol, dx, dy, mod)


# ---- LMC bonus, zone, virtual condition ----------------------------------------------------------
def test_hole_lmc_bonus_zone_and_vc():
    r = g(H, 10.15, 0.2)                                   # 10.20 - 10.15 = 0.05
    assert r.bonus == pytest.approx(0.05) and r.zone_dia == pytest.approx(0.25)
    assert r.virtual_condition == pytest.approx(10.40)     # LMC + tol = 10.20 + 0.2
    assert r.lmc_size == 10.2 and r.mmc_size == 10.0


def test_pin_lmc_bonus_zone_and_vc():
    r = g(P, 9.92, 0.1)                                    # 9.92 - 9.90 = 0.02
    assert r.bonus == pytest.approx(0.02) and r.zone_dia == pytest.approx(0.12)
    assert r.virtual_condition == pytest.approx(9.80)      # LMC - tol = 9.90 - 0.1
    assert r.lmc_size == 9.9


@pytest.mark.parametrize("feat, at_lmc, at_mmc, tol", [(H, 10.2, 10.0, 0.2), (P, 9.9, 10.0, 0.1)])
def test_lmc_bonus_zero_at_lmc_and_full_range_at_mmc(feat, at_lmc, at_mmc, tol):
    assert g(feat, at_lmc, tol).bonus == pytest.approx(0.0)
    full = g(feat, at_mmc, tol)
    assert full.bonus == pytest.approx(0.2 if feat is H else 0.1)
    assert full.bonus == pytest.approx(feat[2] - feat[1])  # whole size range


@pytest.mark.parametrize("feat, actual, tol", [(H, 10.15, 0.2), (P, 9.92, 0.1), (H, 10.03, 0.2)])
def test_mmc_bonus_plus_lmc_bonus_is_size_range(feat, actual, tol):
    mmc, lmc = g(feat, actual, tol, mod=ts.MMC), g(feat, actual, tol, mod=ts.LMC)
    assert mmc.bonus + lmc.bonus == pytest.approx(feat[2] - feat[1])


def test_lmc_pass_fail_hand_cases():
    # hole at 10.15, axis offset (0.12, 0.10): measured 2 x hypot = 0.3124 ; MMC zone 0.35, LMC zone 0.25
    mmc = g(H, 10.15, 0.2, 0.12, 0.10, ts.MMC)
    lmc = g(H, 10.15, 0.2, 0.12, 0.10, ts.LMC)
    assert lmc.measured_dia == pytest.approx(0.31241, abs=1e-4)
    assert mmc.passed and not lmc.passed and not lmc.passed_rfs
    ok = g(H, 10.15, 0.2, 0.08, 0.06, ts.LMC)               # measured 0.2 <= 0.25 but also <= 0.2 stated
    assert ok.passed and ok.passed_rfs
    mid = g(H, 10.15, 0.2, 0.10, 0.05, ts.LMC)              # measured 0.2236: needs the 0.05 bonus
    assert mid.passed and not mid.passed_rfs


def test_lmc_size_out_of_limits_fails():
    assert not g(H, 10.25, 0.2).passed and not g(H, 10.25, 0.2).size_ok


def test_rfs_and_mmc_virtual_condition_unchanged():
    assert g(H, 10.15, 0.2, mod=ts.MMC).virtual_condition == pytest.approx(9.8)    # 10.00 - 0.2
    assert g(H, 10.15, 0.2, mod=ts.RFS).virtual_condition == pytest.approx(9.8)
    assert g(P, 9.92, 0.1, mod=ts.MMC).virtual_condition == pytest.approx(10.1)    # 10.00 + 0.1


def test_bonus_curve_lmc_endpoints_and_direction():
    sizes, zone = ts.bonus_curve(*H, 0.2, modifier=ts.LMC)
    assert zone[0] == pytest.approx(0.4) and zone[-1] == pytest.approx(0.2) and np.all(np.diff(zone) < 0)
    sizes, zone = ts.bonus_curve(*P, 0.1, modifier=ts.LMC)
    assert zone[0] == pytest.approx(0.1) and zone[-1] == pytest.approx(0.2) and np.all(np.diff(zone) > 0)
    _, mmc = ts.bonus_curve(*H, 0.2)                          # default still MMC: grows with size
    assert mmc[0] == pytest.approx(0.2) and mmc[-1] == pytest.approx(0.4)


def test_modifier_validation():
    for bad in ("XYZ", "mmc", None):
        with pytest.raises(ValidationError):
            ts.gdt_position(*H, 10.1, 0.2, 0, 0, bad)


# ---- fastener formula --------------------------------------------------------------------------
def test_floating_fastener_hand_value():
    r = ts.fastener_position_tolerance(8.40, 8.00, ts.FLOATING)       # M8 screw, 8.4 hole
    assert r.clearance == pytest.approx(0.40) and r.tol_per_part == pytest.approx(0.40)


def test_fixed_fastener_hand_value():
    r = ts.fastener_position_tolerance(8.40, 8.00, ts.FIXED)
    assert r.tol_per_part == pytest.approx(0.20) and r.clearance == pytest.approx(0.40)
    assert ts.fastener_position_tolerance(9.0, 8.0, ts.FIXED).tol_per_part == pytest.approx(0.5)   # (9 - 8) / 2


def test_fastener_in_inches_hand_value():
    r = ts.fastener_position_tolerance(0.3125, 0.3, ts.FLOATING)       # 5/16 hole, 0.300 screw
    assert r.tol_per_part == pytest.approx(0.0125)


def test_fastener_default_kind_is_floating():
    assert ts.fastener_position_tolerance(10.0, 9.0).tol_per_part == pytest.approx(1.0)


def test_fastener_equal_sizes_give_zero_tolerance():
    assert ts.fastener_position_tolerance(8.0, 8.0, ts.FLOATING).tol_per_part == 0.0


def test_fastener_larger_than_hole_raises_with_clear_message():
    with pytest.raises(ValidationError, match="no clearance"):
        ts.fastener_position_tolerance(7.9, 8.0)


@pytest.mark.parametrize("args", [(0.0, 8.0), (8.4, 0.0), (-8.4, 8.0), (float("nan"), 8.0), (8.4, float("inf")),
                                  (None, 8.0), ("abc", 8.0), (8.4, 8.0, "loose")])
def test_fastener_bad_inputs_raise(args):
    with pytest.raises(ValidationError):
        ts.fastener_position_tolerance(*args)


def test_floating_formula_matches_geometry_by_grid_search():
    # Two parts, clearance holes of diameter H, axes at the worst case: centre-to-centre offset d = T (each axis T/2 from
    # true position, on opposite sides). The largest round fastener that passes through both holes is the largest circle
    # inside the overlap of the two hole circles, found here by brute force (no formula).
    h, f = 8.4, 8.0
    t = ts.fastener_position_tolerance(h, f, ts.FLOATING).tol_per_part
    r = h / 2
    xs = np.linspace(-r, r, 801)
    X, Y = np.meshgrid(xs, np.linspace(-0.3, 0.3, 121))
    c1, c2 = -t / 2, t / 2
    clearance_radius = np.minimum(r - np.hypot(X - c1, Y), r - np.hypot(X - c2, Y))
    assert 2 * clearance_radius.max() == pytest.approx(f, abs=1e-3)            # largest fastener = F exactly


def test_fixed_formula_matches_geometry():
    # Fastener (diameter F) fixed in part 1 with its axis up to T/2 off true position; clearance hole (diameter H) in part 2
    # with its axis up to T/2 off the other way: centre distance d = T. The fastener touches the hole wall when
    # d + F/2 = H/2.
    h, f = 8.4, 8.0
    t = ts.fastener_position_tolerance(h, f, ts.FIXED).tol_per_part
    assert t + f / 2 == pytest.approx(h / 2)
