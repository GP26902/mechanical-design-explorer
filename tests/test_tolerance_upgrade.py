"""Chapter 3 upgrade: distributions, mean shift, Cp/ppm/sigma level, budget solver, examples, CSV.
Every expected value below is a hand calculation (numbers written out), not a call to the code under test."""
import io
import math

import numpy as np
import pandas as pd
import pytest

from chapters import tolerance_stack as ts
from common import units
from common.validation import ValidationError

# Mixed stack, nominal gap 0: A uniform, B triangular, C normal (k = 3), all +-0.3
MIXED = [ts.Dim("A", 10.0, 0.3, 0.3, +1, ts.UNIFORM), ts.Dim("B", 4.0, 0.3, 0.3, -1, ts.TRIANGULAR),
         ts.Dim("C", 6.0, 0.3, 0.3, -1, ts.NORMAL)]
STACK = [ts.Dim("A", 10.0, 0.3, 0.3, +1), ts.Dim("B", 4.0, 0.3, 0.3, -1), ts.Dim("C", 6.0, 0.3, 0.3, -1)]


# ---- variances -------------------------------------------------------------------------------
def test_distribution_sigmas_match_hand_values():
    assert ts.dim_sigma(MIXED[0], 3) == pytest.approx(math.sqrt(0.09 / 3))     # uniform t^2/3 = 0.03
    assert ts.dim_sigma(MIXED[1], 3) == pytest.approx(math.sqrt(0.09 / 6))     # triangular t^2/6 = 0.015
    assert ts.dim_sigma(MIXED[2], 3) == pytest.approx(0.1)                     # normal (t/k)^2 = 0.01
    assert ts.dim_sigma(MIXED[2], 2) == pytest.approx(0.15)                    # k = 2
    asym = ts.Dim("X", 10.0, 0.3, 0.1, +1, ts.UNIFORM)                         # t = 0.2
    assert ts.dim_sigma(asym, 3) ** 2 == pytest.approx(0.04 / 3)


def test_rss_of_mixed_stack():
    r = ts.rss(MIXED, 3)
    assert r.mean == pytest.approx(0.0, abs=1e-12)
    assert r.sigma == pytest.approx(math.sqrt(0.03 + 0.015 + 0.01))            # sqrt(0.055)
    assert ts.cpk(r.mean, r.sigma, -1.2, 1.2) == pytest.approx(1.2 / (3 * math.sqrt(0.055)))


def test_default_distribution_is_normal_and_unchanged():
    assert ts.Dim("A", 1.0, 0.1, 0.1).dist == ts.NORMAL and ts.Dim("A", 1.0, 0.1, 0.1).shift == 0.0
    base = [ts.Dim("H", 100.0, 0.1, 0.1, +1), ts.Dim("A", 40.0, 0.05, 0.05, -1)]
    s = ts.monte_carlo(base, 3, 2000, seed=1)
    rng = np.random.default_rng(1)                                             # the original sampling recipe
    expect = np.zeros(2000)
    for mean, sigma in ((100.0, 0.1 / 3), (-40.0, 0.05 / 3)):
        expect += mean + sigma * rng.standard_normal(2000)
    assert np.allclose(s, expect)


# ---- mean shift ------------------------------------------------------------------------------
def test_shift_moves_mean_not_sigma_and_changes_cpk():
    d = ts.Dim("X", 10.0, 0.3, 0.3, +1, ts.NORMAL, 0.06)
    r = ts.rss([d], 3)
    assert r.mean == pytest.approx(10.06) and r.sigma == pytest.approx(0.1)
    assert ts.cpk(r.mean, r.sigma, 9.5, 10.5) == pytest.approx(0.44 / 0.3)     # 1.4667
    neg = ts.Dim("X", 10.0, 0.3, 0.3, -1, ts.NORMAL, 0.06)                     # subtracting dimension: gap falls
    assert ts.rss([neg], 3).mean == pytest.approx(-10.06)


def test_worst_case_ignores_shift():
    a = ts.worst_case([ts.Dim("X", 10.0, 0.3, 0.1, +1)])
    b = ts.worst_case([ts.Dim("X", 10.0, 0.3, 0.1, +1, ts.UNIFORM, 0.05)])
    assert a == b == (pytest.approx(9.9), pytest.approx(10.3))


# ---- Monte Carlo against the analytic values --------------------------------------------------
@pytest.mark.parametrize("dist, direction, shift", [(ts.NORMAL, 1, 0.0), (ts.UNIFORM, 1, 0.0), (ts.TRIANGULAR, 1, 0.0),
                                                    (ts.UNIFORM, -1, 0.05), (ts.TRIANGULAR, -1, -0.04), (ts.NORMAL, -1, 0.02)])
def test_monte_carlo_matches_analytic_mean_and_sigma(dist, direction, shift):
    d = ts.Dim("X", 10.0, 0.3, 0.1, direction, dist, shift)                   # t = 0.2, band centre 10.1
    n = 200_000
    s = ts.monte_carlo([d], 3, n, seed=7)
    mean, sigma = ts.dim_stats(d, 3)
    assert mean == pytest.approx(direction * (10.1 + shift))
    assert s.mean() == pytest.approx(mean, abs=5 * sigma / math.sqrt(n))
    assert s.std(ddof=1) == pytest.approx(sigma, rel=0.01)


@pytest.mark.parametrize("dist", [ts.UNIFORM, ts.TRIANGULAR])
@pytest.mark.parametrize("direction", [1, -1])
def test_uniform_and_triangular_samples_stay_inside_the_shifted_band(dist, direction):
    d = ts.Dim("X", 10.0, 0.3, 0.1, direction, dist, 0.05)                    # size band 9.9 .. 10.3, moved by +0.05
    s = ts.monte_carlo([d], 3, 20_000, seed=3)
    sizes = direction * s
    assert sizes.min() >= 9.95 - 1e-12 and sizes.max() <= 10.35 + 1e-12


def test_uniform_gap_tail_fraction_matches_hand_value():
    d = ts.Dim("X", 0.5, 1.0, 1.0, +1, ts.UNIFORM)                             # uniform on -0.5 .. 1.5, centre 0.5
    s = ts.monte_carlo([d], 3, 200_000, seed=11)
    assert ts.fraction_out_of_spec(s, 0.0, 1.5) == pytest.approx(0.25, abs=0.005)   # below 0: 0.5 of 2.0 wide = 0.25


def test_triangular_tail_fraction_matches_hand_value():
    d = ts.Dim("X", 10.0, 1.0, 1.0, +1, ts.TRIANGULAR)                         # triangular 9 .. 11, mode 10
    s = ts.monte_carlo([d], 3, 200_000, seed=5)
    assert ts.fraction_out_of_spec(s, 9.5, 20.0) == pytest.approx(0.125, abs=0.004)   # (9.5 - 9)^2 / (2 x 1) = 0.125


# ---- Cp, ppm, sigma level ---------------------------------------------------------------------
def test_normal_ppm_hand_values():
    assert ts.normal_ppm(0.0, 1.0, -3.0, 3.0) == pytest.approx(2699.796, rel=1e-4)       # Cpk 1.0 centred: 2 x 0.00135
    assert ts.normal_ppm(1.0, 1.0, -3.0, 3.0) == pytest.approx(22781.8, rel=1e-3)        # 0.02275 + 0.0000317
    assert ts.normal_ppm(0.0, 1.0, -3.99, 3.99) == pytest.approx(66.0, rel=0.03)         # Cpk 1.33 centred
    assert ts.normal_ppm(0.0, 0.0, -1.0, 1.0) == 0.0 and ts.normal_ppm(2.0, 0.0, -1.0, 1.0) == 1e6


def test_normal_ppm_validation():
    with pytest.raises(ValidationError):
        ts.normal_ppm(0.0, -1.0, -1.0, 1.0)
    with pytest.raises(ValidationError):
        ts.normal_ppm(0.0, 1.0, 1.0, -1.0)


def test_sigma_level_is_three_times_cpk():
    assert ts.sigma_level(1.33) == pytest.approx(3.99) and ts.sigma_level(2.0) == pytest.approx(6.0)
    assert ts.sigma_level(float("inf")) == float("inf")


def test_analyse_stack_reports_cp_ppm_and_sigma_level():
    res = ts.analyse_stack(STACK, -1.2, 1.2, k=3, n=200_000)                   # sigma_RSS = sqrt(3) * 0.1, mean 0
    sigma = math.sqrt(0.03)
    assert res.cp_rss == pytest.approx(2.4 / (6 * sigma))
    assert res.cpk_rss == pytest.approx(1.2 / (3 * sigma)) and res.sigma_level_rss == pytest.approx(3 * res.cpk_rss)
    assert res.ppm_rss == pytest.approx(ts.normal_ppm(0.0, sigma, -1.2, 1.2))
    assert res.cp_mc == pytest.approx(res.cp_rss, rel=0.02)
    assert res.ppm_mc == pytest.approx(res.frac_out_mc * 1e6)
    assert res.sigma_level_mc == pytest.approx(3 * res.cpk_mc)


def test_cp_exceeds_cpk_when_off_centre():
    res = ts.analyse_stack([ts.Dim("X", 10.0, 0.3, 0.3, +1, ts.NORMAL, 0.06)], 9.5, 10.5, n=5000)
    assert res.cp_rss == pytest.approx(1.0 / 0.6) and res.cpk_rss == pytest.approx(0.44 / 0.3)


# ---- budget solver ----------------------------------------------------------------------------
ONE = [ts.Dim("A", 10.0, 0.3, 0.3, +1)]                                        # sigma 0.1, spec 9.5..10.5, margin 0.5
TWO = [ts.Dim("A", 20.0, 0.3, 0.3, +1), ts.Dim("B", 10.0, 0.3, 0.3, -1)]       # gap 10, V = 0.02


def test_budget_scale_factor_hand_value_tighten():
    b = ts.tolerance_budget(TWO, 9.5, 10.5, 1.33)
    assert b.current_cpk == pytest.approx(0.5 / (3 * math.sqrt(0.02)))         # 1.1785
    assert b.feasible and b.action == "tighten"
    assert b.scale == pytest.approx((0.5 / 3.99) / math.sqrt(0.02))            # 0.8861
    assert list(b.rows["half_width_scaled"]) == pytest.approx([0.3 * b.scale] * 2)


def test_budget_single_row_factor_hand_value():
    b = ts.tolerance_budget(TWO, 9.5, 10.5, 1.33)
    sigma_req = 0.5 / 3.99
    f = math.sqrt((sigma_req ** 2 - 0.01) / 0.01)                              # (0.0157034 - 0.01) / 0.01 -> 0.7552
    assert f == pytest.approx(0.7552, abs=1e-4)
    assert list(b.rows["factor_alone"]) == pytest.approx([f, f])
    assert list(b.rows["half_width_alone"]) == pytest.approx([0.3 * f, 0.3 * f])


def test_budget_single_row_not_enough_alone():
    b = ts.tolerance_budget(TWO, 9.5, 10.5, 3.0)                               # sigma_req^2 = 0.003086 < other row's 0.01
    assert b.feasible and b.action == "tighten" and b.scale == pytest.approx((0.5 / 9.0) / math.sqrt(0.02))
    assert np.isnan(b.rows["factor_alone"]).all() and np.isnan(b.rows["half_width_alone"]).all()


def test_budget_loosen():
    b = ts.tolerance_budget(ONE, 9.5, 10.5, 1.0)
    assert b.action == "loosen" and b.scale == pytest.approx(0.5 / 0.3)        # 1.6667
    assert b.rows["factor_alone"].iloc[0] == pytest.approx(b.scale)            # one row: alone = all


def test_budget_already_on_target():
    b = ts.tolerance_budget(ONE, 9.5, 10.5, 0.5 / 0.3)
    assert b.action == "none" and b.scale == pytest.approx(1.0)


def test_budget_round_trip_all_rows_scaled():
    spec = (-1.2, 1.2)
    stack = [ts.Dim("A", 10.0, 0.3, 0.3, +1, ts.UNIFORM), ts.Dim("B", 4.0, 0.3, 0.3, -1, ts.TRIANGULAR),
             ts.Dim("C", 6.0, 0.4, 0.2, -1, ts.NORMAL, 0.01)]                  # asymmetric band and a shift
    b = ts.tolerance_budget(stack, *spec, 1.33, k=3)
    assert b.feasible and b.scale > 0.35                                       # keeps both tolerances of C non-negative
    scaled = []
    for d, h in zip(stack, b.rows["half_width_scaled"]):
        m = (d.tol_plus - d.tol_minus) / 2                                     # band centre offset stays fixed
        scaled.append(ts.Dim(d.name, d.nominal, h + m, h - m, d.direction, d.dist, d.shift))
    r = ts.rss(scaled, 3)
    assert ts.cpk(r.mean, r.sigma, *spec) == pytest.approx(1.33)
    assert r.mean == pytest.approx(ts.rss(stack, 3).mean)


def test_budget_round_trip_single_row():
    spec = (-1.2, 1.2)
    b = ts.tolerance_budget(MIXED, *spec, 2.2, k=3)
    assert b.feasible
    done = 0
    for i, (d, f) in enumerate(zip(MIXED, b.rows["factor_alone"])):
        if np.isnan(f):
            continue
        dims = list(MIXED)
        dims[i] = ts.Dim(d.name, d.nominal, d.tol_plus * f, d.tol_minus * f, d.direction, d.dist, d.shift)
        r = ts.rss(dims, 3)
        assert ts.cpk(r.mean, r.sigma, *spec) == pytest.approx(2.2)
        done += 1
    assert done >= 1


def test_budget_largest_variance_row_is_the_uniform_one():
    b = ts.tolerance_budget(MIXED, -1.2, 1.2, 1.33)
    top = b.rows.sort_values("variance_share", ascending=False, kind="stable").iloc[0]
    assert top["name"] == "A" and top["variance_share"] == pytest.approx(0.03 / 0.055)


def test_budget_infeasible_cases_give_messages():
    out = ts.tolerance_budget(TWO, 11.0, 12.0, 1.33)                           # mean 10 outside the spec
    assert not out.feasible and "outside" in out.message and np.isnan(out.scale)
    zero = ts.tolerance_budget([ts.Dim("A", 10.0, 0.0, 0.0, +1)], 9.0, 11.0, 1.33)
    assert not zero.feasible and "zero" in zero.message


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), None, "abc"])
def test_budget_bad_target_raises(target):
    with pytest.raises(ValidationError):
        ts.tolerance_budget(TWO, 9.5, 10.5, target)


def test_budget_bad_spec_and_k_raise():
    with pytest.raises(ValidationError):
        ts.tolerance_budget(TWO, 10.5, 9.5, 1.33)
    with pytest.raises(ValidationError):
        ts.tolerance_budget(TWO, 9.5, 10.5, 1.33, k=1.0)


# ---- table rows with distribution and shift ---------------------------------------------------
def test_rows_to_dims_seven_columns_and_unit_scale():
    rows = [("A", "+", 4.0, 0.002, 0.004, "uniform", 0.001), ("B", "−", 1.0, 0.001, 0.001, None, None),
            ("C", "+", 2.0, 0.001, 0.001, " Triangular ", float("nan")), (None,) * 7]
    dims = ts.rows_to_dims(rows, 0.0254)                                       # inches -> metres
    assert len(dims) == 3
    assert dims[0].dist == ts.UNIFORM and dims[0].shift == pytest.approx(0.001 * 0.0254)
    assert dims[1].dist == ts.NORMAL and dims[1].shift == 0.0
    assert dims[2].dist == ts.TRIANGULAR and dims[2].shift == 0.0


@pytest.mark.parametrize("row", [
    ("A", "+", 4.0, 0.1, 0.1, "gaussian", 0.0),          # unknown distribution
    ("A", "+", 4.0, 0.1, 0.1, "normal", "abc"),          # text in the shift cell
    ("A", "+", 4.0, 0.1, 0.1, "normal", float("inf")),
    ("A", "+", 4.0, 0.1, 0.1, "normal"),                 # wrong number of columns
])
def test_rows_to_dims_bad_new_columns_raise(row):
    with pytest.raises(ValidationError):
        ts.rows_to_dims([row])


def test_validate_dims_checks_dist_and_shift():
    with pytest.raises(ValidationError):
        ts.validate_dims([ts.Dim("A", 10.0, 0.1, 0.1, 1, "gaussian")])
    with pytest.raises(ValidationError):
        ts.validate_dims([ts.Dim("A", 10.0, 0.1, 0.1, 1, ts.NORMAL, float("nan"))])


# ---- example stacks ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, nominal, lo, hi, mean", [
    ("Housing + 3 parts (default)", 5.0, 4.75, 5.25, 5.0),
    ("Shaft / bearing / housing end float", 0.5, 0.30, 0.94, 0.64),             # 60.02 - 2 x 17.94 - 23.5
    ("Pin in hole (clearance)", 0.01, 0.005, 0.029, 0.017),                     # 10.0075 - 9.9905
    ("Shim stack", 0.30, 0.08, 0.62, 0.37),                                     # 45 - 20 - 11.95 - 11.18 - 0.75 - 0.75
])
def test_example_stacks_hand_values(name, nominal, lo, hi, mean):
    dims, lsl, usl = ts.example_dims(name)
    assert ts.nominal_gap(dims) == pytest.approx(nominal * 1e-3)
    wc = ts.worst_case(dims)
    assert wc[0] == pytest.approx(lo * 1e-3) and wc[1] == pytest.approx(hi * 1e-3)
    assert ts.rss(dims, 3).mean == pytest.approx(mean * 1e-3)
    assert lsl < usl
    res = ts.analyse_stack(dims, lsl, usl, 3, 5000)                              # runs end to end
    assert np.isfinite(res.cpk_rss) and res.ppm_rss >= 0


def test_examples_are_labelled_and_default_matches_original_stack():
    assert all(ex["note"] for ex in ts.EXAMPLES.values())
    assert ts.DEFAULT_EXAMPLE == "Housing + 3 parts (default)"
    dims, lsl, usl = ts.example_dims(ts.DEFAULT_EXAMPLE)
    assert (lsl, usl) == (pytest.approx(4.8e-3), pytest.approx(5.2e-3))
    assert all(d.dist == ts.NORMAL and d.shift == 0.0 for d in dims)
    assert [d.name for d in dims] == ["Housing", "Part A", "Part B", "Part C"]


def test_unknown_example_raises():
    with pytest.raises(ValidationError):
        ts.example_dims("nope")


# ---- CSV ---------------------------------------------------------------------------------------
def test_results_csv_contents_and_units():
    dims = [ts.Dim("H", 100e-3, 0.1e-3, 0.1e-3, +1), ts.Dim("A", 40e-3, 0.05e-3, 0.05e-3, -1),
            ts.Dim("B", 30e-3, 0.05e-3, 0.05e-3, -1), ts.Dim("C", 25e-3, 0.05e-3, 0.05e-3, -1)]
    res = ts.analyse_stack(dims, 4.8e-3, 5.2e-3, 3, 5000)
    raw = ts.results_csv(res, "mm", 1e-3)
    assert raw.decode("ascii")                                                  # plain ASCII
    df = pd.read_csv(io.BytesIO(raw))
    assert list(df.columns) == ["Method", "Min gap (mm)", "Max gap (mm)", "Mean (mm)", "Sigma (mm)", "Cp", "Cpk",
                                "ppm out of spec", "Sigma level", "Inside spec"]
    assert len(df) == 3
    assert df.loc[0, "Min gap (mm)"] == pytest.approx(4.75) and df.loc[0, "Max gap (mm)"] == pytest.approx(5.25)
    assert df.loc[0, "Inside spec"] == False and pd.isna(df.loc[0, "Cpk"])      # noqa: E712 (worst-case has no Cpk)
    sigma = math.sqrt(0.1 ** 2 + 3 * 0.05 ** 2) / 3                              # mm
    assert df.loc[1, "Sigma (mm)"] == pytest.approx(sigma) and df.loc[1, "Mean (mm)"] == pytest.approx(5.0)
    assert df.loc[1, "Cpk"] == pytest.approx(0.2 / (3 * sigma)) and df.loc[1, "Sigma level"] == pytest.approx(0.2 / sigma)
    assert df.loc[1, "Min gap (mm)"] == pytest.approx(5.0 - 3 * sigma)
    assert df.loc[2, "Cpk"] == pytest.approx(res.cpk_mc)
    inch = pd.read_csv(io.BytesIO(ts.results_csv(res, "in", 0.0254)))
    assert "Min gap (in)" in inch.columns and inch.loc[0, "Min gap (in)"] == pytest.approx(4.75 / 25.4)
