"""Chapter 3: Tolerance stack-up & GD&T position.

Learning tool based on ASME Y14.5 concepts; NOT a certified interpretation of the standard.
Size stack-up (one-dimensional, linear): worst-case, RSS (root-sum-square) and Monte Carlo.
GD&T: position tolerance zone with bonus tolerance at MMC (or RFS, no bonus) for a hole or pin.
No FEA/CFD of any kind; the Monte Carlo is plain random sampling of the dimension distributions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `streamlit run chapters/x.py` find common/

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import units
from common.theme import ACCENT, FAIL_COLOR, PASS_COLOR, apply_theme
from common.ui import get_unit_system, run_standalone
from common.validation import ValidationError, require_in_range, require_non_negative, require_positive

TITLE = "3 · Tolerance stack-up & GD&T"
DESCRIPTION = ("Worst-case, RSS and Monte Carlo stack-ups with Cpk and contribution ranking, plus a "
               "GD&T position tolerance zone with bonus tolerance at MMC (learning tool, ASME Y14.5 concepts).")

# =============================================================================================
# CALCULATIONS (pure functions, SI units in and out, no Streamlit)
# =============================================================================================
MIN_SAMPLES, MAX_SAMPLES = 1_000, 1_000_000


@dataclass(frozen=True)
class Dim:
    """One link in the stack. direction=+1 adds to the gap, -1 subtracts from it."""
    name: str
    nominal: float
    tol_plus: float
    tol_minus: float
    direction: int = 1


def _finite(v, name: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number.") from None
    if not math.isfinite(x):
        raise ValidationError(f"{name} must be a finite number (no blank or NaN cells).")
    return x


def validate_dims(dims: Sequence[Dim]) -> list:
    """Check every row; return a clean list. Raises ValidationError with the row name."""
    dims = list(dims)
    if not dims:
        raise ValidationError("Add at least one dimension to the table.")
    for i, d in enumerate(dims, start=1):
        who = f"Row {i} ('{d.name}')" if d.name else f"Row {i}"
        for v, lab in ((d.nominal, "nominal size"), (d.tol_plus, "+tolerance"), (d.tol_minus, "−tolerance")):
            _finite(v, f"{who}: {lab}")
        require_positive(d.nominal, f"{who}: nominal size")
        require_non_negative(d.tol_plus, f"{who}: +tolerance")
        require_non_negative(d.tol_minus, f"{who}: −tolerance")
        if d.direction not in (1, -1):
            raise ValidationError(f"{who}: direction must be +1 or −1.")
    return dims


DIRECTIONS = {"+": 1, "−": -1, "-": -1}


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def rows_to_dims(rows, scale: float = 1.0) -> list:
    """Table rows (name, direction symbol, nominal, +tol, -tol) in the user's units -> list of Dim in SI.
    `scale` converts one user length unit to metres. Fully blank rows are skipped."""
    dims = []
    for i, row in enumerate(rows, start=1):
        name, sign, nom, tp, tm = row
        if all(_blank(v) for v in row):
            continue
        who = f"Row {i}" + (f" ('{name}')" if not _blank(name) else "")
        if sign not in DIRECTIONS:
            raise ValidationError(f"{who}: choose a direction, + or −.")
        nom, tp, tm = (_finite(v, f"{who}: {lab}") * scale
                       for v, lab in ((nom, "nominal size"), (tp, "+tolerance"), (tm, "−tolerance")))
        dims.append(Dim("" if _blank(name) else str(name), nom, tp, tm, DIRECTIONS[sign]))
    return dims


def nominal_gap(dims: Sequence[Dim]) -> float:
    return float(sum(d.direction * d.nominal for d in validate_dims(dims)))


def worst_case(dims: Sequence[Dim]) -> tuple:
    """(min gap, max gap) with every dimension at its own extreme."""
    lo = hi = 0.0
    for d in validate_dims(dims):
        if d.direction == 1:
            lo += d.nominal - d.tol_minus
            hi += d.nominal + d.tol_plus
        else:
            lo -= d.nominal + d.tol_plus
            hi -= d.nominal - d.tol_minus
    return lo, hi


def dim_stats(d: Dim, k: float) -> tuple:
    """(mean contribution to the gap, sigma contribution). The tolerance band is taken as ±k·sigma,
    centred on the middle of the band, so an asymmetric tolerance shifts the mean."""
    mid = d.nominal + (d.tol_plus - d.tol_minus) / 2
    sigma = (d.tol_plus + d.tol_minus) / (2 * k)
    return d.direction * mid, sigma


def _check_k(k):
    return require_in_range(k, "Sigma multiplier k", 2.0, 6.0)


@dataclass(frozen=True)
class RSSResult:
    mean: float
    sigma: float
    lo: float      # mean - k*sigma
    hi: float      # mean + k*sigma


def rss(dims: Sequence[Dim], k: float = 3.0) -> RSSResult:
    dims, k = validate_dims(dims), _check_k(k)
    stats = [dim_stats(d, k) for d in dims]
    mean = float(sum(m for m, _ in stats))
    sigma = math.sqrt(sum(s * s for _, s in stats))
    return RSSResult(mean, sigma, mean - k * sigma, mean + k * sigma)


def monte_carlo(dims: Sequence[Dim], k: float = 3.0, n: int = 50_000, seed: int = 1) -> np.ndarray:
    """Gap samples: every dimension drawn from a normal distribution (not truncated at its limits)."""
    dims, k = validate_dims(dims), _check_k(k)
    if not (MIN_SAMPLES <= n <= MAX_SAMPLES) or int(n) != n:
        raise ValidationError(f"Number of samples must be a whole number between {MIN_SAMPLES:,} and {MAX_SAMPLES:,}.")
    rng = np.random.default_rng(seed)
    gap = np.zeros(int(n))
    for d in dims:
        mean, sigma = dim_stats(d, k)
        gap += mean + sigma * rng.standard_normal(int(n)) if sigma > 0 else mean
    return gap


def check_spec(lsl: float, usl: float) -> tuple:
    lsl, usl = _finite(lsl, "Lower spec limit"), _finite(usl, "Upper spec limit")
    if lsl >= usl:
        raise ValidationError("Lower spec limit must be less than the upper spec limit.")
    return lsl, usl


def cpk(mean: float, sigma: float, lsl: float, usl: float) -> float:
    """min(USL - mean, mean - LSL) / (3 sigma). With sigma = 0: inf if inside the limits, else 0."""
    lsl, usl = check_spec(lsl, usl)
    if sigma < 0:
        raise ValidationError("Standard deviation cannot be negative.")
    if sigma == 0:
        return float("inf") if lsl <= mean <= usl else 0.0
    return min(usl - mean, mean - lsl) / (3 * sigma)


def cp(sigma: float, lsl: float, usl: float) -> float:
    lsl, usl = check_spec(lsl, usl)
    return float("inf") if sigma == 0 else (usl - lsl) / (6 * sigma)


def fraction_out_of_spec(samples: np.ndarray, lsl: float, usl: float) -> float:
    lsl, usl = check_spec(lsl, usl)
    s = np.asarray(samples, dtype=float)
    return float(np.mean((s < lsl) | (s > usl)))


def contributions(dims: Sequence[Dim], k: float = 3.0) -> pd.DataFrame:
    """Share of the RSS variance and of the worst-case total for each dimension, largest first."""
    dims, k = validate_dims(dims), _check_k(k)
    var = np.array([dim_stats(d, k)[1] ** 2 for d in dims])
    band = np.array([d.tol_plus + d.tol_minus for d in dims])
    df = pd.DataFrame({
        "name": [d.name or f"Row {i}" for i, d in enumerate(dims, start=1)],
        "variance_share": var / var.sum() if var.sum() > 0 else np.zeros(len(dims)),
        "worst_case_share": band / band.sum() if band.sum() > 0 else np.zeros(len(dims)),
    })
    return df.sort_values("variance_share", ascending=False, kind="stable").reset_index(drop=True)


@dataclass
class StackResult:
    nominal: float
    wc_lo: float
    wc_hi: float
    rss: RSSResult
    samples: np.ndarray
    mc_mean: float
    mc_sigma: float
    cpk_rss: float
    cpk_mc: float
    cp_rss: float
    out_wc: bool                # worst-case range falls outside the spec limits
    out_rss: bool
    frac_out_mc: float
    contrib: pd.DataFrame
    lsl: float
    usl: float


def analyse_stack(dims: Sequence[Dim], lsl: float, usl: float, k: float = 3.0,
                  n: int = 50_000, seed: int = 1) -> StackResult:
    dims = validate_dims(dims)
    lsl, usl = check_spec(lsl, usl)
    wc_lo, wc_hi = worst_case(dims)
    r = rss(dims, k)
    s = monte_carlo(dims, k, n, seed)
    mu, sd = float(s.mean()), float(s.std(ddof=1))
    return StackResult(nominal_gap(dims), wc_lo, wc_hi, r, s, mu, sd,
                       cpk(r.mean, r.sigma, lsl, usl), cpk(mu, sd, lsl, usl), cp(r.sigma, lsl, usl),
                       wc_lo < lsl or wc_hi > usl, r.lo < lsl or r.hi > usl,
                       fraction_out_of_spec(s, lsl, usl), contributions(dims, k), lsl, usl)


# ---- GD&T position (hole or pin) -------------------------------------------------------------
HOLE, PIN = "hole", "pin"
MMC, RFS = "MMC", "RFS"


@dataclass(frozen=True)
class GdtResult:
    feature: str
    modifier: str
    mmc_size: float
    size_ok: bool           # actual size within its size limits
    bonus: float
    zone_dia: float         # allowed position tolerance diameter (stated + bonus)
    measured_dia: float     # 2 * radial offset of the actual axis from true position
    virtual_condition: float
    passed: bool
    passed_rfs: bool        # same part judged without any bonus, for comparison


def position_offset_diameter(dx: float, dy: float) -> float:
    """Diameter of the smallest circle about true position that contains the measured axis."""
    return 2.0 * math.hypot(_finite(dx, "Offset x"), _finite(dy, "Offset y"))


def gdt_position(feature: str, size_min: float, size_max: float, actual: float, pos_tol: float,
                 dx: float, dy: float, modifier: str = MMC) -> GdtResult:
    """Position tolerance zone for a hole (internal) or pin (external) feature of size.
    Hole: MMC = smallest size, bonus = actual - size_min.  Pin: MMC = largest size, bonus = size_max - actual.
    VC (virtual condition) = MMC -/+ position tolerance (hole minus, pin plus). RFS gives no bonus."""
    if feature not in (HOLE, PIN):
        raise ValidationError("Feature must be a hole or a pin.")
    if modifier not in (MMC, RFS):
        raise ValidationError("Modifier must be MMC or RFS.")
    for v, lab in ((size_min, "Minimum size"), (size_max, "Maximum size"), (actual, "Actual size"),
                   (pos_tol, "Position tolerance")):
        _finite(v, lab)
    require_positive(size_min, "Minimum size")
    require_positive(size_max, "Maximum size")
    if size_min >= size_max:
        raise ValidationError("Minimum size must be less than maximum size.")
    require_positive(actual, "Actual size")
    require_non_negative(pos_tol, "Position tolerance")
    measured = position_offset_diameter(dx, dy)

    size_ok = size_min <= actual <= size_max
    raw = actual - size_min if feature == HOLE else size_max - actual
    bonus = float(np.clip(raw, 0.0, size_max - size_min)) if modifier == MMC else 0.0
    mmc_size = size_min if feature == HOLE else size_max
    vc = mmc_size - pos_tol if feature == HOLE else mmc_size + pos_tol
    zone = pos_tol + bonus
    eps = 1e-12
    return GdtResult(feature, modifier, mmc_size, size_ok, bonus, zone, measured, vc,
                     size_ok and measured <= zone + eps, size_ok and measured <= pos_tol + eps)


def bonus_curve(feature: str, size_min: float, size_max: float, pos_tol: float, n: int = 50) -> tuple:
    """(sizes, allowed zone diameters) across the size range, for the bonus-vs-size chart."""
    gdt_position(feature, size_min, size_max, size_min, pos_tol, 0.0, 0.0)   # validates inputs
    sizes = np.linspace(size_min, size_max, n)
    bonus = sizes - size_min if feature == HOLE else size_max - sizes
    return sizes, pos_tol + bonus


# =============================================================================================
# FIGURES
# =============================================================================================
GREY, AMBER = "#6b7280", "#f59e0b"
N_SAMPLES, SEED = 50_000, 1


def _L(v, system):
    return units.from_si(v, "length", system)


def histogram_figure(res: StackResult, k: float, system) -> go.Figure:
    """Monte Carlo gap histogram. Bars whose centre lies outside the spec limits are red."""
    s = _L(res.samples, system)
    lsl, usl = _L(res.lsl, system), _L(res.usl, system)
    counts, edges = np.histogram(s, bins=60, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    out = (centres < lsl) | (centres > usl)
    fig = go.Figure()
    for mask, colour, name in ((~out, ACCENT, "Monte Carlo gap, inside spec"), (out, FAIL_COLOR, "Monte Carlo gap, outside spec")):
        fig.add_trace(go.Bar(x=centres[mask], y=counts[mask], width=np.diff(edges)[mask], marker_color=colour,
                             opacity=0.85, name=name, hovertemplate="%{x:.4f}<extra></extra>"))
    wc = (_L(res.wc_lo, system), _L(res.wc_hi, system))
    rs = (_L(res.rss.lo, system), _L(res.rss.hi, system))
    xmin, xmax = min(edges[0], lsl, wc[0]), max(edges[-1], usl, wc[1])
    pad = 0.04 * (xmax - xmin)
    ymax = float(counts.max()) * 1.18
    if res.rss.sigma > 0:                                   # normal curve from the RSS mean and sigma
        mu, sd = _L(res.rss.mean, system), _L(res.rss.sigma, system)
        xs = np.linspace(xmin - pad, xmax + pad, 400)
        pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines", line=dict(color="#111", width=1.5),
                                 name="Normal curve (RSS mean, σ)", hoverinfo="skip"))
        ymax = max(ymax, float(pdf.max()) * 1.1)
    for pair, colour, dash, name in ((wc, GREY, "dot", "Worst-case limits"),
                                     (rs, AMBER, "dot", f"RSS limits (±{k:g}σ)"),
                                     ((lsl, usl), FAIL_COLOR, "dash", "Spec limits (LSL / USL)")):
        for j, x in enumerate(pair):
            fig.add_trace(go.Scatter(x=[x, x], y=[0, ymax], mode="lines", line=dict(color=colour, dash=dash, width=2),
                                     name=name, legendgroup=name, showlegend=(j == 0), hovertemplate=f"{name}: %{{x:.4f}}<extra></extra>"))
    c = lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.98, xanchor="right", yanchor="top", showarrow=False,
                       text=f"<b>Cpk (Monte Carlo) = {c(res.cpk_mc)}</b><br>Cpk (RSS) = {c(res.cpk_rss)}",
                       align="right", bordercolor="#999", borderwidth=1, bgcolor="rgba(255,255,255,0.85)")
    apply_theme(fig, f"Gap distribution vs spec limits ({N_SAMPLES:,} Monte Carlo samples)", 430)
    fig.update_xaxes(range=[xmin - pad, xmax + pad], title=f"Gap ({units.label('length', system)})")
    fig.update_yaxes(title="Probability density", showticklabels=False, range=[0, ymax])
    fig.update_layout(bargap=0, legend=dict(orientation="h", y=-0.22))
    return fig


def contribution_figure(res: StackResult) -> go.Figure:
    c = res.contrib
    fig = go.Figure([
        go.Bar(name="Share of RSS variance (drives Cpk)", x=c["name"], y=c["variance_share"] * 100,
               marker_color=ACCENT, text=[f"{v * 100:.1f}%" for v in c["variance_share"]], textposition="outside"),
        go.Bar(name="Share of worst-case total", x=c["name"], y=c["worst_case_share"] * 100,
               marker_color="#9ca3af", text=[f"{v * 100:.1f}%" for v in c["worst_case_share"]], textposition="outside")])
    apply_theme(fig, "Which dimension drives the variation (ranked)", 380)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(title="Contribution (%)", range=[0, max(115.0, float(c["variance_share"].max()) * 100 * 1.2)])
    return fig


def zone_figure(g: GdtResult, pos_tol: float, dx: float, dy: float, system) -> go.Figure:
    """Axis position vs tolerance zones, drawn to scale. The feature size itself is not drawn."""
    u = units.label("length", system)
    r0, r1 = _L(pos_tol, system) / 2, _L(g.zone_dia, system) / 2
    px, py = _L(dx, system), _L(dy, system)
    rm = math.hypot(px, py)
    lim = max(1.3 * max(r1, rm), _L(0.05e-3, system))
    colour = PASS_COLOR if g.passed else FAIL_COLOR
    fig = go.Figure()
    if g.bonus > 0:
        fig.add_shape(type="circle", x0=-r1, x1=r1, y0=-r1, y1=r1, fillcolor="rgba(31,119,180,0.10)",
                      line=dict(color=ACCENT, dash="dash", width=2))
        fig.add_annotation(x=0, y=r1, text=f"Zone with bonus: Ø{2 * r1:.4f} {u}", showarrow=False, yshift=12,
                           font=dict(color=ACCENT, size=12))
    fig.add_shape(type="circle", x0=-r0, x1=r0, y0=-r0, y1=r0, fillcolor="rgba(31,119,180,0.15)",
                  line=dict(color=ACCENT, width=2))
    if r0 > 0:
        fig.add_annotation(x=0, y=0, text=f"Stated Ø{2 * r0:.4f}", showarrow=False, font=dict(color=ACCENT, size=12))
    if rm > 0:
        fig.add_shape(type="circle", x0=-rm, x1=rm, y0=-rm, y1=rm, line=dict(color=colour, dash="dot", width=1.5))
    fig.add_shape(type="line", x0=-lim, x1=lim, y0=0, y1=0, line=dict(color="#bbb", width=1))
    fig.add_shape(type="line", x0=0, x1=0, y0=-lim, y1=lim, line=dict(color="#bbb", width=1))
    fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers", marker=dict(size=13, color=colour, symbol="x",
                 line=dict(width=2, color=colour)), name="Actual axis",
                 hovertemplate=f"Axis offset: x={px:.4f}, y={py:.4f} {u}<extra></extra>", showlegend=False))
    apply_theme(fig, "Position tolerance zones (to scale; true position at the centre)", 430)
    fig.update_xaxes(range=[-lim, lim], title=f"x ({u})", zeroline=False, constrain="domain")
    fig.update_yaxes(range=[-lim, lim], title=f"y ({u})", scaleanchor="x", scaleratio=1, zeroline=False)
    return fig


def bonus_figure(g: GdtResult, feature, size_min, size_max, actual, pos_tol, system) -> go.Figure:
    u = units.label("length", system)
    sizes, zone = bonus_curve(feature, size_min, size_max, pos_tol)
    if g.modifier == RFS:
        zone = np.full_like(zone, pos_tol)
    fig = go.Figure(go.Scatter(x=_L(sizes, system), y=_L(zone, system), mode="lines",
                               line=dict(color=ACCENT, width=3), name="Allowed zone Ø"))
    fig.add_hline(y=_L(pos_tol, system), line=dict(color=GREY, dash="dash"),
                  annotation_text="stated tolerance", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=[_L(actual, system)], y=[_L(g.zone_dia, system)], mode="markers",
                             marker=dict(size=12, color=PASS_COLOR if g.passed else FAIL_COLOR),
                             name="This part", hovertemplate="%{x:.4f}, %{y:.4f}<extra></extra>"))
    apply_theme(fig, "Allowed position zone vs actual feature size", 330)
    fig.update_xaxes(title=f"Actual feature size ({u})")
    fig.update_yaxes(title=f"Allowed position zone Ø ({u})")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return fig


# =============================================================================================
# UI
# =============================================================================================
ILLUSTRATIVE = [Dim("Housing", 100e-3, 0.10e-3, 0.10e-3, +1), Dim("Part A", 40e-3, 0.05e-3, 0.05e-3, -1),
                Dim("Part B", 30e-3, 0.05e-3, 0.05e-3, -1), Dim("Part C", 25e-3, 0.05e-3, 0.05e-3, -1)]


def _columns(system):
    u = units.label("length", system)
    return ["Name", "Dir", f"Nominal ({u})", f"+Tol ({u})", f"−Tol ({u})"]


def _default_df(system) -> pd.DataFrame:
    r = lambda v: round(_L(v, system), 6)
    return pd.DataFrame([[d.name, "+" if d.direction == 1 else "−", r(d.nominal), r(d.tol_plus), r(d.tol_minus)]
                         for d in ILLUSTRATIVE], columns=_columns(system))


def _step(system):
    return 0.01 if system == units.SI else 0.0005


def _len_input(label, default_si, key, system):
    v = st.number_input(f"{label} ({units.label('length', system)})", value=float(_L(default_si, system)),
                        step=_step(system), format="%.4f", key=f"{key}_{system}")
    return units.to_si(v, "length", system)


def _len_slider(label, lo, hi, default, key, system):
    lo_u, hi_u, d_u = (_L(v, system) for v in (lo, hi, default))
    v = st.slider(f"{label} ({units.label('length', system)})", float(lo_u), float(hi_u), float(d_u),
                  step=float((hi_u - lo_u) / 240), format="%.3f" if system == units.SI else "%.5f",
                  key=f"{key}_{system}")
    return units.to_si(v, "length", system)


def _cpk_text(v):
    return "∞" if not np.isfinite(v) else f"{v:.2f}"


def _guard(fn, system):
    """Run one tab; show a validation problem inside that tab so the other tab keeps working."""
    try:
        fn(system)
    except ValidationError as e:
        st.error(f"Input problem: {e}")


def _render_stack(system) -> None:
    u = units.label("length", system)
    st.caption("Illustrative example values, not from a real project. Edit any cell; use the + row at the bottom "
               "to add a dimension. Dir: + adds to the gap, − subtracts from it.")
    cols = _columns(system)
    cfg = {cols[0]: st.column_config.TextColumn(),
           cols[1]: st.column_config.SelectboxColumn(options=["+", "−"], width="small"),
           **{c: st.column_config.NumberColumn(format="%.4f") for c in cols[2:]}}
    edited = st.data_editor(_default_df(system), num_rows="dynamic", hide_index=True, width="stretch",
                            column_config=cfg, key=f"ts_table_{system}")
    c1, c2, c3 = st.columns(3)
    lsl = units.to_si(c1.number_input(f"Lower spec limit of the gap ({u})", value=float(_L(4.8e-3, system)),
                                      step=_step(system), format="%.4f", key=f"ts_lsl_{system}"), "length", system)
    usl = units.to_si(c2.number_input(f"Upper spec limit of the gap ({u})", value=float(_L(5.2e-3, system)),
                                      step=_step(system), format="%.4f", key=f"ts_usl_{system}"), "length", system)
    k = c3.slider("Sigma multiplier k (tolerance = ±k·σ)", 2.0, 6.0, 3.0, 0.5, key="ts_k")

    dims = rows_to_dims(edited.itertuples(index=False, name=None), units.to_si(1.0, "length", system))
    res = analyse_stack(dims, lsl, usl, k, N_SAMPLES, SEED)

    if not res.out_wc:
        st.success("Worst-case stack is inside the spec limits: every assembly passes if all parts are within tolerance.")
    elif not res.out_rss:
        st.warning("Worst-case exceeds the spec limits but RSS is inside them. Acceptable only if the statistical "
                   "assumptions hold (independent, roughly normal, centred processes).")
    else:
        st.error("Even the RSS range exceeds the spec limits: expect a real fraction of assemblies out of spec.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nominal gap", units.fmt(res.nominal, "length", system, 4))
    m2.metric("Cpk (RSS)", _cpk_text(res.cpk_rss))
    m3.metric("Cpk (Monte Carlo)", _cpk_text(res.cpk_mc))
    m4.metric("Out of spec (Monte Carlo)", f"{res.frac_out_mc * 100:.2f} %")

    st.plotly_chart(histogram_figure(res, k, system), width="stretch")

    f = lambda v: f"{_L(v, system):.4f}"
    mc_lo, mc_hi = float(res.samples.min()), float(res.samples.max())
    rows = [("Worst-case", res.wc_lo, res.wc_hi, not res.out_wc),
            (f"RSS (mean ± {k:g}σ)", res.rss.lo, res.rss.hi, not res.out_rss),
            (f"Monte Carlo (observed min / max of {N_SAMPLES:,})", mc_lo, mc_hi, res.lsl <= mc_lo and mc_hi <= res.usl)]
    left, right = st.columns(2, gap="large")
    left.dataframe(pd.DataFrame({"Method": [r[0] for r in rows], f"Min gap ({u})": [f(r[1]) for r in rows],
                                 f"Max gap ({u})": [f(r[2]) for r in rows],
                                 "Inside spec": ["Yes" if r[3] else "No" for r in rows]}),
                   hide_index=True, width="stretch")
    top = res.contrib.iloc[0]
    left.caption(f"Largest contributor: **{top['name'] or 'Row 1'}** ({top['variance_share'] * 100:.1f} % of the variance). "
                 "Tightening it improves Cpk the most.")
    right.plotly_chart(contribution_figure(res), width="stretch")
    with st.expander("Method and limits"):
        st.markdown(
            "- One-dimensional, linear stack: gap = Σ (±) dimension. Worst-case puts every dimension at its own limit.\n"
            "- RSS and Monte Carlo treat each tolerance band as ±k·σ of a **normal** distribution centred on the middle of the "
            "band (an asymmetric tolerance shifts the mean). RSS limits = mean ± k·√Σσ²; Cpk = min(USL − μ, μ − LSL) / 3σ.\n"
            "- Monte Carlo draws 50,000 gaps with a fixed seed. Samples are **not** truncated at the tolerance limits, and it "
            "resolves only about 1 in 50,000 (0.002 %); for rarer events rely on Cpk.\n"
            "- Not modelled: process drift or shift, correlated dimensions, non-normal distributions, assembly "
            "geometry (angles, 2D/3D stacks), thermal or elastic effects.")


def _render_gdt(system) -> None:
    u = units.label("length", system)
    st.caption("Learning tool based on ASME Y14.5 concepts (position tolerance, MMC bonus tolerance); not a certified "
               "interpretation of the standard. Illustrative example values, not from a real project.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        feature = st.radio("Feature of size", [HOLE, PIN], format_func=str.capitalize, horizontal=True, key="ts_g_feature")
        modifier = st.radio("Position tolerance modifier", [MMC, RFS], horizontal=True, key="ts_g_mod",
                            format_func=lambda m: "At MMC (bonus allowed)" if m == MMC else "RFS (no bonus)")
        d_min, d_max, d_act, d_tol = ((10.0e-3, 10.2e-3, 10.15e-3, 0.2e-3) if feature == HOLE
                                      else (9.90e-3, 10.0e-3, 9.92e-3, 0.1e-3))
        size_min = _len_input(f"{feature.capitalize()} minimum size", d_min, f"ts_g_smin_{feature}", system)
        size_max = _len_input(f"{feature.capitalize()} maximum size", d_max, f"ts_g_smax_{feature}", system)
        actual = _len_input("Actual size of this part", d_act, f"ts_g_act_{feature}", system)
        pos_tol = _len_input("Position tolerance Ø", d_tol, f"ts_g_tol_{feature}", system)
        dx = _len_slider("Axis offset x from true position", -0.6e-3, 0.6e-3, 0.12e-3 if feature == HOLE else 0.05e-3,
                         f"ts_g_dx_{feature}", system)
        dy = _len_slider("Axis offset y from true position", -0.6e-3, 0.6e-3, 0.10e-3 if feature == HOLE else 0.05e-3,
                         f"ts_g_dy_{feature}", system)
    g = gdt_position(feature, size_min, size_max, actual, pos_tol, dx, dy, modifier)
    fl = lambda v: f"Ø{_L(v, system):.4f} {u}"
    with right:
        if g.passed:
            st.success(f"PASS: measured position {fl(g.measured_dia)} ≤ allowed zone {fl(g.zone_dia)}")
        elif not g.size_ok:
            st.error("FAIL: actual size is outside its size limits; the part is rejected whatever its position.")
        else:
            st.error(f"FAIL: measured position {fl(g.measured_dia)} exceeds the allowed zone {fl(g.zone_dia)}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bonus tolerance", units.fmt(g.bonus, "length", system, 4))
        c2.metric("Allowed zone Ø", units.fmt(g.zone_dia, "length", system, 4))
        c3.metric("Measured position Ø", units.fmt(g.measured_dia, "length", system, 4))
        c4.metric("Virtual condition", units.fmt(g.virtual_condition, "length", system, 4))
        if modifier == MMC and g.passed and not g.passed_rfs:
            st.info(f"Without the MMC bonus this part would fail the stated Ø{_L(pos_tol, system):.4f} {u}: the bonus "
                    "tolerance is what makes it acceptable.")
        st.plotly_chart(zone_figure(g, pos_tol, dx, dy, system), width="stretch")
        st.plotly_chart(bonus_figure(g, feature, size_min, size_max, actual, pos_tol, system), width="stretch")
        with st.expander("Method and limits"):
            st.markdown(
                "- Hole: MMC is the smallest size, bonus = actual − minimum. Pin: MMC is the largest size, bonus = maximum − "
                "actual. Allowed zone Ø = stated tolerance + bonus. Virtual condition = MMC − tolerance (hole) or "
                "MMC + tolerance (pin).\n"
                "- Measured position Ø = 2 × distance of the actual axis from true position; pass if it is ≤ the allowed zone "
                "and the actual size is within its limits.\n"
                "- Single feature of size, one position zone. Not modelled: datum reference frames and datum shift, "
                "composite or projected tolerance zones, orientation and form, LMC.")


def render() -> None:
    system = get_unit_system()
    tab_stack, tab_gdt = st.tabs(["Size stack-up", "GD&T position"])
    with tab_stack:
        _guard(_render_stack, system)
    with tab_gdt:
        _guard(_render_gdt, system)


if __name__ == "__main__":
    run_standalone(TITLE, DESCRIPTION, render)
