"""Unit handling. All calculations run in SI; conversion happens only at the input/display edge.

Pure module: no Streamlit dependency, so it is unit-testable.
"""
from __future__ import annotations

from common.validation import ValidationError

SI = "SI"
US = "US"
SYSTEMS = (SI, US)

# kind -> system -> (label, factor such that value_in_SI = value_in_unit * factor)
UNITS: dict[str, dict[str, tuple[str, float]]] = {
    "length":       {SI: ("mm", 1e-3),   US: ("in", 0.0254)},       # section sizes, diameters
    "length_long":  {SI: ("m", 1.0),     US: ("ft", 0.3048)},       # spans, pipe runs
    "force":        {SI: ("N", 1.0),     US: ("lbf", 4.4482216152605)},
    "line_load":    {SI: ("N/m", 1.0),   US: ("lbf/ft", 14.593902937206)},   # force per length, uniform loads
    "torque":       {SI: ("N·m", 1.0),   US: ("lbf·in", 0.1129848290276)},
    "pressure":     {SI: ("kPa", 1e3),   US: ("psi", 6894.757293168)},
    "stress":       {SI: ("MPa", 1e6),   US: ("ksi", 6.894757293168e6)},
    "flow":         {SI: ("L/s", 1e-3),  US: ("gpm", 6.30901964e-5)},
    "velocity":     {SI: ("m/s", 1.0),   US: ("ft/s", 0.3048)},
    "density":      {SI: ("kg/m³", 1.0), US: ("lb/ft³", 16.01846337396)},
    "power":        {SI: ("kW", 1e3),    US: ("hp", 745.69987158227)},   # mechanical horsepower
}


def _lookup(kind: str, system: str) -> tuple[str, float]:
    if kind not in UNITS:
        raise ValidationError(f"Unknown quantity type '{kind}'.")
    if system not in SYSTEMS:
        raise ValidationError(f"Unknown unit system '{system}'. Use 'SI' or 'US'.")
    return UNITS[kind][system]


def label(kind: str, system: str) -> str:
    """Unit label shown to the user, e.g. label('stress', 'US') -> 'ksi'."""
    return _lookup(kind, system)[0]


def to_si(value: float, kind: str, system: str) -> float:
    """Convert a value entered in the user's unit system to SI base units (m, N, Pa, ...)."""
    return value * _lookup(kind, system)[1]


def from_si(value_si: float, kind: str, system: str) -> float:
    """Convert an SI base-unit value to the user's unit system for display."""
    return value_si / _lookup(kind, system)[1]


def fmt(value_si: float, kind: str, system: str, digits: int = 3) -> str:
    """Format an SI value for display in the user's units, e.g. '25.4 mm' or '1 in'."""
    x = from_si(value_si, kind, system)
    text = f"{x:,.0f}" if abs(x) >= 10**digits else f"{x:.{digits}g}"   # avoid '5e+03'
    return f"{text} {label(kind, system)}"
