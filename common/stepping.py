"""Slider step helpers. Pure module (no Streamlit) so it can be unit tested.

A slider whose step is range / N lands on awkward values (9.96, 10.02). Here the step is a round
1-2-5 number and the slider limits are moved onto that grid, so every slider position is a clean value.
Exact values are typed in the number box beside the slider (see common/ui.py: slider_box).
"""
from __future__ import annotations

import math


def nice_step(span: float, target: int = 300) -> float:
    """Largest 1-2-5 step (1, 2, 5 x 10^n) that is no bigger than span / target."""
    if not span > 0:
        raise ValueError("span must be positive")
    raw = span / target
    base = 10.0 ** math.floor(math.log10(raw))
    for m in (5, 2, 1):
        if m * base <= raw * (1 + 1e-9):
            return float(f"{m * base:.10g}")
    return float(f"{base:.10g}")


def decimals(step: float) -> int:
    """Decimal places needed to show multiples of step, e.g. 0.02 -> 2, 0.5 -> 1, 100 -> 0."""
    text = f"{step:.10f}".rstrip("0")
    return max(0, len(text.split(".")[1])) if "." in text else 0


def snap_range(lo: float, hi: float, step: float) -> tuple[float, float]:
    """Move lo up and hi down onto multiples of step (so the slider grid contains round numbers)."""
    d = decimals(step) + 2
    lo_s = round(math.ceil(lo / step - 1e-9) * step, d)
    hi_s = round(math.floor(hi / step + 1e-9) * step, d)
    return (lo_s, hi_s) if hi_s > lo_s else (lo, hi)


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)
