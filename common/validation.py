"""Input validation. Raise ValidationError with a plain-language message; chapters show it
with st.error instead of crashing. Pure module (no Streamlit)."""
from __future__ import annotations

import math


class ValidationError(ValueError):
    """An input value is impossible or out of range. The message is safe to show to the user."""


def _check_number(value: float, name: str) -> None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number (not NaN or infinity).")


def require_positive(value: float, name: str) -> float:
    _check_number(value, name)
    if value <= 0:
        raise ValidationError(f"{name} must be greater than zero (got {value:g}).")
    return value


def require_non_negative(value: float, name: str) -> float:
    _check_number(value, name)
    if value < 0:
        raise ValidationError(f"{name} cannot be negative (got {value:g}).")
    return value


def require_in_range(value: float, name: str, low: float, high: float) -> float:
    _check_number(value, name)
    if not (low <= value <= high):
        raise ValidationError(f"{name} must be between {low:g} and {high:g} (got {value:g}).")
    return value


def require_less_than(small: float, small_name: str, big: float, big_name: str) -> None:
    _check_number(small, small_name)
    _check_number(big, big_name)
    if not small < big:
        raise ValidationError(
            f"{small_name} ({small:g}) must be smaller than {big_name} ({big:g})."
        )
