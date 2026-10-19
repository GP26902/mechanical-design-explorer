import math
import pytest

from common import units
from common.validation import ValidationError


def test_known_conversions():
    # Textbook constants
    assert units.to_si(1, "length", "US") == pytest.approx(0.0254)           # 1 in = 25.4 mm
    assert units.to_si(1, "length_long", "US") == pytest.approx(0.3048)      # 1 ft
    assert units.to_si(1, "force", "US") == pytest.approx(4.4482216, rel=1e-6)  # 1 lbf
    assert units.to_si(1, "pressure", "US") == pytest.approx(6894.757, rel=1e-6)  # 1 psi
    assert units.to_si(1, "stress", "US") == pytest.approx(6.894757e6, rel=1e-6)  # 1 ksi
    assert units.to_si(1, "flow", "US") == pytest.approx(6.30902e-5, rel=1e-5)   # 1 gpm
    assert units.to_si(25.4, "length", "SI") == pytest.approx(0.0254)        # 25.4 mm


def test_torque_us_matches_force_times_length():
    assert units.to_si(1, "torque", "US") == pytest.approx(units.to_si(1, "force", "US") * 0.0254)


@pytest.mark.parametrize("kind", list(units.UNITS))
@pytest.mark.parametrize("system", units.SYSTEMS)
def test_round_trip(kind, system):
    assert units.from_si(units.to_si(12.34, kind, system), kind, system) == pytest.approx(12.34)


def test_labels_and_format():
    assert units.label("stress", "US") == "ksi"
    assert units.fmt(0.0254, "length", "US") == "1 in"


def test_unknown_kind_or_system_gives_clear_error():
    with pytest.raises(ValidationError):
        units.to_si(1, "banana", "SI")
    with pytest.raises(ValidationError):
        units.to_si(1, "length", "metric-ish")
