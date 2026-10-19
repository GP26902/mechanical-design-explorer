import math
import pytest

from common.validation import (ValidationError, require_in_range, require_less_than,
                               require_non_negative, require_positive)


@pytest.mark.parametrize("bad", [0, -1, -0.001, math.nan, math.inf, None, "5", True])
def test_require_positive_rejects(bad):
    with pytest.raises(ValidationError):
        require_positive(bad, "Diameter")


def test_require_positive_accepts_and_message_names_input():
    assert require_positive(2.5, "Diameter") == 2.5
    with pytest.raises(ValidationError, match="Span"):
        require_positive(0, "Span")


def test_non_negative():
    assert require_non_negative(0, "Torque") == 0
    with pytest.raises(ValidationError):
        require_non_negative(-1, "Torque")


def test_in_range_and_less_than():
    assert require_in_range(5, "Position", 0, 10) == 5
    with pytest.raises(ValidationError):
        require_in_range(11, "Position", 0, 10)
    with pytest.raises(ValidationError):
        require_less_than(5, "Inner diameter", 3, "Outer diameter")
    require_less_than(3, "Inner diameter", 5, "Outer diameter")
