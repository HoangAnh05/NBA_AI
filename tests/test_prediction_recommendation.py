"""Tests for betting recommendation classification."""

import pytest

from src.web_app.game_data_processor import _classify_betting_recommendation


@pytest.mark.parametrize(
    ("win_probability", "predicted_margin", "betting_line", "expected"),
    [
        # LAL predicted by 4, with LAL -6.5: 2.5 points apart.
        (0.80, -4.0, 6.5, "POSITIVE"),
        # The threshold is strict: exactly 70% does not activate the override.
        (0.70, -4.0, 6.5, "NEGATIVE"),
        # Exactly three points is within the inclusive tolerance.
        (0.80, -3.5, 6.5, "POSITIVE"),
        # More than three points apart does not activate the override.
        (0.80, -2.0, 6.5, "NEGATIVE"),
        # Existing positive classifications remain positive.
        (0.60, 8.0, -5.0, "POSITIVE"),
    ],
)
def test_classifies_prediction_against_signed_betting_line(
    win_probability, predicted_margin, betting_line, expected
):
    assert (
        _classify_betting_recommendation(
            win_probability, predicted_margin, betting_line
        )
        == expected
    )


def test_missing_prediction_or_line_has_no_recommendation():
    assert _classify_betting_recommendation("", 4.0, -3.0) == ""
    assert _classify_betting_recommendation(0.8, "", -3.0) == ""
    assert _classify_betting_recommendation(0.8, 4.0, None) == ""
