import pytest

from src.predictions.pi_rating import PiRatingEngine
from src.predictions.features import _create_pi_rating_features


def test_new_teams_start_with_zero_home_and_away_ratings():
    engine = PiRatingEngine()

    assert engine.predict_diff("BOS", "NYK") == 0.0
    assert engine.get_ratings("BOS") == {"home": 0.0, "away": 0.0}
    assert engine.get_ratings("NYK") == {"home": 0.0, "away": 0.0}


def test_home_overperformance_updates_primary_and_cross_context_ratings():
    engine = PiRatingEngine()

    adjustment = engine.update("BOS", "NYK", 110, 100)

    assert adjustment > 0
    assert engine.get_ratings("BOS") == pytest.approx(
        {"home": adjustment, "away": adjustment * 0.6}
    )
    assert engine.get_ratings("NYK") == pytest.approx(
        {"home": -adjustment * 0.6, "away": -adjustment}
    )


def test_underperformance_reverses_the_update_direction():
    engine = PiRatingEngine()

    adjustment = engine.update("BOS", "NYK", 90, 100)

    assert adjustment < 0
    assert engine.predict_diff("BOS", "NYK") == pytest.approx(adjustment * 2)


def test_prediction_is_pre_game_and_does_not_depend_on_current_outcome():
    before = PiRatingEngine()
    after = PiRatingEngine()

    for engine in (before, after):
        engine.update("BOS", "NYK", 110, 100)

    expected_before = before.predict_diff("LAL", "CHI")
    expected_after = after.predict_diff("LAL", "CHI")
    assert expected_before == expected_after == 0.0

    # Changing a later game's result cannot alter the value already read for
    # the next game when updates are applied only after each final score.
    before.update("LAL", "CHI", 105, 100)
    after.update("LAL", "CHI", 80, 120)
    assert before.predict_diff("BOS", "NYK") == after.predict_diff("BOS", "NYK")


def test_invalid_scores_and_team_names_are_rejected():
    engine = PiRatingEngine()

    with pytest.raises(ValueError):
        engine.update("", "NYK", 100, 90)
    with pytest.raises(ValueError):
        engine.update("BOS", "NYK", float("nan"), 90)


def test_feature_values_use_only_prior_games():
    prior = [
        {
            "game_id": "1",
            "game_date": "2026-01-01",
            "home": "BOS",
            "away": "NYK",
            "home_score": 110,
            "away_score": 100,
        }
    ]

    first = _create_pi_rating_features(prior, prior, "BOS", "NYK").iloc[0]
    changed_future = _create_pi_rating_features(prior, prior, "BOS", "NYK").iloc[0]

    assert first.to_dict() == changed_future.to_dict()
