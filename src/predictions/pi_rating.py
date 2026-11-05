"""Chronological Pi-rating engine for NBA game predictions.

The engine keeps separate home and away ratings for every team. Callers must
read the ratings before a game and call :meth:`update` only after that game's
final score is known.
"""

import math
from numbers import Real


class PiRatingEngine:
    """Track home and away Pi-ratings without using future game results."""

    def __init__(self, lr_lambda: float = 0.06, gamma: float = 0.60):
        if lr_lambda < 0:
            raise ValueError("lr_lambda must be non-negative.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")

        self.lambda_ = float(lr_lambda)
        self.gamma = float(gamma)
        self.ratings: dict[str, dict[str, float]] = {}

    def _get_team(self, team: str) -> dict[str, float]:
        """Return a team's ratings, initializing unseen teams at zero."""
        if not isinstance(team, str) or not team:
            raise ValueError("team must be a non-empty string.")
        return self.ratings.setdefault(team, {"home": 0.0, "away": 0.0})

    def get_ratings(self, team: str) -> dict[str, float]:
        """Return a copy of a team's current home and away ratings."""
        return self._get_team(team).copy()

    def predict_diff(self, home_team: str, away_team: str) -> float:
        """Return the expected home-minus-away point differential."""
        home = self._get_team(home_team)
        away = self._get_team(away_team)
        return home["home"] - away["away"]

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: Real,
        away_score: Real,
    ) -> float:
        """Update ratings after a completed game and return the adjustment.

        The adjustment is positive when the home team exceeds its expected
        point differential and negative when it underperforms.
        """
        if not isinstance(home_score, Real) or not math.isfinite(float(home_score)):
            raise ValueError("home_score must be a finite number.")
        if not isinstance(away_score, Real) or not math.isfinite(float(away_score)):
            raise ValueError("away_score must be a finite number.")

        expected_diff = self.predict_diff(home_team, away_team)
        actual_diff = float(home_score) - float(away_score)
        error = actual_diff - expected_diff
        psi = 3.0 * math.log10(1.0 + abs(error))
        adjustment = math.copysign(self.lambda_ * psi, error) if error else 0.0

        home = self._get_team(home_team)
        away = self._get_team(away_team)

        home["home"] += adjustment
        away["away"] -= adjustment
        home["away"] += adjustment * self.gamma
        away["home"] -= adjustment * self.gamma
        return adjustment

