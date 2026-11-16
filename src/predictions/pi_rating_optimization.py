"""Chronological backtesting and parameter evaluation for NBA Pi-ratings.

This module provides tools for evaluating and optimizing PiRatingEngine
parameters (lr_lambda, gamma) using chronological walk-forward validation
as specified in docs/pi_rating_optimization.md.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from src.predictions.pi_rating import PiRatingEngine

# Default grid search parameters recommended in docs/pi_rating_optimization.md
DEFAULT_LAMBDA_VALUES: tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.4,
    0.8,
    1.2,
    2.0,
)
DEFAULT_GAMMA_VALUES: tuple[float, ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)


class GameRecord(NamedTuple):
    """Normalized representation of a completed NBA game."""

    date: Any
    home_team: str
    away_team: str
    home_score: float
    away_score: float


def _get_field(item: Any, *field_names: str) -> Any:
    """Extract field value from a dict-like or object-like record."""
    for field in field_names:
        if isinstance(item, Mapping):
            if field in item:
                return item[field]
        elif hasattr(item, field):
            return getattr(item, field)
    return None


def prepare_games(games: Iterable[Any]) -> list[GameRecord]:
    """Clean, filter, de-duplicate, and chronologically sort game records.

    Accepts an iterable of dicts, namedtuples, objects, or a pandas DataFrame.
    Filters out games with missing or non-finite scores, removes duplicate
    matchups on the same date, and sorts all games chronologically by date.
    """
    if hasattr(games, "to_dict"):
        raw_items = games.to_dict(orient="records")
    else:
        raw_items = games

    prepared: list[GameRecord] = []
    seen: set[tuple[Any, str, str]] = set()

    for item in raw_items:
        date = _get_field(item, "date", "game_date", "date_time_utc")
        home = _get_field(item, "home_team", "home")
        away = _get_field(item, "away_team", "away")
        home_score = _get_field(item, "home_score")
        away_score = _get_field(item, "away_score")

        if date is None or home is None or away is None:
            continue
        if home_score is None or away_score is None:
            continue

        if not isinstance(home, str) or not isinstance(away, str):
            continue
        home = home.strip()
        away = away.strip()
        if not home or not away:
            continue

        if not isinstance(home_score, Real) or not isinstance(away_score, Real):
            continue
        h_score = float(home_score)
        a_score = float(away_score)
        if not (math.isfinite(h_score) and math.isfinite(a_score)):
            continue

        dedup_key = (date, home, away)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        prepared.append(
            GameRecord(
                date=date,
                home_team=home,
                away_team=away,
                home_score=h_score,
                away_score=a_score,
            )
        )

    # Sort chronologically by date
    try:
        prepared.sort(key=lambda g: g.date)
    except TypeError:
        # Fallback to string representation if dates are unorderable mixed types
        prepared.sort(key=lambda g: str(g.date))

    return prepared


def _resolve_validation_index(
    total_games: int,
    validation_start: float | int | str,
    dates: Sequence[Any],
) -> int:
    """Resolve the split index where the validation period begins."""
    if total_games == 0:
        raise ValueError("The games dataset contains no completed games.")

    if isinstance(validation_start, float):
        if not (0.0 <= validation_start < 1.0):
            raise ValueError(
                "validation_start as a float fraction must be in [0.0, 1.0)."
            )
        split_idx = int(total_games * validation_start)
    elif isinstance(validation_start, int) and not isinstance(validation_start, bool):
        split_idx = validation_start
    elif isinstance(validation_start, str):
        split_idx = next(
            (i for i, d in enumerate(dates) if str(d) >= validation_start),
            total_games,
        )
    else:
        raise TypeError(
            f"Unsupported validation_start type: {type(validation_start).__name__}"
        )

    if split_idx < 0:
        raise ValueError("validation_start index must be non-negative.")
    if split_idx >= total_games:
        raise ValueError("The validation period contains no games.")

    return split_idx


def calculate_metrics(
    actuals: Sequence[float],
    predicteds: Sequence[float],
) -> dict[str, float | int]:
    """Calculate MAE, RMSE, bias, and winner accuracy for predicted differentials.

    Error is defined as: actual_diff - predicted_diff.
    Winner accuracy counts predictions where the predicted sign matches
    the actual sign (positive for home win, negative for away win).
    """
    if len(actuals) != len(predicteds):
        raise ValueError("actuals and predicteds must have the same length.")
    if not actuals:
        raise ValueError("Cannot calculate metrics on an empty sequence.")

    n = len(actuals)
    errors = [a - p for a, p in zip(actuals, predicteds)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    bias = sum(errors) / n

    correct_winners = sum(
        1
        for a, p in zip(actuals, predicteds)
        if (a > 0 and p > 0) or (a < 0 and p < 0) or (a == 0 and p == 0)
    )
    winner_accuracy = correct_winners / n

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "winner_accuracy": float(winner_accuracy),
        "count": n,
    }


def _evaluate_records(
    records: Sequence[GameRecord],
    lr_lambda: float,
    gamma: float,
    validation_start: int,
    validation_end: int,
) -> tuple[list[float], list[float]]:
    """Return actual and predicted differentials for one chronological window."""
    engine = PiRatingEngine(lr_lambda=lr_lambda, gamma=gamma)

    for game in records[:validation_start]:
        engine.update(
            game.home_team,
            game.away_team,
            game.home_score,
            game.away_score,
        )

    actual_diffs: list[float] = []
    predicted_diffs: list[float] = []
    for game in records[validation_start:validation_end]:
        predicted_diffs.append(
            engine.predict_diff(game.home_team, game.away_team)
        )
        actual_diffs.append(game.home_score - game.away_score)
        engine.update(
            game.home_team,
            game.away_team,
            game.home_score,
            game.away_score,
        )

    return actual_diffs, predicted_diffs


def _resolve_window_size(
    total_games: int,
    value: float | int,
    name: str,
) -> int:
    """Resolve a positive window size from an item count or dataset fraction."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer or fraction.")
    if isinstance(value, float):
        if not (0.0 < value <= 1.0):
            raise ValueError(f"{name} as a fraction must be in (0.0, 1.0].")
        size = int(total_games * value)
    elif isinstance(value, int):
        size = value
    else:
        raise TypeError(f"{name} must be a positive integer or fraction.")

    if size <= 0:
        raise ValueError(f"{name} must resolve to at least one game.")
    return size


def _rolling_windows(
    total_games: int,
    initial_train_size: float | int,
    validation_size: float | int,
    step_size: int | None,
) -> list[tuple[int, int]]:
    """Build expanding-training rolling-origin validation windows."""
    initial_size = _resolve_window_size(
        total_games, initial_train_size, "initial_train_size"
    )
    window_size = _resolve_window_size(
        total_games, validation_size, "validation_size"
    )
    if initial_size >= total_games:
        raise ValueError("initial_train_size must leave validation games.")

    step = window_size if step_size is None else step_size
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("step_size must be a positive integer.")

    windows: list[tuple[int, int]] = []
    validation_start = initial_size
    while validation_start < total_games:
        validation_end = min(validation_start + window_size, total_games)
        windows.append((validation_start, validation_end))
        validation_start += step
    return windows


def evaluate_pi_rating(
    games: Iterable[Any],
    lr_lambda: float = 0.06,
    gamma: float = 0.60,
    validation_start: float | int | str = 0.70,
) -> dict[str, Any]:
    """Evaluate PiRatingEngine performance on a chronological validation segment.

    Warm-up games (before validation_start) are replayed to establish baseline
    ratings. For validation games, pre-game predicted differentials are recorded
    BEFORE the engine is updated with the final scores, preventing data leakage.
    """
    records = prepare_games(games)
    split_idx = _resolve_validation_index(
        len(records), validation_start, [r.date for r in records]
    )

    engine = PiRatingEngine(lr_lambda=lr_lambda, gamma=gamma)

    # 1. Warm-up phase: replay games chronologically to build ratings
    for game in records[:split_idx]:
        engine.update(
            game.home_team,
            game.away_team,
            game.home_score,
            game.away_score,
        )

    # 2. Validation phase: predict first, record error, then update
    actual_diffs: list[float] = []
    predicted_diffs: list[float] = []

    for game in records[split_idx:]:
        predicted = engine.predict_diff(game.home_team, game.away_team)
        actual = game.home_score - game.away_score

        predicted_diffs.append(predicted)
        actual_diffs.append(actual)

        engine.update(
            game.home_team,
            game.away_team,
            game.home_score,
            game.away_score,
        )

    metrics = calculate_metrics(actual_diffs, predicted_diffs)

    return {
        "lr_lambda": float(lr_lambda),
        "gamma": float(gamma),
        **metrics,
    }


def evaluate_zero_baseline(
    games: Iterable[Any],
    validation_start: float | int | str = 0.70,
) -> dict[str, Any]:
    """Evaluate a naive baseline that always predicts a zero point differential."""
    records = prepare_games(games)
    split_idx = _resolve_validation_index(
        len(records), validation_start, [r.date for r in records]
    )

    validation_games = records[split_idx:]
    actual_diffs = [g.home_score - g.away_score for g in validation_games]
    predicted_diffs = [0.0] * len(validation_games)

    metrics = calculate_metrics(actual_diffs, predicted_diffs)
    return {
        "baseline": "zero_diff",
        **metrics,
    }


def run_grid_search(
    games: Iterable[Any],
    lambda_values: Sequence[float] = DEFAULT_LAMBDA_VALUES,
    gamma_values: Sequence[float] = DEFAULT_GAMMA_VALUES,
    validation_start: float | int | str = 0.70,
) -> dict[str, Any]:
    """Run a grid search over lr_lambda and gamma combinations.

    Returns all evaluation results alongside the best parameters and baseline
    comparisons (current default parameters and zero-diff baseline).
    """
    records = prepare_games(games)
    split_idx = _resolve_validation_index(
        len(records), validation_start, [r.date for r in records]
    )

    results: list[dict[str, Any]] = []

    for lr in lambda_values:
        for gamma in gamma_values:
            # Create a fresh engine for each combination
            engine = PiRatingEngine(lr_lambda=lr, gamma=gamma)

            for game in records[:split_idx]:
                engine.update(
                    game.home_team,
                    game.away_team,
                    game.home_score,
                    game.away_score,
                )

            actual_diffs: list[float] = []
            predicted_diffs: list[float] = []

            for game in records[split_idx:]:
                pred = engine.predict_diff(game.home_team, game.away_team)
                act = game.home_score - game.away_score
                predicted_diffs.append(pred)
                actual_diffs.append(act)
                engine.update(
                    game.home_team,
                    game.away_team,
                    game.home_score,
                    game.away_score,
                )

            metrics = calculate_metrics(actual_diffs, predicted_diffs)
            results.append({"lr_lambda": float(lr), "gamma": float(gamma), **metrics})

    best = min(results, key=lambda r: (r["mae"], r["rmse"]))

    # Default engine comparison (lr_lambda=0.06, gamma=0.60)
    default_metrics = next(
        (
            r
            for r in results
            if math.isclose(r["lr_lambda"], 0.06) and math.isclose(r["gamma"], 0.60)
        ),
        None,
    )
    if default_metrics is None:
        default_metrics = evaluate_pi_rating(
            records, lr_lambda=0.06, gamma=0.60, validation_start=split_idx
        )

    zero_baseline = evaluate_zero_baseline(records, validation_start=split_idx)

    return {
        "results": results,
        "best_params": best,
        "default_metrics": default_metrics,
        "zero_baseline": zero_baseline,
        "total_games": len(records),
        "validation_games": len(records) - split_idx,
    }


def run_rolling_grid_search(
    games: Iterable[Any],
    lambda_values: Sequence[float] = DEFAULT_LAMBDA_VALUES,
    gamma_values: Sequence[float] = DEFAULT_GAMMA_VALUES,
    initial_train_size: float | int = 0.50,
    validation_size: float | int = 0.10,
    step_size: int | None = None,
) -> dict[str, Any]:
    """Run an expanding-training rolling-origin grid search.

    Each window trains on all games before its validation segment. Validation
    segments are evaluated chronologically and never update an engine before
    their own prediction. Results are aggregated over all windows, while
    ``window_results`` retains per-window metrics for stability analysis.
    """
    records = prepare_games(games)
    if not records:
        raise ValueError("The games dataset contains no completed games.")
    windows = _rolling_windows(
        len(records), initial_train_size, validation_size, step_size
    )

    window_results: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for lr in lambda_values:
        for gamma in gamma_values:
            all_actuals: list[float] = []
            all_predictions: list[float] = []
            for window_index, (start, end) in enumerate(windows):
                actuals, predictions = _evaluate_records(
                    records, lr, gamma, start, end
                )
                all_actuals.extend(actuals)
                all_predictions.extend(predictions)
                window_results.append(
                    {
                        "window": window_index,
                        "train_games": start,
                        "validation_start": start,
                        "validation_end": end,
                        "lr_lambda": float(lr),
                        "gamma": float(gamma),
                        **calculate_metrics(actuals, predictions),
                    }
                )

            results.append(
                {
                    "lr_lambda": float(lr),
                    "gamma": float(gamma),
                    "window_count": len(windows),
                    **calculate_metrics(all_actuals, all_predictions),
                }
            )

    best = min(results, key=lambda r: (r["mae"], r["rmse"]))

    def aggregate_for_parameters(
        lr_lambda: float, gamma: float
    ) -> dict[str, Any]:
        actuals: list[float] = []
        predictions: list[float] = []
        for start, end in windows:
            window_actuals, window_predictions = _evaluate_records(
                records, lr_lambda, gamma, start, end
            )
            actuals.extend(window_actuals)
            predictions.extend(window_predictions)
        return {
            "lr_lambda": float(lr_lambda),
            "gamma": float(gamma),
            **calculate_metrics(actuals, predictions),
        }

    default_metrics = next(
        (
            result
            for result in results
            if math.isclose(result["lr_lambda"], 0.06)
            and math.isclose(result["gamma"], 0.60)
        ),
        None,
    )
    if default_metrics is None:
        default_metrics = aggregate_for_parameters(0.06, 0.60)

    baseline_actuals = [
        game.home_score - game.away_score
        for start, end in windows
        for game in records[start:end]
    ]
    zero_baseline = {
        "baseline": "zero_diff",
        **calculate_metrics(baseline_actuals, [0.0] * len(baseline_actuals)),
    }

    return {
        "results": results,
        "window_results": window_results,
        "best_params": best,
        "default_metrics": default_metrics,
        "zero_baseline": zero_baseline,
        "total_games": len(records),
        "validation_games": sum(end - start for start, end in windows),
        "window_count": len(windows),
        "initial_train_size": windows[0][0],
        "validation_size": windows[0][1] - windows[0][0],
        "step_size": windows[1][0] - windows[0][0] if len(windows) > 1 else (
            step_size or windows[0][1] - windows[0][0]
        ),
    }
