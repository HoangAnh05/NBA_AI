import unittest
import math

from src.predictions.pi_rating_optimization import (
    DEFAULT_LAMBDA_VALUES,
    DEFAULT_GAMMA_VALUES,
    prepare_games,
    calculate_metrics,
    evaluate_pi_rating,
    evaluate_zero_baseline,
    run_grid_search,
)


class TestPiRatingOptimization(unittest.TestCase):
    def setUp(self):
        # A sequence of 10 chronological games
        self.sample_games = [
            {"date": f"2026-01-{i+1:02d}", "home_team": "BOS", "away_team": "NYK", "home_score": 110 + i, "away_score": 100}
            for i in range(10)
        ]

    def test_prepare_games_normalizes_and_sorts_chronologically(self):
        unsorted_games = [
            {"date": "2026-01-05", "home": "BOS", "away": "NYK", "home_score": 105, "away_score": 100},
            {"date": "2026-01-02", "home_team": "LAL", "away_team": "GSW", "home_score": 98, "away_score": 102},
            {"date": "2026-01-08", "game_date": "2026-01-08", "home": "MIA", "away": "CHI", "home_score": None, "away_score": 90},  # unplayed
            {"date": "2026-01-02", "home": "LAL", "away": "GSW", "home_score": 98, "away_score": 102},  # duplicate
        ]

        cleaned = prepare_games(unsorted_games)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0].date, "2026-01-02")
        self.assertEqual(cleaned[0].home_team, "LAL")
        self.assertEqual(cleaned[0].away_team, "GSW")
        self.assertEqual(cleaned[1].date, "2026-01-05")

    def test_calculate_metrics_values(self):
        actuals = [10.0, -5.0, 4.0]
        predicteds = [6.0, -2.0, 8.0]
        # errors: [4.0, -3.0, -4.0]
        # abs_errors: [4.0, 3.0, 4.0] -> sum = 11.0, mae = 11/3
        # squared_errors: [16.0, 9.0, 16.0] -> sum = 41.0, rmse = sqrt(41/3)
        # bias: (4 - 3 - 4)/3 = -1.0
        # signs: actual (10 > 0, pred 6 > 0 -> match), (-5 < 0, pred -2 < 0 -> match), (4 > 0, pred 8 > 0 -> match) -> 3/3 = 1.0

        metrics = calculate_metrics(actuals, predicteds)
        self.assertAlmostEqual(metrics["mae"], 11.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(41.0 / 3.0))
        self.assertAlmostEqual(metrics["bias"], -1.0)
        self.assertAlmostEqual(metrics["winner_accuracy"], 1.0)
        self.assertEqual(metrics["count"], 3)

    def test_calculate_metrics_validations(self):
        with self.assertRaises(ValueError):
            calculate_metrics([], [])
        with self.assertRaises(ValueError):
            calculate_metrics([1.0], [1.0, 2.0])

    def test_evaluate_pi_rating_respects_warmup_and_chronology(self):
        # 10 games, validation_start=0.7 -> 7 warm-up games, 3 validation games
        result = evaluate_pi_rating(self.sample_games, lr_lambda=0.06, gamma=0.60, validation_start=0.7)

        self.assertIn("mae", result)
        self.assertIn("rmse", result)
        self.assertIn("bias", result)
        self.assertIn("winner_accuracy", result)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["lr_lambda"], 0.06)
        self.assertEqual(result["gamma"], 0.60)

    def test_evaluate_pi_rating_date_split(self):
        result = evaluate_pi_rating(self.sample_games, validation_start="2026-01-08")
        # Games from 2026-01-08 to 2026-01-10 are 3 validation games
        self.assertEqual(result["count"], 3)

    def test_evaluate_pi_rating_empty_validation_raises(self):
        with self.assertRaises(ValueError):
            evaluate_pi_rating(self.sample_games, validation_start=1.0)

        with self.assertRaises(ValueError):
            evaluate_pi_rating(self.sample_games, validation_start=10)

    def test_evaluate_zero_baseline(self):
        result = evaluate_zero_baseline(self.sample_games, validation_start=0.7)
        self.assertEqual(result["baseline"], "zero_diff")
        self.assertEqual(result["count"], 3)
        # Validation games are games 7, 8, 9 with diffs 17, 18, 19
        # Baseline predicts 0, so errors are 17, 18, 19
        expected_mae = (17 + 18 + 19) / 3.0
        self.assertAlmostEqual(result["mae"], expected_mae)

    def test_run_grid_search(self):
        lambdas = [0.05, 0.1]
        gammas = [0.5, 0.6]

        search = run_grid_search(
            self.sample_games,
            lambda_values=lambdas,
            gamma_values=gammas,
            validation_start=0.7,
        )

        self.assertEqual(len(search["results"]), 4)
        self.assertIn("best_params", search)
        self.assertIn("zero_baseline", search)
        self.assertIn("default_metrics", search)
        self.assertEqual(search["total_games"], 10)
        self.assertEqual(search["validation_games"], 3)

    def test_prediction_does_not_leak_current_outcome(self):
        # Even if the final score for game 0 in validation changes, the prediction
        # for game 0 is made from prior games only.
        games_a = [
            {"date": "2026-01-01", "home": "BOS", "away": "NYK", "home_score": 100, "away_score": 90},
            {"date": "2026-01-02", "home": "LAL", "away": "GSW", "home_score": 110, "away_score": 100},
        ]
        games_b = [
            {"date": "2026-01-01", "home": "BOS", "away": "NYK", "home_score": 100, "away_score": 90},
            {"date": "2026-01-02", "home": "LAL", "away": "GSW", "home_score": 80, "away_score": 120},
        ]

        # For LAL vs GSW (never played before), pre-game prediction diff must be 0.0 in both cases
        res_a = evaluate_pi_rating(games_a, validation_start=1)
        res_b = evaluate_pi_rating(games_b, validation_start=1)

        # In res_a: actual diff is 10, pred was 0 -> error 10, mae 10
        # In res_b: actual diff is -40, pred was 0 -> error -40, mae 40
        self.assertEqual(res_a["mae"], 10.0)
        self.assertEqual(res_b["mae"], 40.0)

    def test_dataframe_like_input_support(self):
        class DummyDataFrame:
            def __init__(self, records):
                self._records = records

            def to_dict(self, orient="records"):
                return self._records

        df = DummyDataFrame(self.sample_games)
        res = evaluate_pi_rating(df, validation_start=0.7)
        self.assertEqual(res["count"], 3)



if __name__ == "__main__":
    unittest.main()
