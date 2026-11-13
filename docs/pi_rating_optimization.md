# Optimizing the NBA Pi Rating Parameters

## Objective

Optimize the two parameters in the chronological `PiRatingEngine`:

-   `lr_lambda` (`lambda_`): controls how strongly ratings react to each
    game's prediction error.
-   `gamma`: controls how much of the update transfers to the team's
    opposite home/away rating.

The engine predicts the home-minus-away point differential before a
game, then updates ratings only after the final score is known.

The goal is to find parameter values that improve predictions of
**future games**, not values that best explain games already used to
update the ratings.

------------------------------------------------------------------------

## 1. Understand the update rule

The engine calculates:

\[ error = actual_diff - expected_diff \]

\[ `\psi `{=tex}= 3`\log`{=tex}\_{10}(1 + \|error\|) \]

\[ adjustment =
`\lambda `{=tex}`\cdot `{=tex}`\psi `{=tex}`\cdot `{=tex}sign(error) \]

The adjustment is applied as follows:

-   Home team's home rating: `+= adjustment`
-   Away team's away rating: `-= adjustment`
-   Home team's away rating: `+= adjustment * gamma`
-   Away team's home rating: `-= adjustment * gamma`

A small `lr_lambda` makes ratings change slowly; a larger value makes
them react more strongly. A small `gamma` limits cross-transfer between
home and away ratings; a larger value increases it.

The current defaults (`lr_lambda=0.06`, `gamma=0.60`) should be included
as a baseline experiment.

------------------------------------------------------------------------

## 2. Prepare the data

The games dataset should contain at least:

-   `date`
-   `home_team`
-   `away_team`
-   `home_score`
-   `away_score`

Before running experiments:

1.  Remove games without final scores.
2.  Ensure team names are consistent across both seasons.
3.  Sort all games by date and, if necessary, tip-off time.
4.  Ensure each game appears only once.
5.  Never use a game's final score to calculate that same game's
    pre-game prediction.

The engine must process games in chronological order.

------------------------------------------------------------------------

## 3. Recommended chronological evaluation

Use the complete 2025--2026 season to select parameters and the ongoing
2026--2027 season as a later, out-of-sample check.

A practical first experiment:

1.  Sort the 2025--2026 games chronologically.
2.  Use the first 70% as rating warm-up/history. Replay these games to
    build ratings.
3.  Evaluate candidate parameter combinations on the final 30% of the
    2025--2026 season. For every game, predict first, record the error,
    then update the ratings.
4.  Select parameters using validation performance.
5.  Freeze the selected parameters.
6.  Evaluate them on the completed games from 2026--2027, without tuning
    against that season.

This is a simple chronological holdout. For a more reliable estimate,
repeat the experiment with several rolling-origin validation windows
(each window trains on earlier games and validates on later games).

### Important: season transition

The current engine initializes unseen teams at zero and has no
season-transition mechanism. Decide explicitly how to handle the
transition between seasons.

Possible approaches:

-   **Reset:** start every team at zero at the beginning of the new
    season.
-   **Carry over:** retain the previous season's ratings.
-   **Regress toward the mean:** shrink previous ratings toward zero or
    a chosen league-average rating.

If testing season-to-season carryover or regression, treat its strength
as another parameter to evaluate using chronological validation. Do not
choose it based on the 2026--2027 test results.

------------------------------------------------------------------------

## 4. Start with a grid search

Because there are only two parameters, begin with a grid search. It is
easy to inspect and visualize the entire parameter space.

Example search values:

``` python
lambda_values = [
    0.01, 0.025, 0.05, 0.1,
    0.2, 0.4, 0.8, 1.2, 2.0
]

gamma_values = [
    0.0, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]
```

This produces 99 combinations. These are exploratory ranges, not
guaranteed optimal bounds. Expand or refine them if the best result lies
at an edge.

For each combination:

1.  Create a fresh `PiRatingEngine`.
2.  Replay the warm-up games chronologically.
3.  For each validation game, call `predict_diff()` before `update()`.
4.  Record the prediction error.
5.  Call `update()` after recording the prediction.
6.  Calculate validation metrics.

Do not reuse the same engine state between parameter combinations.

------------------------------------------------------------------------

## 5. Metrics to compare

### Mean Absolute Error (MAE)

\[ MAE = `\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N} \|ActualDiff_i -
PredictedDiff_i\| \]

MAE is a good primary metric because it measures the average absolute
error in predicted point differential. Lower is better.

### Root Mean Squared Error (RMSE)

\[ RMSE = `\sqrt{\frac{1}{N}\sum_{i=1}^{N}
(ActualDiff_i - PredictedDiff_i)^2}`{=tex} \]

RMSE penalizes large misses more heavily. Lower is better.

Also consider tracking:

-   Winner accuracy, based on the sign of the predicted differential.
-   Bias, the mean signed prediction error.
-   Performance by month or season segment, to see whether results are
    stable.

If the Pi Rating is intended to predict point differential, use MAE or
RMSE as the primary optimization objective. If it is intended to predict
win probabilities, add a probability-mapping step and evaluate log loss
or Brier score.

------------------------------------------------------------------------

## 6. Backtesting implementation

Assume `games` is a Pandas DataFrame with the columns listed above. This
function evaluates one parameter combination on a chronological
validation segment. The warm-up games build ratings but are not included
in the metrics.

``` python
import numpy as np
import pandas as pd

def evaluate_pi_rating(games, lr_lambda, gamma, validation_start):
    engine = PiRatingEngine(
        lr_lambda=lr_lambda,
        gamma=gamma
    )

    errors = []

    games = games.sort_values("date").reset_index(drop=True)

    for i, game in games.iterrows():
        # Predict BEFORE updating ratings.
        predicted = engine.predict_diff(
            game.home_team,
            game.away_team
        )

        actual = game.home_score - game.away_score

        # Evaluate only the validation period.
        if i >= validation_start:
            errors.append(actual - predicted)

        # Update only after the actual result is known.
        engine.update(
            game.home_team,
            game.away_team,
            game.home_score,
            game.away_score
        )

    if not errors:
        raise ValueError("The validation period contains no games.")

    errors = np.asarray(errors)

    return {
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "bias": np.mean(errors),
    }
```

Run the grid search on the 2025--2026 season:

``` python
results = []

season_games = season_games.sort_values("date").reset_index(drop=True)
validation_start = int(len(season_games) * 0.70)

for lr in lambda_values:
    for gamma in gamma_values:
        metrics = evaluate_pi_rating(
            season_games,
            lr_lambda=lr,
            gamma=gamma,
            validation_start=validation_start
        )

        results.append({
            "lr_lambda": lr,
            "gamma": gamma,
            **metrics
        })

results_df = pd.DataFrame(results)

best = results_df.loc[results_df["mae"].idxmin()]

print("Best parameters:")
print(best)
```

Here, `season_games` must contain only the 2025--2026 games. The first
70% are replayed to build ratings; the final 30% are scored as
validation games. The prediction for each validation game is still made
before its result is used to update the engine.

------------------------------------------------------------------------

## 7. Visualize the results

Use three complementary visualizations.

### A. Heatmap --- parameter performance

-   X-axis: `gamma`
-   Y-axis: `lr_lambda`
-   Cell color: validation MAE

This shows which parameter combinations perform well. Lower MAE is
better. A broad region of good performance is generally more reassuring
than one isolated best cell.

Example:

``` python
import matplotlib.pyplot as plt
import seaborn as sns

heatmap_data = results_df.pivot(
    index="lr_lambda",
    columns="gamma",
    values="mae"
)

sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis_r")
plt.title("Pi Rating Validation MAE")
plt.xlabel("gamma")
plt.ylabel("lr_lambda")
plt.tight_layout()
plt.show()
```

### B. Scatter plot --- parameter combinations

-   X-axis: `lr_lambda`
-   Y-axis: `gamma`
-   Color: validation MAE

This helps identify clusters of good parameter combinations and whether
performance changes smoothly across the search space.

``` python
plt.scatter(
    results_df["lr_lambda"],
    results_df["gamma"],
    c=results_df["mae"]
)
plt.colorbar(label="Validation MAE")
plt.xlabel("lr_lambda")
plt.ylabel("gamma")
plt.title("Pi Rating Parameter Search")
plt.tight_layout()
plt.show()
```

### C. Baseline comparison

Compare:

1.  The current defaults (`0.06`, `0.60`).
2.  The best grid-search parameters.
3.  A no-update baseline that always predicts a zero point differential.

The optimized engine should improve on the baseline on the same
validation games. Also compare winner accuracy if predicting winners is
an important use case.

------------------------------------------------------------------------

## 8. After the grid search: Optuna

Once the heatmap reveals a promising region, use Optuna for a finer
search. With only two parameters, Optuna is optional; the grid search is
a useful first step because it makes the full search space visible.

For a fair comparison, use the same chronological validation procedure
and metric. Do not tune on the 2026--2027 test season.

------------------------------------------------------------------------

## 9. Recommended final pipeline

``` text
Historical NBA games
        |
        v
Data cleaning and chronological sorting
        |
        v
2025–2026 season
        |
        +--> First 70%: rating warm-up
        |
        +--> Final 30%: parameter validation
        |
        v
Grid search over lr_lambda and gamma
        |
        v
Compare MAE / RMSE / winner accuracy
        |
        v
Heatmap + scatter plot + baseline comparison
        |
        v
Select and freeze parameters
        |
        v
2026–2027 season: out-of-sample evaluation
        |
        v
Use Pi Rating features in the NBA prediction model
```

## 10. Important considerations

-   Always predict before updating. Updating first would leak the game's
    result into its own prediction.
-   Reset the engine for every parameter combination.
-   Keep validation games chronological.
-   Do not use the current season both to select parameters and to claim
    independent test performance.
-   Compare against a simple baseline.
-   Check whether good performance occurs across a broad parameter
    region rather than at one isolated point.
-   Once the parameters are selected, replay all available historical
    games chronologically to produce up-to-date ratings for upcoming
    games. For each new game, read ratings before the game and update
    only after its final score is known.

**Recommended first step:** implement the grid search and heatmap using
the complete 2025--2026 season. Confirm that the optimized parameters
improve validation MAE over the current defaults and the zero-rating
baseline before moving on to Optuna or integrating the optimized ratings
into XGBoost.
