import sqlite3
from src.predictions.pi_rating_optimization import run_grid_search

# 1. Load completed 2025-2026 games from the database
conn = sqlite3.connect("data/NBA_AI_starter.sqlite")
cursor = conn.cursor()
cursor.execute("""
    SELECT 
        g.date_time_utc AS date,
        g.home_team,
        g.away_team,
        gs.home_score,
        gs.away_score
    FROM Games g
    JOIN GameStates gs ON g.game_id = gs.game_id AND gs.is_final_state = 1
    WHERE g.season = '2025-2026'
    ORDER BY g.date_time_utc ASC
""")
cols = [desc[0] for desc in cursor.description]
games = [dict(zip(cols, row)) for row in cursor.fetchall()]

# 2. Run the grid search (first 70% warm-up, final 30% validation)
result = run_grid_search(games, validation_start=0.70)

# 3. Print the results
best = result["best_params"]
defaults = result["default_metrics"]
zero_baseline = result["zero_baseline"]

print("=" * 50)
print(f"Optimal Parameters: lr_lambda={best['lr_lambda']}, gamma={best['gamma']}")
print(f"  Best MAE:          {best['mae']:.2f} pts")
print(f"  Default (0.06, 0.6) MAE: {defaults['mae']:.2f} pts")
print(f"  Zero-diff Baseline MAE:  {zero_baseline['mae']:.2f} pts")
print(f"  Winner Accuracy:   {best['winner_accuracy'] * 100:.1f}%")
print("=" * 50)