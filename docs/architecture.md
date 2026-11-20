# NBA AI Architecture

This document describes the architecture of NBA AI and identifies the files
that implement each part of the system.

## 1. System overview

NBA AI is a Python application with four connected concerns:

1. **Data ingestion and persistence** collect NBA data and store it in SQLite.
2. **Feature and state preparation** turn raw game data into pre-game features,
   play-by-play game states, and model-ready datasets.
3. **Prediction engines** generate spread, score, and win-probability forecasts.
4. **Web presentation and monitoring** expose forecasts, game data, and pipeline
   status through a Flask application and dashboard.

The main production flow is:

```text
NBA API / ESPN
       |
       v
database_updater
       |
       v
SQLite database <---- predictions, features, game states
       |
       +--> prediction engines --> saved predictions
       |
       v
Flask web app / API / dashboard
```

The daily coordinator is
[src/pipeline/orchestrator.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/orchestrator.py).
It runs post-game data updates and pre-game prediction generation.

## 2. Application entry points

| Responsibility | File |
|---|---|
| Start the Flask application | [start_app.py](/home/anh/Documents/Proj_nba/NBA_AI/start_app.py) |
| Run the daily post-game/pre-game pipeline | [src/pipeline/orchestrator.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/orchestrator.py) |
| Run database update operations directly | [src/database_updater/database_update_manager.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/database_update_manager.py) |
| Generate predictions directly or from the pipeline | [src/predictions/prediction_manager.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_manager.py) |
| Train or build model artifacts | [scripts/](/home/anh/Documents/Proj_nba/NBA_AI/scripts/) |

`start_app.py` validates command-line options, configures logging, creates the
Flask app, and starts the server. The orchestrator is intended for scheduled
execution, such as cron.

## 3. Configuration and shared infrastructure

### Configuration

- [config.yaml](/home/anh/Documents/Proj_nba/NBA_AI/config.yaml) contains
  project, database, web-app, and predictor settings.
- [.env.example](/home/anh/Documents/Proj_nba/NBA_AI/.env.example) documents
  environment variables.
- [src/config.py](/home/anh/Documents/Proj_nba/NBA_AI/src/config.py) loads YAML,
  substitutes environment variables, resolves paths, and validates the database
  location.

### Database access

- [src/database.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database.py) is the
  shared SQLite connection layer.
- It enables WAL mode and a busy timeout so the web application can read while
  pipeline processes write.
- [data/NBA_AI_starter.sqlite](/home/anh/Documents/Proj_nba/NBA_AI/data/NBA_AI_starter.sqlite)
  is the local starter database.

### Cross-cutting services

- [src/logging_config.py](/home/anh/Documents/Proj_nba/NBA_AI/src/logging_config.py)
  configures application logging.
- [src/utils.py](/home/anh/Documents/Proj_nba/NBA_AI/src/utils.py) contains
  shared date/time, database, validation, timing, and pipeline helpers.
- [src/health_check.py](/home/anh/Documents/Proj_nba/NBA_AI/src/health_check.py)
  provides health/readiness checks.

## 4. Data ingestion and database update layer

The database update layer is responsible for collecting external data and
turning it into internally consistent records. The coordinator and individual
collectors are in
[src/database_updater/](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/).

| Data or operation | File |
|---|---|
| Overall update sequence and stage selection | [database_update_manager.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/database_update_manager.py) |
| Schedule and game metadata | [schedule.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/schedule.py) |
| Players | [players.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/players.py) |
| Player attributes | [player_attributes.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/player_attributes.py) |
| Coaches | [coaches.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/coaches.py) |
| Official injury reports | [nba_official_injuries.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/nba_official_injuries.py) |
| Betting lines and market data | [betting.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/betting.py) |
| Play-by-play retrieval and storage | [pbp.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/pbp.py) |
| Box-score retrieval and storage | [boxscores.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/boxscores.py) |
| Parsed game-state creation | [game_states.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/game_states.py) |
| Historical pre-game state loading | [prior_states.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/prior_states.py) |
| Data quality validation | [validators.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/validators.py) |
| Team cover/matchup data | [covers.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/covers.py) |

The normal update order is schedule, people and context data, play-by-play,
game states, box scores, pre-game features, and optionally predictions.

## 5. Play-by-play processing

The play-by-play subsystem converts raw event streams into structured,
model-friendly observations:

- [src/pbp/event_parser.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pbp/event_parser.py)
  parses individual events.
- [src/pbp/lineup_tracker.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pbp/lineup_tracker.py)
  tracks the players on court.
- [src/pbp/stat_computer.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pbp/stat_computer.py)
  computes derived player and lineup statistics.
- [src/pbp/constants.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pbp/constants.py)
  defines shared event and statistic constants.

The parsed output feeds the database's game-state records and the deep-learning
datasets.

## 6. Feature engineering and prediction orchestration

### Feature preparation

- [src/predictions/features.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/features.py)
  creates and loads pre-game feature sets used by traditional predictors.
- [src/predictions/prediction_utils.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_utils.py)
  provides prediction-specific transformations and utilities.
- [src/predictions/prediction_manager.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_manager.py)
  resolves predictor names, lazily creates predictor instances, validates
  outputs, and saves predictions.

### Predictor abstraction

- [src/predictions/prediction_engines/base_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_engines/base_predictor.py)
  defines `BasePredictor` and `BaseMLPredictor`.
- [src/predictions/prediction_engines/baseline_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_engines/baseline_predictor.py)
  implements a formula-based baseline.
- [src/predictions/prediction_engines/linear_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_engines/linear_predictor.py)
  implements the linear model.
- [src/predictions/prediction_engines/tree_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_engines/tree_predictor.py)
  implements the tree-based model.
- [src/predictions/prediction_engines/mlp_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_engines/mlp_predictor.py)
  implements the PyTorch MLP model.

All predictors expose a common pre-game prediction interface. This lets the
pipeline and web layer select a model by configuration rather than depending on
model-specific implementation details.

### Pi Rating

The chronological Pi Rating implementation is in
[src/predictions/pi_rating.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/pi_rating.py).
Its parameter search and backtesting helpers are in
[src/predictions/pi_rating_optimization.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/pi_rating_optimization.py),
with supporting guidance in
[docs/pi_rating_optimization.md](/home/anh/Documents/Proj_nba/NBA_AI/docs/pi_rating_optimization.md).

## 7. Deep-learning model families

### Phase 3 / Transformer path

The transformer path models sequences of games and play-by-play events:

- [src/transformer/tokenizer.py](/home/anh/Documents/Proj_nba/NBA_AI/src/transformer/tokenizer.py)
  and [src/transformer/gamestates/tokenizer.py](/home/anh/Documents/Proj_nba/NBA_AI/src/transformer/gamestates/tokenizer.py)
  convert structured events into tokens.
- [src/transformer/dataset.py](/home/anh/Documents/Proj_nba/NBA_AI/src/transformer/dataset.py)
  and [src/transformer/gamestates/dataset.py](/home/anh/Documents/Proj_nba/NBA_AI/src/transformer/gamestates/dataset.py)
  define training inputs.
- [src/transformer/models/](/home/anh/Documents/Proj_nba/NBA_AI/src/transformer/models/)
  contains event, roster, temporal-attention, fusion, and prediction-head
  components.
- [src/transformer/models/phase1_model.py](/home/anh/Documents/Proj_nba/NBA_AI/src/transformer/models/phase1_model.py)
  wires the main Phase 1 model together.
- [src/transformer/phase2/](/home/anh/Documents/Proj_nba/NBA_AI/src/transformer/phase2/)
  contains the Phase 2 architecture, datasets, cache builder, pretraining, and
  training code.
- [src/pipeline/phase3_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/phase3_predictor.py)
  adapts the trained Phase 3 model to the production prediction interface.

### Phase 5 hierarchical path

Phase 5 is a hierarchical model that builds from player abilities toward a
game-level prediction:

1. **L1:** player ability and state estimation.
2. **L2:** player form and interaction modeling.
3. **L3:** team representation and effects.
4. **L4:** matchup and game prediction.

The implementation is in
[src/phase5/](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/):

- [model.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/model.py) defines
  shared model components and the player ability encoder.
- [l1_extract.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/l1_extract.py)
  and [l1_store.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/l1_store.py)
  handle L1 vectors.
- [l2_model.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/l2_model.py),
  [l3_model.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/l3_model.py),
  and [l4_model.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/l4_model.py)
  implement successive hierarchy levels.
- [dataset.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/dataset.py),
  [game_dataset.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/game_dataset.py),
  and [cache_builder.py](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/cache_builder.py)
  prepare training and inference data.
- [src/pipeline/phase5_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/phase5_predictor.py)
  exposes Phase 5 predictions to the production pipeline.

## 8. Ensemble and production pipeline

- [src/pipeline/ensemble_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/ensemble_predictor.py)
  combines available component predictions. It averages spreads and combines
  win probabilities in log-odds space.
- [src/pipeline/roster_assembler.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/roster_assembler.py)
  builds roster inputs for model inference.
- [src/pipeline/team_features.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/team_features.py)
  prepares team-level features.
- [src/pipeline/l1_updater.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/l1_updater.py)
  incrementally updates Phase 5 player vectors.
- [src/pipeline/phase3_cache_updater.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/phase3_cache_updater.py)
  refreshes transformer inference caches.
- [src/pipeline/monitoring.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/monitoring.py)
  records pipeline runs, stage results, errors, and metrics.

The orchestrator uses these pieces in two operational modes:

- **Post-game:** collect finalized data, parse/enrich it, and update model state.
- **Pre-game:** refresh injuries and betting lines, find upcoming games, and
  generate predictions.

## 9. Flask web application

The web layer reads prepared data and predictions from SQLite; it does not
perform the daily data collection itself.

- [src/web_app/app.py](/home/anh/Documents/Proj_nba/NBA_AI/src/web_app/app.py)
  creates the Flask app and defines the home, date, and game-data routes.
- [src/games_api/api.py](/home/anh/Documents/Proj_nba/NBA_AI/src/games_api/api.py)
  defines JSON API routes.
- [src/games_api/games.py](/home/anh/Documents/Proj_nba/NBA_AI/src/games_api/games.py)
  queries games and predictions for API and page responses.
- [src/web_app/game_data_processor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/web_app/game_data_processor.py)
  converts database rows into frontend-ready data.
- [src/web_app/dashboard.py](/home/anh/Documents/Proj_nba/NBA_AI/src/web_app/dashboard.py)
  provides dashboard routes and model-performance views.
- [src/web_app/templates/](/home/anh/Documents/Proj_nba/NBA_AI/src/web_app/templates/)
  contains the Jinja HTML templates.
- [src/web_app/static/](/home/anh/Documents/Proj_nba/NBA_AI/src/web_app/static/)
  contains JavaScript, CSS, and image assets.

The browser-facing flow is:

```text
Browser -> Flask route -> games_api/database query
        -> game_data_processor -> JSON or Jinja template
```

## 10. Training, scripts, and model artifacts

- [scripts/](/home/anh/Documents/Proj_nba/NBA_AI/scripts/) contains operational
  scripts for backtesting, data collection, cache generation, and training
  Phase 2, Phase 5, and legacy models.
- [models/](/home/anh/Documents/Proj_nba/NBA_AI/models/) stores trained model
  files, scalers, metadata, and backtest results.
- [TRAINING_COMMANDS.md](/home/anh/Documents/Proj_nba/NBA_AI/TRAINING_COMMANDS.md)
  documents the supported training commands.

Training is intentionally separate from the daily serving path: scripts build
or update artifacts, while predictors load those artifacts during inference.

## 11. Tests and verification

- [tests/](/home/anh/Documents/Proj_nba/NBA_AI/tests/) contains unit,
  integration, pipeline-readiness, predictor, dashboard, and data-processing
  tests.
- [tests/conftest.py](/home/anh/Documents/Proj_nba/NBA_AI/tests/conftest.py)
  provides shared test fixtures and setup.
- [pytest.ini](/home/anh/Documents/Proj_nba/NBA_AI/pytest.ini) defines pytest
  configuration.

Tests are organized around architectural boundaries, including database
updates, predictors, pipeline orchestration, APIs, and frontend integration.

## 12. Responsibility map

| Architectural part | Primary files |
|---|---|
| Runtime configuration | [config.yaml](/home/anh/Documents/Proj_nba/NBA_AI/config.yaml), [src/config.py](/home/anh/Documents/Proj_nba/NBA_AI/src/config.py) |
| Persistence | [src/database.py](/home/anh/Documents/Proj_nba/NBA_AI/src/database.py), [data/](/home/anh/Documents/Proj_nba/NBA_AI/data/) |
| External data ingestion | [src/database_updater/](/home/anh/Documents/Proj_nba/NBA_AI/src/database_updater/) |
| PBP parsing | [src/pbp/](/home/anh/Documents/Proj_nba/NBA_AI/src/pbp/) |
| Feature generation | [src/predictions/features.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/features.py), [src/pipeline/team_features.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/team_features.py) |
| Traditional predictors | [src/predictions/prediction_engines/](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/prediction_engines/) |
| Pi Rating | [src/predictions/pi_rating.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/pi_rating.py), [src/predictions/pi_rating_optimization.py](/home/anh/Documents/Proj_nba/NBA_AI/src/predictions/pi_rating_optimization.py) |
| Transformer models | [src/transformer/](/home/anh/Documents/Proj_nba/NBA_AI/src/transformer/) |
| Hierarchical Phase 5 models | [src/phase5/](/home/anh/Documents/Proj_nba/NBA_AI/src/phase5/) |
| Prediction composition | [src/pipeline/ensemble_predictor.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/ensemble_predictor.py) |
| Daily orchestration | [src/pipeline/orchestrator.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/orchestrator.py) |
| Web/API serving | [src/web_app/](/home/anh/Documents/Proj_nba/NBA_AI/src/web_app/), [src/games_api/](/home/anh/Documents/Proj_nba/NBA_AI/src/games_api/) |
| Monitoring | [src/pipeline/monitoring.py](/home/anh/Documents/Proj_nba/NBA_AI/src/pipeline/monitoring.py), [src/health_check.py](/home/anh/Documents/Proj_nba/NBA_AI/src/health_check.py) |
| Training and operations | [scripts/](/home/anh/Documents/Proj_nba/NBA_AI/scripts/), [models/](/home/anh/Documents/Proj_nba/NBA_AI/models/) |
| Verification | [tests/](/home/anh/Documents/Proj_nba/NBA_AI/tests/) |
