# Legacy Model Training Commands

This guide documents the commands for training the three legacy NBA prediction
models implemented in [`scripts/train_legacy_models.py`](scripts/train_legacy_models.py):

- **Ridge** (saved as the Linear predictor)
- **XGBoost** (saved as the Tree predictor)
- **MLP** (PyTorch neural network predictor)

Run all commands from the repository root.

## Prerequisites

Create and activate the project virtual environment, then install the project
dependencies:

```bash
python -m venv venv
source venv/bin/activate                 # Linux/macOS
# venv\Scripts\activate                  # Windows
pip install -r requirements.txt
pip install optuna                       # required by XGBoost tuning
```

The database configured in `.env` must contain the feature data and completed
game scores. The script uses the configured database automatically.

## Check the training data

Run a dry run before training. This loads the data, reports the training and
validation sizes, and does not write model files:

```bash
python scripts/train_legacy_models.py --dry-run
```

## Train models

### Train all three models

```bash
python scripts/train_legacy_models.py
```

This trains Ridge, XGBoost, and MLP using the script defaults:

- Validation season: `2025-2026`
- Cutoff date: `2026-03-31`
- XGBoost Optuna trials: `50`

### Train only Ridge

```bash
python scripts/train_legacy_models.py --models ridge
```

### Train only XGBoost

```bash
python scripts/train_legacy_models.py --models xgboost
```

### Train only MLP

```bash
python scripts/train_legacy_models.py --models mlp
```

### Train a selected group

```bash
python scripts/train_legacy_models.py --models ridge xgboost
python scripts/train_legacy_models.py --models ridge mlp
python scripts/train_legacy_models.py --models xgboost mlp
```

## Change the validation split

Use `--val-season` to hold out a different season and `--cutoff-date` to
exclude games on or after a specific date:

```bash
python scripts/train_legacy_models.py \
  --models ridge xgboost mlp \
  --val-season 2024-2025 \
  --cutoff-date 2025-04-01
```

The training data contains all other available seasons, while the selected
validation season is kept separate.

## Change XGBoost tuning time

The XGBoost model uses Optuna. Increase the number of trials for a broader
search, or reduce it for a faster experiment:

```bash
# Faster experiment
python scripts/train_legacy_models.py --models xgboost --xgb-trials 10

# More extensive tuning
python scripts/train_legacy_models.py --models xgboost --xgb-trials 100
```

`--xgb-trials` affects only XGBoost. Ridge and MLP are not controlled by this
option.

## Output files

Training overwrites the model files used by the application:

| Model argument | Predictor name | Output directory | Main files |
|---|---|---|---|
| `ridge` | Linear | `models/linear/` | `model.joblib`, `scaler.json`, `metadata.json` |
| `xgboost` | Tree | `models/tree/` | `home_model.json`, `away_model.json`, `scaler.json`, `metadata.json` |
| `mlp` | MLP | `models/mlp/` | `model.pth`, `metadata.json` |

The script prints validation metrics and a comparison table after training.
Review those results before using the newly generated models in production.

## Recommended workflow

```bash
# 1. Confirm that the database and feature data are available
python scripts/train_legacy_models.py --dry-run

# 2. Train the models
python scripts/train_legacy_models.py \
  --models ridge xgboost mlp \
  --val-season 2025-2026 \
  --cutoff-date 2026-03-31 \
  --xgb-trials 50

# 3. Optionally regenerate the predictor performance chart
python scripts/generate_readme_chart.py
```

Do not edit files under `models/` manually. They are generated training
artifacts, and each training run replaces the selected model's files.
