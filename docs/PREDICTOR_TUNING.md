# Predictor Training and Tuning

## Quick Map

| Predictor | Train or tune here | What it controls |
|---|---|---|
| Baseline | `src/predictions/prediction_engines/baseline_predictor.py` | PPG and opponent-PPG formula |
| Linear | `scripts/train_legacy_models.py` | Ridge model and alpha values |
| Tree | `scripts/train_legacy_models.py` | XGBoost search ranges and trials |
| MLP | `scripts/train_legacy_models.py` | Network size, dropout, learning rate, epochs |
| Phase 5 | `scripts/train_phase5_b.py`, `scripts/train_phase5_c.py`, `src/phase5/l4_config.py` | L2/L3/L4 model and loss settings |
| Phase 3 | `scripts/` Phase 3 training scripts and the model config | Transformer architecture and training settings |
| Ensemble | `src/pipeline/ensemble_predictor.py` | Component models and combination weights |
| Win probability | `src/predictions/prediction_utils.py` | Pre-game probability calibration |

## Main Training Commands

From the project root:

```bash
# Train Linear and Tree
python scripts/train_legacy_models.py --models ridge xgboost

# Train MLP
python scripts/train_legacy_models.py --models mlp

# Train all legacy models
python scripts/train_legacy_models.py
```

The legacy training script validates using a later season. Do not evaluate on the same games used for training.

Phase 5 training is split into stages:

```bash
python scripts/train_phase5_b.py
python scripts/train_phase5_c.py
```

Use `--test` with the Phase 5 scripts for a short training smoke test.

## Important Files

- `config.yaml`: selects the default predictor and model checkpoint paths.
- `src/predictions/prediction_manager.py`: registers predictors and loads them.
- `src/predictions/features.py`: shared legacy-model feature loading.
- `src/pipeline/ensemble_predictor.py`: ensemble averaging and weights.
- `scripts/generate_readme_chart.py`: compares real completed-game performance.

Do not edit files in `models/` directly. They are trained outputs. After retraining, update the paths in `config.yaml` if the filenames changed.

## Evaluation Workflow

1. Train one predictor.
2. Save its new model and scaler files.
3. Update `config.yaml` if needed.
4. Generate predictions for completed games without using future data.
5. Run the comparison:

```bash
python scripts/generate_readme_chart.py
```

Compare ATS percentage, ROI, spread MAE, and number of games. Keep a change only if it improves the same held-out date range, not just the training results.

## Safe Tuning Order

1. Tune Linear, Tree, and MLP in `train_legacy_models.py`.
2. Compare them with Phase 5 on the same games.
3. Tune ensemble weights in `ensemble_predictor.py`.
4. Tune win-probability constants in `prediction_utils.py` separately from score models.
