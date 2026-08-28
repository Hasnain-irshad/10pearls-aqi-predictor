# Model Evaluation (walk-forward validation)

- Train rows: 160,002 | Validation rows: 39,998
- **Best model: XGBoost**

| Model | RMSE | MAE | R² |
|-------|-----:|----:|---:|
| XGBoost | 19.69 | 12.80 | 0.850 |
| RandomForest | 20.38 | 13.56 | 0.839 |
| Ridge | 22.85 | 16.01 | 0.797 |
| Persistence (baseline) | 25.27 | 15.71 | 0.752 |
