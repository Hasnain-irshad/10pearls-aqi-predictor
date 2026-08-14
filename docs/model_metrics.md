# Model Evaluation (walk-forward validation)

- Train rows: 160,000 | Validation rows: 40,000
- **Best model: XGBoost**

| Model | RMSE | MAE | R² |
|-------|-----:|----:|---:|
| XGBoost | 20.60 | 11.80 | 0.824 |
| RandomForest | 22.04 | 12.90 | 0.799 |
| Ridge | 25.19 | 15.57 | 0.738 |
| Persistence (baseline) | 27.84 | 15.18 | 0.679 |
