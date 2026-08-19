# Rigorous Evaluation — per-horizon & walk-forward

## Error by forecast horizon

How RMSE grows as we predict further ahead (vs the persistence baseline).

![error by horizon](images/error_by_horizon.png)

| Horizon (h) | RMSE | MAE | R² | Baseline RMSE |
|---:|---:|---:|---:|---:|
| 1.0 | 6.71 | 3.51 | 0.981 | 4.46 |
| 2.0 | 8.05 | 3.98 | 0.974 | 8.16 |
| 3.0 | 8.04 | 4.75 | 0.973 | 11.38 |
| 6.0 | 10.28 | 6.64 | 0.955 | 18.12 |
| 12.0 | 14.44 | 9.37 | 0.914 | 25.68 |
| 24.0 | 22.05 | 14.38 | 0.801 | 27.09 |
| 36.0 | 27.03 | 17.67 | 0.688 | 35.82 |
| 48.0 | 29.71 | 18.61 | 0.639 | 37.49 |
| 60.0 | 27.82 | 19.15 | 0.677 | 41.02 |
| 72.0 | 29.41 | 19.67 | 0.651 | 37.59 |

## Walk-forward backtest

Rolling-window validation — each fold trains on the past, tests on the next window.

![walk-forward](images/walk_forward_backtest.png)

| Fold | Train rows | Test rows | Test from | RMSE | MAE | R² |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 33,335 | 33,333 | 2023-08-12 | 22.93 | 13.73 | 0.785 |
| 2 | 66,668 | 33,333 | 2024-03-18 | 16.01 | 10.78 | 0.766 |
| 3 | 100,001 | 33,333 | 2024-10-23 | 29.16 | 14.45 | 0.748 |
| 4 | 133,334 | 33,333 | 2025-05-30 | 17.97 | 10.95 | 0.837 |
| 5 | 166,667 | 33,333 | 2026-01-05 | 20.74 | 11.83 | 0.795 |

**Mean backtest RMSE: 21.36** (± 5.09)
