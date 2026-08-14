# EDA — Key Findings

Dataset: **696,960 hourly rows**, 22 cities, 2023-01-01 → 2026-08-12.

## Highlights
- **Most polluted city:** Faisalabad (mean AQI 156.9).
- **Cleanest city:** Gilgit (mean AQI 75.7).
- **Worst month:** Jan — clear winter-smog seasonality.
- **Strongest weather driver:** surface_pressure (r=0.32).

## Figures
- `eda_aqi_distribution.png` — overall AQI distribution.
- `eda_city_ranking.png` — mean AQI by city.
- `eda_seasonality.png` — monthly seasonality (winter peak).
- `eda_diurnal.png` — daily (hourly) pattern.
- `eda_weather_correlation.png` — weather ↔ AQI correlations.

## City ranking (mean AQI)

| City | Mean AQI |
|------|---------:|
| Faisalabad | 156.9 |
| Lahore | 151.5 |
| Sargodha | 144.6 |
| Multan | 143.7 |
| Gujranwala | 142.7 |
| Sialkot | 139.5 |
| Larkana | 125.3 |
| Bahawalpur | 121.7 |
| Sukkur | 116.2 |
| Mardan | 115.2 |
| Rawalpindi | 110.7 |
| Islamabad | 110.7 |
| Peshawar | 108.8 |
| Turbat | 103.7 |
| Abbottabad | 95.7 |
| Karachi | 90.1 |
| Hyderabad | 87.5 |
| Gwadar | 82.8 |
| Mingora | 82.5 |
| Muzaffarabad | 78.3 |
| Quetta | 76.5 |
| Gilgit | 75.7 |
