from aqi.statistics import (
    _category_name,
    _category_color,
    compute_distribution,
    compute_city_ranking,
    compute_seasonality,
    compute_diurnal,
    compute_category_distribution,
)
import pandas as pd

def test_category_helpers():
    assert _category_name(30) == 'Good'
    assert _category_name(80) == 'Moderate'
    assert _category_name(130) == 'Unhealthy for Sensitive Groups'
    assert _category_name(180) == 'Unhealthy'
    assert _category_name(250) == 'Very Unhealthy'
    assert _category_name(350) == 'Hazardous'
    assert _category_color(30) == '#00e400'

def test_statistics_on_dummy_data():
    df = pd.DataFrame({
        'city': ['Lahore', 'Karachi', 'Lahore'],
        'aqi': [120, 80, 140],
        'datetime': ['2026-01-01 10:00:00', '2026-01-01 11:00:00', '2026-06-01 12:00:00'],
        'temperature_2m': [25.0, 30.0, 35.0],
        'relative_humidity_2m': [50.0, 60.0, 40.0],
    })
    
    dist = compute_distribution(df, bins=5)
    assert len(dist) == 5
    
    ranking = compute_city_ranking(df)
    assert len(ranking) == 2
    assert ranking[0]['city'] == 'Lahore'
    
    cats = compute_category_distribution(df)
    assert len(cats) == 6
