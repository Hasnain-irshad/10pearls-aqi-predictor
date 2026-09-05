import calendar
import json
import numpy as np
import pandas as pd
from functools import lru_cache

from aqi.data.store import read_features
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

def _category_name(aqi: float) -> str:
    if pd.isna(aqi): return 'Unknown'
    if aqi <= 50: return 'Good'
    if aqi <= 100: return 'Moderate'
    if aqi <= 150: return 'Unhealthy for Sensitive Groups'
    if aqi <= 200: return 'Unhealthy'
    if aqi <= 300: return 'Very Unhealthy'
    return 'Hazardous'

def _category_color(aqi: float) -> str:
    if pd.isna(aqi): return '#999999'
    if aqi <= 50: return '#00e400'
    if aqi <= 100: return '#ffde33'
    if aqi <= 150: return '#ff9933'
    if aqi <= 200: return '#ff5050'
    if aqi <= 300: return '#b25aff'
    return '#c81d3f'

def compute_distribution(df: pd.DataFrame, bins: int = 50) -> list[dict]:
    counts, edges = np.histogram(df['aqi'].dropna(), bins=bins)
    result = []
    for i in range(len(counts)):
        result.append({
            'min': float(edges[i]),
            'max': float(edges[i+1]),
            'count': int(counts[i])
        })
    return result

def compute_correlation_matrix(df: pd.DataFrame) -> dict:
    features = ['aqi', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation', 'surface_pressure', 'cloud_cover', 'wind_speed_10m']
    available_features = [f for f in features if f in df.columns]
    
    corr = df[available_features].corr()
    # Replace NaNs with None for JSON serialization
    matrix = corr.replace({np.nan: None}).values.tolist()
    
    return {
        'features': available_features,
        'matrix': matrix
    }

def compute_city_ranking(df: pd.DataFrame) -> list[dict]:
    city_means = df.groupby('city')['aqi'].mean().sort_values(ascending=False)
    result = []
    for city, mean_aqi in city_means.items():
        if pd.isna(mean_aqi):
            continue
        result.append({
            'city': str(city),
            'mean_aqi': float(mean_aqi),
            'category': _category_name(mean_aqi)
        })
    return result

def compute_seasonality(df: pd.DataFrame) -> list[dict]:
    # Ensure datetime is parsed
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    monthly_means = df.groupby(df['datetime'].dt.month)['aqi'].mean()
    result = []
    for month in range(1, 13):
        mean_aqi = monthly_means.get(month, None)
        if mean_aqi is not None and not pd.isna(mean_aqi):
            result.append({
                'month': int(month),
                'month_name': calendar.month_abbr[month],
                'mean_aqi': float(mean_aqi)
            })
    return result

def compute_diurnal(df: pd.DataFrame) -> list[dict]:
    # Ensure datetime is parsed
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
    hourly_means = df.groupby(df['datetime'].dt.hour)['aqi'].mean()
    result = []
    for hour in range(24):
        mean_aqi = hourly_means.get(hour, None)
        if mean_aqi is not None and not pd.isna(mean_aqi):
            result.append({
                'hour': int(hour),
                'mean_aqi': float(mean_aqi)
            })
    return result

def compute_category_distribution(df: pd.DataFrame) -> list[dict]:
    aqi_dropna = df['aqi'].dropna()
    total = len(aqi_dropna)
    if total == 0:
        return []
        
    categories = [
        {'name': 'Good', 'color': '#00e400', 'min': -np.inf, 'max': 50},
        {'name': 'Moderate', 'color': '#ffde33', 'min': 50, 'max': 100},
        {'name': 'Unhealthy for Sensitive Groups', 'color': '#ff9933', 'min': 100, 'max': 150},
        {'name': 'Unhealthy', 'color': '#ff5050', 'min': 150, 'max': 200},
        {'name': 'Very Unhealthy', 'color': '#b25aff', 'min': 200, 'max': 300},
        {'name': 'Hazardous', 'color': '#c81d3f', 'min': 300, 'max': np.inf}
    ]
    
    result = []
    for cat in categories:
        if cat['name'] == 'Good':
            count = int((aqi_dropna <= 50).sum())
        else:
            count = int(((aqi_dropna > cat['min']) & (aqi_dropna <= cat['max'])).sum())
            
        result.append({
            'category': cat['name'],
            'color': cat['color'],
            'count': count,
            'percentage': float(count / total * 100) if total > 0 else 0.0
        })
    return result

@lru_cache(maxsize=1)
def compute_all() -> dict:
    logger.info("Computing all statistics from features data...")
    df = read_features()
    
    # Pre-parse datetime once if needed for seasonality and diurnal functions
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
        
    return {
        'distribution': compute_distribution(df),
        'correlation': compute_correlation_matrix(df),
        'city_ranking': compute_city_ranking(df),
        'seasonality': compute_seasonality(df),
        'diurnal': compute_diurnal(df),
        'category_distribution': compute_category_distribution(df)
    }

if __name__ == '__main__':
    print("Computing stats...")
    stats = compute_all()
    print(json.dumps(stats, indent=2))
