import pandas as pd
import numpy as np

def add_time_features(df):
    """Add time-based features"""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['season'] = (df.index.month % 12 + 3) // 3  # 1=Winter, 2=Spring, etc.
    return df

def add_lag_features(df, target_col, lags=[1, 24, 168]):
    """Add lag features from target variable"""
    df = df.copy()
    for lag in lags:
        df[f'prev_{lag}h'] = df[target_col].shift(lag)
    return df.dropna()

def add_interaction_features(df):
    """Create interaction and polynomial features"""
    df = df.copy()
    if 'temperature' in df:
        df['temp_squared'] = df['temperature'] ** 2
    if 'wind_speed' in df:
        df['wind_cubed'] = df['wind_speed'] ** 3
    if 'radiation' in df and 'temperature' in df:
        df['rad_times_temp'] = df['radiation'] * df['temperature']
    return df

def add_rolling_features(df, target_col, windows=[24, 168]):
    """Add rolling statistics"""
    df = df.copy()
    for window in windows:
        df[f'rolling_avg_{window}h'] = df[target_col].rolling(window).mean()
        df[f'rolling_std_{window}h'] = df[target_col].rolling(window).std()
    return df.dropna()

def engineer_features(df, target_col):
    """Apply all feature engineering steps"""
    df = add_time_features(df)
    df = add_interaction_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    return df