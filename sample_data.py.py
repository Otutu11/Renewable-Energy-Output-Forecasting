import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_energy_data(num_days=365, start_date="2023-01-01"):
    """Generate synthetic renewable energy dataset"""
    np.random.seed(42)
    timestamps = pd.date_range(start=start_date, periods=num_days*24, freq='H')
    n = len(timestamps)
    
    # Base patterns
    daily = np.sin(2 * np.pi * np.arange(n) / 24)
    weekly = np.sin(2 * np.pi * np.arange(n) / (24*7))
    seasonal = np.sin(2 * np.pi * (np.arange(n) % (24*365)) / (24*365))
    
    # Synthetic features
    temperature = 15 + 10 * seasonal + 5 * daily + np.random.normal(0, 3, n)
    wind_speed = 8 + 4 * weekly + 3 * daily + np.random.normal(0, 2, n)
    radiation = np.maximum(0, 300 * np.sin(np.pi * (np.arange(n) % 24)/24) * 
                          (1 - 0.3 * seasonal) + np.random.normal(0, 30, n))
    
    # Energy output calculation
    energy_output = (
        0.6 * radiation +
        0.4 * wind_speed**1.5 +
        0.1 * temperature * wind_speed -
        0.05 * temperature**2 +
        np.random.normal(0, 20, n)
    )
    energy_output = np.maximum(10, energy_output)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'wind_speed': wind_speed,
        'radiation': radiation,
        'energy_output': energy_output
    })
    
    # Add 5% missing values
    mask = np.random.choice([True, False], size=df.shape, p=[0.05, 0.95])
    df = df.mask(mask)
    
    return df

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    energy_df = generate_energy_data()
    energy_df.to_csv('data/raw/energy_data.csv', index=False)
    print("Generated sample data with", len(energy_df), "records")