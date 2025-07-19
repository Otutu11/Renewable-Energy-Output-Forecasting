import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    """Load and clean energy data"""
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Handle missing values
    df = df.interpolate(method='time').fillna(method='ffill')
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    return df

def create_sequences(data, timesteps):
    """Create time sequences for LSTM"""
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i, -1])  # Target is last column
    return np.array(X), np.array(y)

def prepare_data(df, target_col, test_size=0.2, timesteps=24):
    """Prepare data for modeling"""
    # Train-test split (temporal)
    train_size = int(len(df) * (1 - test_size))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    # Create sequences for LSTM
    X_train_seq, y_train_seq = create_sequences(train_scaled, timesteps)
    X_test_seq, y_test_seq = create_sequences(test_scaled, timesteps)
    
    # Prepare for XGBoost (no sequences)
    X_train = train_scaled[timesteps:]
    y_train = train[target_col].values[timesteps:]
    X_test = test_scaled[timesteps:]
    y_test = test[target_col].values[timesteps:]
    
    return {
        'XGB': (X_train, X_test, y_train, y_test),
        'LSTM': (X_train_seq, X_test_seq, y_train_seq, y_test_seq),
        'scaler': scaler
    }