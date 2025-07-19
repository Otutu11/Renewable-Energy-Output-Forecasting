DATA_PATH = 'data/raw/energy_data.csv'
TARGET = 'energy_output'
FEATURES = [
    'temperature', 'wind_speed', 'radiation', 
    'hour', 'day_of_week', 'month', 'season',
    'prev_hour', 'prev_day', 'prev_week',
    'temp_squared', 'wind_cubed', 'rad_times_temp'
]
TIME_FEATURES = ['hour', 'day_of_week', 'month', 'season']
LAGS = [1, 24, 168]  # 1h, 24h, 1 week

# Model Params
TIMESTEPS = 72  # 3-day lookback for LSTM
TEST_SPLIT = 0.2
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.9
}
LSTM_PARAMS = {
    'units': 128,
    'dropout': 0.2,
    'recurrent_dropout': 0.2,
    'epochs': 100,
    'batch_size': 64
}