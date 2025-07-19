import os
import joblib
import numpy as np
import pandas as pd
from config import *
from src.data_preprocessing import load_and_clean_data, prepare_data
from src.feature_engineering import engineer_features
from src.train import train_xgboost, train_lstm, evaluate_model
from src.visualize import *

def main():
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Generate sample data if needed
    if not os.path.exists(DATA_PATH):
        print("Generating sample data...")
        from generate_sample_data import generate_energy_data
        df = generate_energy_data()
        df.to_csv(DATA_PATH, index=False)
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    
    # Feature engineering
    print("Engineering features...")
    df = engineer_features(df, TARGET)
    df.to_csv('data/processed/processed_data.csv')
    
    # Prepare data
    print("Preparing data for modeling...")
    data_dict = prepare_data(df, TARGET, TEST_SPLIT, TIMESTEPS)
    
    # Split data
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = data_dict['XGB']
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = data_dict['LSTM']
    scaler = data_dict['scaler']
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = train_xgboost(
        X_train_xgb, y_train_xgb, 
        X_test_xgb, y_test_xgb,
        XGB_PARAMS
    )
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    
    # Train LSTM
    print("\nTraining LSTM model...")
    lstm_model, history = train_lstm(
        X_train_lstm, y_train_lstm,
        X_test_lstm, y_test_lstm,
        LSTM_PARAMS
    )
    lstm_model.save('models/lstm_model.h5')
    
    # Evaluate models
    print("\nEvaluating models...")
    xgb_metrics, xgb_pred = evaluate_model(
        xgb_model, X_test_xgb, y_test_xgb, 'XGBoost'
    )
    lstm_metrics, lstm_pred = evaluate_model(
        lstm_model, X_test_lstm, y_test_lstm, 'LSTM'
    )
    
    # Inverse scaling for actual values
    dummy = np.zeros((len(y_test_xgb), len(FEATURES) + 1))
    dummy[:, -1] = y_test_xgb
    y_actual = scaler.inverse_transform(dummy)[:, -1]
    
    # Inverse scaling for predictions
    dummy[:, -1] = xgb_pred
    xgb_pred = scaler.inverse_transform(dummy)[:, -1]
    
    dummy[:, -1] = lstm_pred
    lstm_pred = scaler.inverse_transform(dummy)[:, -1]
    
    # Visualization
    plot_actual_vs_predicted(
        y_actual, xgb_pred, 
        'XGBoost: Actual vs Predicted'
    )
    plot_actual_vs_predicted(
        y_actual, lstm_pred, 
        'LSTM: Actual vs Predicted'
    )
    plot_error_distribution(
        y_actual - xgb_pred, 
        'XGBoost'
    )
    plot_error_distribution(
        y_actual - lstm_pred, 
        'LSTM'
    )
    plot_feature_importance(
        xgb_model, FEATURES, 
        'XGBoost Feature Importance'
    )
    plot_training_history(history)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()