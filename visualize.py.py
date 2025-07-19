import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_actual_vs_predicted(actual, predicted, title):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predicted, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Energy Output')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{title.replace(" ", "_")}.png')
    plt.close()

def plot_error_distribution(errors, title):
    """Plot error distribution"""
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'Error Distribution - {title}')
    plt.xlabel('Prediction Error')
    plt.savefig(f'results/error_dist_{title}.png')
    plt.close()

def plot_feature_importance(model, feature_names, title):
    """Plot feature importance for XGBoost"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        plt.close()

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('results/training_history.png')
    plt.close()