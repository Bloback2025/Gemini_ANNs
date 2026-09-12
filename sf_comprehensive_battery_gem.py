#!/usr/bin/env python3
r"""
Canonical Comprehensive Battery: Sf Manifold
Executes a strict grid search across Lookbacks, Horizons, and Feature Sets.
Enforces absolute train/test isolation, chronological sorting, and naive baseline benchmarks.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Lock seeds for reproducible rigor
tf.random.set_seed(42)
np.random.seed(42)

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    return df

def build_sequences(df, lookback, horizon, base_cols):
    """Constructs flattened lag features and strict forward horizons without leakage."""
    temp_df = df.copy()
    
    # 1. Target Engineering (t + horizon)
    temp_df['Target'] = temp_df['Close'].shift(-horizon)
    
    # 2. Flattened Lookback Engineering (t, t-1, ... t-lookback+1)
    feature_names = []
    for lag in range(lookback):
        for col in base_cols:
            lag_col = f"{col}_t-{lag}"
            temp_df[lag_col] = temp_df[col].shift(lag)
            feature_names.append(lag_col)
            
    # 3. Drop NaNs to ensure perfectly aligned matrices
    temp_df = temp_df.dropna().reset_index(drop=True)
    
    X = temp_df[feature_names].astype(float).values
    y = temp_df['Target'].astype(float).values
    
    # Naive Baseline: Predict that the target (t+h) will equal the current close (t)
    naive_preds = temp_df['Close'].astype(float).values
    
    return X, y, naive_preds

def main():
    print("="*80)
    print(" EXECUTING COMPREHENSIVE BATTERY: Sf MANIFOLD")
    print("="*80)
    
    try:
        train_df = load_and_clean('Sf_training_gem.csv')
        test_df = load_and_clean('Sf_testing_gem.csv')
    except FileNotFoundError:
        print("Error: Could not locate Sf_training_gem.csv or Sf_testing_gem.csv.")
        return

    # Filter available columns dynamically
    drop_cols = ['Date', 'Target']
    all_numeric = [c for c in train_df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    
    features_base = [c for c in all_numeric if c not in ['Vol', 'OpenInt']]
    features_ext = [c for c in all_numeric] # Includes Vol and OpenInt if they exist
    
    configs = {
        "Base (OHLC)": features_base,
        "Extended (Liquidity)": features_ext
    }
    
    lookbacks = [1, 5, 10, 20]
    horizons = [1, 3, 5, 10]
    
    print(f"{'Memory':<8} | {'Horizon':<8} | {'Features':<20} | {'Naive MAE':<12} | {'ANN MAE':<12} | {'Edge'} ")
    print("-" * 80)
    
    for lookback in lookbacks:
        for horizon in horizons:
            for feat_name, feat_cols in configs.items():
                
                # Check if extended features actually exist in this dataset
                if feat_name == "Extended (Liquidity)" and not any(c in feat_cols for c in ['Vol', 'OpenInt']):
                    continue
                
                X_train, y_train, _ = build_sequences(train_df, lookback, horizon, feat_cols)
                X_test, y_test, naive_test = build_sequences(test_df, lookback, horizon, feat_cols)
                
                if len(X_train) == 0 or len(X_test) == 0:
                    continue
                    
                # Strict Train-Only Scaling
                mu = np.mean(X_train, axis=0)
                sigma = np.std(X_train, axis=0) + 1e-8
                X_train_n = (X_train - mu) / sigma
                X_test_n = (X_test - mu) / sigma
                
                # Target Scaling
                t_mu = np.mean(y_train)
                t_sig = np.std(y_train) + 1e-8
                y_train_n = (y_train - t_mu) / t_sig
                
                # Fast, deep architecture
                model = models.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(X_train_n.shape[1],)),
                    layers.Dropout(0.2),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1, activation='linear')
                ])
                
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
                model.fit(X_train_n, y_train_n, epochs=25, batch_size=32, verbose=0)
                
                # Inference and unscaling
                preds_n = model.predict(X_test_n, verbose=0).reshape(-1)
                preds = (preds_n * t_sig) + t_mu
                
                ann_mae = mean_absolute_error(y_test, preds)
                naive_mae = mean_absolute_error(y_test, naive_test)
                
                edge_val = naive_mae - ann_mae
                edge_str = f"+{edge_val:.6f}" if edge_val > 0 else f"{edge_val:.6f}"
                
                print(f"L-{lookback:<6} | t+{horizon:<6} | {feat_name:<20} | {naive_mae:<12.6f} | {ann_mae:<12.6f} | {edge_str}")

if __name__ == "__main__":
    main()
