#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)

def main():
    print("="*70)
    print(" CONTROL TEST: SYNTHETIC GEOMETRY BIAS")
    print("="*70)

    # 1. Generate Synthetic OHLC Data (N=5000)
    N = 5000
    open_price = np.random.uniform(0.5, 1.5, N)
    high = open_price + np.random.uniform(0.01, 0.10, N)
    low = open_price - np.random.uniform(0.01, 0.10, N)
    close = np.random.uniform(low, high, N)
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    # 2. INJECT THE DETERMINISTIC BIAS
    # Rule: Target = Close + (0.8 * Spread * Direction) + Noise
    direction = np.sign(close - open_price)
    spread = high - low
    
    # Adding a 0.001 noise floor to simulate minor market friction
    target = close + (0.8 * spread * direction) + np.random.normal(0, 0.001, N)

    X = np.column_stack([open_price, high, low, close, prev_close])
    y = target

    # 3. Train/Test Split (Strictly Sequential)
    split = int(N * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Naive Baseline: Predicts tomorrow equals today's close
    naive_test = close[split:]

    # 4. Strict Scaling
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0) + 1e-8
    X_train_n = (X_train - mu) / sigma
    X_test_n = (X_test - mu) / sigma

    t_mu = np.mean(y_train)
    t_sig = np.std(y_train) + 1e-8
    y_train_n = (y_train - t_mu) / t_sig

    # 5. THE EXACT SAME ARCHITECTURE FROM YOUR SF BATTERY
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_n.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
    
    print("Training the MLP architecture on biased data...")
    model.fit(X_train_n, y_train_n, epochs=25, batch_size=32, verbose=0)

    # 6. Evaluation
    preds_n = model.predict(X_test_n, verbose=0).reshape(-1)
    preds = (preds_n * t_sig) + t_mu

    ann_mae = mean_absolute_error(y_test, preds)
    naive_mae = mean_absolute_error(y_test, naive_test)
    edge = naive_mae - ann_mae

    print("\nResults:")
    print(f"Data Shape: {X.shape[0]} rows | Features: Open, High, Low, Close, PrevClose")
    print("Bias Rule : Target = Close + 0.8 * (High - Low) * Sign(Close - Open)\n")
    print(f"Naive MAE : {naive_mae:.6f}  (Blind to the geometrical rule)")
    print(f"ANN MAE   : {ann_mae:.6f}  (Should map the geometry)")
    print("-" * 70)
    print(f"Net Edge  : +{edge:.6f}")
    print("="*70)

if __name__ == "__main__":
    main()