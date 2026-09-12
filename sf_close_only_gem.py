#!/usr/bin/env python3
r"""
Date of Creation: September 8, 2026
File Name: sf_close_only_gem.py
Path: C:/Users/loweb/AI_Financial_Sims/Gemini_ANNs/sf_close_only_gem.py

Annotations:
- Sf Benchmark Verification (Close-Only Input) as mandated by README 1.5.
- Evaluates the 256->128->64->1 ANN architecture strictly on a 1D 'Close' feature matrix.
- Sweeps non-adjacent horizons (t-1, t-5, t-10, t-15, t-20) to validate momentum extraction.
- Computes the ANN MAE side-by-side against the Naive Baseline (y_{t+h} = y_t).
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import layers, models, regularizers, optimizers, callbacks

# Load datasets
train_df = pd.read_csv('Sf_training_gem.csv')
val_df = pd.read_csv('Sf_validation_gem.csv')
test_df = pd.read_csv('Sf_testing_gem.csv')

# Clean columns and sort by date to prevent KeyError and ensure alignment
for df in [train_df, val_df, test_df]:
    df.columns = [c.strip() for c in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df.sort_values('Date', inplace=True)

# Enforce 1D Input Constraint
feature_cols = ['Close']
horizons = [1, 5, 10, 15, 20]

print("="*60)
print(" Sf BENCHMARK VERIFICATION (CLOSE-ONLY INPUT)")
print("="*60)
print(f"{'Horizon':<10} | {'ANN MAE (Close Only)':<20} | {'Naive Baseline MAE'}")
print("-" * 60)

for h in horizons:
    tr, va, te = train_df.copy(), val_df.copy(), test_df.copy()
    
    # Target is Close shifted back by h periods
    tr['Target'] = tr['Close'].shift(-h)
    va['Target'] = va['Close'].shift(-h)
    te['Target'] = te['Close'].shift(-h)
    
    tr.dropna(subset=feature_cols + ['Target'], inplace=True)
    va.dropna(subset=feature_cols + ['Target'], inplace=True)
    te.dropna(subset=feature_cols + ['Target'], inplace=True)
    
    X_tr = tr[feature_cols].astype(float).values
    y_tr = tr['Target'].astype(float).values
    X_v = va[feature_cols].astype(float).values
    y_v = va['Target'].astype(float).values
    X_te = te[feature_cols].astype(float).values
    y_te = te['Target'].values
    
    # Standardization geometry
    mu_x = np.nanmean(X_tr, axis=0)
    sigma_x = np.nanstd(X_tr, axis=0) + 1e-12
    X_tr_n = (X_tr - mu_x) / sigma_x
    X_v_n = (X_v - mu_x) / sigma_x
    X_te_n = (X_te - mu_x) / sigma_x
    
    mu_y = float(np.nanmean(y_tr))
    sigma_y = float(np.nanstd(y_tr)) + 1e-12
    y_tr_n = (y_tr - mu_y) / sigma_y
    y_v_n = (y_v - mu_y) / sigma_y
    
    # Architecture Lock
    inp = layers.Input(shape=(1,))
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    out = layers.Dense(1, activation="linear")(x)
    
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-4, clipnorm=0.5), loss="mae")
    
    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_tr_n, y_tr_n, validation_data=(X_v_n, y_v_n), epochs=40, batch_size=64, callbacks=[es], verbose=0)
    
    # Inference & Affine Synchronization
    preds_n = model.predict(X_te_n, verbose=0).flatten()
    preds = preds_n * sigma_y + mu_y
    
    # Evaluate Out-of-Sample
    ann_mae = float(np.mean(np.abs(preds - y_te)))
    naive_mae = float(np.mean(np.abs(te['Close'].values - y_te)))
    
    print(f"t-{h:<8} | {ann_mae:<20.6f} | {naive_mae:.6f}")

print("="*60)