#!/usr/bin/env python3
r"""
Date of Creation: September 8, 2026
File Name: ablation_simulate_sf_gem.py

Annotations:
- Canonical Sf Structural Feature Ablation & Perturbation Audit Script.
- Includes strict whitespace stripping, chronological sorting, and PrevClose engineering.
- Executes expectation clamping (x_i <- E[X_i]) and prints Delta L_MAE directly to the screen.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="Sf_testing_gem.csv")
    parser.add_argument("--model", type=str, default="artifacts/sf_ann_run/model_final.keras")
    parser.add_argument("--scaler", type=str, default="artifacts/sf_ann_run/scaler.json")
    args = parser.parse_args()

    print("="*75)
    print(" SF STRUCTURAL FEATURE ABLATION & PERTURBATION AUDIT")
    print("="*75)

    with open(args.scaler, "r") as f:
        scaler = json.load(f)

    feature_cols = scaler["feature_cols"]
    mu = np.array(scaler["mu"], float)
    sigma = np.array(scaler["sigma"], float)
    target_mu = float(scaler["target_mu"])
    target_sigma = float(scaler["target_sigma"])

    print(f"Loading Sf test data from {args.test_csv}...")
    df_test = pd.read_csv(args.test_csv)
    
    # CRITICAL FIX: Strip whitespace, sort chronologically, and engineer PrevClose
    df_test.columns = df_test.columns.str.strip()
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    df_test = df_test.sort_values('Date', ascending=True).reset_index(drop=True)
    df_test['PrevClose'] = df_test['Close'].shift(1)
    df_test = df_test.dropna(subset=feature_cols)

    X_test = df_test[feature_cols].astype(float).values
    y_test = df_test['Close'].astype(float).values

    # Temporal alignment: X_t predicts y_{t+1}
    X_eval = X_test[:-1]
    y_eval = y_test[1:]
    X_eval_n = (X_eval - mu) / sigma

    model = tf.keras.models.load_model(args.model)
    preds_n = model.predict(X_eval_n, verbose=0).reshape(-1)
    preds_baseline = preds_n * target_sigma + target_mu
    baseline_mae = mean_absolute_error(y_eval, preds_baseline)

    print(f"\nBaseline Out-of-Sample Sf MAE: {baseline_mae:.4f}\n")
    print(f"{'Feature Dimension':<20} | {'Ablation MAE (Delta L)':<25}")
    print("-" * 55)

    for idx, col in enumerate(feature_cols):
        X_ablated_n = X_eval_n.copy()
        X_ablated_n[:, idx] = 0.0 

        preds_ablated_n = model.predict(X_ablated_n, verbose=0).reshape(-1)
        preds_ablated = preds_ablated_n * target_sigma + target_mu
        ablated_mae = mean_absolute_error(y_eval, preds_ablated)

        delta_mae = ablated_mae - baseline_mae
        print(f"{col:<20} | {delta_mae:<25.4f}")

    print("="*55)

if __name__ == "__main__":
    main()