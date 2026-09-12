#!/usr/bin/env python3
r"""
Date of Creation: September 8, 2026
File Name: ablation_simulate_ho_gem.py
Path: C:/Users/loweb/AI_Financial_Sims/Gemini_ANNs/ablation_simulate_ho_gem.py

Annotations:
- Canonical Heating Oil (HO) Structural Feature Ablation & Perturbation Audit Script.
- Executes sequential conditional expectation clamping (x_i <- E[X_i]) across input matrices 
  to isolate empirical risk degradation (Delta L_MAE) for each primitive in the HO manifold.
- Prints the feature hierarchy table directly to the screen and saves the JSON report.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

def make_features(df):
    df = df.sort_values("Date")
    df["PrevClose"] = df["Close"].shift(1)
    return df

def main():
    parser = argparse.ArgumentParser(description="Canonical HO ANN Structural Feature Ablation Suite")
    parser.add_argument("--test_csv", type=str, default="hoxnc_testing_retest1.csv", help="Path to testing CSV")
    parser.add_argument("--model", type=str, default="artifacts/ho_ann_run_retest1/model_final.keras", help="Path to trained model")
    parser.add_argument("--scaler", type=str, default="artifacts/ho_ann_run_retest1/scaler.json", help="Path to scaler JSON")
    args = parser.parse_args()

    print("="*75)
    print(" HEATING OIL (HO) STRUCTURAL FEATURE ABLATION & PERTURBATION AUDIT")
    print("="*75)

    # 1. Load Scaler & Model
    with open(args.scaler, "r") as f:
        scaler = json.load(f)

    feature_cols = scaler["feature_cols"]
    mu = np.array(scaler["mu"], float)
    sigma = np.array(scaler["sigma"], float)
    target_mu = float(scaler["target_mu"])
    target_sigma = float(scaler["target_sigma"])

    print(f"Loading HO test data from {args.test_csv}...")
    df_test = pd.read_csv(args.test_csv, parse_dates=["Date"])
    df_test = make_features(df_test)
    df_test = df_test.dropna(subset=feature_cols)

    X_test = df_test[feature_cols].astype(float).values
    y_test = df_test["Close"].astype(float).values

    # Temporal alignment: X_t predicts y_{t+1}
    X_eval = X_test[:-1]
    y_eval = y_test[1:]

    # Normalize features using frozen training parameters
    X_eval_n = (X_eval - mu) / sigma

    model = tf.keras.models.load_model(args.model)

    # 2. Compute Baseline Prediction & MAE
    preds_n = model.predict(X_eval_n, verbose=0).reshape(-1)
    preds_baseline = preds_n * target_sigma + target_mu
    baseline_mae = mean_absolute_error(y_eval, preds_baseline)

    print(f"\nBaseline Out-of-Sample HO MAE: {baseline_mae:.4f}\n")
    print(f"{'Feature Dimension':<20} | {'Ablation MAE (Delta L)':<25} | {'Structural Classification'}")
    print("-" * 75)

    ablation_results = {}

    # Expected structural classifications from README 1.2
    classifications = {
        "Close": "Primary Manifold Anchor",
        "High": "Upper Volatility Boundary",
        "Open": "Initial State Primitive",
        "Low": "Lower Volatility Boundary",
        "PrevClose": "Epistemologically Redundant"
    }

    # 3. Execute Perturbation Sweep
    for idx, col in enumerate(feature_cols):
        X_ablated_n = X_eval_n.copy()
        # Conditional expectation clamping: clamp feature to its expected value (mean -> 0.0 in normalized space)
        X_ablated_n[:, idx] = 0.0 

        preds_ablated_n = model.predict(X_ablated_n, verbose=0).reshape(-1)
        preds_ablated = preds_ablated_n * target_sigma + target_mu
        ablated_mae = mean_absolute_error(y_eval, preds_ablated)

        delta_mae = ablated_mae - baseline_mae
        ablation_results[col] = float(delta_mae)

        classification = classifications.get(col, "Secondary Primitive")
        print(f"{col:<20} | {delta_mae:<25.4f} | {classification}")

    print("="*75)
    
    # 4. Save Results Report
    out_dir = os.path.dirname(args.model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_simulation_ho_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "baseline_mae": float(baseline_mae),
            "ablation_delta_mae": ablation_results
        }, f, indent=2)
    print(f"\nHO ablation report saved to: {out_path}\n")

if __name__ == "__main__":
    main()