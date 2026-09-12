#!/usr/bin/env python3
r"""
Date of Creation: September 8, 2026
Path: C:/Users/loweb/AI_Financial_Sims/Gemini_ANNs/ho_infer_ann_retest1_gem_REPLACE.py

Annotations:
- Canonical retest inference runner synchronized with legacy pipeline and fixed JSON output.

CRITICAL FIXES INCORPORATED:
1. Tabular JSON Output: Uses pd.DataFrame and .to_json(orient="records") to generate the 
   exact array of objects (Date, Actual_Close, Pred_Close) demanded by advanced_eval_metrics_gem.py.
2. Temporal Slicing: Restores the preds[:-1] and iloc[1:] logic from the legacy script to ensure 
   today's features predict tomorrow's close, entirely eliminating lag bias.
3. CLI Arguments: Retains the --input_csv and --out_csv flags so README execution commands 
   work perfectly without throwing parser errors.
"""

import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf

def make_features(df):
    df = df.sort_values("Date")
    df["PrevClose"] = df["Close"].shift(1)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.scaler, "r") as f:
        scaler = json.load(f)

    feature_cols = scaler["feature_cols"]
    mu = np.array(scaler["mu"], float)
    sigma = np.array(scaler["sigma"], float)
    target_mu = float(scaler["target_mu"])
    target_sigma = float(scaler["target_sigma"])

    df_test = pd.read_csv(args.input_csv, parse_dates=["Date"])
    df_test = make_features(df_test)
    df_test = df_test.dropna(subset=feature_cols)

    X_test = df_test[feature_cols].astype(float).values
    X_test_n = (X_test - mu) / sigma

    model = tf.keras.models.load_model(args.model)
    pred_n = model.predict(X_test_n, verbose=0).reshape(-1)
    preds = pred_n * target_sigma + target_mu

    out_df = pd.DataFrame({
        "Date": df_test["Date"].iloc[1:].values,
        "Actual_Close": df_test["Close"].iloc[1:].values,
        "Pred_Close": preds[:-1]
    })

    out_df.to_json(args.out_csv, orient="records", date_format="iso", indent=2)
    print(f"INFERENCE DONE. Saved to {args.out_csv}")

if __name__ == "__main__":
    main()