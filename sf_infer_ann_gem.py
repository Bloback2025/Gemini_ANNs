#!/usr/bin/env python3
r"""
Date of Creation: September 8, 2026
File Name: sf_infer_ann_gem.py
Path: C:/Users/loweb/AI_Financial_Sims/Gemini_ANNs/sf_infer_ann_gem.py

Annotations:
- Canonical Sf ANN inference runner synchronized with HO pipeline.
- Incorporates strict whitespace stripping, chronological sorting, and PrevClose engineering.
- Enforces proper temporal slicing (preds[:-1]) to prevent lag bias.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    # Load scaler config
    with open(args.scaler, "r") as f:
        scaler = json.load(f)

    feature_cols = scaler["feature_cols"]
    mu = np.array(scaler["mu"], float)
    sigma = np.array(scaler["sigma"], float)
    target_mu = float(scaler["target_mu"])
    target_sigma = float(scaler["target_sigma"])

    print(f"Loading Sf test data from {args.input_csv}...")
    df_test = pd.read_csv(args.input_csv)
    
    # 1. Strip hidden whitespace from columns
    df_test.columns = df_test.columns.str.strip()
    
    # 2. Enforce chronological order
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    df_test = df_test.sort_values('Date', ascending=True).reset_index(drop=True)
    
    # 3. Engineer PrevClose
    df_test['PrevClose'] = df_test['Close'].shift(1)
    df_test = df_test.dropna(subset=feature_cols)

    # 4. Extract and normalize features
    X_test = df_test[feature_cols].astype(float).values
    X_test_n = (X_test - mu) / sigma

    # Run predictions
    model = tf.keras.models.load_model(args.model)
    pred_n = model.predict(X_test_n, verbose=0).reshape(-1)
    preds = pred_n * target_sigma + target_mu

    # 5. Temporal Alignment (X_t predicts y_{t+1}) and Tabular JSON Output
    out_df = pd.DataFrame({
        "Date": df_test["Date"].iloc[1:].values,
        "Actual_Close": df_test["Close"].iloc[1:].values,
        "Pred_Close": preds[:-1]
    })

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_json(args.out_csv, orient="records", date_format="iso", indent=2)
    print(f"INFERENCE DONE. Saved to {args.out_csv}")

if __name__ == "__main__":
    main()