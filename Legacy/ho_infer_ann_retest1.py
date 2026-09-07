#!/usr/bin/env python3
"""
HO_train_phase_ANNs\ho_infer_ann_retest1.py
Canonical HO ANN inference runner. Produces preds_model.json.
"""

import os, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ap = argparse.ArgumentParser()
ap.add_argument("--test_csv", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--scaler", required=True)
ap.add_argument("--outdir", required=True)
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)

with open(args.scaler, "r") as f:
    scaler = json.load(f)

FEATURE_COLS = scaler["feature_cols"]
mu = np.array(scaler["mu"], float)
sigma = np.array(scaler["sigma"], float)
target_mu = float(scaler["target_mu"])
target_sigma = float(scaler["target_sigma"])

def make_features(df):
    df = df.sort_values("Date")
    df["PrevClose"] = df["Close"].shift(1)
    return df.dropna(subset=FEATURE_COLS)

df_test = pd.read_csv(args.test_csv, parse_dates=["Date"])
df_test = make_features(df_test)

X_test = df_test[FEATURE_COLS].astype(float).values
X_test_n = (X_test - mu) / sigma

model = tf.keras.models.load_model(args.model)
pred_n = model.predict(X_test_n, verbose=0).reshape(-1)
pred = pred_n * target_sigma + target_mu

out = pd.DataFrame({
    "Date": df_test["Date"].iloc[1:].values,
    "Actual_Close": df_test["Close"].iloc[1:].values,
    "Pred_Normalized": pred_n[:-1],
    "Pred_Close": pred[:-1]
})

out_path = os.path.join(args.outdir, "preds_model_retest1.json")
out.to_json(out_path, orient="records", indent=2)

print("INFERENCE DONE")
print(out_path)



