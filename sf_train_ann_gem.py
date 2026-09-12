#!/usr/bin/env python3
r"""
Date of Creation: September 8, 2026
File Name: sf_train_ann_gem.py
Path: C:/Users/loweb/AI_Financial_Sims/Gemini_ANNs/sf_train_ann_gem.py

Annotations:
- Canonical Sf ANN production training script matching the deep 256->128->64 topology.
- Incorporates L2 regularization (1e-5), Adam optimization (1e-5, clipnorm=0.5), and MAE loss.
- Serializes model checkpoints, final weights, and normalization scaler parameters.
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

# -------------------------
# CLI & Paths
# -------------------------
parser = argparse.ArgumentParser(description="Canonical Sf ANN production trainer.")
parser.add_argument("--train_csv", type=str, default="Sf_training_gem.csv", help="Path to training CSV")
parser.add_argument("--val_csv", type=str, default="Sf_validation_gem.csv", help="Path to validation CSV")
parser.add_argument("--outdir", type=str, default="artifacts/sf_ann_run", help="Output directory for artifacts")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

TRAIN_CSV = Path(args.train_csv)
VAL_CSV   = Path(args.val_csv)
OUTDIR    = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
SEED = int(args.seed)
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Data helpers
# -------------------------
def load_xy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    
    # 1. Strip hidden whitespace from column headers (Fixes KeyError)
    df.columns = df.columns.str.strip()
    
    # 2. Enforce chronological order (Fixes reverse-date target leakage)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    # 3. Engineer PrevClose to match HO baseline feature set
    df['PrevClose'] = df['Close'].shift(1)
    df = df.dropna()
    
    # 4. Extract aligned matrices
    feature_cols = ['Open', 'High', 'Low', 'Close', 'PrevClose']
    X = df[feature_cols].astype(float).values
    y = df['Close'].astype(float).values
    
    # 5. Temporal alignment: X_t predicts y_{t+1}
    return X[:-1], y[1:], feature_cols

# -------------------------
# Load Data
# -------------------------
print(f"Loading training data from {TRAIN_CSV}...")
X_train, y_train, feature_cols = load_xy(TRAIN_CSV)
print(f"Loading validation data from {VAL_CSV}...")
X_val, y_val, _ = load_xy(VAL_CSV)

# -------------------------
# Scaling
# -------------------------
mu = np.nanmean(X_train, axis=0)
sigma = np.nanstd(X_train, axis=0) + 1e-12
X_train_n = (X_train - mu) / sigma
X_val_n   = (X_val   - mu) / sigma

y_mu = float(np.nanmean(y_train))
y_sigma = float(np.nanstd(y_train)) + 1e-12
y_train_n = (y_train - y_mu) / y_sigma
y_val_n   = (y_val   - y_mu) / y_sigma

# -------------------------
# Model Architecture (Production Topology)
# -------------------------
inp = layers.Input(shape=(X_train_n.shape[1],))
x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
out = layers.Dense(1, activation="linear")(x)
model = models.Model(inp, out)

model.compile(optimizer=optimizers.Adam(1e-5, clipnorm=0.5), loss="mae", metrics=["mae"])

# -------------------------
# Callbacks & Training
# -------------------------
es = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
ckpt_path = OUTDIR / "model_best.keras"
ckpt = callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_loss")

print("Starting production training loop...")
history = model.fit(
    X_train_n, y_train_n,
    validation_data=(X_val_n, y_val_n),
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=[es, ckpt],
    verbose=2
)

final_model_path = OUTDIR / "model_final.keras"
model.save(str(final_model_path))

# -------------------------
# Artifact Export (Scaler & Runlog)
# -------------------------
scaler = {
    "feature_cols": feature_cols,
    "mu": mu.tolist(),
    "sigma": sigma.tolist(),
    "target_mu": y_mu,
    "target_sigma": y_sigma,
    "target_type": "standardized_target"
}
scaler_file_path = OUTDIR / "scaler.json"
with open(scaler_file_path, "w") as f:
    json.dump(scaler, f, indent=2)

runlog = {
    "model_file": str(final_model_path),
    "best_ckpt": str(ckpt_path),
    "scaler_file": str(scaler_file_path),
    "training_rows": int(X_train.shape[0]),
    "validation_rows": int(X_val.shape[0]),
    "epochs_ran": int(len(history.history.get("loss", []))),
    "run_timestamp_utc": datetime.utcnow().isoformat() + "Z"
}
runlog_file_path = OUTDIR / "RUNLOG.json"
with open(runlog_file_path, "w") as f:
    json.dump(runlog, f, indent=2)

print("TRAIN DONE")
print(str(final_model_path))
print(str(scaler_file_path))