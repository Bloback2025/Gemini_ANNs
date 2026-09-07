#!/usr/bin/env python3
"""
train_phase2b2_HO_v5_heavy_fixed.py

Deterministic, auditable training script for the 2bANN2 HO v5 heavy model.
- Uses canonical FEATURE_COLS = ["Open","High","Low","Close"]
- Persists scaler.json (mu/sigma/feature_cols) into outdir
- Writes RUNLOG_2bANN2_HO_v5_heavy_{timestamp}.json with model_file, model_hash, feature_cols, scaler_file, metrics
- Produces model as .keras (zip-based SavedModel)
- Deterministic when --seed and --deterministic are used
"""
import argparse
import json
import os
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Canonical feature list
FEATURE_COLS = ["Open", "High", "Low", "Close"]

def set_deterministic(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # TensorFlow deterministic ops (best-effort)
    try:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except Exception:
        pass

def shift_forward(X: np.ndarray, y: np.ndarray):
    """
    Align features X[t] to predict y[t+1].
    Returns X_shifted (len-1, dim) and y_shifted (len-1,)
    """
    if len(X) < 2 or len(y) < 2:
        return np.empty((0, X.shape[1])), np.empty((0,))
    Xs = X[:-1]
    ys = y[1:]
    return Xs, ys

def build_model(input_dim: int):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def compute_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="Path to training CSV")
    p.add_argument("--val_csv", required=False, help="Path to validation CSV")
    p.add_argument("--test_csv", required=False, help="Path to test CSV")
    p.add_argument("--outdir", required=True, help="Output directory for model and runlog")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic ops where possible")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.deterministic:
        set_deterministic(args.seed)

    # Load training CSV
    df_train = pd.read_csv(args.train_csv, parse_dates=["Date"])
    df_train.columns = df_train.columns.str.strip().str.capitalize()
    missing = [c for c in FEATURE_COLS + ["Close"] if c not in df_train.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in training CSV: {missing}")

    X = df_train[FEATURE_COLS].astype(float).values
    y = df_train["Close"].astype(float).values

    # Shift forward so X[t] -> y[t+1]
    X_s, y_s = shift_forward(X, y)
    if X_s.shape[0] == 0:
        raise RuntimeError("Not enough rows after shift_forward to train")

    # Compute and persist scaler (mu/sigma) from training-shifted features
    mu = X_s.mean(axis=0)
    sigma = X_s.std(axis=0) + 1e-9
    scaler = {"mu": mu.tolist(), "sigma": sigma.tolist(), "feature_cols": FEATURE_COLS}
    scaler_path = outdir / "scaler.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler, f, indent=2)

    # Build model
    model = build_model(input_dim=len(FEATURE_COLS))

    # Train
    history = model.fit(
        (X_s - mu) / sigma,
        y_s,
        validation_split=0.05 if args.val_csv is None else 0.0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=not args.deterministic,
        verbose=2
    )

    # Save model
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"2bANN2_HO_model_v5_heavy_{timestamp}.keras"
    model_path = outdir / model_filename
    model.save(str(model_path))

    # Compute model SHA256 and write sidecar
    model_hash = compute_sha256(model_path)
    with open(str(model_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(model_hash)

    # Collect metrics (use last epoch)
    metrics = {
        "loss": float(history.history.get("loss", [None])[-1]),
        "mae": float(history.history.get("mae", [None])[-1]),
        "val_loss": float(history.history.get("val_loss", [None])[-1]) if "val_loss" in history.history else None,
        "val_mae": float(history.history.get("val_mae", [None])[-1]) if "val_mae" in history.history else None
    }

    # Write RUNLOG
    runlog = {
        "model_file": str(model_path),
        "model_hash": model_hash,
        "train_file": args.train_csv,
        "val_file": args.val_csv,
        "test_file": args.test_csv,
        "metrics": metrics,
        "feature_cols": FEATURE_COLS,
        "scaler_file": str(scaler_path),
        "timestamp": timestamp
    }
    runlog_path = outdir / f"RUNLOG_2bANN2_HO_v5_heavy_{timestamp}.json"
    with open(runlog_path, "w") as f:
        json.dump(runlog, f, indent=2)

    # Also write runlog sha256
    runlog_hash = compute_sha256(runlog_path)
    with open(str(runlog_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(runlog_hash)

    print("Training complete")
    print("model:", model_path)
    print("runlog:", runlog_path)
    print("scaler:", scaler_path)

if __name__ == "__main__":
    main()



