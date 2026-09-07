"""
Date of Creation: September 6, 2026, 7:10 PM
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\HO_v5_heavy_fixed_gem.py
Original Legacy Source: C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\train_phase2b2_HO_v5_heavy_fixed.py

Annotations:
- This is a clean rewrite of the stable v5 heavy training script (train_phase2b2_HO_v5_heavy_fixed.py), adapted for the Gemini_ANNs project root directory structure.
- Preserves deterministic training controls (`--deterministic`, `--seed`), SHA-256 model and runlog hashing sidecars, scaler persistence (`scaler.json`), and dynamic timestamped runlog creation.
- Enhanced with multi-ticker safety checking: if a `Ticker` column exists in the dataset, shifting is grouped per-ticker to prevent cross-ticker data leakage.
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
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except Exception:
        pass

def shift_forward_grouped(df: pd.DataFrame, feature_cols: list[str]):
    """
    Align features X[t] to predict y[t+1].
    Safely handles multi-ticker grouping if 'Ticker' is present to prevent data leakage.
    """
    if "Ticker" in df.columns:
        # Shift per ticker group
        df_shifted_list = []
        for _, group in df.groupby("Ticker"):
            g = group.copy()
            g_X = g[feature_cols].values
            g_y = g["Close"].values
            if len(g_X) >= 2:
                xs = g_X[:-1]
                ys = g_y[1:]
                # Create a sub-dataframe or arrays matching the length
                sub_df = g.iloc[:-1].copy()
                sub_df[feature_cols] = xs
                sub_df["Target_Close"] = ys
                df_shifted_list.append(sub_df)
        if not df_shifted_list:
            return np.empty((0, len(feature_cols))), np.empty((0,))
        combined = pd.concat(df_shifted_list, ignore_index=True)
        return combined[feature_cols].astype(float).values, combined["Target_Close"].astype(float).values
    else:
        X = df[feature_cols].astype(float).values
        y = df["Close"].astype(float).values
        if len(X) < 2 or len(y) < 2:
            return np.empty((0, X.shape[1])), np.empty((0,))
        return X[:-1], y[1:]

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
    p = argparse.ArgumentParser(description="Deterministic training script for HO v5 heavy model (Gemini version).")
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

    # Shift forward safely with ticker consideration
    X_s, y_s = shift_forward_grouped(df_train, FEATURE_COLS)
    if X_s.shape[0] == 0:
        raise RuntimeError("Not enough rows after shift forward to train")

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