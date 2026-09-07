Python
"""
Date of Creation: September 6, 2026, 6:08 PM
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\ho_train_ann_retest1_gem.py
Original Legacy Source: C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\ho_train_ann_retest1.py

Annotations:
- This is the canonical retest training script rewritten for the Gemini_ANNs project root.
- Fixes the historical GPU engagement issue by explicitly adding TensorFlow hardware diagnostics and memory growth configuration for WSL2.
- Eliminates hardcoded absolute legacy Windows paths in favor of clean CLI arguments.
- Protects against cross-ticker data leakage by incorporating safe `groupby("Ticker")` logic during feature lagging/shifting.
- Produces model_final.keras, model_best.keras, scaler.json, and RUNLOG.json.
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
parser = argparse.ArgumentParser(description="Canonical HO ANN trainer (retest1 - Gemini version).")
parser.add_argument("--train_csv", type=str, default="hoxnc_training_retest1.csv", help="Path to training CSV")
parser.add_argument("--val_csv", type=str, default="hoxnc_validation_retest1.csv", help="Path to validation CSV")
parser.add_argument("--outdir", type=str, default="artifacts/ho_ann_run_retest1", help="Output directory for artifacts")
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
# GPU Debug & Hardware Safety (Resolves past GPU issues)
# -------------------------
print("DEBUG TF version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("DEBUG TF GPUs:", gpus)
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("✅ GPU Memory Growth Enabled Successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not set memory growth: {e}")
else:
    print("❌ WARNING: No GPU detected by TensorFlow. Execution will fall back to CPU.")

# -------------------------
# Data helpers
# -------------------------
FEATURE_COLS = ["Open", "High", "Low", "Close", "PrevClose"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date")
    # CRITICAL FIX: Safe shifting per ticker if multi-ticker dataset is passed
    if "Ticker" in df.columns:
        df["PrevClose"] = df.groupby("Ticker")["Close"].shift(1)
    else:
        df["PrevClose"] = df["Close"].shift(1)
    return df.dropna(subset=FEATURE_COLS)

def load_xy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = make_features(df)
    X = df[FEATURE_COLS].astype(float).values
    y = df["Close"].astype(float).values
    return X[:-1], y[1:]

# -------------------------
# Load Data
# -------------------------
print(f"Loading training data from {TRAIN_CSV}...")
X_train, y_train = load_xy(TRAIN_CSV)
print(f"Loading validation data from {VAL_CSV}...")
X_val,   y_val   = load_xy(VAL_CSV)

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
# Model Architecture
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

print("Starting model training loop...")
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
    "feature_cols": FEATURE_COLS,
    "mu": mu.tolist(),
    "sigma": sigma.tolist(),
    "target_mu": y_mu,
    "target_sigma": y_sigma,
    "target_type": "standardized_close"
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
    "run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "gpu_devices": [str(g) for g in gpus] if gpus else []
}
runlog_file_path = OUTDIR / "RUNLOG.json"
with open(runlog_file_path, "w") as f:
    json.dump(runlog, f, indent=2)

print("TRAIN DONE")
print(str(final_model_path))
print(str(scaler_file_path))