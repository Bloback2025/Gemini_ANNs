Python
"""
Date of Creation: September 6, 2026, 5:42 PM
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\ho_train_ann_gputest_gem.py
Original Legacy Source: C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\ho_train_ann_gputest.py

Annotations:
- This is the recovered, Gemini-audited production training script (originally the now-lost file C:\\Users\\loweb\\AI_Financial_Sims\\HO\\HO_train_phase_ANNs\\ho_train_ann_gputest.py), integrated into the Gemini_ANNs root structure.
- Updated to prevent data leakage across multiple tickers by incorporating safe `groupby("Ticker")` logic during feature generation (shifting/lagging) when a Ticker column is present.
- Preserves Windows-to-WSL path translation helpers, reproducible random seeding, TensorFlow GPU memory growth allocation, checkpointing, and JSON artifact exports (`scaler.json`, `RUNLOG.json`).
"""

import os
import re
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
# Helpers
# -------------------------
def windows_to_wsl(p: str) -> str:
    if not p:
        return p
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
    if m and os.name == 'posix':
        drive = m.group(1).lower()
        rest = m.group(2).replace('\\', '/')
        return f'/mnt/{drive}/{rest}'
    return p

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="HO ANN GPU trainer (Gemini Clean Rewrite).")
parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV")
parser.add_argument("--outdir", type=str, required=True, help="Output directory for artifacts")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

TRAIN_CSV = Path(windows_to_wsl(args.train_csv))
VAL_CSV   = Path(windows_to_wsl(args.val_csv))
OUTDIR    = Path(windows_to_wsl(args.outdir))
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
# GPU debug & safety
# -------------------------
print("DEBUG TF version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("DEBUG TF GPUs:", gpus)
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# -------------------------
# Data helpers
# -------------------------
FEATURE_COLS = ["Open", "High", "Low", "Close", "PrevClose"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date")
    # CRITICAL FIX: Prevent cross-ticker data leakage if Ticker column exists
    if "Ticker" in df.columns:
        df["PrevClose"] = df.groupby("Ticker")["Close"].shift(1)
    else:
        df["PrevClose"] = df["Close"].shift(1)
    return df.dropna(subset=FEATURE_COLS)

def load_xy(path: Path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = make_features(df)
    X = df[FEATURE_COLS].astype(float).values
    y = df["Close"].astype(float).values
    # keep original alignment behavior: X[t] -> predict y[t+1]
    return X[:-1], y[1:]

# -------------------------
# Load data (explicit checks)
# -------------------------
if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")
if not VAL_CSV.exists():
    raise FileNotFoundError(f"Validation CSV not found: {VAL_CSV}")

X_train, y_train = load_xy(TRAIN_CSV)
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
# Model
# -------------------------
inp = layers.Input(shape=(X_train_n.shape[1],))
x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
out = layers.Dense(1, activation="linear")(x)
model = models.Model(inp, out)

model.compile(optimizer=optimizers.Adam(1e-5, clipnorm=0.5), loss="mae", metrics=["mae"])

# -------------------------
# Callbacks & training
# -------------------------
es = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
ckpt_path = OUTDIR / "model_best.keras"
ckpt = callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_loss")

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
# Export scaler and runlog
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