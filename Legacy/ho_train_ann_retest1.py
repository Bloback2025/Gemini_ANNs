^X^X#!/usr/bin/env python3
r"""
HO_train_phase_ANNs\ho_train_ann_retest1.py
Canonical HO ANN trainer. Produces model_final.keras, model_best.keras, scaler.json, RUNLOG.json.
"""

import os, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_CSV = r"C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\hoxnc_training_retest1.csv"
VAL_CSV   = r"C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\hoxnc_validation_retest1.csv"
OUTDIR = r"C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\artifacts\ho_ann_run_retest1"

os.makedirs(OUTDIR, exist_ok=True)

FEATURE_COLS = ["Open", "High", "Low", "Close", "PrevClose"]

def make_features(df):
    df = df.sort_values("Date")
    df["PrevClose"] = df["Close"].shift(1)
    return df.dropna(subset=FEATURE_COLS)

def load_xy(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = make_features(df)
    X = df[FEATURE_COLS].astype(float).values
    y = df["Close"].astype(float).values
    return X[:-1], y[1:]

X_train, y_train = load_xy(TRAIN_CSV)
X_val,   y_val   = load_xy(VAL_CSV)

mu = np.nanmean(X_train, axis=0)
sigma = np.nanstd(X_train, axis=0) + 1e-12
X_train_n = (X_train - mu) / sigma
X_val_n   = (X_val   - mu) / sigma

y_mu = float(np.nanmean(y_train))
y_sigma = float(np.nanstd(y_train)) + 1e-12
y_train_n = (y_train - y_mu) / y_sigma
y_val_n   = (y_val   - y_mu) / y_sigma

inp = layers.Input(shape=(X_train_n.shape[1],))
x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
out = layers.Dense(1, activation="linear")(x)
model = models.Model(inp, out)

model.compile(optimizer=optimizers.Adam(1e-5, clipnorm=0.5), loss="mae", metrics=["mae"])

es = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
ckpt_path = os.path.join(OUTDIR, "model_best.keras")
ckpt = callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss")

model.fit(
    X_train_n, y_train_n,
    validation_split=0.1,
    epochs=200,
    batch_size=64,
    callbacks=[es, ckpt],
    verbose=2
)

final_model_path = os.path.join(OUTDIR, "model_final.keras")
model.save(final_model_path)

scaler = {
    "feature_cols": FEATURE_COLS,
    "mu": mu.tolist(),
    "sigma": sigma.tolist(),
    "target_mu": y_mu,
    "target_sigma": y_sigma,
    "target_type": "standardized_close"
}
with open(os.path.join(OUTDIR, "scaler.json"), "w") as f:
    json.dump(scaler, f, indent=2)

runlog = {
    "model_file": final_model_path,
    "best_ckpt": ckpt_path,
    "scaler_file": os.path.join(OUTDIR, "scaler.json"),
    "training_rows": int(X_train.shape[0]),
    "validation_rows": int(X_val.shape[0])
}
with open(os.path.join(OUTDIR, "RUNLOG.json"), "w") as f:
    json.dump(runlog, f, indent=2)

print("TRAIN DONE")
print(final_model_path)
print(os.path.join(OUTDIR, "scaler.json"))



