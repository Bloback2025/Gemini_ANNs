cp ho_infer_ann_gputest.py ho_infer_ann_gputest.py.bak && cat > ho_infer_ann_gputest.py <<'PY'
#!/usr/bin/env python3
"""
ho_infer_ann_gputest.py
GPU-capable inference script for HO ANN. Loads scaler.json and model_final.keras and runs inference on CSV.
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

def windows_to_wsl(p: str) -> str:
    if not p:
        return p
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
    if m and os.name == 'posix':
        drive = m.group(1).lower()
        rest = m.group(2).replace('\\', '/')
        return f'/mnt/{drive}/{rest}'
    return p

parser = argparse.ArgumentParser(description="HO ANN GPU inference.")
parser.add_argument("--input_csv", type=str, required=True, help="CSV to run inference on")
parser.add_argument("--model", type=str, required=True, help="Path to model_final.keras")
parser.add_argument("--scaler", type=str, required=True, help="Path to scaler.json")
parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
args = parser.parse_args()

INPUT_CSV = Path(windows_to_wsl(args.input_csv))
MODEL_PATH = Path(windows_to_wsl(args.model))
SCALER_PATH = Path(windows_to_wsl(args.scaler))
OUT_CSV = Path(windows_to_wsl(args.out_csv))
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

print("DEBUG TF version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("DEBUG TF GPUs:", gpus)
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

df = pd.read_csv(INPUT_CSV, parse_dates=["Date"])
with open(SCALER_PATH, "r") as f:
    scaler = json.load(f)

FEATURE_COLS = scaler.get("feature_cols", ["Open","High","Low","Close","PrevClose"])
mu = np.array(scaler["mu"], dtype=float)
sigma = np.array(scaler["sigma"], dtype=float)
target_mu = float(scaler.get("target_mu", 0.0))
target_sigma = float(scaler.get("target_sigma", 1.0))

df = df.sort_values("Date")
df["PrevClose"] = df["Close"].shift(1)
df = df.dropna(subset=FEATURE_COLS)
X = df[FEATURE_COLS].astype(float).values
X_n = (X - mu) / (sigma + 1e-12)

model = tf.keras.models.load_model(str(MODEL_PATH))
pred_n = model.predict(X_n, batch_size=64)
pred = (pred_n.flatten() * target_sigma) + target_mu

out_df = df.copy()
out_df["pred_close"] = np.nan
# align predictions so pred[t] corresponds to the row where features were available
out_df.iloc[1:1+len(pred), out_df.columns.get_loc("pred_close")] = pred

out_df.to_csv(OUT_CSV, index=False)

runlog = {
    "model": str(MODEL_PATH),
    "scaler": str(SCALER_PATH),
    "input_csv": str(INPUT_CSV),
    "output_csv": str(OUT_CSV),
    "run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "gpu_devices": [str(g) for g in gpus] if gpus else []
}
with open(OUT_CSV.parent / "INFER_RUNLOG.json", "w") as f:
    json.dump(runlog, f, indent=2)

print("INFERENCE DONE")
print(str(OUT_CSV))
PY

