# check_preds.py
import pandas as pd
import numpy as np
from pathlib import Path

PRED_PATH = Path(r"C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\artifacts\ho_ann_run\preds_from_model.csv")
TEST_PATH = Path(r"C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\artifacts\ho_ann_run\processed_test.csv")

p = pd.read_csv(PRED_PATH)
t = pd.read_csv(TEST_PATH)

# align by date if present, else by row order
if 'date' in p.columns and 'date' in t.columns:
    p['date'] = pd.to_datetime(p['date'], errors='coerce')
    t['date'] = pd.to_datetime(t['date'], errors='coerce')
    merged = pd.merge(p, t[['date','Close']], on='date', how='inner', suffixes=('_pred','_act'))
else:
    n = min(len(p), len(t))
    merged = pd.concat([p.iloc[:n].reset_index(drop=True), t[['Close']].iloc[:n].reset_index(drop=True)], axis=1)
    merged = merged.rename(columns={'Close':'Close_act'})

# detect prediction column
pred_col = None
for c in ['pred_close','pred','prediction','y_pred']:
    if c in merged.columns:
        pred_col = c
        break
if pred_col is None:
    numeric = merged.select_dtypes(include=[np.number]).columns.tolist()
    pred_col = numeric[0] if numeric else None

# determine actual close column
if 'Close' in merged.columns:
    actual_col = 'Close'
elif 'Close_act' in merged.columns:
    actual_col = 'Close_act'
else:
    actual_col = None

if pred_col and actual_col:
    merged = merged.dropna(subset=[pred_col, actual_col])
    mae = (merged[pred_col].astype(float) - merged[actual_col].astype(float)).abs().mean()
    rmse = np.sqrt(((merged[pred_col].astype(float) - merged[actual_col].astype(float))**2).mean())
    print(f"Rows compared: {len(merged)}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print("\nSample comparisons (first 10):")
    print(merged[[pred_col, actual_col]].head(10).to_string(index=False))
else:
    print("Could not find prediction or actual Close column. Columns in merged file:", merged.columns.tolist())
