#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

def main():
    target_file = "artifacts/sf_ann_run/inference_rebuild/preds_model_sf.json"
    
    try:
        with open(target_file, "r") as f:
            d = json.load(f)
    except FileNotFoundError:
        print(f"Error: {target_file} not found. Run inference first.")
        return

    y_true_full = np.array([row['Actual_Close'] for row in d])
    y_pred_full = np.array([row['Pred_Close'] for row in d])

    # Establish naive persistence (yesterday's Close)
    y_naive = np.roll(y_true_full, 1)[1:]
    y_true = y_true_full[1:]
    y_pred = y_pred_full[1:]

    ann_mae = mean_absolute_error(y_true, y_pred)
    naive_mae = mean_absolute_error(y_true, y_naive)
    ann_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    naive_rmse = np.sqrt(np.mean((y_true - y_naive) ** 2))

    actual_change = y_true - y_naive
    pred_change = y_pred - y_naive
    hit_rate = np.mean(np.sign(actual_change) == np.sign(pred_change))
    ic, p_val = stats.spearmanr(y_pred, y_true)

    print("\n" + "="*55)
    print(" SF METRICS & NAIVE BASELINE REPORT")
    print("="*55)
    print(f"ANN   MAE:  {ann_mae:.6f}  |  Naive MAE:  {naive_mae:.6f}")
    print(f"ANN   RMSE: {ann_rmse:.6f}  |  Naive RMSE: {naive_rmse:.6f}")
    print("-" * 55)
    print(f"Directional Hit Rate:    {hit_rate:.2%}")
    print(f"Spearman Rank IC:        {ic:.4f} (p-val: {p_val:.4g})")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()