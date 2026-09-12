import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("hoxnc_testing_retest1.csv")
prices = df["Close"].values

def evaluate_family(name, intervals, max_lag):
    print(f"\n--- {name} ---")
    results = {}
    for interval in intervals:
        for k in range(interval, max_lag + 1, interval):
            if k >= len(prices):
                continue
            y_true = prices[k:]
            y_lag = prices[:-k]
            mae = mean_absolute_error(y_true, y_lag)
            results[f"t-{k}"] = mae
            print(f"  HO (Close, t-{k}) MAE: {mae:.6f}")
    return results

# 1. Individual Decimated Lags & Families
evaluate_family("5-Day Intervals", [5], 25)
evaluate_family("10-Day Intervals", [10], 50)
evaluate_family("20-Day Intervals", [20], 100)

