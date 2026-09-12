import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("hoxnc_testing_retest1.csv")
prices = df["Close"].values

for k in range(1, 16):
    y_true = prices[k:]
    y_lag = prices[:-k]
    mae = mean_absolute_error(y_true, y_lag)
    print(f"HO Lag t-{k}: MAE = {mae:.6f}")
