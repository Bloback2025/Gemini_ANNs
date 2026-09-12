#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading Sf test data...")
    df = pd.read_csv("Sf_testing_gem.csv")
    
    # Preprocessing sync: strip whitespace and enforce chronological order
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    prices = df['Close'].values

    def evaluate_family(name, interval, steps):
        print(f"\n--- {name} ---")
        lags = [interval * i for i in range(1, steps + 1)]
        
        # 1. Individual Decimated Lags
        for k in lags:
            if k >= len(prices): continue
            y_true = prices[k:]
            y_lag = prices[:-k]
            mae = mean_absolute_error(y_true, y_lag)
            print(f"  Sf1 (Close, t-{k}) MAE: {mae:.6f}")
        
        # 2. Family Combo (Unmixed Ensemble)
        max_lag = max(lags)
        if max_lag < len(prices):
            valid_len = len(prices) - max_lag
            y_true_family = prices[-valid_len:]
            
            ensemble = np.zeros(valid_len)
            for k in lags:
                ensemble += prices[-valid_len - k : len(prices) - k]
            ensemble /= len(lags)
            
            combo_mae = mean_absolute_error(y_true_family, ensemble)
            print(f"\n  Sf1 {interval}-Day Family Combo MAE: {combo_mae:.6f}")

    # Execute the structural families
    evaluate_family("10-Day Intervals", 10, 5)
    evaluate_family("5-Day Intervals", 5, 5)
    evaluate_family("20-Day Intervals", 20, 5)

if __name__ == "__main__":
    main()
    