import sys, pandas as pd
p="/mnt/c/Users/loweb/AI_Financial_Sims/HO/HO_train_phase_ANNs/hoxnc_testing_retest1.csv"
try:
    df=pd.read_csv(p, parse_dates=["Date"])
except Exception as e:
    print("ERROR: cannot read", p, e); sys.exit(2)
req=["Date","Open","High","Low","Close"]
miss=[c for c in req if c not in df.columns]
if miss:
    print("MISSING_COLUMNS:", miss); sys.exit(3)
print("ROWS:", len(df))
print("COLUMNS OK")
if "Ticker" in df.columns:
    groups=df["Ticker"].nunique(); print("TICKERS:", groups)
    if groups>1: print("WARNING: multiple tickers present — group-by needed before shift")
